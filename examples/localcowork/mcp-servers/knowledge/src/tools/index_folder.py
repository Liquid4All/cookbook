"""
knowledge.index_folder — Index a folder of documents for semantic search.

Reads all files in a directory (optionally filtered by extension),
splits their text into chunks, generates mock embeddings, and stores
everything in the local SQLite database.

Non-destructive / read-only from the filesystem perspective.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# ─── Shared base import ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "_shared", "py"))

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes  # noqa: E402

from db import get_db  # noqa: E402
from embeddings import generate_embedding, serialize_embedding  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_CHUNK_CHARS: int = 500
DEFAULT_FILE_TYPES: list[str] = [".txt", ".md", ".py", ".ts", ".js", ".html", ".css", ".json"]


# ─── Params / Result ─────────────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for knowledge.index_folder."""

    path: str = Field(description="Absolute path to the folder to index")
    recursive: bool = Field(default=True, description="Recurse into subdirectories")
    file_types: list[str] | None = Field(
        default=None,
        description="File extensions to include (e.g. ['.txt', '.md']). Defaults to common text types.",
    )


class Result(BaseModel):
    """Return value for knowledge.index_folder."""

    documents_indexed: int
    chunks_created: int


# ─── Tool ─────────────────────────────────────────────────────────────────────


class IndexFolder(MCPTool[Params, Result]):
    """Index a folder of documents for semantic search."""

    name = "knowledge.index_folder"
    description = "Index a folder of documents for semantic search"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Scan *params.path*, chunk each file, and store embeddings."""
        folder = Path(params.path)
        if not folder.is_dir():
            raise MCPError(
                ErrorCodes.FILE_NOT_FOUND,
                f"Directory not found: {params.path}",
            )

        extensions = {e.lower() for e in (params.file_types or DEFAULT_FILE_TYPES)}
        files = _collect_files(folder, extensions, params.recursive)

        db = get_db()
        documents_indexed = 0
        chunks_created = 0

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            abs_path = str(file_path.resolve())

            # Upsert document
            existing = db.execute(
                "SELECT id, file_hash FROM documents WHERE path = ?", (abs_path,)
            ).fetchone()

            if existing and existing["file_hash"] == file_hash:
                continue  # unchanged

            if existing:
                doc_id: int = existing["id"]
                db.execute(
                    "UPDATE documents SET content=?, file_hash=?, indexed_at=datetime('now') "
                    "WHERE id=?",
                    (text, file_hash, doc_id),
                )
                db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
            else:
                cursor = db.execute(
                    "INSERT INTO documents (path, filename, content, file_hash) VALUES (?,?,?,?)",
                    (abs_path, file_path.name, text, file_hash),
                )
                doc_id = cursor.lastrowid  # type: ignore[assignment]

            # Chunk and embed
            text_chunks = chunk_text(text)
            for idx, chunk in enumerate(text_chunks):
                emb = generate_embedding(chunk)
                emb_bytes = serialize_embedding(emb)
                db.execute(
                    "INSERT INTO chunks (document_id, content, chunk_index, embedding) "
                    "VALUES (?,?,?,?)",
                    (doc_id, chunk, idx, emb_bytes),
                )
                chunks_created += 1

            documents_indexed += 1

        db.commit()
        return MCPResult(success=True, data=Result(
            documents_indexed=documents_indexed,
            chunks_created=chunks_created,
        ))


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _collect_files(
    folder: Path, extensions: set[str], recursive: bool
) -> list[Path]:
    """Return all files under *folder* whose suffix is in *extensions*."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    )


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters.

    Splitting strategy: split on double newlines first (paragraphs),
    then merge small paragraphs and split large ones.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            # Flush current buffer first
            if current:
                chunks.append(current)
                current = ""
            # Split oversized paragraph by sentences / hard limit
            for sub in _split_large(para, max_chars):
                chunks.append(sub)
        elif len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]] if text else []


def _split_large(text: str, max_chars: int) -> list[str]:
    """Force-split text that exceeds *max_chars*."""
    parts: list[str] = []
    while len(text) > max_chars:
        # Try to split at a newline or space near the limit
        split_at = text.rfind("\n", 0, max_chars)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        parts.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        parts.append(text)
    return parts
