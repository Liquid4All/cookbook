"""
knowledge.update_index — Update the index for changed or new files.

Scans the filesystem under *path*, compares file hashes against the
stored values, and adds / updates / removes documents as needed.

Non-destructive from the filesystem perspective (only the DB changes).
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
from tools.index_folder import chunk_text  # noqa: E402

# ─── Params / Result ─────────────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for knowledge.update_index."""

    path: str = Field(description="Absolute path to re-scan")


class Result(BaseModel):
    """Return value for knowledge.update_index."""

    added: int
    updated: int
    removed: int


# ─── Tool ─────────────────────────────────────────────────────────────────────


class UpdateIndex(MCPTool[Params, Result]):
    """Update the index for changed, new, or removed files."""

    name = "knowledge.update_index"
    description = "Update index for changed/new files"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Re-scan *params.path* and reconcile with the DB."""
        scan_root = Path(params.path)
        if not scan_root.exists():
            raise MCPError(
                ErrorCodes.FILE_NOT_FOUND,
                f"Path not found: {params.path}",
            )

        db = get_db()

        # Normalise to directory or single file
        if scan_root.is_file():
            disk_files = {str(scan_root.resolve()): scan_root}
        else:
            disk_files = {
                str(p.resolve()): p
                for p in scan_root.rglob("*")
                if p.is_file()
            }

        # Fetch indexed documents under this path prefix
        prefix = str(scan_root.resolve())
        indexed_rows = db.execute(
            "SELECT id, path, file_hash FROM documents WHERE path LIKE ? || '%'",
            (prefix,),
        ).fetchall()

        indexed_map: dict[str, tuple[int, str]] = {
            row["path"]: (row["id"], row["file_hash"]) for row in indexed_rows
        }

        added = 0
        updated = 0
        removed = 0

        # Add / update pass
        for abs_path, file_path in disk_files.items():
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            if abs_path in indexed_map:
                doc_id, old_hash = indexed_map[abs_path]
                if old_hash == file_hash:
                    continue  # unchanged
                # File changed — re-index
                db.execute(
                    "UPDATE documents SET content=?, file_hash=?, "
                    "indexed_at=datetime('now') WHERE id=?",
                    (text, file_hash, doc_id),
                )
                db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                _insert_chunks(db, doc_id, text)
                updated += 1
            else:
                # New file
                cursor = db.execute(
                    "INSERT INTO documents (path, filename, content, file_hash) "
                    "VALUES (?,?,?,?)",
                    (abs_path, file_path.name, text, file_hash),
                )
                doc_id = cursor.lastrowid  # type: ignore[assignment]
                _insert_chunks(db, doc_id, text)
                added += 1

        # Remove pass — documents that no longer exist on disk
        for abs_path, (doc_id, _) in indexed_map.items():
            if abs_path not in disk_files:
                db.execute("DELETE FROM chunks WHERE document_id=?", (doc_id,))
                db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
                removed += 1

        db.commit()

        return MCPResult(success=True, data=Result(
            added=added,
            updated=updated,
            removed=removed,
        ))


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _insert_chunks(
    db: "sqlite3.Connection",  # noqa: F821  # forward ref for type hint
    doc_id: int,
    text: str,
) -> None:
    """Chunk *text*, embed each chunk, and insert into the chunks table."""
    for idx, chunk in enumerate(chunk_text(text)):
        emb = generate_embedding(chunk)
        emb_bytes = serialize_embedding(emb)
        db.execute(
            "INSERT INTO chunks (document_id, content, chunk_index, embedding) "
            "VALUES (?,?,?,?)",
            (doc_id, chunk, idx, emb_bytes),
        )
