"""
knowledge.get_related_chunks — Find chunks related to a text passage.

Generates an embedding for the input text and returns the most similar
indexed chunks via cosine similarity.

Non-destructive: no confirmation required.
"""

from __future__ import annotations

import os
import sys

from pydantic import BaseModel, Field

# ─── Shared base import ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "_shared", "py"))

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes  # noqa: E402

from db import get_db  # noqa: E402
from embeddings import (  # noqa: E402
    cosine_similarity,
    deserialize_embedding,
    generate_embedding,
)

# ─── Params / Result ─────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """A related chunk result."""

    path: str
    chunk_text: str
    chunk_index: int
    score: float


class Params(BaseModel):
    """Parameters for knowledge.get_related_chunks."""

    text: str = Field(description="Text passage to find related chunks for")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to return")


class Result(BaseModel):
    """Return value for knowledge.get_related_chunks."""

    chunks: list[Chunk]


# ─── Tool ─────────────────────────────────────────────────────────────────────


class GetRelatedChunks(MCPTool[Params, Result]):
    """Find chunks related to a specific document or passage."""

    name = "knowledge.get_related_chunks"
    description = "Find chunks related to a specific document or passage"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Find chunks similar to *params.text*."""
        try:
            text_emb = generate_embedding(params.text)
            chunks = _find_related(text_emb, params.top_k)
            return MCPResult(success=True, data=Result(chunks=chunks))
        except Exception as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR, f"get_related_chunks failed: {e}"
            ) from e


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _find_related(query_emb: list[float], top_k: int) -> list[Chunk]:
    """Score all indexed chunks and return the top-k by similarity."""
    db = get_db()

    rows = db.execute(
        """
        SELECT c.content, c.chunk_index, c.embedding, d.path
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        """
    ).fetchall()

    scored: list[Chunk] = []
    for row in rows:
        if row["embedding"] is None:
            continue
        chunk_emb = deserialize_embedding(row["embedding"])
        score = cosine_similarity(query_emb, chunk_emb)
        scored.append(
            Chunk(
                path=row["path"],
                chunk_text=row["content"],
                chunk_index=row["chunk_index"],
                score=round(score, 6),
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]
