"""
knowledge.search_documents — Semantic search across indexed documents.

Generates an embedding for the query, computes cosine similarity against
every stored chunk, and returns the top-k results ordered by score.

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


class SearchResult(BaseModel):
    """A single search hit."""

    path: str
    chunk_text: str
    score: float
    chunk_index: int


class Params(BaseModel):
    """Parameters for knowledge.search_documents."""

    query: str = Field(description="Semantic search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    filter_path: str | None = Field(
        default=None,
        description="Only return results whose document path starts with this prefix",
    )


class Result(BaseModel):
    """Return value for knowledge.search_documents."""

    results: list[SearchResult]


# ─── Tool ─────────────────────────────────────────────────────────────────────


class SearchDocuments(MCPTool[Params, Result]):
    """Semantic search across indexed documents."""

    name = "knowledge.search_documents"
    description = "Semantic search across indexed documents"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Search chunks by cosine similarity to *params.query*."""
        try:
            query_emb = generate_embedding(params.query)
            scored = _score_chunks(query_emb, params.filter_path)

            # Sort descending by score, take top_k
            scored.sort(key=lambda r: r.score, reverse=True)
            top = scored[: params.top_k]

            return MCPResult(success=True, data=Result(results=top))
        except Exception as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR, f"Search failed: {e}"
            ) from e


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _score_chunks(
    query_emb: list[float], filter_path: str | None
) -> list[SearchResult]:
    """Compute similarity for every chunk (optionally path-filtered)."""
    db = get_db()

    if filter_path:
        rows = db.execute(
            """
            SELECT c.content, c.chunk_index, c.embedding, d.path
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.path LIKE ? || '%'
            """,
            (filter_path,),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT c.content, c.chunk_index, c.embedding, d.path
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()

    results: list[SearchResult] = []
    for row in rows:
        if row["embedding"] is None:
            continue
        chunk_emb = deserialize_embedding(row["embedding"])
        score = cosine_similarity(query_emb, chunk_emb)
        results.append(
            SearchResult(
                path=row["path"],
                chunk_text=row["content"],
                score=round(score, 6),
                chunk_index=row["chunk_index"],
            )
        )

    return results
