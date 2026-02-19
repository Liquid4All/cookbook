"""
knowledge.ask_about_files — Ask a question grounded in indexed documents.

Retrieves the most relevant chunks for the question, concatenates them
as supporting context, and returns a stub "answer" alongside source
references.  A real LLM integration will replace the answer generation
in a later workstream.

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


class Source(BaseModel):
    """A source document used to ground the answer."""

    path: str
    chunk_text: str
    score: float


class Params(BaseModel):
    """Parameters for knowledge.ask_about_files."""

    question: str = Field(description="The question to answer")
    context_docs: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )


class Result(BaseModel):
    """Return value for knowledge.ask_about_files."""

    answer: str
    sources: list[Source]


# ─── Tool ─────────────────────────────────────────────────────────────────────


class AskAboutFiles(MCPTool[Params, Result]):
    """Ask a question and get an answer grounded in indexed documents."""

    name = "knowledge.ask_about_files"
    description = "Ask a question and get an answer grounded in indexed documents"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Retrieve context chunks and synthesise an answer stub."""
        try:
            query_emb = generate_embedding(params.question)
            sources = _retrieve_sources(query_emb, params.context_docs)

            if not sources:
                answer = (
                    "No indexed documents matched the question. "
                    "Try indexing a folder first with knowledge.index_folder."
                )
            else:
                answer = _build_answer_stub(params.question, sources)

            return MCPResult(success=True, data=Result(
                answer=answer,
                sources=sources,
            ))
        except Exception as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR, f"ask_about_files failed: {e}"
            ) from e


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _retrieve_sources(
    query_emb: list[float], top_k: int
) -> list[Source]:
    """Return top-k chunks ranked by cosine similarity."""
    db = get_db()
    rows = db.execute(
        """
        SELECT c.content, c.embedding, d.path
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        """
    ).fetchall()

    scored: list[Source] = []
    for row in rows:
        if row["embedding"] is None:
            continue
        chunk_emb = deserialize_embedding(row["embedding"])
        score = cosine_similarity(query_emb, chunk_emb)
        scored.append(Source(
            path=row["path"],
            chunk_text=row["content"],
            score=round(score, 6),
        ))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]


def _build_answer_stub(question: str, sources: list[Source]) -> str:
    """
    Build a placeholder answer from retrieved context.

    This concatenates the source chunks under a header.  In a future
    workstream the real LLM will generate a grounded answer instead.
    """
    lines: list[str] = [
        f"Question: {question}",
        "",
        "Based on the following indexed documents:",
        "",
    ]
    for idx, src in enumerate(sources, start=1):
        lines.append(f"[{idx}] ({src.path})")
        lines.append(src.chunk_text)
        lines.append("")

    return "\n".join(lines)
