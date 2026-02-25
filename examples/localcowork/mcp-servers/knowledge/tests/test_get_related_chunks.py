"""
Tests for knowledge.get_related_chunks.

Verifies that related chunks are found, properly ranked, and that
the tool handles edge cases (empty index, varying top_k).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.get_related_chunks import GetRelatedChunks, Params


class TestGetRelatedChunks:
    """Test the get_related_chunks tool execution."""

    @pytest.mark.asyncio
    async def test_returns_chunks(self, indexed_dir: Path) -> None:
        tool = GetRelatedChunks()
        params = Params(text="project readme documentation")
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert len(result.data.chunks) > 0

    @pytest.mark.asyncio
    async def test_chunks_have_required_fields(self, indexed_dir: Path) -> None:
        tool = GetRelatedChunks()
        params = Params(text="meeting notes")
        result = await tool.execute(params)

        assert result.data is not None
        for chunk in result.data.chunks:
            assert chunk.path != ""
            assert chunk.chunk_text != ""
            assert isinstance(chunk.chunk_index, int)
            assert isinstance(chunk.score, float)

    @pytest.mark.asyncio
    async def test_respects_top_k(self, indexed_dir: Path) -> None:
        tool = GetRelatedChunks()
        params = Params(text="anything", top_k=2)
        result = await tool.execute(params)

        assert result.data is not None
        assert len(result.data.chunks) <= 2

    @pytest.mark.asyncio
    async def test_sorted_by_score_descending(self, indexed_dir: Path) -> None:
        tool = GetRelatedChunks()
        params = Params(text="subdirectory recursive", top_k=10)
        result = await tool.execute(params)

        assert result.data is not None
        scores = [c.score for c in result.data.chunks]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_index(self) -> None:
        """With no indexed documents, should return an empty list."""
        tool = GetRelatedChunks()
        params = Params(text="something")
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert len(result.data.chunks) == 0

    @pytest.mark.asyncio
    async def test_deterministic(self, indexed_dir: Path) -> None:
        """Same text should produce identical results."""
        tool = GetRelatedChunks()
        params = Params(text="Q3 roadmap action items")

        r1 = await tool.execute(params)
        r2 = await tool.execute(params)

        assert r1.data is not None and r2.data is not None
        assert len(r1.data.chunks) == len(r2.data.chunks)
        for a, b in zip(r1.data.chunks, r2.data.chunks):
            assert a.score == b.score
            assert a.path == b.path

    @pytest.mark.asyncio
    async def test_exact_text_match_scores_highest(self, indexed_dir: Path) -> None:
        """A chunk's own text should be its best match (score == 1.0)."""
        # Read a chunk directly from the DB
        from db import get_db

        db = get_db()
        row = db.execute("SELECT content FROM chunks LIMIT 1").fetchone()
        assert row is not None
        exact_text: str = row["content"]

        tool = GetRelatedChunks()
        params = Params(text=exact_text, top_k=1)
        result = await tool.execute(params)

        assert result.data is not None
        assert len(result.data.chunks) == 1
        # Cosine similarity of a normalised vector with itself is 1.0
        assert result.data.chunks[0].score == pytest.approx(1.0, abs=1e-4)
