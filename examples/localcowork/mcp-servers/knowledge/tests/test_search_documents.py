"""
Tests for knowledge.search_documents.

Verifies semantic search returns ranked results, respects top_k,
and supports path filtering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.search_documents import SearchDocuments, Params


class TestSearchDocuments:
    """Test the search_documents tool execution."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, indexed_dir: Path) -> None:
        tool = SearchDocuments()
        params = Params(query="project readme", top_k=5)
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert len(result.data.results) > 0

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, indexed_dir: Path) -> None:
        tool = SearchDocuments()
        params = Params(query="roadmap", top_k=2)
        result = await tool.execute(params)

        assert result.data is not None
        assert len(result.data.results) <= 2

    @pytest.mark.asyncio
    async def test_search_results_have_scores(self, indexed_dir: Path) -> None:
        tool = SearchDocuments()
        params = Params(query="meeting notes")
        result = await tool.execute(params)

        assert result.data is not None
        for hit in result.data.results:
            assert isinstance(hit.score, float)
            assert hit.path != ""
            assert hit.chunk_text != ""

    @pytest.mark.asyncio
    async def test_search_results_sorted_descending(self, indexed_dir: Path) -> None:
        tool = SearchDocuments()
        params = Params(query="testing", top_k=10)
        result = await tool.execute(params)

        assert result.data is not None
        scores = [r.score for r in result.data.results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_filter_path(self, indexed_dir: Path) -> None:
        tool = SearchDocuments()
        sub_dir = str(indexed_dir / "sub")
        params = Params(query="subdirectory", filter_path=sub_dir)
        result = await tool.execute(params)

        assert result.data is not None
        for hit in result.data.results:
            assert hit.path.startswith(sub_dir)

    @pytest.mark.asyncio
    async def test_search_empty_index(self) -> None:
        """Searching an empty index should return no results."""
        tool = SearchDocuments()
        params = Params(query="anything")
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert len(result.data.results) == 0

    @pytest.mark.asyncio
    async def test_search_identical_query_deterministic(self, indexed_dir: Path) -> None:
        """Same query should produce identical results (deterministic embeddings)."""
        tool = SearchDocuments()
        params = Params(query="roadmap Q3")

        r1 = await tool.execute(params)
        r2 = await tool.execute(params)

        assert r1.data is not None and r2.data is not None
        assert len(r1.data.results) == len(r2.data.results)
        for a, b in zip(r1.data.results, r2.data.results):
            assert a.score == b.score
            assert a.path == b.path
