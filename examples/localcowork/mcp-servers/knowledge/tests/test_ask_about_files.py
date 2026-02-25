"""
Tests for knowledge.ask_about_files.

Verifies that questions are answered with context chunks and sources,
and that the stub answer includes relevant information.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.ask_about_files import AskAboutFiles, Params


class TestAskAboutFiles:
    """Test the ask_about_files tool execution."""

    @pytest.mark.asyncio
    async def test_ask_returns_answer_and_sources(self, indexed_dir: Path) -> None:
        tool = AskAboutFiles()
        params = Params(question="What was discussed in the meeting?")
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert result.data.answer != ""
        assert len(result.data.sources) > 0

    @pytest.mark.asyncio
    async def test_answer_contains_question(self, indexed_dir: Path) -> None:
        """The stub answer should echo the original question."""
        tool = AskAboutFiles()
        question = "What is the project about?"
        params = Params(question=question)
        result = await tool.execute(params)

        assert result.data is not None
        assert question in result.data.answer

    @pytest.mark.asyncio
    async def test_sources_have_path_and_score(self, indexed_dir: Path) -> None:
        tool = AskAboutFiles()
        params = Params(question="readme sample project")
        result = await tool.execute(params)

        assert result.data is not None
        for src in result.data.sources:
            assert src.path != ""
            assert isinstance(src.score, float)
            assert src.chunk_text != ""

    @pytest.mark.asyncio
    async def test_context_docs_limits_sources(self, indexed_dir: Path) -> None:
        tool = AskAboutFiles()
        params = Params(question="anything", context_docs=2)
        result = await tool.execute(params)

        assert result.data is not None
        assert len(result.data.sources) <= 2

    @pytest.mark.asyncio
    async def test_ask_empty_index(self) -> None:
        """With no indexed docs, the answer should indicate nothing was found."""
        tool = AskAboutFiles()
        params = Params(question="what is this?")
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert "No indexed documents" in result.data.answer
        assert len(result.data.sources) == 0

    @pytest.mark.asyncio
    async def test_sources_sorted_by_score(self, indexed_dir: Path) -> None:
        tool = AskAboutFiles()
        params = Params(question="testing knowledge server", context_docs=10)
        result = await tool.execute(params)

        assert result.data is not None
        scores = [s.score for s in result.data.sources]
        assert scores == sorted(scores, reverse=True)
