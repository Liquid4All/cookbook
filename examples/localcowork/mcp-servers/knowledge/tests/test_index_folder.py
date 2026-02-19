"""
Tests for knowledge.index_folder.

Verifies folder scanning, chunking, embedding storage, and
idempotent re-indexing behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from db import get_db
from tools.index_folder import IndexFolder, Params, chunk_text


# ─── Unit: chunk_text ─────────────────────────────────────────────────────────


class TestChunkText:
    """Verify the paragraph-based chunking logic."""

    def test_single_short_paragraph(self) -> None:
        chunks = chunk_text("Hello world")
        assert chunks == ["Hello world"]

    def test_multiple_paragraphs(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, max_chars=500)
        assert len(chunks) == 1  # all fit in one chunk
        assert "Para one." in chunks[0]

    def test_respects_max_chars(self) -> None:
        text = "A" * 600
        chunks = chunk_text(text, max_chars=500)
        assert all(len(c) <= 500 for c in chunks)
        assert len(chunks) >= 2

    def test_empty_string(self) -> None:
        chunks = chunk_text("")
        assert chunks == []

    def test_splits_on_paragraph_boundary(self) -> None:
        para_a = "Short paragraph A."
        para_b = "Short paragraph B."
        text = f"{para_a}\n\n{para_b}"
        # With a tiny limit, each para should be its own chunk
        chunks = chunk_text(text, max_chars=25)
        assert len(chunks) == 2


# ─── Integration: IndexFolder tool ──────────────────────────────────────────


class TestIndexFolder:
    """Test the full index_folder tool execution."""

    @pytest.mark.asyncio
    async def test_index_sample_dir(self, sample_dir: Path) -> None:
        tool = IndexFolder()
        params = Params(path=str(sample_dir), recursive=True)
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert result.data.documents_indexed == 3
        assert result.data.chunks_created > 0

    @pytest.mark.asyncio
    async def test_index_non_recursive(self, sample_dir: Path) -> None:
        tool = IndexFolder()
        params = Params(path=str(sample_dir), recursive=False)
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        # Only top-level files (readme.md, notes.txt), not sub/deep.txt
        assert result.data.documents_indexed == 2

    @pytest.mark.asyncio
    async def test_index_with_file_type_filter(self, sample_dir: Path) -> None:
        tool = IndexFolder()
        params = Params(path=str(sample_dir), file_types=[".md"])
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert result.data.documents_indexed == 1  # only readme.md

    @pytest.mark.asyncio
    async def test_index_idempotent(self, sample_dir: Path) -> None:
        """Running index twice on unchanged files should index 0 the second time."""
        tool = IndexFolder()
        params = Params(path=str(sample_dir))

        r1 = await tool.execute(params)
        assert r1.data is not None
        assert r1.data.documents_indexed == 3

        r2 = await tool.execute(params)
        assert r2.data is not None
        assert r2.data.documents_indexed == 0  # already indexed

    @pytest.mark.asyncio
    async def test_index_stores_in_db(self, sample_dir: Path) -> None:
        tool = IndexFolder()
        params = Params(path=str(sample_dir))
        await tool.execute(params)

        db = get_db()
        doc_count = db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunk_count = db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

        assert doc_count == 3
        assert chunk_count > 0

    @pytest.mark.asyncio
    async def test_index_nonexistent_dir(self) -> None:
        tool = IndexFolder()
        params = Params(path="/nonexistent/path/abc123")

        with pytest.raises(Exception, match="Directory not found"):
            await tool.execute(params)
