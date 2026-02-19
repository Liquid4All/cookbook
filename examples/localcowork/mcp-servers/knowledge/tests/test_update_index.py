"""
Tests for knowledge.update_index.

Verifies detection of added, modified, and removed files,
and correct reconciliation of the index.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from db import get_db
from tools.index_folder import IndexFolder
from tools.index_folder import Params as IndexParams
from tools.update_index import UpdateIndex, Params


class TestUpdateIndex:
    """Test the update_index tool execution."""

    @pytest.mark.asyncio
    async def test_no_changes(self, indexed_dir: Path) -> None:
        """When nothing changed, all counts should be zero."""
        tool = UpdateIndex()
        params = Params(path=str(indexed_dir))
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert result.data.added == 0
        assert result.data.updated == 0
        assert result.data.removed == 0

    @pytest.mark.asyncio
    async def test_detects_new_file(self, indexed_dir: Path) -> None:
        # Add a new file after initial indexing
        (indexed_dir / "new_file.txt").write_text("Brand new content here.")

        tool = UpdateIndex()
        params = Params(path=str(indexed_dir))
        result = await tool.execute(params)

        assert result.data is not None
        assert result.data.added == 1
        assert result.data.updated == 0
        assert result.data.removed == 0

    @pytest.mark.asyncio
    async def test_detects_modified_file(self, indexed_dir: Path) -> None:
        # Modify an existing file
        readme = indexed_dir / "readme.md"
        readme.write_text("Completely rewritten content.")

        tool = UpdateIndex()
        params = Params(path=str(indexed_dir))
        result = await tool.execute(params)

        assert result.data is not None
        assert result.data.updated == 1

    @pytest.mark.asyncio
    async def test_detects_removed_file(self, indexed_dir: Path) -> None:
        # Delete an existing file
        (indexed_dir / "notes.txt").unlink()

        tool = UpdateIndex()
        params = Params(path=str(indexed_dir))
        result = await tool.execute(params)

        assert result.data is not None
        assert result.data.removed == 1

    @pytest.mark.asyncio
    async def test_combined_add_update_remove(self, indexed_dir: Path) -> None:
        """Test all three operations in a single update call."""
        # Add
        (indexed_dir / "added.txt").write_text("I am new.")
        # Update
        (indexed_dir / "readme.md").write_text("Updated readme.")
        # Remove
        (indexed_dir / "notes.txt").unlink()

        tool = UpdateIndex()
        params = Params(path=str(indexed_dir))
        result = await tool.execute(params)

        assert result.data is not None
        assert result.data.added >= 1
        assert result.data.updated >= 1
        assert result.data.removed >= 1

    @pytest.mark.asyncio
    async def test_update_refreshes_chunks(self, indexed_dir: Path) -> None:
        """When a file is updated, its chunks should reflect the new content."""
        readme = indexed_dir / "readme.md"
        abs_path = str(readme.resolve())
        readme.write_text("Completely unique replacement text for testing.")

        tool = UpdateIndex()
        params = Params(path=str(indexed_dir))
        await tool.execute(params)

        db = get_db()
        row = db.execute(
            "SELECT id FROM documents WHERE path = ?", (abs_path,)
        ).fetchone()
        assert row is not None

        chunks = db.execute(
            "SELECT content FROM chunks WHERE document_id = ?", (row["id"],)
        ).fetchall()
        chunk_texts = [c["content"] for c in chunks]
        assert any("unique replacement" in t for t in chunk_texts)

    @pytest.mark.asyncio
    async def test_update_nonexistent_path(self) -> None:
        tool = UpdateIndex()
        params = Params(path="/nonexistent/xyz")

        with pytest.raises(Exception, match="Path not found"):
            await tool.execute(params)

    @pytest.mark.asyncio
    async def test_update_single_file(self, sample_dir: Path) -> None:
        """update_index should work when pointed at a single file."""
        file_path = sample_dir / "notes.txt"

        tool = UpdateIndex()
        params = Params(path=str(file_path))
        result = await tool.execute(params)

        assert result.data is not None
        assert result.data.added == 1
