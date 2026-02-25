"""Tests for document.diff_documents tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.diff_documents import DiffDocuments


@pytest.fixture()
def tool() -> DiffDocuments:
    return DiffDocuments()


async def test_diff_identical_files(tool: DiffDocuments, tmp_dir: Path) -> None:
    """Should return no changes for identical files."""
    f = tmp_dir / "same.txt"
    f.write_text("Hello world.\n\nParagraph two.", encoding="utf-8")

    result = await tool.execute(
        tool.get_params_model()(path_a=str(f), path_b=str(f), granularity="paragraph")
    )

    assert result.success is True
    assert result.data is not None
    assert len(result.data.changes) == 0


async def test_diff_paragraph_level(tool: DiffDocuments, tmp_dir: Path) -> None:
    """Should detect paragraph-level changes."""
    file_a = tmp_dir / "a.txt"
    file_b = tmp_dir / "b.txt"
    file_a.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    file_b.write_text("First paragraph.\n\nModified paragraph.\n\nNew paragraph.", encoding="utf-8")

    result = await tool.execute(
        tool.get_params_model()(path_a=str(file_a), path_b=str(file_b), granularity="paragraph")
    )

    assert result.success is True
    assert result.data is not None
    assert len(result.data.changes) > 0
    assert "added" in result.data.summary or "removed" in result.data.summary


async def test_diff_word_level(tool: DiffDocuments, tmp_dir: Path) -> None:
    """Should detect word-level changes."""
    file_a = tmp_dir / "wa.txt"
    file_b = tmp_dir / "wb.txt"
    file_a.write_text("the quick brown fox", encoding="utf-8")
    file_b.write_text("the slow brown fox", encoding="utf-8")

    result = await tool.execute(
        tool.get_params_model()(path_a=str(file_a), path_b=str(file_b), granularity="word")
    )

    assert result.success is True
    assert result.data is not None
    changes = result.data.changes
    # "quick" removed, "slow" added
    removed = [c for c in changes if c.type == "removed"]
    added = [c for c in changes if c.type == "added"]
    assert len(removed) >= 1
    assert len(added) >= 1


async def test_diff_file_not_found(tool: DiffDocuments, tmp_dir: Path) -> None:
    """Should raise error for missing file."""
    from mcp_base import MCPError

    real_file = tmp_dir / "real.txt"
    real_file.write_text("content", encoding="utf-8")

    with pytest.raises(MCPError, match="File not found"):
        await tool.execute(
            tool.get_params_model()(
                path_a=str(real_file), path_b=str(tmp_dir / "missing.txt"), granularity="paragraph"
            )
        )


def test_metadata(tool: DiffDocuments) -> None:
    """Should have correct metadata."""
    assert tool.name == "document.diff_documents"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
