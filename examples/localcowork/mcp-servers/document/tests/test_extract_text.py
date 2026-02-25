"""Tests for document.extract_text tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.extract_text import ExtractText


@pytest.fixture()
def tool() -> ExtractText:
    return ExtractText()


async def test_extract_text_from_txt(tool: ExtractText, tmp_dir: Path) -> None:
    """Should extract text from a .txt file."""
    txt_file = tmp_dir / "sample.txt"
    txt_file.write_text("Hello, world!\nSecond line.", encoding="utf-8")

    result = await tool.execute(tool.get_params_model()(path=str(txt_file)))

    assert result.success is True
    assert result.data is not None
    assert "Hello, world!" in result.data.text
    assert result.data.format == "txt"


async def test_extract_text_from_md(tool: ExtractText, tmp_dir: Path) -> None:
    """Should extract text from a .md file."""
    md_file = tmp_dir / "readme.md"
    md_file.write_text("# Title\n\nSome content.", encoding="utf-8")

    result = await tool.execute(tool.get_params_model()(path=str(md_file)))

    assert result.success is True
    assert result.data is not None
    assert "Title" in result.data.text
    assert result.data.format == "md"


async def test_extract_text_from_html(tool: ExtractText, tmp_dir: Path) -> None:
    """Should strip tags from HTML."""
    html_file = tmp_dir / "page.html"
    html_file.write_text(
        "<html><body><h1>Title</h1><p>Content here.</p></body></html>",
        encoding="utf-8",
    )

    result = await tool.execute(tool.get_params_model()(path=str(html_file)))

    assert result.success is True
    assert result.data is not None
    assert "Title" in result.data.text
    assert "Content here" in result.data.text
    assert "<h1>" not in result.data.text


async def test_extract_text_file_not_found(tool: ExtractText, tmp_dir: Path) -> None:
    """Should raise error for missing file."""
    from mcp_base import MCPError

    with pytest.raises(MCPError, match="File not found"):
        await tool.execute(tool.get_params_model()(path=str(tmp_dir / "nope.txt")))


def test_metadata(tool: ExtractText) -> None:
    """Should have correct metadata."""
    assert tool.name == "document.extract_text"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
