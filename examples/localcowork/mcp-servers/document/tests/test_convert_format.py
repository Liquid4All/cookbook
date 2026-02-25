"""Tests for document.convert_format tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.convert_format import ConvertFormat


@pytest.fixture()
def tool() -> ConvertFormat:
    return ConvertFormat()


async def test_convert_txt_to_md(tool: ConvertFormat, tmp_dir: Path) -> None:
    """Should convert txt to md (copy)."""
    src = tmp_dir / "source.txt"
    src.write_text("Hello world", encoding="utf-8")
    out = str(tmp_dir / "output.md")

    result = await tool.execute(
        tool.get_params_model()(path=str(src), target_format="md", output_path=out)
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.path == out
    assert Path(out).read_text(encoding="utf-8") == "Hello world"


async def test_convert_md_to_html(tool: ConvertFormat, tmp_dir: Path) -> None:
    """Should convert markdown to HTML."""
    src = tmp_dir / "source.md"
    src.write_text("# Title\n\nContent", encoding="utf-8")
    out = str(tmp_dir / "output.html")

    result = await tool.execute(
        tool.get_params_model()(path=str(src), target_format="html", output_path=out)
    )

    assert result.success is True
    content = Path(out).read_text(encoding="utf-8")
    assert "<h1>" in content or "Title" in content


async def test_convert_txt_to_docx(tool: ConvertFormat, tmp_dir: Path) -> None:
    """Should convert txt to DOCX."""
    src = tmp_dir / "source.txt"
    src.write_text("Line one\nLine two", encoding="utf-8")
    out = str(tmp_dir / "output.docx")

    result = await tool.execute(
        tool.get_params_model()(path=str(src), target_format="docx", output_path=out)
    )

    assert result.success is True
    assert Path(out).exists()

    from docx import Document

    doc = Document(out)
    texts = [p.text for p in doc.paragraphs]
    assert "Line one" in texts


async def test_convert_auto_output_path(tool: ConvertFormat, tmp_dir: Path) -> None:
    """Should auto-generate output path when not provided."""
    src = tmp_dir / "file.txt"
    src.write_text("content", encoding="utf-8")

    result = await tool.execute(
        tool.get_params_model()(path=str(src), target_format="md")
    )

    assert result.success is True
    assert result.data is not None
    assert result.data.path.endswith(".md")


async def test_convert_unsupported_format(tool: ConvertFormat, tmp_dir: Path) -> None:
    """Should reject unsupported target format."""
    from mcp_base import MCPError

    src = tmp_dir / "file.txt"
    src.write_text("content", encoding="utf-8")

    with pytest.raises(MCPError, match="Unsupported target format"):
        await tool.execute(
            tool.get_params_model()(
                path=str(src), target_format="xlsx", output_path=str(tmp_dir / "out.xlsx")
            )
        )


def test_metadata(tool: ConvertFormat) -> None:
    """Should have correct metadata."""
    assert tool.name == "document.convert_format"
    assert tool.confirmation_required is True
    assert tool.undo_supported is False
