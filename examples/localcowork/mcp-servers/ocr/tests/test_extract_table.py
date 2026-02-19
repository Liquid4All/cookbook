"""Tests for ocr.extract_table tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.extract_table import ExtractTable


@pytest.fixture()
def tool() -> ExtractTable:
    return ExtractTable()


async def test_extract_csv_table(tool: ExtractTable, tmp_dir: Path) -> None:
    """Should extract table from a CSV file."""
    csv_path = tmp_dir / "data.csv"
    csv_path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\n", encoding="utf-8")

    result = await tool.execute(tool.get_params_model()(path=str(csv_path)))

    assert result.success is True
    assert result.data is not None
    assert result.data.headers == ["name", "age", "city"]
    assert len(result.data.rows) == 2
    assert result.data.rows[0] == ["Alice", "30", "NYC"]


async def test_extract_tsv_table(tool: ExtractTable, tmp_dir: Path) -> None:
    """Should extract table from a TSV file."""
    tsv_path = tmp_dir / "data.tsv"
    tsv_path.write_text("id\tvalue\n1\tfoo\n2\tbar\n", encoding="utf-8")

    result = await tool.execute(tool.get_params_model()(path=str(tsv_path)))

    assert result.success is True
    assert result.data is not None
    assert result.data.headers == ["id", "value"]
    assert len(result.data.rows) == 2


async def test_extract_pipe_table(tool: ExtractTable, tmp_dir: Path) -> None:
    """Should extract table from pipe-delimited text."""
    # Write as a generic text file with pipe-delimited content
    txt_path = tmp_dir / "table.csv"
    txt_path.write_text("name | score | grade\nAlice | 95 | A\nBob | 82 | B\n", encoding="utf-8")

    result = await tool.execute(tool.get_params_model()(path=str(txt_path)))

    assert result.success is True
    assert result.data is not None
    assert len(result.data.headers) >= 2


async def test_extract_from_pdf(tool: ExtractTable, tmp_dir: Path) -> None:
    """Should handle PDF table extraction."""
    from pypdf import PdfWriter

    pdf_path = str(tmp_dir / "table.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    result = await tool.execute(tool.get_params_model()(path=pdf_path, page=1))

    assert result.success is True
    assert result.data is not None


async def test_file_not_found(tool: ExtractTable, tmp_dir: Path) -> None:
    """Should raise error for missing file."""
    from mcp_base import MCPError

    with pytest.raises(MCPError, match="File not found"):
        await tool.execute(tool.get_params_model()(path=str(tmp_dir / "missing.csv")))


def test_metadata(tool: ExtractTable) -> None:
    """Should have correct metadata."""
    assert tool.name == "ocr.extract_table"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
