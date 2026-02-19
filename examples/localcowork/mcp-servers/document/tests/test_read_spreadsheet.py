"""Tests for document.read_spreadsheet tool."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tools.read_spreadsheet import ReadSpreadsheet


@pytest.fixture()
def tool() -> ReadSpreadsheet:
    return ReadSpreadsheet()


async def test_read_csv(tool: ReadSpreadsheet, tmp_dir: Path) -> None:
    """Should read a CSV file."""
    csv_path = tmp_dir / "data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "age", "city"])
        writer.writerow(["Alice", "30", "NYC"])
        writer.writerow(["Bob", "25", "LA"])

    result = await tool.execute(tool.get_params_model()(path=str(csv_path)))

    assert result.success is True
    assert result.data is not None
    assert result.data.headers == ["name", "age", "city"]
    assert result.data.total_rows == 2
    assert result.data.rows[0]["name"] == "Alice"


async def test_read_tsv(tool: ReadSpreadsheet, tmp_dir: Path) -> None:
    """Should read a TSV file."""
    tsv_path = tmp_dir / "data.tsv"
    tsv_path.write_text("id\tvalue\n1\tfoo\n2\tbar\n", encoding="utf-8")

    result = await tool.execute(tool.get_params_model()(path=str(tsv_path)))

    assert result.success is True
    assert result.data is not None
    assert result.data.headers == ["id", "value"]
    assert result.data.total_rows == 2


async def test_read_xlsx(tool: ReadSpreadsheet, tmp_dir: Path) -> None:
    """Should read an Excel file."""
    from openpyxl import Workbook

    xlsx_path = tmp_dir / "data.xlsx"
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.append(["product", "price"])
    ws.append(["Widget", 9.99])
    ws.append(["Gadget", 24.99])
    wb.save(xlsx_path)

    result = await tool.execute(tool.get_params_model()(path=str(xlsx_path)))

    assert result.success is True
    assert result.data is not None
    assert result.data.headers == ["product", "price"]
    assert result.data.total_rows == 2


async def test_read_file_not_found(tool: ReadSpreadsheet, tmp_dir: Path) -> None:
    """Should raise error for missing file."""
    from mcp_base import MCPError

    with pytest.raises(MCPError, match="File not found"):
        await tool.execute(tool.get_params_model()(path=str(tmp_dir / "nope.csv")))


def test_metadata(tool: ReadSpreadsheet) -> None:
    """Should have correct metadata."""
    assert tool.name == "document.read_spreadsheet"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
