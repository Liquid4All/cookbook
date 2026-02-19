"""Tests for ocr.extract_text_from_pdf tool."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.extract_text_from_pdf import ExtractTextFromPdf


@pytest.fixture()
def tool() -> ExtractTextFromPdf:
    return ExtractTextFromPdf()


@pytest.fixture()
def sample_pdf(tmp_dir: Path) -> str:
    """Create a simple PDF with text content for testing."""
    from pypdf import PdfWriter

    pdf_path = str(tmp_dir / "sample.pdf")
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # Note: blank pages won't have extractable text; this tests the pipeline
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


async def test_extract_from_pdf(tool: ExtractTextFromPdf, sample_pdf: str) -> None:
    """Should process a PDF file."""
    result = await tool.execute(tool.get_params_model()(path=sample_pdf))

    assert result.success is True
    assert result.data is not None
    assert len(result.data.pages) >= 1
    assert result.data.pages[0].page == 1


async def test_extract_specific_pages(tool: ExtractTextFromPdf, sample_pdf: str) -> None:
    """Should handle page selection."""
    result = await tool.execute(tool.get_params_model()(path=sample_pdf, pages=[1]))

    assert result.success is True
    assert result.data is not None
    assert len(result.data.pages) == 1


async def test_file_not_found(tool: ExtractTextFromPdf, tmp_dir: Path) -> None:
    """Should raise error for missing file."""
    from mcp_base import MCPError

    with pytest.raises(MCPError, match="File not found"):
        await tool.execute(tool.get_params_model()(path=str(tmp_dir / "missing.pdf")))


async def test_reject_non_pdf(tool: ExtractTextFromPdf, tmp_dir: Path) -> None:
    """Should reject non-PDF files."""
    from mcp_base import MCPError

    txt_file = tmp_dir / "test.txt"
    txt_file.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(MCPError, match="must be a PDF"):
        await tool.execute(tool.get_params_model()(path=str(txt_file)))


def test_metadata(tool: ExtractTextFromPdf) -> None:
    """Should have correct metadata."""
    assert tool.name == "ocr.extract_text_from_pdf"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
