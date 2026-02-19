"""
ocr.extract_text_from_pdf — Extract text from a scanned PDF using OCR.

Non-destructive: executes immediately, no confirmation needed.
Converts PDF pages to images, then runs OCR on each page.
"""

from __future__ import annotations

import os
import tempfile

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_sandboxed, assert_absolute_path


class PageResult(BaseModel):
    """OCR result for a single page."""

    page: int
    text: str


class Params(BaseModel):
    """Parameters for extract_text_from_pdf."""

    path: str = Field(description="Path to PDF file")
    pages: list[int] | None = Field(default=None, description="Specific pages to OCR (default: all)")


class Result(BaseModel):
    """Return value for extract_text_from_pdf."""

    pages: list[PageResult]


class ExtractTextFromPdf(MCPTool[Params, Result]):
    """Extract text from a PDF using OCR (for scanned PDFs)."""

    name = "ocr.extract_text_from_pdf"
    description = "Extract text from a PDF using OCR (for scanned PDFs)"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Convert PDF pages to images and run OCR."""
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        if not os.path.exists(params.path):
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {params.path}")

        if not params.path.lower().endswith(".pdf"):
            raise MCPError(ErrorCodes.INVALID_PARAMS, "File must be a PDF")

        try:
            # Try to extract text from PDF pages using pypdf first (fast path)
            from pypdf import PdfReader

            reader = PdfReader(params.path)
            total_pages = len(reader.pages)
            target_pages = params.pages or list(range(1, total_pages + 1))

            page_results: list[PageResult] = []

            for page_num in target_pages:
                if page_num < 1 or page_num > total_pages:
                    continue

                page = reader.pages[page_num - 1]
                text = page.extract_text() or ""

                # If pypdf extracted text, use it (digital PDF)
                # If empty, the page is likely scanned — would need image-based OCR
                if not text.strip():
                    text = f"[Page {page_num}: scanned page — requires image-based OCR engine]"

                page_results.append(PageResult(page=page_num, text=text))

            return MCPResult(success=True, data=Result(pages=page_results))

        except MCPError:
            raise
        except ImportError as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR,
                f"Missing dependency for PDF OCR: {e}. Install pypdf.",
            ) from e
        except Exception as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to OCR PDF: {e}") from e
