"""
document.extract_text — Extract plain text from a document.

Supports: PDF, DOCX, TXT, MD, HTML.
Non-destructive: executes immediately, no confirmation needed.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_sandboxed, assert_absolute_path


class Params(BaseModel):
    """Parameters for extract_text."""

    path: str = Field(description="Path to document")


class Result(BaseModel):
    """Return value for extract_text."""

    text: str
    format: str
    pages: int | None = None


class ExtractText(MCPTool[Params, Result]):
    """Extract plain text from a document (PDF, DOCX, HTML, TXT, MD)."""

    name = "document.extract_text"
    description = "Extract plain text from a document (PDF, DOCX, HTML, etc.)"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Extract text from the specified document."""
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        if not os.path.exists(params.path):
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {params.path}")

        ext = Path(params.path).suffix.lower()

        try:
            if ext == ".pdf":
                text, pages = _extract_pdf(params.path)
                return MCPResult(success=True, data=Result(text=text, format="pdf", pages=pages))

            if ext in (".docx",):
                text = _extract_docx(params.path)
                return MCPResult(success=True, data=Result(text=text, format="docx"))

            if ext in (".txt", ".md", ".csv", ".tsv", ".log"):
                text = Path(params.path).read_text(encoding="utf-8")
                return MCPResult(success=True, data=Result(text=text, format=ext.lstrip(".")))

            if ext in (".html", ".htm"):
                text = _extract_html(params.path)
                return MCPResult(success=True, data=Result(text=text, format="html"))

            # Fallback: try reading as text
            text = Path(params.path).read_text(encoding="utf-8", errors="replace")
            return MCPResult(success=True, data=Result(text=text, format="text"))

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to extract text: {e}") from e


def _extract_pdf(file_path: str) -> tuple[str, int]:
    """Extract text from PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages_text: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n\n".join(pages_text), len(reader.pages)


def _extract_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def _extract_html(file_path: str) -> str:
    """Extract text from HTML by stripping tags."""
    import re

    content = Path(file_path).read_text(encoding="utf-8")
    # Remove script and style elements
    content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
    content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
    # Remove tags
    content = re.sub(r"<[^>]+>", " ", content)
    # Collapse whitespace
    content = re.sub(r"\s+", " ", content).strip()
    return content
