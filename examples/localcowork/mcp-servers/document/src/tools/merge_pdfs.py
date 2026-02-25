"""
document.merge_pdfs — Merge multiple PDFs into one.

Mutable: requires user confirmation (writes a file).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_sandboxed, assert_absolute_path


class Params(BaseModel):
    """Parameters for merge_pdfs."""

    paths: list[str] = Field(description="Paths to PDFs to merge (in order)", min_length=2)
    output_path: str = Field(description="Where to save merged PDF")


class Result(BaseModel):
    """Return value for merge_pdfs."""

    path: str
    total_pages: int


class MergePdfs(MCPTool[Params, Result]):
    """Merge multiple PDFs into one."""

    name = "document.merge_pdfs"
    description = "Merge multiple PDFs into one"
    confirmation_required = True
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Merge PDF files into a single output."""
        assert_absolute_path(params.output_path, "output_path")
        assert_sandboxed(params.output_path)

        for p in params.paths:
            assert_absolute_path(p, "paths[]")
            assert_sandboxed(p)
            if not os.path.exists(p):
                raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {p}")

        try:
            from pypdf import PdfWriter

            writer = PdfWriter()
            total_pages = 0

            for pdf_path in params.paths:
                writer.append(pdf_path)
                # Count pages added (pypdf tracks this internally)

            total_pages = len(writer.pages)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(params.output_path), exist_ok=True)

            with open(params.output_path, "wb") as f:
                writer.write(f)

            writer.close()

            return MCPResult(
                success=True,
                data=Result(path=params.output_path, total_pages=total_pages),
            )

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to merge PDFs: {e}") from e
