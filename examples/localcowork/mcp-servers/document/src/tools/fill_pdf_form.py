"""
document.fill_pdf_form — Fill fields in a PDF form.

Mutable: requires user confirmation (writes a file).
Uses pypdf to fill in form fields.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_sandboxed, assert_absolute_path


class Params(BaseModel):
    """Parameters for fill_pdf_form."""

    path: str = Field(description="Path to PDF form")
    fields: dict[str, Any] = Field(description="Map of field names to values")
    output_path: str = Field(description="Where to save filled PDF")


class Result(BaseModel):
    """Return value for fill_pdf_form."""

    path: str
    fields_filled: int


class FillPdfForm(MCPTool[Params, Result]):
    """Fill fields in a PDF form."""

    name = "document.fill_pdf_form"
    description = "Fill fields in a PDF form"
    confirmation_required = True
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Fill form fields in a PDF and write output."""
        assert_absolute_path(params.path, "path")
        assert_absolute_path(params.output_path, "output_path")
        assert_sandboxed(params.path)
        assert_sandboxed(params.output_path)

        if not os.path.exists(params.path):
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {params.path}")

        try:
            from pypdf import PdfReader, PdfWriter

            reader = PdfReader(params.path)
            writer = PdfWriter()

            # Copy all pages
            for page in reader.pages:
                writer.add_page(page)

            # Fill form fields
            fields_filled = 0
            if reader.get_fields():
                for field_name, value in params.fields.items():
                    try:
                        writer.update_page_form_field_values(
                            writer.pages[0],
                            {field_name: str(value)},
                        )
                        fields_filled += 1
                    except Exception:
                        # Skip fields that don't exist or can't be set
                        pass

            # Ensure output directory exists
            os.makedirs(os.path.dirname(params.output_path), exist_ok=True)

            with open(params.output_path, "wb") as f:
                writer.write(f)

            return MCPResult(
                success=True,
                data=Result(path=params.output_path, fields_filled=fields_filled),
            )

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to fill PDF form: {e}") from e
