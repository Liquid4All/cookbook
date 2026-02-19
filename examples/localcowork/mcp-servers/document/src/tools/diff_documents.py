"""
document.diff_documents — Produce a structured diff between two documents.

Non-destructive: executes immediately, no confirmation needed.
Supports paragraph, sentence, and word-level diffing.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_sandboxed, assert_absolute_path


class Change(BaseModel):
    """A single change between documents."""

    type: str  # "added", "removed", "modified"
    text: str
    context: str = ""


class Params(BaseModel):
    """Parameters for diff_documents."""

    path_a: str = Field(description="Path to first document")
    path_b: str = Field(description="Path to second document")
    granularity: str = Field(default="paragraph", description="Diff level: paragraph, sentence, or word")


class Result(BaseModel):
    """Return value for diff_documents."""

    changes: list[Change]
    summary: str


class DiffDocuments(MCPTool[Params, Result]):
    """Produce a structured diff between two documents."""

    name = "document.diff_documents"
    description = "Produce a structured diff between two documents"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Diff two documents and return structured changes."""
        assert_absolute_path(params.path_a, "path_a")
        assert_absolute_path(params.path_b, "path_b")
        assert_sandboxed(params.path_a)
        assert_sandboxed(params.path_b)

        for p in (params.path_a, params.path_b):
            if not os.path.exists(p):
                raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {p}")

        allowed = ("paragraph", "sentence", "word")
        if params.granularity not in allowed:
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                f"Invalid granularity: {params.granularity}. Allowed: {', '.join(allowed)}",
            )

        try:
            text_a = _read_text(params.path_a)
            text_b = _read_text(params.path_b)

            units_a = _split(text_a, params.granularity)
            units_b = _split(text_b, params.granularity)

            changes = _compute_diff(units_a, units_b)

            added = sum(1 for c in changes if c.type == "added")
            removed = sum(1 for c in changes if c.type == "removed")
            summary = f"{len(changes)} changes: {added} added, {removed} removed"

            return MCPResult(success=True, data=Result(changes=changes, summary=summary))

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to diff: {e}") from e


def _read_text(file_path: str) -> str:
    """Read document as text (supports PDF, DOCX, plain text)."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)

    if ext == ".docx":
        from docx import Document

        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    return Path(file_path).read_text(encoding="utf-8")


def _split(text: str, granularity: str) -> list[str]:
    """Split text into units based on granularity."""
    if granularity == "paragraph":
        return [p.strip() for p in text.split("\n\n") if p.strip()]
    if granularity == "sentence":
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    # word
    return text.split()


def _compute_diff(units_a: list[str], units_b: list[str]) -> list[Change]:
    """Compute diff between two lists of text units using LCS."""
    import difflib

    changes: list[Change] = []
    matcher = difflib.SequenceMatcher(None, units_a, units_b)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            for i in range(i1, i2):
                changes.append(Change(type="removed", text=units_a[i]))
        elif tag == "insert":
            for j in range(j1, j2):
                changes.append(Change(type="added", text=units_b[j]))
        elif tag == "replace":
            for i in range(i1, i2):
                changes.append(Change(type="removed", text=units_a[i]))
            for j in range(j1, j2):
                changes.append(Change(type="added", text=units_b[j]))

    return changes
