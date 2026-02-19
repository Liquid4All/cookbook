"""
ocr.extract_table — Extract tabular data from an image or PDF page.

Non-destructive: executes immediately, no confirmation needed.

Engine priority (ADR-004):
  1. LFM Vision via model-gateway — primary, native table understanding
  2. Text-based parsing           — fallback, heuristic line detection
"""

from __future__ import annotations

import os
import re

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_sandboxed, assert_absolute_path


class Params(BaseModel):
    """Parameters for extract_table."""

    path: str = Field(description="Path to image or PDF")
    page: int | None = Field(default=None, description="Page number (for PDFs)")


class Result(BaseModel):
    """Return value for extract_table."""

    headers: list[str]
    rows: list[list[str]]


class ExtractTable(MCPTool[Params, Result]):
    """Extract tabular data from an image or PDF page."""

    name = "ocr.extract_table"
    description = "Extract tabular data from an image or PDF page"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Extract table data from the specified file.

        Engine priority (ADR-004): LFM Vision → text-based parsing.
        """
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        if not os.path.exists(params.path):
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"File not found: {params.path}")

        ext = os.path.splitext(params.path)[1].lower()

        try:
            # 1. Try LFM Vision for image-based table extraction (ADR-004)
            if ext not in (".csv", ".tsv"):
                try:
                    headers, rows = await _extract_table_with_vision(params.path, params.page)
                    return MCPResult(success=True, data=Result(headers=headers, rows=rows))
                except (ImportError, MCPError):
                    pass  # Vision not available, fall through

            # 2. Text-based fallback: extract text then parse table structure
            if ext == ".pdf":
                text = _extract_pdf_page_text(params.path, params.page or 1)
            elif ext in (".csv", ".tsv"):
                text = _read_text_file(params.path)
            else:
                text = _extract_image_text(params.path)

            headers, rows = _parse_table_from_text(text)

            return MCPResult(success=True, data=Result(headers=headers, rows=rows))

        except MCPError:
            raise
        except Exception as e:
            raise MCPError(ErrorCodes.INTERNAL_ERROR, f"Failed to extract table: {e}") from e


# ─── Engine: LFM Vision (via model-gateway) ─────────────────────────────────


async def _extract_table_with_vision(
    path: str, page: int | None
) -> tuple[list[str], list[list[str]]]:
    """Extract table using a vision-capable model via OpenAI-compatible API.

    Sends the image to the vision model and asks it to extract tabular data
    as JSON. Reads endpoint from LOCALCOWORK_VISION_ENDPOINT env var.

    NOTE(WS-8): Will be replaced by model-gateway service for unified routing.
    """
    import base64
    import json
    from pathlib import Path

    import aiohttp  # type: ignore[import-untyped]

    endpoint = os.environ.get("LOCALCOWORK_VISION_ENDPOINT", "http://localhost:8081/v1")
    model = os.environ.get("LOCALCOWORK_VISION_MODEL", "LFM2.5-VL-1.6B")
    url = f"{endpoint}/chat/completions"

    # Read and encode image
    image_bytes = Path(path).read_bytes()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    ext = os.path.splitext(path)[1].lower()
    mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_types.get(ext, "image/png")
    data_url = f"data:{mime};base64,{b64}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract the table from this image. Return ONLY valid JSON in this "
                            'format: {"headers": ["col1", "col2"], "rows": [["val1", "val2"]]}. '
                            "No commentary, no markdown, just the JSON object."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    raise ImportError(f"Vision model returned {resp.status}")
                result = await resp.json()
    except (aiohttp.ClientError, OSError) as exc:
        raise ImportError(f"Vision model unavailable: {exc}") from exc

    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content.strip():
        raise ImportError("Vision model returned empty response")

    # Parse the JSON response
    try:
        data = json.loads(content.strip())
        headers = data.get("headers", [])
        rows = data.get("rows", [])
        return headers, rows
    except json.JSONDecodeError as exc:
        raise ImportError(f"Vision model returned invalid JSON: {exc}") from exc


# ─── Text-based Helpers ──────────────────────────────────────────────────────


def _extract_pdf_page_text(path: str, page_num: int) -> str:
    """Extract text from a specific PDF page."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    if page_num < 1 or page_num > len(reader.pages):
        raise MCPError(
            ErrorCodes.INVALID_PARAMS,
            f"Page {page_num} out of range (1-{len(reader.pages)})",
        )
    return reader.pages[page_num - 1].extract_text() or ""


def _read_text_file(path: str) -> str:
    """Read a text-based file."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def _extract_image_text(path: str) -> str:
    """Try to OCR an image file."""
    try:
        import pytesseract  # type: ignore[import-untyped]
        from PIL import Image

        image = Image.open(path)
        return pytesseract.image_to_string(image)
    except ImportError:
        pass

    try:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(path, cls=True)
        lines = []
        if result and result[0]:
            for line_result in result[0]:
                lines.append(line_result[1][0])
        return "\n".join(lines)
    except ImportError:
        pass

    raise MCPError(
        ErrorCodes.INTERNAL_ERROR,
        "No OCR engine available for image processing. Install pytesseract or paddleocr.",
    )


def _parse_table_from_text(text: str) -> tuple[list[str], list[list[str]]]:
    """Parse tabular data from text using heuristic line detection.

    Detects tables by looking for consistent delimiters (|, tabs, multiple spaces).
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    if not lines:
        return [], []

    # Try pipe-delimited tables first (markdown-style)
    if "|" in lines[0]:
        return _parse_pipe_table(lines)

    # Try tab-delimited
    if "\t" in lines[0]:
        return _parse_delimited_table(lines, "\t")

    # Try comma-delimited
    if "," in lines[0]:
        return _parse_delimited_table(lines, ",")

    # Try multi-space delimited (common in OCR output)
    return _parse_space_table(lines)


def _parse_pipe_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse a pipe-delimited table."""
    parsed_lines: list[list[str]] = []
    for line in lines:
        # Skip separator lines (--- | ---)
        if re.match(r"^[\s|:-]+$", line):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            parsed_lines.append(cells)

    if not parsed_lines:
        return [], []

    return parsed_lines[0], parsed_lines[1:]


def _parse_delimited_table(
    lines: list[str], delimiter: str
) -> tuple[list[str], list[list[str]]]:
    """Parse a table with a specific delimiter."""
    parsed = [[c.strip() for c in line.split(delimiter)] for line in lines]
    if not parsed:
        return [], []
    return parsed[0], parsed[1:]


def _parse_space_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    """Parse a table delimited by multiple spaces."""
    parsed: list[list[str]] = []
    for line in lines:
        cells = re.split(r"\s{2,}", line.strip())
        if cells:
            parsed.append(cells)

    if not parsed:
        return [], []

    return parsed[0], parsed[1:]
