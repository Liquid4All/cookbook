"""
OCR MCP Server — Entry Point

Registers all OCR tools and starts the JSON-RPC listener.
This server provides OCR text extraction using Tesseract and PaddleOCR.

Tools (4):
  ocr.extract_text_from_image   — OCR text from image
  ocr.extract_text_from_pdf     — OCR text from scanned PDF
  ocr.extract_structured_data   — structured data extraction from OCR text
  ocr.extract_table             — tabular data extraction from image/PDF
"""

from __future__ import annotations

import os
import sys

# Add shared path and own package root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from mcp_base import MCPServer  # noqa: E402
from validation import init_sandbox  # noqa: E402

from tools.extract_text_from_image import ExtractTextFromImage  # noqa: E402
from tools.extract_text_from_pdf import ExtractTextFromPdf  # noqa: E402
from tools.extract_structured_data import ExtractStructuredData  # noqa: E402
from tools.extract_table import ExtractTable  # noqa: E402

# ─── Sandbox Initialization ─────────────────────────────────────────────────

allowed_paths_str = os.environ.get("LOCALCOWORK_ALLOWED_PATHS", os.path.expanduser("~"))
allowed_paths = allowed_paths_str.split(os.pathsep)
init_sandbox(allowed_paths)

# ─── Server Setup ───────────────────────────────────────────────────────────

server = MCPServer(
    name="ocr",
    version="1.0.0",
    tools=[
        ExtractTextFromImage(),
        ExtractTextFromPdf(),
        ExtractStructuredData(),
        ExtractTable(),
    ],
)

if __name__ == "__main__":
    server.start()
