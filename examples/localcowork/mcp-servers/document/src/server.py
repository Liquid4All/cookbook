"""
Document MCP Server — Entry Point

Registers all document tools and starts the JSON-RPC listener.
This server provides document extraction, conversion, diffing, and PDF generation.

Tools (8):
  document.extract_text      — extract text from PDF/DOCX/HTML
  document.convert_format    — convert between document formats (confirm)
  document.diff_documents    — diff two documents
  document.create_pdf        — create PDF from markdown (confirm)
  document.fill_pdf_form     — fill PDF form fields (confirm)
  document.merge_pdfs        — merge multiple PDFs (confirm)
  document.create_docx       — create Word doc from markdown (confirm)
  document.read_spreadsheet  — read Excel/CSV spreadsheet
"""

from __future__ import annotations

import os
import sys

# Add shared path and own package root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared", "py"))
sys.path.insert(0, os.path.dirname(__file__))

from mcp_base import MCPServer  # noqa: E402
from validation import init_sandbox  # noqa: E402

from tools.extract_text import ExtractText  # noqa: E402
from tools.convert_format import ConvertFormat  # noqa: E402
from tools.diff_documents import DiffDocuments  # noqa: E402
from tools.create_pdf import CreatePdf  # noqa: E402
from tools.fill_pdf_form import FillPdfForm  # noqa: E402
from tools.merge_pdfs import MergePdfs  # noqa: E402
from tools.create_docx import CreateDocx  # noqa: E402
from tools.read_spreadsheet import ReadSpreadsheet  # noqa: E402

# ─── Sandbox Initialization ─────────────────────────────────────────────────

allowed_paths_str = os.environ.get("LOCALCOWORK_ALLOWED_PATHS", os.path.expanduser("~"))
allowed_paths = allowed_paths_str.split(os.pathsep)
init_sandbox(allowed_paths)

# ─── Server Setup ───────────────────────────────────────────────────────────

server = MCPServer(
    name="document",
    version="1.0.0",
    tools=[
        ExtractText(),
        ConvertFormat(),
        DiffDocuments(),
        CreatePdf(),
        FillPdfForm(),
        MergePdfs(),
        CreateDocx(),
        ReadSpreadsheet(),
    ],
)

if __name__ == "__main__":
    server.start()
