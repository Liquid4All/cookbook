"""
Screenshot Pipeline MCP Server — Entry Point

Registers all screenshot-to-action pipeline tools and starts the JSON-RPC listener.
This server composes screenshot capture (via system server) with OCR extraction
(via OCR server) and provides heuristic action suggestions.

Tools (3):
  screenshot.capture_and_extract  — capture screenshot + OCR text extraction
  screenshot.extract_ui_elements  — detect UI elements in a screenshot image
  screenshot.suggest_actions      — suggest actions from extracted text/elements
"""

from __future__ import annotations

import os
import sys

# Add shared path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "_shared", "py"))

# Add src path for pipeline modules
sys.path.insert(0, os.path.dirname(__file__))

from mcp_base import MCPServer  # noqa: E402

from tools.capture_and_extract import CaptureAndExtract  # noqa: E402
from tools.extract_ui_elements import ExtractUIElements  # noqa: E402
from tools.suggest_actions import SuggestActions  # noqa: E402

# ---- Server Setup ------------------------------------------------------------

server = MCPServer(
    name="screenshot-pipeline",
    version="0.1.0",
    tools=[
        CaptureAndExtract(),
        ExtractUIElements(),
        SuggestActions(),
    ],
)

if __name__ == "__main__":
    server.start()
