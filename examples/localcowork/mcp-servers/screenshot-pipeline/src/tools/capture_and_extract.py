"""
screenshot.capture_and_extract — Capture a screenshot and extract text via OCR.

Composes system.take_screenshot with ocr.extract_text_from_image.
Currently uses stubs that simulate both operations; the real implementation
will call the respective MCP servers when Tauri integration is complete.

Non-destructive: no confirmation required.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "_shared", "py")
)

from pydantic import BaseModel, Field  # noqa: E402

from mcp_base import MCPResult, MCPTool  # noqa: E402
from pipeline_types import CaptureAndExtractResult, CaptureRegion  # noqa: E402


# ---- Params / Result Models -------------------------------------------------


class Params(BaseModel):
    """Parameters for screenshot.capture_and_extract."""

    region: CaptureRegion | None = Field(
        default=None,
        description="Optional region for partial screenshot (x, y, width, height)",
    )
    language: str = Field(
        default="eng",
        description="OCR language code (e.g., 'eng', 'fra', 'deu')",
    )


# ---- Stub Functions ---------------------------------------------------------
# These stubs simulate screenshot capture and OCR extraction.
# In production, they will call system.take_screenshot and
# ocr.extract_text_from_image via the MCP client.


def _stub_take_screenshot(region: CaptureRegion | None) -> str:
    """Simulate taking a screenshot. Returns a mock file path."""
    screenshot_dir = os.path.join(tempfile.gettempdir(), "localcowork_screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    filename = f"screenshot_{uuid.uuid4().hex[:8]}.png"
    screenshot_path = os.path.join(screenshot_dir, filename)

    # Write a minimal stub file to represent the screenshot
    region_desc = "full_screen"
    if region:
        region_desc = f"{region.x}_{region.y}_{region.width}x{region.height}"

    with open(screenshot_path, "w", encoding="utf-8") as f:
        f.write(f"STUB_SCREENSHOT:region={region_desc}\n")

    return screenshot_path


def _stub_extract_text(screenshot_path: str, language: str) -> tuple[str, float]:
    """
    Simulate OCR text extraction from a screenshot.

    Returns a tuple of (extracted_text, confidence).
    The stub returns sample text for testing the pipeline;
    real implementation will invoke OCR server.
    """
    sample_text = (
        "Meeting Notes - Project Review\n"
        "Date: 2026-01-15\n"
        "Attendees: alice@company.com, bob@team.org\n"
        "\n"
        "TODO: Update the project timeline\n"
        "Action item: Review budget spreadsheet at /Users/shared/budget.xlsx\n"
        "Next meeting: https://meet.example.com/room-42\n"
    )
    confidence = 0.92
    return sample_text, confidence


# ---- Tool Implementation ----------------------------------------------------


class CaptureAndExtract(MCPTool[Params, CaptureAndExtractResult]):
    """Capture a screenshot and extract text using OCR."""

    name = "screenshot.capture_and_extract"
    description = (
        "Capture a screenshot (full screen or region) and extract text using OCR. "
        "Returns the screenshot path, extracted text, and confidence score."
    )
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[CaptureAndExtractResult]:
        """Capture screenshot and run OCR extraction."""
        # Step 1: Capture screenshot (stub)
        screenshot_path = _stub_take_screenshot(params.region)

        # Step 2: Extract text via OCR (stub)
        text, confidence = _stub_extract_text(screenshot_path, params.language)

        result = CaptureAndExtractResult(
            screenshot_path=screenshot_path,
            text=text,
            confidence=confidence,
        )
        return MCPResult(success=True, data=result)
