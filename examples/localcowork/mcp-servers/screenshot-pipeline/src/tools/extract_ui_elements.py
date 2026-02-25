"""
screenshot.extract_ui_elements — Extract UI elements from a screenshot image.

Detects buttons, text fields, labels, links, icons, and checkboxes
with their positions. Currently uses stubs; the real implementation
will use OCR + a vision model for UI element detection.

Non-destructive: no confirmation required.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "_shared", "py")
)

from pydantic import BaseModel, Field  # noqa: E402

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes  # noqa: E402
from pipeline_types import BoundingBox, ExtractUIElementsResult, UIElement  # noqa: E402


# ---- Params Model -----------------------------------------------------------


class Params(BaseModel):
    """Parameters for screenshot.extract_ui_elements."""

    image_path: str = Field(description="Absolute path to the screenshot image file")


# ---- Stub Functions ---------------------------------------------------------


def _stub_detect_elements(image_path: str) -> list[UIElement]:
    """
    Simulate UI element detection from a screenshot.

    Returns a list of mock UI elements for testing the pipeline.
    Real implementation will use OCR + vision model.
    """
    return [
        UIElement(
            type="button",
            text="Submit",
            bounds=BoundingBox(x=350, y=420, width=120, height=40),
            confidence=0.95,
        ),
        UIElement(
            type="text_field",
            text="Search...",
            bounds=BoundingBox(x=100, y=50, width=300, height=32),
            confidence=0.90,
        ),
        UIElement(
            type="label",
            text="Project Name",
            bounds=BoundingBox(x=100, y=100, width=150, height=20),
            confidence=0.88,
        ),
        UIElement(
            type="link",
            text="View Documentation",
            bounds=BoundingBox(x=100, y=300, width=180, height=18),
            confidence=0.92,
        ),
        UIElement(
            type="checkbox",
            text="I agree to terms",
            bounds=BoundingBox(x=100, y=380, width=200, height=22),
            confidence=0.87,
        ),
        UIElement(
            type="button",
            text="Cancel",
            bounds=BoundingBox(x=200, y=420, width=100, height=40),
            confidence=0.93,
        ),
    ]


# ---- Tool Implementation ----------------------------------------------------


class ExtractUIElements(MCPTool[Params, ExtractUIElementsResult]):
    """Extract UI elements and their positions from a screenshot."""

    name = "screenshot.extract_ui_elements"
    description = (
        "Extract UI elements (buttons, text fields, labels, links, checkboxes) "
        "and their bounding boxes from a screenshot image."
    )
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[ExtractUIElementsResult]:
        """Detect UI elements in the provided screenshot image."""
        # Validate the image path exists
        if not os.path.isabs(params.image_path):
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                f"image_path must be an absolute path, got: {params.image_path}",
            )

        if not os.path.exists(params.image_path):
            raise MCPError(
                ErrorCodes.FILE_NOT_FOUND,
                f"Image file not found: {params.image_path}",
            )

        # Detect UI elements (stub)
        elements = _stub_detect_elements(params.image_path)

        result = ExtractUIElementsResult(elements=elements)
        return MCPResult(success=True, data=result)
