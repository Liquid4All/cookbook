"""
Shared Pydantic models for the screenshot-to-action pipeline.

Named pipeline_types.py (NOT types.py) to avoid shadowing the stdlib types module.
All tools import their shared data models from here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---- Bounding box for screen regions and UI elements -------------------------


class BoundingBox(BaseModel):
    """Rectangular region on screen, in pixels."""

    x: int = Field(description="X coordinate of the top-left corner")
    y: int = Field(description="Y coordinate of the top-left corner")
    width: int = Field(description="Width of the region in pixels")
    height: int = Field(description="Height of the region in pixels")


# ---- Screenshot capture + OCR result ----------------------------------------


class CaptureRegion(BaseModel):
    """Optional region for partial screenshot capture."""

    x: int = Field(description="X coordinate of the top-left corner")
    y: int = Field(description="Y coordinate of the top-left corner")
    width: int = Field(ge=1, description="Width of the capture region in pixels")
    height: int = Field(ge=1, description="Height of the capture region in pixels")


class CaptureAndExtractResult(BaseModel):
    """Result of screenshot capture + OCR extraction."""

    screenshot_path: str = Field(description="Path to the captured screenshot image")
    text: str = Field(description="Extracted text from OCR")
    confidence: float = Field(
        ge=0.0, le=1.0, description="OCR confidence score (0.0 to 1.0)"
    )


# ---- UI element extraction ---------------------------------------------------


class UIElement(BaseModel):
    """A single UI element detected in a screenshot."""

    type: str = Field(description="Element type (button, text_field, label, link, icon, checkbox)")
    text: str = Field(description="Visible text content of the element")
    bounds: BoundingBox = Field(description="Bounding box of the element on screen")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Detection confidence score (0.0 to 1.0)"
    )


class ExtractUIElementsResult(BaseModel):
    """Result of UI element extraction from a screenshot."""

    elements: list[UIElement] = Field(
        default_factory=list, description="Detected UI elements"
    )


# ---- Action suggestions -----------------------------------------------------


class ActionSuggestion(BaseModel):
    """A suggested action based on screenshot analysis."""

    action: str = Field(description="Short action label (e.g., 'Draft reply email')")
    description: str = Field(description="Human-readable explanation of the suggestion")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence that this action is relevant (0.0 to 1.0)"
    )
    tool_chain: list[str] = Field(
        min_length=1, description="Sequence of MCP tool names to execute this action"
    )


class SuggestActionsResult(BaseModel):
    """Result of action suggestion analysis."""

    suggestions: list[ActionSuggestion] = Field(
        default_factory=list, description="Suggested actions ranked by confidence"
    )
