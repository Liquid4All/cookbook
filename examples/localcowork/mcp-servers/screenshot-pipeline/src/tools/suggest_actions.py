"""
screenshot.suggest_actions — Suggest actions based on extracted text and UI elements.

Uses the heuristic action classifier to analyze text (and optional UI elements)
from a screenshot, returning ranked action suggestions with tool chains.

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

from mcp_base import MCPResult, MCPTool  # noqa: E402
from action_classifier import classify_with_elements  # noqa: E402
from pipeline_types import SuggestActionsResult, UIElement  # noqa: E402


# ---- Params Model -----------------------------------------------------------


class Params(BaseModel):
    """Parameters for screenshot.suggest_actions."""

    text: str = Field(description="Extracted text from the screenshot")
    elements: list[UIElement] | None = Field(
        default=None,
        description="Optional list of detected UI elements from the screenshot",
    )


# ---- Tool Implementation ----------------------------------------------------


class SuggestActions(MCPTool[Params, SuggestActionsResult]):
    """Analyze screenshot text and elements to suggest possible actions."""

    name = "screenshot.suggest_actions"
    description = (
        "Analyze extracted text and UI elements from a screenshot and suggest "
        "possible actions the user might want to take, with corresponding tool chains."
    )
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[SuggestActionsResult]:
        """Classify text and elements into actionable suggestions."""
        suggestions = classify_with_elements(params.text, params.elements)

        result = SuggestActionsResult(suggestions=suggestions)
        return MCPResult(success=True, data=result)
