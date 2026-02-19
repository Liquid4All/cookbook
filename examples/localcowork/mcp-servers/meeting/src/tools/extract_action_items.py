"""
meeting.extract_action_items — Extract action items from a meeting transcript.

Scans transcript text for explicit markers (ACTION:, TODO:) and implicit
phrases ("will do", "needs to", etc.) to identify action items with
assignees, deadlines, and priority levels.

Non-destructive: no confirmation required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "_shared" / "py"))

from pydantic import BaseModel, Field  # noqa: E402

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes  # noqa: E402

# Add parent src dir for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction import extract_action_items_from_text  # noqa: E402


# ─── Params / Result Models ──────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for meeting.extract_action_items."""

    transcript: str = Field(description="Meeting transcript text")


class ActionItemDict(BaseModel):
    """A single action item in dict-compatible format."""

    assignee: str
    task: str
    deadline: str
    context: str
    priority: str


class Result(BaseModel):
    """Return value for meeting.extract_action_items."""

    items: list[ActionItemDict]


# ─── Tool Implementation ─────────────────────────────────────────────────────


class ExtractActionItems(MCPTool[Params, Result]):
    """Extract action items from a meeting transcript."""

    name = "meeting.extract_action_items"
    description = "Extract action items from a meeting transcript"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Extract action items from the provided transcript text."""
        if not params.transcript.strip():
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                "Transcript must not be empty",
            )

        action_items = extract_action_items_from_text(params.transcript)

        result_items = [
            ActionItemDict(
                assignee=item.assignee,
                task=item.task,
                deadline=item.deadline,
                context=item.context,
                priority=item.priority,
            )
            for item in action_items
        ]

        return MCPResult(success=True, data=Result(items=result_items))
