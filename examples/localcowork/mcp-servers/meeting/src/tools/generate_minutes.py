"""
meeting.generate_minutes — Generate formatted meeting minutes document.

Produces a markdown-formatted meeting minutes document from transcript text.
Extracts attendees from speaker labels, splits discussion into sections,
and includes action items and decisions.

Writes to the filesystem: confirmation required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "_shared" / "py"))

from pydantic import BaseModel, Field  # noqa: E402

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes  # noqa: E402

# Add parent src dir for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extraction import generate_minutes_text  # noqa: E402


# ─── Params / Result Models ──────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for meeting.generate_minutes."""

    transcript: str = Field(description="Meeting transcript text")
    template: str | None = Field(
        default=None,
        description="Minutes template name (optional)",
    )
    output_path: str = Field(description="Where to save the minutes file")


class Result(BaseModel):
    """Return value for meeting.generate_minutes."""

    path: str = Field(description="Path to the generated minutes file")


# ─── Tool Implementation ─────────────────────────────────────────────────────


class GenerateMinutes(MCPTool[Params, Result]):
    """Generate formatted meeting minutes document."""

    name = "meeting.generate_minutes"
    description = "Generate formatted meeting minutes document"
    confirmation_required = True
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Generate minutes from transcript and write to output_path."""
        if not params.transcript.strip():
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                "Transcript must not be empty",
            )

        output = Path(params.output_path)

        # Validate output directory exists
        if not output.parent.exists():
            raise MCPError(
                ErrorCodes.FILE_NOT_FOUND,
                f"Output directory does not exist: {output.parent}",
            )

        # Generate minutes text
        minutes_content = generate_minutes_text(
            params.transcript,
            template=params.template,
        )

        # Write to file
        try:
            output.write_text(minutes_content, encoding="utf-8")
        except OSError as e:
            raise MCPError(
                ErrorCodes.INTERNAL_ERROR,
                f"Failed to write minutes file: {e}",
            ) from e

        return MCPResult(success=True, data=Result(path=str(output)))
