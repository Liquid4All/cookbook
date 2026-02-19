"""
meeting.extract_commitments — Extract commitments, decisions, and open questions.

Scans transcript text for commitment phrases ("I will", "I'll"),
decision markers ("We decided", "Agreed:"), and open questions
("?", "TBD", "to be determined").

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
from extraction import extract_commitments_from_text  # noqa: E402


# ─── Params / Result Models ──────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for meeting.extract_commitments."""

    transcript: str = Field(description="Meeting transcript text")


class CommitmentDict(BaseModel):
    """A commitment in dict-compatible format."""

    person: str
    commitment: str
    deadline: str
    context: str


class DecisionDict(BaseModel):
    """A decision in dict-compatible format."""

    decision: str
    made_by: str
    context: str


class Result(BaseModel):
    """Return value for meeting.extract_commitments."""

    commitments: list[CommitmentDict]
    decisions: list[DecisionDict]
    open_questions: list[str]


# ─── Tool Implementation ─────────────────────────────────────────────────────


class ExtractCommitments(MCPTool[Params, Result]):
    """Extract commitments, decisions, and open questions from a transcript."""

    name = "meeting.extract_commitments"
    description = (
        "Extract commitments, decisions, and open questions from a transcript"
    )
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Extract commitments, decisions, and open questions."""
        if not params.transcript.strip():
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                "Transcript must not be empty",
            )

        commitments, decisions, open_questions = extract_commitments_from_text(
            params.transcript
        )

        result_commitments = [
            CommitmentDict(
                person=c.person,
                commitment=c.commitment,
                deadline=c.deadline,
                context=c.context,
            )
            for c in commitments
        ]

        result_decisions = [
            DecisionDict(
                decision=d.decision,
                made_by=d.made_by,
                context=d.context,
            )
            for d in decisions
        ]

        return MCPResult(
            success=True,
            data=Result(
                commitments=result_commitments,
                decisions=result_decisions,
                open_questions=open_questions,
            ),
        )
