"""
Shared Pydantic models for the meeting MCP server.

Defines types used by both transcription (WS-5A) and extraction (WS-5B) tools:
- Segment, TranscriptionResult — transcription output
- ActionItem, Commitment, Decision — extraction output
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Transcription Types (WS-5A) ─────────────────────────────────────────────


class Segment(BaseModel):
    """A single transcription segment with timing and speaker info."""

    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    speaker: str = Field(description="Speaker label (e.g. Speaker_1)")
    text: str = Field(description="Transcribed text for this segment")


class TranscriptionResult(BaseModel):
    """Full transcription result returned by the transcription engine."""

    segments: list[Segment] = Field(description="Ordered list of transcript segments")
    duration_seconds: float = Field(description="Total audio duration in seconds")


# ─── Extraction Types (WS-5B) ────────────────────────────────────────────────


class ActionItem(BaseModel):
    """An action item extracted from a meeting transcript."""

    assignee: str = Field(description="Person assigned to the action")
    task: str = Field(description="Description of the task")
    deadline: str = Field(default="", description="When the task is due")
    context: str = Field(default="", description="Surrounding context from the transcript")
    priority: str = Field(default="medium", description="high | medium | low")


class Commitment(BaseModel):
    """A commitment made by a person during the meeting."""

    person: str = Field(description="Person who made the commitment")
    commitment: str = Field(description="What was committed to")
    deadline: str = Field(default="", description="When it should be done")
    context: str = Field(default="", description="Surrounding context from the transcript")


class Decision(BaseModel):
    """A decision made during the meeting."""

    decision: str = Field(description="The decision that was made")
    made_by: str = Field(default="", description="Person or group who made the decision")
    context: str = Field(default="", description="Surrounding context from the transcript")
