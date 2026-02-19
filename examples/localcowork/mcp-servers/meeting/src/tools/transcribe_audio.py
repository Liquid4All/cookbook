"""
meeting.transcribe_audio — Transcribe audio to text using Whisper.cpp.

Accepts an audio file path, optional language code, and optional speaker
diarization flag.  Returns a list of timed transcript segments and total
duration.  Non-destructive: no confirmation required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

# ─── Shared base import ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "_shared", "py"))

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes  # noqa: E402

from meeting_types import Segment  # noqa: E402
from transcription import SUPPORTED_EXTENSIONS, get_engine  # noqa: E402


# ─── Params / Result Models ──────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for meeting.transcribe_audio."""

    path: str = Field(description="Absolute path to the audio file")
    language: str = Field(default="en", description="ISO 639-1 language code")
    diarize: bool = Field(default=False, description="Enable speaker diarization")


class SegmentDict(BaseModel):
    """A single transcript segment returned to the caller."""

    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    speaker: str = Field(description="Speaker label (e.g. Speaker_1)")
    text: str = Field(description="Transcribed text for this segment")


class Result(BaseModel):
    """Return value for meeting.transcribe_audio."""

    transcript: list[SegmentDict] = Field(description="Ordered list of transcript segments")
    duration_seconds: float = Field(description="Total audio duration in seconds")


# ─── Tool Implementation ─────────────────────────────────────────────────────


class TranscribeAudio(MCPTool[Params, Result]):
    """Transcribe audio to text using Whisper.cpp."""

    name = "meeting.transcribe_audio"
    description = "Transcribe audio to text using Whisper.cpp"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Transcribe the audio file at *params.path*."""
        # Validate file exists
        audio_path = Path(params.path)
        if not audio_path.exists():
            raise MCPError(
                ErrorCodes.FILE_NOT_FOUND,
                f"Audio file not found: {params.path}",
            )

        # Validate file extension
        ext = audio_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        # Transcribe
        engine = get_engine()
        try:
            transcription = engine.transcribe(
                path=params.path,
                language=params.language,
                diarize=params.diarize,
            )
        except FileNotFoundError as exc:
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, str(exc)) from exc
        except ValueError as exc:
            raise MCPError(ErrorCodes.INVALID_PARAMS, str(exc)) from exc

        # Convert Segment models to SegmentDict for the result
        segments = [
            SegmentDict(
                start_time=seg.start_time,
                end_time=seg.end_time,
                speaker=seg.speaker,
                text=seg.text,
            )
            for seg in transcription.segments
        ]

        return MCPResult(
            success=True,
            data=Result(
                transcript=segments,
                duration_seconds=transcription.duration_seconds,
            ),
        )
