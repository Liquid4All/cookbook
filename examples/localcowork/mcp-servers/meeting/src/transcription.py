"""
Transcription engine for the meeting MCP server.

Wraps Whisper.cpp (via pywhispercpp) for audio-to-text transcription and
pyannote.audio for speaker diarization. When the real libraries are not
installed, falls back to a stub implementation that validates inputs and
returns mock transcripts for development and testing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from meeting_types import Segment, TranscriptionResult

# ── Supported audio extensions ───────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
)

# ── Try to import real libraries ─────────────────────────────────────────────

_WHISPER_AVAILABLE: bool = False
_DIARIZE_AVAILABLE: bool = False

try:
    from pywhispercpp.model import Model as WhisperModel  # type: ignore[import-untyped]

    _WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None  # type: ignore[assignment, misc]

try:
    from pyannote.audio import Pipeline as DiarizePipeline  # type: ignore[import-untyped]

    _DIARIZE_AVAILABLE = True
except ImportError:
    DiarizePipeline = None  # type: ignore[assignment, misc]

_logger = logging.getLogger("meeting.transcription")

# ── Constants ────────────────────────────────────────────────────────────────

# Rough estimate: 16-bit PCM mono at 16 kHz => ~32 KB/sec
_BYTES_PER_SECOND_ESTIMATE: float = 32_000.0

# Stub transcript templates by language
_STUB_SEGMENTS_EN: list[dict[str, str]] = [
    {"text": "Hello, thank you for joining today's meeting."},
    {"text": "Let's start by reviewing the agenda."},
    {"text": "The first item is the quarterly report."},
    {"text": "Are there any questions before we proceed?"},
]

_STUB_SEGMENTS_OTHER: list[dict[str, str]] = [
    {"text": "[Transcribed audio segment 1]"},
    {"text": "[Transcribed audio segment 2]"},
    {"text": "[Transcribed audio segment 3]"},
]


# ── Transcription Engine ─────────────────────────────────────────────────────


class TranscriptionEngine:
    """
    Audio transcription engine with optional speaker diarization.

    Uses Whisper.cpp when available; otherwise returns a stub transcript
    that validates inputs and produces mock output for development.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """
        Initialize the transcription engine.

        Args:
            model_path: Path to the Whisper model file. If None, uses the
                default model or falls back to stub mode.
        """
        self._model_path = model_path
        self._whisper_model: object | None = None
        self._using_stub = not _WHISPER_AVAILABLE

        if _WHISPER_AVAILABLE and model_path and WhisperModel is not None:
            try:
                self._whisper_model = WhisperModel(model_path)
            except Exception:
                _logger.warning(
                    "Failed to load Whisper model at %s; falling back to stub",
                    model_path,
                )
                self._using_stub = True

    @property
    def is_stub(self) -> bool:
        """Return True if the engine is running in stub/mock mode."""
        return self._using_stub

    def transcribe(
        self,
        path: str,
        language: str = "en",
        diarize: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            path: Absolute path to the audio file.
            language: ISO 639-1 language code (default: "en").
            diarize: Whether to add speaker diarization labels.

        Returns:
            TranscriptionResult with segments and duration.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the file extension is not supported.
        """
        audio_path = Path(path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        ext = audio_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if self._using_stub:
            return self._transcribe_stub(audio_path, language, diarize)

        return self._transcribe_real(audio_path, language, diarize)

    # ── Real implementation ──────────────────────────────────────────────────

    def _transcribe_real(
        self,
        audio_path: Path,
        language: str,
        diarize: bool,
    ) -> TranscriptionResult:
        """Transcribe using real Whisper.cpp and optional pyannote diarization."""
        # This path is only reached when pywhispercpp is installed
        if self._whisper_model is None and WhisperModel is not None:
            self._whisper_model = WhisperModel("large-v3")

        # Call whisper — the real API returns a list of (start, end, text) tuples
        raw_segments: list[tuple[float, float, str]] = self._whisper_model.transcribe(  # type: ignore[union-attr]
            str(audio_path), language=language
        )

        segments: list[Segment] = []
        for start, end, text in raw_segments:
            segments.append(
                Segment(
                    start_time=start,
                    end_time=end,
                    speaker="Speaker_1",
                    text=text.strip(),
                )
            )

        # Apply diarization if requested and available
        if diarize and _DIARIZE_AVAILABLE and DiarizePipeline is not None:
            segments = self._apply_diarization(audio_path, segments)
        elif diarize:
            _logger.warning("Diarization requested but pyannote.audio is not installed")

        duration = segments[-1].end_time if segments else 0.0
        return TranscriptionResult(segments=segments, duration_seconds=duration)

    def _apply_diarization(
        self,
        audio_path: Path,
        segments: list[Segment],
    ) -> list[Segment]:
        """Apply pyannote speaker diarization to existing segments."""
        if DiarizePipeline is None:
            return segments

        try:
            pipeline = DiarizePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            diarization = pipeline(str(audio_path))

            # Map each segment to its most likely speaker
            labeled: list[Segment] = []
            for seg in segments:
                mid = (seg.start_time + seg.end_time) / 2.0
                speaker = _find_speaker_at(diarization, mid)
                labeled.append(
                    Segment(
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        speaker=speaker,
                        text=seg.text,
                    )
                )
            return labeled
        except Exception:
            _logger.warning("Diarization failed; returning segments without labels")
            return segments

    # ── Stub implementation ──────────────────────────────────────────────────

    def _transcribe_stub(
        self,
        audio_path: Path,
        language: str,
        diarize: bool,
    ) -> TranscriptionResult:
        """
        Generate a mock transcript for development/testing.

        Estimates duration from file size and produces template segments.
        """
        file_size = audio_path.stat().st_size
        estimated_duration = max(1.0, file_size / _BYTES_PER_SECOND_ESTIMATE)

        templates = _STUB_SEGMENTS_EN if language == "en" else _STUB_SEGMENTS_OTHER
        num_segments = len(templates)
        segment_duration = estimated_duration / num_segments

        segments: list[Segment] = []
        for i, template in enumerate(templates):
            start = i * segment_duration
            end = (i + 1) * segment_duration
            speaker = f"Speaker_{(i % 2) + 1}" if diarize else "Speaker_1"

            segments.append(
                Segment(
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    speaker=speaker,
                    text=template["text"],
                )
            )

        return TranscriptionResult(
            segments=segments,
            duration_seconds=round(estimated_duration, 3),
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_speaker_at(diarization: object, time_seconds: float) -> str:
    """Find the speaker label at a given time in the diarization output."""
    # pyannote diarization objects are iterable as (turn, _, speaker)
    for turn, _, speaker in diarization:  # type: ignore[union-attr]
        if turn.start <= time_seconds <= turn.end:  # type: ignore[union-attr]
            return str(speaker)
    return "Speaker_1"


def get_engine(model_path: str | None = None) -> TranscriptionEngine:
    """
    Create or return a TranscriptionEngine instance.

    Args:
        model_path: Optional path to a Whisper model file.

    Returns:
        A configured TranscriptionEngine.
    """
    resolved_path = model_path or os.environ.get("WHISPER_MODEL_PATH")
    return TranscriptionEngine(model_path=resolved_path)
