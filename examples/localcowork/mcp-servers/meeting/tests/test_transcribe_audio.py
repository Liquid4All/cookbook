"""
Tests for meeting.transcribe_audio tool.

Covers: basic transcription, diarization, non-existent file handling,
unsupported extensions, language parameter, and result structure validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_base import MCPError
from tools.transcribe_audio import Params, Result, SegmentDict, TranscribeAudio


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_tool() -> TranscribeAudio:
    """Create a fresh TranscribeAudio tool instance."""
    return TranscribeAudio()


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestTranscribeAudioBasic:
    """Basic transcription without diarization."""

    @pytest.mark.asyncio
    async def test_basic_transcription(self, wav_file: Path) -> None:
        """Transcribing a valid WAV file returns segments and duration."""
        tool = _make_tool()
        params = Params(path=str(wav_file))
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert len(result.data.transcript) > 0
        assert result.data.duration_seconds > 0.0

    @pytest.mark.asyncio
    async def test_mp3_transcription(self, mp3_file: Path) -> None:
        """Transcribing a valid MP3 file is accepted."""
        tool = _make_tool()
        params = Params(path=str(mp3_file))
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        assert len(result.data.transcript) > 0

    @pytest.mark.asyncio
    async def test_default_language_is_english(self, wav_file: Path) -> None:
        """Default language should be 'en'."""
        params = Params(path=str(wav_file))
        assert params.language == "en"

    @pytest.mark.asyncio
    async def test_default_diarize_is_false(self, wav_file: Path) -> None:
        """Default diarize should be False."""
        params = Params(path=str(wav_file))
        assert params.diarize is False


class TestTranscribeAudioDiarization:
    """Transcription with speaker diarization enabled."""

    @pytest.mark.asyncio
    async def test_diarization_adds_speaker_labels(self, wav_file: Path) -> None:
        """When diarize=True, segments should have different speaker labels."""
        tool = _make_tool()
        params = Params(path=str(wav_file), diarize=True)
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None

        speakers = {seg.speaker for seg in result.data.transcript}
        # Stub produces alternating Speaker_1 / Speaker_2
        assert len(speakers) >= 2, f"Expected multiple speakers, got: {speakers}"
        assert "Speaker_1" in speakers
        assert "Speaker_2" in speakers

    @pytest.mark.asyncio
    async def test_no_diarization_single_speaker(self, wav_file: Path) -> None:
        """When diarize=False, all segments should have Speaker_1."""
        tool = _make_tool()
        params = Params(path=str(wav_file), diarize=False)
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None

        speakers = {seg.speaker for seg in result.data.transcript}
        assert speakers == {"Speaker_1"}


class TestTranscribeAudioErrors:
    """Error handling for invalid inputs."""

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_error(self, nonexistent_path: Path) -> None:
        """Transcribing a non-existent file raises MCPError."""
        tool = _make_tool()
        params = Params(path=str(nonexistent_path))

        with pytest.raises(MCPError) as exc_info:
            await tool.execute(params)

        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_unsupported_extension_raises_error(self, unsupported_file: Path) -> None:
        """Transcribing an unsupported format raises MCPError."""
        tool = _make_tool()
        params = Params(path=str(unsupported_file))

        with pytest.raises(MCPError) as exc_info:
            await tool.execute(params)

        assert "unsupported" in str(exc_info.value).lower()


class TestTranscribeAudioLanguage:
    """Language parameter handling."""

    @pytest.mark.asyncio
    async def test_english_language(self, wav_file: Path) -> None:
        """English transcription returns English-like segment text."""
        tool = _make_tool()
        params = Params(path=str(wav_file), language="en")
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        # Stub uses English templates for language="en"
        first_text = result.data.transcript[0].text
        assert len(first_text) > 0

    @pytest.mark.asyncio
    async def test_non_english_language(self, wav_file: Path) -> None:
        """Non-English language returns generic segment text."""
        tool = _make_tool()
        params = Params(path=str(wav_file), language="fr")
        result = await tool.execute(params)

        assert result.success is True
        assert result.data is not None
        # Stub uses generic templates for non-English
        first_text = result.data.transcript[0].text
        assert "[Transcribed audio segment" in first_text

    @pytest.mark.asyncio
    async def test_language_parameter_passed_through(self, wav_file: Path) -> None:
        """Language param is correctly stored on the Params model."""
        params = Params(path=str(wav_file), language="de")
        assert params.language == "de"


class TestTranscribeAudioResultStructure:
    """Validate the shape and content of the result."""

    @pytest.mark.asyncio
    async def test_result_has_transcript_and_duration(self, wav_file: Path) -> None:
        """Result must contain both 'transcript' and 'duration_seconds'."""
        tool = _make_tool()
        params = Params(path=str(wav_file))
        result = await tool.execute(params)

        assert result.data is not None
        assert hasattr(result.data, "transcript")
        assert hasattr(result.data, "duration_seconds")
        assert isinstance(result.data.transcript, list)
        assert isinstance(result.data.duration_seconds, float)

    @pytest.mark.asyncio
    async def test_segment_fields(self, wav_file: Path) -> None:
        """Each segment must have start_time, end_time, speaker, and text."""
        tool = _make_tool()
        params = Params(path=str(wav_file))
        result = await tool.execute(params)

        assert result.data is not None
        for seg in result.data.transcript:
            assert isinstance(seg, SegmentDict)
            assert isinstance(seg.start_time, float)
            assert isinstance(seg.end_time, float)
            assert isinstance(seg.speaker, str)
            assert isinstance(seg.text, str)
            assert seg.end_time > seg.start_time
            assert len(seg.speaker) > 0
            assert len(seg.text) > 0

    @pytest.mark.asyncio
    async def test_segments_are_chronological(self, wav_file: Path) -> None:
        """Segments should be in chronological order."""
        tool = _make_tool()
        params = Params(path=str(wav_file))
        result = await tool.execute(params)

        assert result.data is not None
        times = [seg.start_time for seg in result.data.transcript]
        assert times == sorted(times), "Segments are not in chronological order"

    @pytest.mark.asyncio
    async def test_duration_matches_last_segment(self, wav_file: Path) -> None:
        """Duration should be close to the last segment's end_time."""
        tool = _make_tool()
        params = Params(path=str(wav_file))
        result = await tool.execute(params)

        assert result.data is not None
        last_end = result.data.transcript[-1].end_time
        assert abs(result.data.duration_seconds - last_end) < 0.01

    @pytest.mark.asyncio
    async def test_tool_metadata(self) -> None:
        """Tool has correct MCP metadata."""
        tool = _make_tool()
        assert tool.name == "meeting.transcribe_audio"
        assert tool.confirmation_required is False
        assert tool.undo_supported is False
        assert len(tool.description) > 0
