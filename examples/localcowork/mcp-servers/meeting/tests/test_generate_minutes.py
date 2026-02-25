"""Tests for meeting.generate_minutes tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.generate_minutes import GenerateMinutes


@pytest.fixture()
def tool() -> GenerateMinutes:
    """Create a GenerateMinutes tool instance."""
    return GenerateMinutes()


async def test_basic_minutes_generation(
    tool: GenerateMinutes, full_meeting_transcript: str, tmp_path: Path
) -> None:
    """Should generate minutes markdown and write to file."""
    output_path = str(tmp_path / "minutes.md")
    params = tool.get_params_model()(
        transcript=full_meeting_transcript,
        output_path=output_path,
    )
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert result.data.path == output_path

    content = Path(output_path).read_text(encoding="utf-8")
    assert "## Meeting Minutes" in content


async def test_attendees_extracted(
    tool: GenerateMinutes, full_meeting_transcript: str, tmp_path: Path
) -> None:
    """Should extract speaker names as attendees."""
    output_path = str(tmp_path / "minutes.md")
    params = tool.get_params_model()(
        transcript=full_meeting_transcript,
        output_path=output_path,
    )
    result = await tool.execute(params)

    assert result.success is True
    content = Path(output_path).read_text(encoding="utf-8")

    assert "### Attendees" in content
    assert "John" in content
    assert "Sarah" in content
    assert "Mike" in content


async def test_default_template(
    tool: GenerateMinutes, full_meeting_transcript: str, tmp_path: Path
) -> None:
    """Should use default template when none specified."""
    output_path = str(tmp_path / "minutes.md")
    params = tool.get_params_model()(
        transcript=full_meeting_transcript,
        output_path=output_path,
    )
    result = await tool.execute(params)

    assert result.success is True
    content = Path(output_path).read_text(encoding="utf-8")

    # Default template sections
    assert "### Attendees" in content
    assert "### Discussion" in content
    assert "### Action Items" in content
    assert "### Decisions" in content


async def test_output_file_created(
    tool: GenerateMinutes, full_meeting_transcript: str, tmp_path: Path
) -> None:
    """Should create the output file at the specified path."""
    output_path = str(tmp_path / "output" / "minutes.md")
    (tmp_path / "output").mkdir()

    params = tool.get_params_model()(
        transcript=full_meeting_transcript,
        output_path=output_path,
    )
    result = await tool.execute(params)

    assert result.success is True
    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 0


async def test_empty_transcript_raises(
    tool: GenerateMinutes, tmp_path: Path
) -> None:
    """Should raise error for empty transcript."""
    from mcp_base import MCPError

    output_path = str(tmp_path / "minutes.md")
    params = tool.get_params_model()(
        transcript="",
        output_path=output_path,
    )
    with pytest.raises(MCPError, match="Transcript must not be empty"):
        await tool.execute(params)


async def test_invalid_output_directory(
    tool: GenerateMinutes, full_meeting_transcript: str
) -> None:
    """Should raise error when output directory does not exist."""
    from mcp_base import MCPError

    params = tool.get_params_model()(
        transcript=full_meeting_transcript,
        output_path="/nonexistent/path/minutes.md",
    )
    with pytest.raises(MCPError, match="Output directory does not exist"):
        await tool.execute(params)


async def test_custom_template_name(
    tool: GenerateMinutes, full_meeting_transcript: str, tmp_path: Path
) -> None:
    """Should accept a custom template name parameter."""
    output_path = str(tmp_path / "minutes.md")
    params = tool.get_params_model()(
        transcript=full_meeting_transcript,
        template="custom",
        output_path=output_path,
    )
    result = await tool.execute(params)

    # Should still succeed (template is reserved for future use)
    assert result.success is True
    assert result.data is not None
    assert result.data.path == output_path


async def test_action_items_in_minutes(
    tool: GenerateMinutes, tmp_path: Path
) -> None:
    """Should include extracted action items in the minutes."""
    transcript = (
        "John: ACTION: Sarah will review the design docs by Monday.\n"
        "Sarah: Understood, I'll get it done.\n"
    )
    output_path = str(tmp_path / "minutes.md")
    params = tool.get_params_model()(
        transcript=transcript,
        output_path=output_path,
    )
    result = await tool.execute(params)

    assert result.success is True
    content = Path(output_path).read_text(encoding="utf-8")
    assert "### Action Items" in content
    assert "review the design docs" in content


def test_metadata(tool: GenerateMinutes) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "meeting.generate_minutes"
    assert tool.confirmation_required is True
    assert tool.undo_supported is False
