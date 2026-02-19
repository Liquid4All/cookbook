"""Tests for meeting.extract_action_items tool."""

from __future__ import annotations

import pytest

from tools.extract_action_items import ExtractActionItems


@pytest.fixture()
def tool() -> ExtractActionItems:
    """Create an ExtractActionItems tool instance."""
    return ExtractActionItems()


async def test_extract_action_marker(
    tool: ExtractActionItems, action_items_transcript: str
) -> None:
    """Should detect ACTION: markers in transcript."""
    params = tool.get_params_model()(transcript=action_items_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    tasks = [item.task for item in result.data.items]
    # ACTION: John will review the proposal by Friday
    assert any("review the proposal" in t for t in tasks)


async def test_extract_todo_marker(
    tool: ExtractActionItems, action_items_transcript: str
) -> None:
    """Should detect TODO: markers in transcript."""
    params = tool.get_params_model()(transcript=action_items_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    tasks = [item.task for item in result.data.items]
    # TODO: Sarah needs to update the budget spreadsheet
    assert any("update the budget" in t for t in tasks)


async def test_extract_at_person_pattern(
    tool: ExtractActionItems, action_items_transcript: str
) -> None:
    """Should extract assignee from @person pattern."""
    params = tool.get_params_model()(transcript=action_items_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # @Lisa will handle the client presentation
    lisa_items = [item for item in result.data.items if item.assignee == "Lisa"]
    assert len(lisa_items) >= 1


async def test_extract_deadline(
    tool: ExtractActionItems, action_items_transcript: str
) -> None:
    """Should extract deadline from 'by Friday' pattern."""
    params = tool.get_params_model()(transcript=action_items_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    items_with_deadline = [item for item in result.data.items if item.deadline]
    assert len(items_with_deadline) >= 1
    assert any("Friday" in item.deadline for item in items_with_deadline)


async def test_extract_priority_high(tool: ExtractActionItems) -> None:
    """Should detect high priority from 'urgent' and 'ASAP' keywords."""
    transcript = "John: ACTION: Fix the production bug urgently by tonight."
    params = tool.get_params_model()(transcript=transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.items) >= 1
    assert result.data.items[0].priority == "high"


async def test_extract_priority_low(tool: ExtractActionItems) -> None:
    """Should detect low priority from 'when possible' keyword."""
    transcript = "John: TODO: Update the README when possible."
    params = tool.get_params_model()(transcript=transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.items) >= 1
    assert result.data.items[0].priority == "low"


async def test_empty_transcript_raises(tool: ExtractActionItems) -> None:
    """Should raise error for empty transcript."""
    from mcp_base import MCPError

    params = tool.get_params_model()(transcript="")
    with pytest.raises(MCPError, match="Transcript must not be empty"):
        await tool.execute(params)


async def test_whitespace_only_transcript_raises(tool: ExtractActionItems) -> None:
    """Should raise error for whitespace-only transcript."""
    from mcp_base import MCPError

    params = tool.get_params_model()(transcript="   \n  \n  ")
    with pytest.raises(MCPError, match="Transcript must not be empty"):
        await tool.execute(params)


async def test_no_action_items_found(
    tool: ExtractActionItems, no_matches_transcript: str
) -> None:
    """Should return empty list when no action items found."""
    params = tool.get_params_model()(transcript=no_matches_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.items) == 0


async def test_result_structure(
    tool: ExtractActionItems, action_items_transcript: str
) -> None:
    """Should return properly structured ActionItem objects."""
    params = tool.get_params_model()(transcript=action_items_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.items) > 0

    for item in result.data.items:
        assert isinstance(item.assignee, str)
        assert isinstance(item.task, str)
        assert isinstance(item.deadline, str)
        assert isinstance(item.context, str)
        assert item.priority in ("high", "medium", "low")


async def test_should_phrase_pattern(tool: ExtractActionItems) -> None:
    """Should detect 'should' action phrases."""
    transcript = "Manager: The intern should review the onboarding docs."
    params = tool.get_params_model()(transcript=transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.items) >= 1


def test_metadata(tool: ExtractActionItems) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "meeting.extract_action_items"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
