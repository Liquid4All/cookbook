"""Tests for screenshot.suggest_actions tool."""

from __future__ import annotations

import pytest

from tools.suggest_actions import SuggestActions
from pipeline_types import BoundingBox, UIElement


@pytest.fixture()
def tool() -> SuggestActions:
    """Create a SuggestActions tool instance."""
    return SuggestActions()


async def test_email_suggests_draft_email(tool: SuggestActions) -> None:
    """Should suggest 'Draft reply email' when text contains email addresses."""
    params = tool.get_params_model()(text="Please contact alice@example.com for details.")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Draft reply email" in actions
    email_suggestion = next(s for s in result.data.suggestions if s.action == "Draft reply email")
    assert "email.draft_email" in email_suggestion.tool_chain


async def test_file_path_suggests_open_file(tool: SuggestActions) -> None:
    """Should suggest 'Open file' when text contains file paths."""
    params = tool.get_params_model()(text="See the report at /Users/shared/reports/q4.pdf")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Open file" in actions
    file_suggestion = next(s for s in result.data.suggestions if s.action == "Open file")
    assert "system.open_file_with" in file_suggestion.tool_chain


async def test_date_time_suggests_create_event(tool: SuggestActions) -> None:
    """Should suggest 'Create calendar event' when text contains dates/times."""
    params = tool.get_params_model()(text="Meeting scheduled for 2026-03-15 at 2:30 PM")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Create calendar event" in actions
    event_suggestion = next(
        s for s in result.data.suggestions if s.action == "Create calendar event"
    )
    assert "calendar.create_event" in event_suggestion.tool_chain


async def test_todo_suggests_create_task(tool: SuggestActions) -> None:
    """Should suggest 'Create task' when text contains TODO patterns."""
    params = tool.get_params_model()(text="TODO: Update the project timeline before Friday")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Create task" in actions
    task_suggestion = next(s for s in result.data.suggestions if s.action == "Create task")
    assert "task.create_task" in task_suggestion.tool_chain


async def test_url_suggests_open_application(tool: SuggestActions) -> None:
    """Should suggest 'Open URL' when text contains URLs."""
    params = tool.get_params_model()(text="Check the docs at https://docs.example.com/api")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Open URL" in actions
    url_suggestion = next(s for s in result.data.suggestions if s.action == "Open URL")
    assert "system.open_application" in url_suggestion.tool_chain


async def test_table_data_suggests_write_csv(tool: SuggestActions) -> None:
    """Should suggest 'Extract to spreadsheet' when text contains tabular data."""
    table_text = (
        "Name|Age|Department\n"
        "Alice|32|Engineering\n"
        "Bob|28|Marketing\n"
    )
    params = tool.get_params_model()(text=table_text)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Extract to spreadsheet" in actions
    csv_suggestion = next(
        s for s in result.data.suggestions if s.action == "Extract to spreadsheet"
    )
    assert "data.write_csv" in csv_suggestion.tool_chain


async def test_empty_text_returns_no_suggestions(tool: SuggestActions) -> None:
    """Should return empty suggestions for empty or whitespace-only text."""
    params = tool.get_params_model()(text="")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.suggestions) == 0


async def test_whitespace_only_returns_no_suggestions(tool: SuggestActions) -> None:
    """Should return empty suggestions for whitespace-only text."""
    params = tool.get_params_model()(text="   \n\t  \n  ")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.suggestions) == 0


async def test_multiple_patterns_return_multiple_suggestions(tool: SuggestActions) -> None:
    """Should return multiple suggestions when text matches multiple patterns."""
    multi_text = (
        "Contact alice@example.com\n"
        "TODO: Review the budget\n"
        "Meeting on 2026-03-15\n"
        "Visit https://example.com\n"
    )
    params = tool.get_params_model()(text=multi_text)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.suggestions) >= 3


async def test_confidence_values_between_0_and_1(tool: SuggestActions) -> None:
    """All confidence values should be between 0.0 and 1.0."""
    params = tool.get_params_model()(
        text="alice@example.com TODO review https://example.com 2026-01-01"
    )
    result = await tool.execute(params)

    assert result.data is not None
    for suggestion in result.data.suggestions:
        assert 0.0 <= suggestion.confidence <= 1.0


async def test_tool_chain_is_nonempty(tool: SuggestActions) -> None:
    """Each suggestion should have a non-empty tool_chain."""
    params = tool.get_params_model()(text="alice@example.com TODO: fix bug")
    result = await tool.execute(params)

    assert result.data is not None
    for suggestion in result.data.suggestions:
        assert len(suggestion.tool_chain) > 0


async def test_suggestions_sorted_by_confidence(tool: SuggestActions) -> None:
    """Suggestions should be sorted by confidence in descending order."""
    params = tool.get_params_model()(
        text="alice@example.com TODO review https://example.com 2026-01-01"
    )
    result = await tool.execute(params)

    assert result.data is not None
    if len(result.data.suggestions) > 1:
        confidences = [s.confidence for s in result.data.suggestions]
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1]


async def test_with_ui_elements(tool: SuggestActions) -> None:
    """Should incorporate UI elements into suggestions."""
    elements = [
        UIElement(
            type="button",
            text="Send Email",
            bounds=BoundingBox(x=0, y=0, width=100, height=30),
            confidence=0.9,
        ),
    ]
    params = tool.get_params_model()(text="No patterns in main text", elements=elements)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    # "Send Email" in a button text won't necessarily trigger the email pattern,
    # but the test verifies elements are processed without error


async def test_checkbox_pattern_suggests_task(tool: SuggestActions) -> None:
    """Should suggest 'Create task' for checkbox patterns like [ ] or [x]."""
    params = tool.get_params_model()(text="[ ] Buy groceries\n[x] Clean desk")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Create task" in actions


async def test_action_item_pattern_suggests_task(tool: SuggestActions) -> None:
    """Should suggest 'Create task' for 'Action item:' text."""
    params = tool.get_params_model()(text="Action item: schedule the review meeting")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    actions = [s.action for s in result.data.suggestions]
    assert "Create task" in actions


def test_metadata(tool: SuggestActions) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "screenshot.suggest_actions"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
    assert len(tool.description) > 0
