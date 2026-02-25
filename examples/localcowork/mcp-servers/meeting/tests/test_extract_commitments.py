"""Tests for meeting.extract_commitments tool."""

from __future__ import annotations

import pytest

from tools.extract_commitments import ExtractCommitments


@pytest.fixture()
def tool() -> ExtractCommitments:
    """Create an ExtractCommitments tool instance."""
    return ExtractCommitments()


async def test_extract_commitment_i_will(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should detect 'I will' commitment patterns."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    commitments_text = [c.commitment for c in result.data.commitments]
    assert any("finish the report" in c for c in commitments_text)


async def test_extract_commitment_ill(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should detect I'll commitment patterns."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    commitments_text = [c.commitment for c in result.data.commitments]
    assert any("send the updated numbers" in c for c in commitments_text)


async def test_extract_commitment_i_commit_to(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should detect 'I commit to' patterns."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    commitments_text = [c.commitment for c in result.data.commitments]
    assert any("delivering the API docs" in c for c in commitments_text)


async def test_extract_decision_we_decided(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should detect 'We decided' decision patterns."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    decisions_text = [d.decision for d in result.data.decisions]
    assert any("React" in d for d in decisions_text)


async def test_extract_decision_agreed(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should detect 'Agreed:' decision patterns."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    decisions_text = [d.decision for d in result.data.decisions]
    assert any("deadline" in d or "March" in d for d in decisions_text)


async def test_extract_open_question_mark(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should detect open questions containing '?'."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    assert any("authentication" in q for q in result.data.open_questions)


async def test_extract_open_question_tbd(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should detect 'TBD' as open question marker."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # "TBD" line should be captured as open question
    assert len(result.data.open_questions) >= 1


async def test_empty_transcript_raises(tool: ExtractCommitments) -> None:
    """Should raise error for empty transcript."""
    from mcp_base import MCPError

    params = tool.get_params_model()(transcript="")
    with pytest.raises(MCPError, match="Transcript must not be empty"):
        await tool.execute(params)


async def test_no_matches_found(
    tool: ExtractCommitments, no_matches_transcript: str
) -> None:
    """Should return empty lists when nothing found."""
    params = tool.get_params_model()(transcript=no_matches_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.commitments) == 0
    assert len(result.data.decisions) == 0


async def test_multiple_commitments_same_person(tool: ExtractCommitments) -> None:
    """Should capture multiple commitments from the same person."""
    transcript = (
        "Alice: I will write the tests.\n"
        "Alice: I'll also update the documentation.\n"
        "Alice: I promise to review Bob's PR.\n"
    )
    params = tool.get_params_model()(transcript=transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    alice_commitments = [
        c for c in result.data.commitments if c.person == "Alice"
    ]
    assert len(alice_commitments) >= 2


async def test_mixed_content(
    tool: ExtractCommitments, full_meeting_transcript: str
) -> None:
    """Should extract commitments, decisions, and questions from mixed content."""
    params = tool.get_params_model()(transcript=full_meeting_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # Should find at least one of each
    assert len(result.data.commitments) >= 1
    assert len(result.data.decisions) >= 1
    assert len(result.data.open_questions) >= 1


async def test_result_structure(
    tool: ExtractCommitments, commitments_transcript: str
) -> None:
    """Should return properly structured result objects."""
    params = tool.get_params_model()(transcript=commitments_transcript)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    for c in result.data.commitments:
        assert isinstance(c.person, str)
        assert isinstance(c.commitment, str)
        assert isinstance(c.deadline, str)
        assert isinstance(c.context, str)

    for d in result.data.decisions:
        assert isinstance(d.decision, str)
        assert isinstance(d.made_by, str)
        assert isinstance(d.context, str)

    for q in result.data.open_questions:
        assert isinstance(q, str)


def test_metadata(tool: ExtractCommitments) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "meeting.extract_commitments"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
