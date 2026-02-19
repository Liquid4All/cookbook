"""Tests for security.propose_cleanup tool."""

from __future__ import annotations

import pytest

from patterns import Finding
from tools.propose_cleanup import ProposeCleanup


@pytest.fixture()
def tool() -> ProposeCleanup:
    """Create a ProposeCleanup tool instance."""
    return ProposeCleanup()


def _make_finding(
    finding_type: str,
    file_path: str = "/tmp/test.txt",
    line_number: int = 1,
) -> Finding:
    """Create a test Finding."""
    return Finding(
        file_path=file_path,
        line_number=line_number,
        finding_type=finding_type,
        matched_text="12****89",
        context="some context line",
    )


async def test_propose_cleanup_for_ssn(tool: ProposeCleanup) -> None:
    """Should propose redaction for SSN findings."""
    findings = [_make_finding("ssn")]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.actions) == 1
    assert result.data.actions[0].action_type == "redact"
    assert result.data.actions[0].severity == "high"


async def test_propose_cleanup_for_aws_key(tool: ProposeCleanup) -> None:
    """Should propose rotation for AWS key findings."""
    findings = [_make_finding("aws_key")]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.actions) == 1
    assert result.data.actions[0].action_type == "rotate"
    assert result.data.actions[0].severity == "high"


async def test_propose_cleanup_for_private_key(tool: ProposeCleanup) -> None:
    """Should propose moving private key findings."""
    findings = [_make_finding("private_key")]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.actions) == 1
    assert result.data.actions[0].action_type == "move"
    assert result.data.actions[0].severity == "high"


async def test_propose_cleanup_for_email(tool: ProposeCleanup) -> None:
    """Should propose redaction with low severity for email findings."""
    findings = [_make_finding("email")]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.actions) == 1
    assert result.data.actions[0].action_type == "redact"
    assert result.data.actions[0].severity == "low"


async def test_deduplicates_same_file_same_type(tool: ProposeCleanup) -> None:
    """Should produce one action per (file, type) even with multiple findings."""
    findings = [
        _make_finding("ssn", line_number=1),
        _make_finding("ssn", line_number=5),
        _make_finding("ssn", line_number=10),
    ]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    # Three SSN findings in the same file should produce only one action
    assert len(result.data.actions) == 1


async def test_multiple_types_multiple_actions(tool: ProposeCleanup) -> None:
    """Should produce separate actions for different finding types."""
    findings = [
        _make_finding("ssn"),
        _make_finding("email"),
        _make_finding("aws_key"),
    ]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.actions) == 3

    action_types = {a.action_type for a in result.data.actions}
    assert "redact" in action_types
    assert "rotate" in action_types


async def test_different_files_same_type(tool: ProposeCleanup) -> None:
    """Should produce separate actions for same type in different files."""
    findings = [
        _make_finding("ssn", file_path="/tmp/file1.txt"),
        _make_finding("ssn", file_path="/tmp/file2.txt"),
    ]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.actions) == 2


async def test_empty_findings(tool: ProposeCleanup) -> None:
    """Should return empty actions for empty findings list."""
    params = tool.get_params_model()(findings=[])
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.actions) == 0


async def test_action_includes_target_path(tool: ProposeCleanup) -> None:
    """ProposedAction should reference the correct file path."""
    findings = [_make_finding("ssn", file_path="/tmp/important.csv")]
    params = tool.get_params_model()(findings=findings)
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert result.data.actions[0].target_path == "/tmp/important.csv"


def test_metadata(tool: ProposeCleanup) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "security.propose_cleanup"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
