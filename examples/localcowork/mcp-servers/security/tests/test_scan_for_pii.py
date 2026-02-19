"""Tests for security.scan_for_pii tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.scan_for_pii import ScanForPii


@pytest.fixture()
def tool() -> ScanForPii:
    """Create a ScanForPii tool instance."""
    return ScanForPii()


async def test_scan_detects_ssn(tool: ScanForPii, pii_file: Path) -> None:
    """Should detect SSN patterns in files."""
    params = tool.get_params_model()(path=str(pii_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    ssn_findings = [f for f in result.data.findings if f.finding_type == "ssn"]
    assert len(ssn_findings) >= 1
    # SSN should be masked
    assert "123-45-6789" not in ssn_findings[0].matched_text
    assert "*" in ssn_findings[0].matched_text


async def test_scan_detects_email(tool: ScanForPii, pii_file: Path) -> None:
    """Should detect email addresses in files."""
    params = tool.get_params_model()(path=str(pii_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    email_findings = [f for f in result.data.findings if f.finding_type == "email"]
    assert len(email_findings) >= 1
    # Email should be masked
    assert "john.doe@example.com" not in email_findings[0].matched_text


async def test_scan_detects_phone(tool: ScanForPii, pii_file: Path) -> None:
    """Should detect phone numbers in files."""
    params = tool.get_params_model()(path=str(pii_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    phone_findings = [f for f in result.data.findings if f.finding_type == "phone"]
    assert len(phone_findings) >= 1


async def test_scan_detects_credit_card(tool: ScanForPii, pii_file: Path) -> None:
    """Should detect credit card numbers (validated with Luhn)."""
    params = tool.get_params_model()(path=str(pii_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    cc_findings = [f for f in result.data.findings if f.finding_type == "credit_card"]
    assert len(cc_findings) >= 1


async def test_scan_filters_by_type(tool: ScanForPii, pii_file: Path) -> None:
    """Should only return findings for requested PII types."""
    params = tool.get_params_model()(path=str(pii_file), types=["ssn"])
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # All findings should be SSN type
    for finding in result.data.findings:
        assert finding.finding_type == "ssn"


async def test_scan_directory(tool: ScanForPii, tmp_dir: Path) -> None:
    """Should recursively scan directories."""
    sub = tmp_dir / "subdir"
    sub.mkdir()
    (sub / "data.txt").write_text("SSN: 987-65-4321\n", encoding="utf-8")
    (tmp_dir / "top.txt").write_text("SSN: 111-22-3333\n", encoding="utf-8")

    params = tool.get_params_model()(path=str(tmp_dir), types=["ssn"])
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.findings) >= 2


async def test_scan_no_findings(tool: ScanForPii, tmp_dir: Path) -> None:
    """Should return empty findings for clean files."""
    clean = tmp_dir / "clean.txt"
    clean.write_text("This file has no PII at all.\n", encoding="utf-8")

    params = tool.get_params_model()(path=str(clean))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.findings) == 0


async def test_scan_file_not_found(tool: ScanForPii, tmp_dir: Path) -> None:
    """Should raise error for missing path."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(tmp_dir / "nonexistent.txt"))
    with pytest.raises(MCPError, match="Path not found"):
        await tool.execute(params)


async def test_scan_skips_binary_files(tool: ScanForPii, tmp_dir: Path) -> None:
    """Should skip binary file extensions."""
    binary_file = tmp_dir / "image.png"
    binary_file.write_bytes(b"SSN: 123-45-6789")

    params = tool.get_params_model()(path=str(tmp_dir))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    # Should not report findings from binary files
    png_findings = [f for f in result.data.findings if f.file_path.endswith(".png")]
    assert len(png_findings) == 0


async def test_finding_has_line_number(tool: ScanForPii, pii_file: Path) -> None:
    """Should include correct line numbers in findings."""
    params = tool.get_params_model()(path=str(pii_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    ssn_findings = [f for f in result.data.findings if f.finding_type == "ssn"]
    assert len(ssn_findings) >= 1
    # SSN is on line 2 of the fixture
    assert ssn_findings[0].line_number == 2


def test_metadata(tool: ScanForPii) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "security.scan_for_pii"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
