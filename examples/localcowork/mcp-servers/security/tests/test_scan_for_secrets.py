"""Tests for security.scan_for_secrets tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.scan_for_secrets import ScanForSecrets


@pytest.fixture()
def tool() -> ScanForSecrets:
    """Create a ScanForSecrets tool instance."""
    return ScanForSecrets()


async def test_scan_detects_aws_key(tool: ScanForSecrets, secrets_file: Path) -> None:
    """Should detect AWS access key IDs."""
    params = tool.get_params_model()(path=str(secrets_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    aws_findings = [f for f in result.data.findings if f.finding_type == "aws_key"]
    assert len(aws_findings) >= 1
    # Key should be masked
    assert "AKIAIOSFODNN7EXAMPLE" not in aws_findings[0].matched_text
    assert "*" in aws_findings[0].matched_text


async def test_scan_detects_private_key(tool: ScanForSecrets, secrets_file: Path) -> None:
    """Should detect private key headers."""
    params = tool.get_params_model()(path=str(secrets_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    pk_findings = [f for f in result.data.findings if f.finding_type == "private_key"]
    assert len(pk_findings) >= 1


async def test_scan_detects_generic_api_key(tool: ScanForSecrets, secrets_file: Path) -> None:
    """Should detect generic API key patterns."""
    params = tool.get_params_model()(path=str(secrets_file))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    api_findings = [f for f in result.data.findings if f.finding_type == "generic_api_key"]
    assert len(api_findings) >= 1


async def test_scan_directory(tool: ScanForSecrets, tmp_dir: Path) -> None:
    """Should scan all files in a directory recursively."""
    sub = tmp_dir / "config"
    sub.mkdir()
    (sub / ".env").write_text("AWS_KEY=AKIAIOSFODNN7EXAMPLE\n", encoding="utf-8")
    (tmp_dir / "app.py").write_text(
        '# Nothing secret here\nx = 42\n', encoding="utf-8"
    )

    params = tool.get_params_model()(path=str(tmp_dir))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    aws_findings = [f for f in result.data.findings if f.finding_type == "aws_key"]
    assert len(aws_findings) >= 1


async def test_scan_no_secrets(tool: ScanForSecrets, tmp_dir: Path) -> None:
    """Should return empty findings for clean files."""
    clean = tmp_dir / "clean.py"
    clean.write_text("# This file has no secrets\nx = 42\n", encoding="utf-8")

    params = tool.get_params_model()(path=str(clean))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.findings) == 0


async def test_scan_file_not_found(tool: ScanForSecrets, tmp_dir: Path) -> None:
    """Should raise error for missing path."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(tmp_dir / "nonexistent.py"))
    with pytest.raises(MCPError, match="Path not found"):
        await tool.execute(params)


async def test_scan_skips_binary_files(tool: ScanForSecrets, tmp_dir: Path) -> None:
    """Should skip binary file extensions."""
    binary = tmp_dir / "data.bin"
    binary.write_bytes(b"AKIAIOSFODNN7EXAMPLE")

    params = tool.get_params_model()(path=str(tmp_dir))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    bin_findings = [f for f in result.data.findings if f.file_path.endswith(".bin")]
    assert len(bin_findings) == 0


async def test_finding_context_trimmed(tool: ScanForSecrets, tmp_dir: Path) -> None:
    """Should include trimmed context in findings."""
    f = tmp_dir / "config.txt"
    f.write_text("    AKIAIOSFODNN7EXAMPLE    \n", encoding="utf-8")

    params = tool.get_params_model()(path=str(f))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.findings) >= 1
    # Context should be trimmed
    assert result.data.findings[0].context == "AKIAIOSFODNN7EXAMPLE"


async def test_scan_github_token(tool: ScanForSecrets, tmp_dir: Path) -> None:
    """Should detect GitHub personal access tokens."""
    f = tmp_dir / "gh.txt"
    f.write_text(
        "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl\n",
        encoding="utf-8",
    )

    params = tool.get_params_model()(path=str(f))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    gh_findings = [f for f in result.data.findings if f.finding_type == "github_token"]
    assert len(gh_findings) >= 1


def test_metadata(tool: ScanForSecrets) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "security.scan_for_secrets"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
