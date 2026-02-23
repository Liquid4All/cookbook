"""
security.scan_for_secrets — Scan files for exposed secrets.

Walks files in a directory (or scans a single file), applying secret
detection patterns (AWS keys, GitHub tokens, Stripe keys, private key
headers, generic API keys/passwords). Returns findings with masked values.
Non-destructive: no confirmation required.
"""

from __future__ import annotations

from pathlib import Path

from mcp_base import ErrorCodes, MCPError, MCPResult, MCPTool
from pydantic import BaseModel, Field
from validation import assert_absolute_path, assert_sandboxed

from patterns import (
    SECRET_PATTERNS,
    Finding,
    mask_sensitive_value,
    safe_collect_files,
)

# ─── Params / Result Models ────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for security.scan_for_secrets."""

    path: str = Field(description="Absolute path to file or directory to scan")


class Result(BaseModel):
    """Return value for security.scan_for_secrets."""

    findings: list[Finding]


# ─── Tool Implementation ───────────────────────────────────────────────────


class ScanForSecrets(MCPTool[Params, Result]):
    """Scan files for exposed secrets (API keys, private keys, passwords)."""

    name = "security.scan_for_secrets"
    description = "Scan files for exposed secrets (API keys, private keys, passwords)"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Scan the target path for secret matches."""
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        target = Path(params.path)
        if not target.exists():
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"Path not found: {params.path}")

        # Collect files to scan (with safe limits on depth/count/dir exclusions)
        files = safe_collect_files(target)

        # Scan each file
        findings: list[Finding] = []
        for file_path in files:
            file_findings = _scan_file_for_secrets(file_path)
            findings.extend(file_findings)

        return MCPResult(success=True, data=Result(findings=findings))


# ─── Helper Functions ──────────────────────────────────────────────────────


def _scan_file_for_secrets(file_path: Path) -> list[Finding]:
    """Scan a single file for secret patterns."""
    findings: list[Finding] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return findings

    lines = content.splitlines()

    for line_num, line in enumerate(lines, start=1):
        for secret_type, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(line):
                matched = match.group()
                findings.append(
                    Finding(
                        file_path=str(file_path),
                        line_number=line_num,
                        finding_type=secret_type,
                        matched_text=mask_sensitive_value(matched),
                        context=line.strip()[:200],
                    )
                )

    return findings
