"""
security.scan_for_pii — Scan files for personally identifiable information.

Walks files in a directory (or scans a single file), applying PII regex
patterns (SSN, credit card, email, phone). Returns findings with masked values.
Non-destructive: no confirmation required.
"""

from __future__ import annotations

from pathlib import Path

from mcp_base import ErrorCodes, MCPError, MCPResult, MCPTool
from pydantic import BaseModel, Field
from validation import assert_absolute_path, assert_sandboxed

from patterns import (
    ALLOWED_PII_TYPES,
    PII_PATTERNS,
    Finding,
    luhn_check,
    mask_sensitive_value,
    safe_collect_files,
)

# ─── Params / Result Models ────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for security.scan_for_pii."""

    path: str = Field(description="Absolute path to file or directory to scan")
    types: list[str] | None = Field(
        default=None,
        description="PII types to scan for (ssn, credit_card, email, phone). All if omitted.",
    )


class Result(BaseModel):
    """Return value for security.scan_for_pii."""

    findings: list[Finding]


# ─── Tool Implementation ───────────────────────────────────────────────────


class ScanForPii(MCPTool[Params, Result]):
    """Scan files for personally identifiable information."""

    name = "security.scan_for_pii"
    description = "Scan files for PII (SSN, credit cards, emails, phone numbers)"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Scan the target path for PII matches."""
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        target = Path(params.path)
        if not target.exists():
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"Path not found: {params.path}")

        # Determine which PII types to scan for
        scan_types = _resolve_scan_types(params.types)

        # Collect files to scan (with safe limits on depth/count/dir exclusions)
        files = safe_collect_files(target)

        # Scan each file
        findings: list[Finding] = []
        for file_path in files:
            file_findings = _scan_file_for_pii(file_path, scan_types)
            findings.extend(file_findings)

        return MCPResult(success=True, data=Result(findings=findings))


# ─── Helper Functions ──────────────────────────────────────────────────────


def _resolve_scan_types(types: list[str] | None) -> set[str]:
    """Resolve requested PII types to a validated set."""
    if types is None:
        return set(ALLOWED_PII_TYPES)

    resolved: set[str] = set()
    for t in types:
        normalized = t.lower().strip()
        if normalized in ALLOWED_PII_TYPES:
            resolved.add(normalized)
    return resolved if resolved else set(ALLOWED_PII_TYPES)


def _scan_file_for_pii(file_path: Path, scan_types: set[str]) -> list[Finding]:
    """Scan a single file for PII patterns."""
    findings: list[Finding] = []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return findings

    lines = content.splitlines()

    for line_num, line in enumerate(lines, start=1):
        for pii_type in scan_types:
            pattern = PII_PATTERNS.get(pii_type)
            if pattern is None:
                continue

            for match in pattern.finditer(line):
                matched = match.group()

                # Validate credit card numbers with Luhn check
                if pii_type == "credit_card" and not luhn_check(matched):
                    continue

                findings.append(
                    Finding(
                        file_path=str(file_path),
                        line_number=line_num,
                        finding_type=pii_type,
                        matched_text=mask_sensitive_value(matched),
                        context=line.strip()[:200],
                    )
                )

    return findings
