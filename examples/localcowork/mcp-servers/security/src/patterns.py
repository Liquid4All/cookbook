"""
Regex pattern library and data models for security scanning.

Defines PII and secret detection patterns, plus Pydantic models
shared across all security tools: Finding, FileInfo, ProposedAction.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# ─── PII Detection Patterns ────────────────────────────────────────────────

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}

# ─── Secret Detection Patterns ─────────────────────────────────────────────

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}"),
    "stripe_key": re.compile(r"(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
    "generic_api_key": re.compile(
        r"(?i)(?:password|secret|token|api_key|apikey)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}

# Allowed PII types that callers can filter by
ALLOWED_PII_TYPES: set[str] = {"ssn", "credit_card", "email", "phone"}

# ─── Binary file extensions to skip when scanning ──────────────────────────

BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".tiff", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat",
    ".mp3", ".wav", ".ogg", ".mp4", ".avi", ".mkv", ".mov",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pyc", ".pyo", ".class", ".o", ".obj",
    ".enc", ".key",
})


# ─── Pydantic Models ───────────────────────────────────────────────────────


class Finding(BaseModel):
    """A single PII or secret finding in a scanned file."""

    file_path: str = Field(description="Absolute path to the file")
    line_number: int = Field(description="1-based line number of the match")
    finding_type: str = Field(description="Category of the finding (e.g. ssn, aws_key)")
    matched_text: str = Field(description="Masked version of the matched text")
    context: str = Field(description="The line containing the match (trimmed)")


class FileInfo(BaseModel):
    """Metadata about a file for duplicate detection."""

    path: str = Field(description="Absolute path to the file")
    size: int = Field(description="File size in bytes")
    hash: str = Field(description="SHA-256 hex digest (empty for name-only method)")


class ProposedAction(BaseModel):
    """A proposed cleanup action for a security finding."""

    action_type: str = Field(description="Action: redact, delete, rotate, or move")
    target_path: str = Field(description="Path to the affected file")
    description: str = Field(description="Human-readable description of the action")
    severity: str = Field(description="Severity level: high, medium, or low")


# ─── Masking Utility ───────────────────────────────────────────────────────


def mask_sensitive_value(value: str) -> str:
    """Mask a sensitive value, showing only first 2 and last 2 characters."""
    stripped = value.strip()
    if len(stripped) <= 4:
        return "*" * len(stripped)
    return stripped[:2] + "*" * (len(stripped) - 4) + stripped[-2:]


def is_binary_file(file_path: str) -> bool:
    """Check if a file should be skipped based on its extension."""
    from pathlib import Path

    return Path(file_path).suffix.lower() in BINARY_EXTENSIONS


def luhn_check(number: str) -> bool:
    """Validate a credit card number using the Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0
