"""
Shared Validation Utilities — Python

Path sandboxing, input sanitization, and common validators
used across all Python MCP servers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

try:
    from .mcp_base import MCPError, ErrorCodes
except ImportError:
    from mcp_base import MCPError, ErrorCodes  # type: ignore[no-redef]

# ─── Sandbox Validation ──────────────────────────────────────────────────────

_allowed_paths: list[Path] = []


def init_sandbox(paths: list[str]) -> None:
    """Initialize sandbox with user-granted directories."""
    global _allowed_paths
    _allowed_paths = [Path(p).resolve() for p in paths]


def assert_sandboxed(target_path: str) -> None:
    """
    Assert that a path is within the sandboxed directories.
    Raises MCPError with SANDBOX_VIOLATION code if not.
    """
    resolved = Path(target_path).resolve()

    is_allowed = any(
        resolved == allowed or str(resolved).startswith(str(allowed) + os.sep)
        for allowed in _allowed_paths
    )

    if not is_allowed:
        allowed_str = ", ".join(str(p) for p in _allowed_paths)
        raise MCPError(
            ErrorCodes.SANDBOX_VIOLATION,
            f'Path "{resolved}" is outside the sandboxed directories. Allowed: {allowed_str}',
        )


# ─── Input Sanitization ─────────────────────────────────────────────────────


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and invalid characters."""
    sanitized = filename
    sanitized = sanitized.replace("..", "")
    sanitized = re.sub(r'[<>:"|?*\x00]', "", sanitized)
    sanitized = sanitized.strip()
    return sanitized


def is_absolute_path(p: str) -> bool:
    """Check if a path is absolute."""
    return os.path.isabs(p)


def assert_absolute_path(p: str, param_name: str) -> None:
    """Assert a path is absolute, raise ValueError if not."""
    if not is_absolute_path(p):
        raise ValueError(f'Parameter "{param_name}" must be an absolute path. Got: "{p}"')


# ─── File Type Helpers ───────────────────────────────────────────────────────

FILE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "document": (".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".odt", ".html"),
    "spreadsheet": (".xlsx", ".xls", ".csv", ".tsv", ".ods"),
    "image": (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".svg"),
    "audio": (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma"),
    "video": (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"),
    "archive": (".zip", ".tar", ".gz", ".rar", ".7z", ".bz2"),
    "code": (".ts", ".js", ".py", ".rs", ".go", ".java", ".c", ".cpp", ".h"),
}


def get_file_category(file_path: str) -> str:
    """Get the category of a file by its extension."""
    ext = Path(file_path).suffix.lower()
    for category, extensions in FILE_CATEGORIES.items():
        if ext in extensions:
            return category
    return "other"


# ─── PII Pattern Helpers (used by security server) ──────────────────────────

# These are detection patterns, not validation patterns
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
}

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "aws_secret": re.compile(r"(?i:aws)(.{0,20})?['\"][0-9a-zA-Z/+]{40}['\"]"),
    "gcp_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "stripe_key": re.compile(r"(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
    "generic_secret": re.compile(
        r"(?i)(?:password|secret|token|api_key|apikey)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
}


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
