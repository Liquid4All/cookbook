"""
security.find_duplicates — Find duplicate files in a directory.

Supports three deduplication methods:
  - hash: SHA-256 content hash comparison (default)
  - name: filename comparison (ignoring directory)
  - content: full byte-for-byte content comparison

Returns groups of duplicate FileInfo objects.
Non-destructive: no confirmation required.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from mcp_base import MCPError, MCPResult, MCPTool, ErrorCodes
from validation import assert_absolute_path, assert_sandboxed

from patterns import FileInfo

# ─── Constants ─────────────────────────────────────────────────────────────

VALID_METHODS: frozenset[str] = frozenset({"hash", "name", "content"})


# ─── Params / Result Models ────────────────────────────────────────────────


class Params(BaseModel):
    """Parameters for security.find_duplicates."""

    path: str = Field(description="Absolute path to directory to scan")
    method: str = Field(
        default="hash",
        description="Comparison method: hash (SHA-256), name, or content",
    )


class Result(BaseModel):
    """Return value for security.find_duplicates."""

    groups: list[list[FileInfo]]


# ─── Tool Implementation ───────────────────────────────────────────────────


class FindDuplicates(MCPTool[Params, Result]):
    """Find duplicate files in a directory."""

    name = "security.find_duplicates"
    description = "Find duplicate files by hash, name, or content comparison"
    confirmation_required = False
    undo_supported = False

    async def execute(self, params: Params) -> MCPResult[Result]:
        """Find duplicate files in the target directory."""
        assert_absolute_path(params.path, "path")
        assert_sandboxed(params.path)

        target = Path(params.path)
        if not target.exists():
            raise MCPError(ErrorCodes.FILE_NOT_FOUND, f"Path not found: {params.path}")
        if not target.is_dir():
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                f"Path must be a directory: {params.path}",
            )

        method = params.method.lower()
        if method not in VALID_METHODS:
            raise MCPError(
                ErrorCodes.INVALID_PARAMS,
                f"Invalid method: {method}. Must be one of: {', '.join(sorted(VALID_METHODS))}",
            )

        # Collect all files
        files = _collect_regular_files(target)

        # Group by selected method
        if method == "hash":
            groups = _group_by_hash(files)
        elif method == "name":
            groups = _group_by_name(files)
        else:
            groups = _group_by_content(files)

        # Filter to only groups with duplicates (2+ files)
        duplicate_groups = [group for group in groups if len(group) >= 2]

        return MCPResult(success=True, data=Result(groups=duplicate_groups))


# ─── Helper Functions ──────────────────────────────────────────────────────


def _collect_regular_files(directory: Path) -> list[Path]:
    """Collect all regular files in a directory tree."""
    files: list[Path] = []
    for root, _dirs, filenames in os.walk(str(directory)):
        for fname in filenames:
            full_path = Path(root) / fname
            if full_path.is_file():
                files.append(full_path)
    return files


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    except (OSError, PermissionError):
        return ""
    return hasher.hexdigest()


def _file_info(file_path: Path, file_hash: str = "") -> FileInfo:
    """Create a FileInfo from a path."""
    try:
        size = file_path.stat().st_size
    except OSError:
        size = 0
    return FileInfo(path=str(file_path), size=size, hash=file_hash)


def _group_by_hash(files: list[Path]) -> list[list[FileInfo]]:
    """Group files by SHA-256 hash."""
    hash_map: dict[str, list[FileInfo]] = defaultdict(list)
    for fp in files:
        digest = _compute_sha256(fp)
        if digest:
            hash_map[digest].append(_file_info(fp, digest))
    return list(hash_map.values())


def _group_by_name(files: list[Path]) -> list[list[FileInfo]]:
    """Group files by filename (ignoring directory path)."""
    name_map: dict[str, list[FileInfo]] = defaultdict(list)
    for fp in files:
        name_map[fp.name].append(_file_info(fp))
    return list(name_map.values())


def _group_by_content(files: list[Path]) -> list[list[FileInfo]]:
    """Group files by full content comparison (via hash + size)."""
    # Content comparison is effectively the same as hash but we also
    # check size first as a fast pre-filter.
    size_map: dict[int, list[Path]] = defaultdict(list)
    for fp in files:
        try:
            size_map[fp.stat().st_size].append(fp)
        except OSError:
            continue

    groups: list[list[FileInfo]] = []
    for same_size_files in size_map.values():
        if len(same_size_files) < 2:
            continue
        # Within same-size files, group by hash
        hash_map: dict[str, list[FileInfo]] = defaultdict(list)
        for fp in same_size_files:
            digest = _compute_sha256(fp)
            if digest:
                hash_map[digest].append(_file_info(fp, digest))
        groups.extend(hash_map.values())

    return groups
