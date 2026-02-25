"""Tests for security.find_duplicates tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.find_duplicates import FindDuplicates


@pytest.fixture()
def tool() -> FindDuplicates:
    """Create a FindDuplicates tool instance."""
    return FindDuplicates()


async def test_find_duplicates_by_hash(tool: FindDuplicates, duplicate_dir: Path) -> None:
    """Should find files with identical content via hash comparison."""
    params = tool.get_params_model()(path=str(duplicate_dir), method="hash")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # file1.txt and file2.txt have the same content
    assert len(result.data.groups) >= 1
    # At least one group should have 2+ files
    has_dup = any(len(group) >= 2 for group in result.data.groups)
    assert has_dup


async def test_find_duplicates_by_name(tool: FindDuplicates, duplicate_dir: Path) -> None:
    """Should find files with the same filename in different directories."""
    params = tool.get_params_model()(path=str(duplicate_dir), method="name")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # file1.txt exists in root and subdir
    name_groups = result.data.groups
    file1_group = [
        g for g in name_groups if any("file1.txt" in fi.path for fi in g)
    ]
    assert len(file1_group) >= 1
    assert len(file1_group[0]) >= 2


async def test_find_duplicates_by_content(tool: FindDuplicates, duplicate_dir: Path) -> None:
    """Should find files with identical content via content comparison."""
    params = tool.get_params_model()(path=str(duplicate_dir), method="content")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    # file1.txt and file2.txt have the same content and size
    has_dup = any(len(group) >= 2 for group in result.data.groups)
    assert has_dup


async def test_no_duplicates(tool: FindDuplicates, tmp_dir: Path) -> None:
    """Should return empty groups when all files are unique."""
    (tmp_dir / "unique1.txt").write_text("content A", encoding="utf-8")
    (tmp_dir / "unique2.txt").write_text("content B", encoding="utf-8")
    (tmp_dir / "unique3.txt").write_text("content C", encoding="utf-8")

    params = tool.get_params_model()(path=str(tmp_dir))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    assert len(result.data.groups) == 0


async def test_default_method_is_hash(tool: FindDuplicates, duplicate_dir: Path) -> None:
    """Should default to hash method when not specified."""
    params = tool.get_params_model()(path=str(duplicate_dir))
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None
    # Should behave like hash method: find content duplicates
    has_dup = any(len(group) >= 2 for group in result.data.groups)
    assert has_dup


async def test_invalid_method(tool: FindDuplicates, duplicate_dir: Path) -> None:
    """Should reject invalid method parameter."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(duplicate_dir), method="invalid")
    with pytest.raises(MCPError, match="Invalid method"):
        await tool.execute(params)


async def test_path_not_found(tool: FindDuplicates, tmp_dir: Path) -> None:
    """Should raise error for missing path."""
    from mcp_base import MCPError

    params = tool.get_params_model()(path=str(tmp_dir / "nonexistent"))
    with pytest.raises(MCPError, match="Path not found"):
        await tool.execute(params)


async def test_path_must_be_directory(tool: FindDuplicates, tmp_dir: Path) -> None:
    """Should raise error when path is a file, not a directory."""
    from mcp_base import MCPError

    f = tmp_dir / "file.txt"
    f.write_text("hello", encoding="utf-8")

    params = tool.get_params_model()(path=str(f))
    with pytest.raises(MCPError, match="must be a directory"):
        await tool.execute(params)


async def test_file_info_has_hash(tool: FindDuplicates, duplicate_dir: Path) -> None:
    """FileInfo objects should include SHA-256 hash for hash method."""
    params = tool.get_params_model()(path=str(duplicate_dir), method="hash")
    result = await tool.execute(params)

    assert result.success is True
    assert result.data is not None

    for group in result.data.groups:
        for file_info in group:
            assert len(file_info.hash) == 64  # SHA-256 hex digest length


def test_metadata(tool: FindDuplicates) -> None:
    """Should have correct tool metadata."""
    assert tool.name == "security.find_duplicates"
    assert tool.confirmation_required is False
    assert tool.undo_supported is False
