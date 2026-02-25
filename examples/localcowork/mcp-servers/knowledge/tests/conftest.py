"""
Shared fixtures for knowledge server tests.

Sets up sys.path, provides an in-memory SQLite database, and creates
temporary directories populated with sample text files.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from pathlib import Path

import pytest

# ─── Setup Import Paths ─────────────────────────────────────────────────────
# The _shared/py/ modules use relative imports (from .mcp_base import ...),
# so we need to load them as a proper package.

_shared_py_dir = Path(__file__).resolve().parent.parent.parent / "_shared" / "py"
_src = str(Path(__file__).resolve().parent.parent / "src")
_tools = str(Path(__file__).resolve().parent.parent / "src" / "tools")

# Add src paths for tool imports
for p in (_src, _tools):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_shared_module(name: str, file_name: str) -> types.ModuleType:
    """Load a module from _shared/py/ and register it in sys.modules."""
    module_path = _shared_py_dir / file_name
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load mcp_base first (no dependencies)
_load_shared_module("mcp_base", "mcp_base.py")

# Load validation (depends on mcp_base — now in sys.modules)
_load_shared_module("validation", "validation.py")


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _in_memory_db() -> None:  # type: ignore[misc]
    """
    Provide a fresh in-memory SQLite DB for every test.

    Automatically injects the connection via ``db.set_db`` and tears
    it down after each test so tests never share state.
    """
    import db as db_module

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    db_module._init_schema(conn)
    db_module.set_db(conn)
    yield
    db_module.close_db()


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory with sample text files for indexing tests.

    Structure:
        tmp_path/
        ├── readme.md
        ├── notes.txt
        └── sub/
            └── deep.txt
    """
    (tmp_path / "readme.md").write_text(
        "# Project README\n\n"
        "This is a sample project for testing the knowledge server.\n\n"
        "It includes multiple files across directories."
    )
    (tmp_path / "notes.txt").write_text(
        "Meeting notes from Monday.\n\n"
        "Discussed the roadmap for Q3.\n\n"
        "Action items:\n- Review PRD\n- Update backlog\n- Schedule follow-up"
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.txt").write_text(
        "This file lives in a subdirectory.\n\n"
        "It tests recursive indexing behaviour."
    )
    return tmp_path


@pytest.fixture()
def indexed_dir(sample_dir: Path) -> Path:
    """
    Return a sample directory that has already been indexed.

    Runs index_folder under the hood so search / ask / related tests
    can assume data is present.
    """
    import asyncio

    from tools.index_folder import IndexFolder, Params

    tool = IndexFolder()
    params = Params(path=str(sample_dir), recursive=True)
    asyncio.get_event_loop().run_until_complete(tool.execute(params))
    return sample_dir
