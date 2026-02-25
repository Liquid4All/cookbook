"""
Shared fixtures for document server tests.

Sets up sys.path, sandbox, and temp directories.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest

# ─── Setup Import Paths ─────────────────────────────────────────────────────
# The _shared/py/ modules use relative imports (from .mcp_base import ...),
# so we need to load them as a proper package. We register them under
# "localcowork_shared" to avoid conflicts with the system 'py' package.

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
mcp_base = _load_shared_module("mcp_base", "mcp_base.py")

# Load validation (depends on mcp_base — now in sys.modules)
validation = _load_shared_module("validation", "validation.py")


@pytest.fixture(autouse=True)
def _setup_sandbox() -> None:
    """Initialize sandbox with temp dirs for all tests."""
    validation.init_sandbox([  # type: ignore[attr-defined]
        tempfile.gettempdir(),
        "/private/var/folders",
        "/private/tmp",
        "/tmp",
    ])


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory inside the sandbox."""
    return tmp_path
