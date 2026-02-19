"""
Shared fixtures for screenshot-pipeline server tests.

Sets up sys.path so tools and shared modules can be imported correctly.
Provides temp directories and sample image files for testing.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ---- Setup Import Paths -----------------------------------------------------
# Load _shared/py/ modules explicitly to avoid import conflicts.

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


# Load mcp_base (no dependencies)
_load_shared_module("mcp_base", "mcp_base.py")

# Load validation (depends on mcp_base)
_load_shared_module("validation", "validation.py")


# ---- Fixtures ----------------------------------------------------------------


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture()
def sample_image(tmp_dir: Path) -> Path:
    """Create a sample image file (stub) for UI element extraction tests."""
    img = tmp_dir / "test_screenshot.png"
    img.write_text("STUB_SCREENSHOT:region=full_screen\n", encoding="utf-8")
    return img
