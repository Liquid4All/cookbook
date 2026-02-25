"""
Shared fixtures for security server tests.

Sets up sys.path, sandbox, and temp directories with sample files
containing PII, secrets, and duplicates for testing.
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
# their module names to avoid conflicts.

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

# Load validation (depends on mcp_base -- now in sys.modules)
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


# ─── Sample Files ──────────────────────────────────────────────────────────


@pytest.fixture()
def pii_file(tmp_dir: Path) -> Path:
    """Create a sample file containing PII data."""
    f = tmp_dir / "pii_sample.txt"
    f.write_text(
        "Customer records:\n"
        "Name: John Doe, SSN: 123-45-6789\n"
        "Email: john.doe@example.com\n"
        "Phone: (555) 123-4567\n"
        "Card: 4111 1111 1111 1111\n"
        "Notes: nothing sensitive here\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture()
def secrets_file(tmp_dir: Path) -> Path:
    """Create a sample file containing exposed secrets."""
    f = tmp_dir / "secrets_sample.env"
    f.write_text(
        "# Configuration\n"
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
        'api_key = "sk_test_FAKE0000000000000000000000"\n'
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBogIBAAJBAL0FGRAa...\n"
        "-----END RSA PRIVATE KEY-----\n"
        "NORMAL_VAR=hello\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture()
def duplicate_dir(tmp_dir: Path) -> Path:
    """Create a directory with duplicate files."""
    content_a = "This is file content A.\n"
    content_b = "This is file content B.\n"

    # Create duplicates by content
    (tmp_dir / "file1.txt").write_text(content_a, encoding="utf-8")
    (tmp_dir / "file2.txt").write_text(content_a, encoding="utf-8")
    (tmp_dir / "file3.txt").write_text(content_b, encoding="utf-8")

    # Create duplicates by name in subdirectories
    sub = tmp_dir / "subdir"
    sub.mkdir()
    (sub / "file1.txt").write_text("different content", encoding="utf-8")

    return tmp_dir


@pytest.fixture()
def plaintext_file(tmp_dir: Path) -> Path:
    """Create a sample plaintext file for encryption tests."""
    f = tmp_dir / "secret_document.txt"
    f.write_text(
        "This is a confidential document.\n"
        "It contains sensitive business information.\n",
        encoding="utf-8",
    )
    return f
