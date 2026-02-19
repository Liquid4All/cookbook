"""
Shared fixtures for meeting server tests.

Sets up sys.path so that shared MCP base classes and the server's own
source modules are importable.  Provides temporary audio files for
transcription tests and sample transcripts for extraction tests.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ─── Setup Import Paths ─────────────────────────────────────────────────────
# The _shared/py/ modules need to be loaded explicitly so they can be found
# when the tool modules import from ``mcp_base``.

_shared_py_dir = Path(__file__).resolve().parent.parent.parent / "_shared" / "py"
_src = str(Path(__file__).resolve().parent.parent / "src")
_tools = str(Path(__file__).resolve().parent.parent / "src" / "tools")

# Add src paths for tool imports (meeting_types, transcription, tools.*)
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


# ─── Audio File Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    """Create a minimal WAV file for transcription tests.

    Writes a valid-enough file header followed by dummy audio bytes.
    The stub engine only checks existence and file size, so the actual
    audio content doesn't matter.
    """
    f = tmp_path / "sample.wav"
    # Write ~64 KB of dummy data (~2 seconds at the stub's estimate rate)
    f.write_bytes(b"\x00" * 64_000)
    return f


@pytest.fixture()
def mp3_file(tmp_path: Path) -> Path:
    """Create a dummy MP3 file for extension-validation tests."""
    f = tmp_path / "sample.mp3"
    f.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 32_000)
    return f


@pytest.fixture()
def unsupported_file(tmp_path: Path) -> Path:
    """Create a file with an unsupported extension."""
    f = tmp_path / "sample.aac"
    f.write_bytes(b"\x00" * 1_000)
    return f


@pytest.fixture()
def nonexistent_path(tmp_path: Path) -> Path:
    """Return a path that does not exist."""
    return tmp_path / "does_not_exist.wav"


# ─── Transcript Fixtures (WS-5B extraction tests) ───────────────────────────


@pytest.fixture()
def action_items_transcript() -> str:
    """Transcript containing various action item patterns."""
    return (
        "John: Let's review the project status.\n"
        "Sarah: ACTION: John will review the proposal by Friday.\n"
        "John: I agree. TODO: Sarah needs to update the budget spreadsheet.\n"
        "Mike: @Lisa will handle the client presentation.\n"
        "Sarah: This is urgent, we need to finalize ASAP.\n"
        "John: Mike should coordinate with the design team.\n"
    )


@pytest.fixture()
def commitments_transcript() -> str:
    """Transcript containing commitments, decisions, and open questions."""
    return (
        "Alice: I will finish the report by Monday.\n"
        "Bob: I'll send the updated numbers to the team.\n"
        "Alice: We decided to use React for the frontend.\n"
        "Bob: Agreed: the deadline is March 15th.\n"
        "Alice: How should we handle authentication?\n"
        "Bob: That's TBD, we need to figure out the SSO integration.\n"
        "Alice: I commit to delivering the API docs by Wednesday.\n"
    )


@pytest.fixture()
def full_meeting_transcript() -> str:
    """A complete meeting transcript with speakers, actions, and decisions."""
    return (
        "John: Welcome everyone. Let's start with the sprint review.\n"
        "\n"
        "Sarah: The backend API is 90% complete. I will finish the remaining\n"
        "endpoints by Tuesday.\n"
        "\n"
        "Mike: The frontend needs more work. TODO: Mike will fix the login page.\n"
        "John: We decided to delay the launch by one week.\n"
        "Sarah: How should we handle the database migration?\n"
        "Mike: That's still TBD.\n"
        "\n"
        "John: ACTION: Sarah will write the migration script by Thursday.\n"
        "John: Let's wrap up. Thanks everyone.\n"
    )


@pytest.fixture()
def empty_transcript() -> str:
    """An empty transcript."""
    return ""


@pytest.fixture()
def no_matches_transcript() -> str:
    """A transcript with no action items, commitments, or decisions."""
    return (
        "John: Good morning everyone.\n"
        "Sarah: Good morning.\n"
        "Mike: Hi there.\n"
        "John: Let's get started.\n"
    )
