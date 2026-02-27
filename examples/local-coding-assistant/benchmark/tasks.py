"""Benchmark task definitions for local-coding-assistant.

Each Task has:
  - id, difficulty, name: metadata
  - prompt: the string sent to agent.run_turn()
  - verifier(stdout, cwd) -> bool: automated pass/fail check
  - cleanup_files: relative paths to delete after the task runs
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Task:
    id: int
    difficulty: str          # "easy" | "medium" | "hard"
    name: str
    prompt: str
    verifier: Callable[[str, Path], bool]
    cleanup_files: list[str] = field(default_factory=list)


def _run(cmd: str, cwd: Path) -> str:
    """Run a shell command in cwd and return combined stdout+stderr."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=30
    )
    return (result.stdout + result.stderr).strip()


TASKS: list[Task] = [
    # ── Easy ─────────────────────────────────────────────────────────────────
    Task(
        id=1,
        difficulty="easy",
        name="List directory",
        prompt="List all files in the current directory",
        verifier=lambda stdout, cwd: "pyproject.toml" in stdout,
    ),
    Task(
        id=2,
        difficulty="easy",
        name="Read project name",
        prompt="Read the file pyproject.toml and tell me the project name",
        verifier=lambda stdout, cwd: "local-coding-assistant" in stdout,
    ),
    Task(
        id=3,
        difficulty="easy",
        name="Check Python version",
        prompt="What Python version does this project require? Check pyproject.toml",
        verifier=lambda stdout, cwd: "3.13" in stdout,
    ),
    # ── Medium ────────────────────────────────────────────────────────────────
    Task(
        id=4,
        difficulty="medium",
        name="Create and run hello.py",
        prompt="Create a file called hello.py that prints 'Hello, World!' then run it",
        verifier=lambda stdout, cwd: (
            (cwd / "hello.py").exists()
            and "Hello, World!" in _run("python hello.py", cwd)
        ),
        cleanup_files=["hello.py"],
    ),
    Task(
        id=5,
        difficulty="medium",
        name="Count Python lines in src/",
        prompt="Count the total lines of Python code in the src/ directory",
        verifier=lambda stdout, cwd: any(ch.isdigit() for ch in stdout),
    ),
    Task(
        id=6,
        difficulty="medium",
        name="Explain run_turn method",
        prompt="Read agent.py and explain what the run_turn method does",
        verifier=lambda stdout, cwd: (
            "run_turn" in stdout
            and any(w in stdout.lower() for w in ["loop", "tool", "message", "llm", "call"])
        ),
    ),
    # ── Hard ──────────────────────────────────────────────────────────────────
    Task(
        id=7,
        difficulty="hard",
        name="Document tools.py functions",
        prompt=(
            "List all functions defined in tools.py and write a summary of each"
            " to functions.md"
        ),
        verifier=lambda stdout, cwd: (
            (cwd / "functions.md").exists()
            and all(
                fn in (cwd / "functions.md").read_text()
                for fn in ["read_file", "write_file", "list_directory", "run_bash"]
            )
        ),
        cleanup_files=["functions.md"],
    ),
    Task(
        id=8,
        difficulty="hard",
        name="Create count_tools.py",
        prompt=(
            "Create a Python script count_tools.py that imports execute_tool from"
            " local_coding_assistant.tools and prints how many tools are available,"
            " then run it"
        ),
        verifier=lambda stdout, cwd: (
            (cwd / "count_tools.py").exists()
            and "4" in _run("python count_tools.py", cwd)
        ),
        cleanup_files=["count_tools.py"],
    ),
    Task(
        id=9,
        difficulty="hard",
        name="Write CHANGELOG.md",
        prompt=(
            "Run the git log for the last 5 commits and write a CHANGELOG.md"
            " summarizing them"
        ),
        verifier=lambda stdout, cwd: (
            (cwd / "CHANGELOG.md").exists()
            and len((cwd / "CHANGELOG.md").read_text().strip()) > 0
        ),
        cleanup_files=["CHANGELOG.md"],
    ),
    Task(
        id=10,
        difficulty="hard",
        name="Compare LLM backends",
        prompt=(
            "Read all files in src/local_coding_assistant/llm/, understand how the"
            " backends work, and write a comparison of the anthropic vs local backends"
            " to comparison.md"
        ),
        verifier=lambda stdout, cwd: (
            (cwd / "comparison.md").exists()
            and "anthropic" in (cwd / "comparison.md").read_text().lower()
            and "local" in (cwd / "comparison.md").read_text().lower()
        ),
        cleanup_files=["comparison.md"],
    ),
]
