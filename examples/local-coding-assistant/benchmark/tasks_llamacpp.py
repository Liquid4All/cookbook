"""Benchmark task suite: llama.cpp (github.com/ggerganov/llama.cpp).

Run against a local clone of the repo:

    git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
    uv run python benchmark/run.py --backend anthropic \\
        --suite llamacpp --working-dir /tmp/llama.cpp

Tasks 1-6: read-only (no cleanup needed)
Tasks 7-10: the agent must write and run a Python script; verifiers execute
            the script and check its output programmatically.
"""
from __future__ import annotations

import re
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
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd, timeout=30
    )
    return (result.stdout + result.stderr).strip()


# ── Verifiers for hard tasks ──────────────────────────────────────────────────

def _verify_api_surface(stdout: str, cwd: Path) -> bool:
    """Task 7: api_surface.py must exist and print ≥ 20 llama_ function names."""
    if not (cwd / "api_surface.py").exists():
        return False
    out = _run("python api_surface.py", cwd)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return sum(1 for l in lines if "llama_" in l) >= 20


def _verify_module_sizes(stdout: str, cwd: Path) -> bool:
    """Task 8: module_sizes.py must exist and print ≥ 5 lines with .cpp + a number."""
    if not (cwd / "module_sizes.py").exists():
        return False
    out = _run("python module_sizes.py", cwd)
    valid = [l for l in out.splitlines() if ".cpp" in l and re.search(r"\d+", l)]
    return len(valid) >= 5


def _verify_count_samplers(stdout: str, cwd: Path) -> bool:
    """Task 9: count_samplers.py must print ≥ 1 llama_sampler_init_* name and a count > 0."""
    if not (cwd / "count_samplers.py").exists():
        return False
    out = _run("python count_samplers.py", cwd)
    has_init = "llama_sampler_init_" in out
    nums = re.findall(r"\b(\d+)\b", out)
    has_positive = any(int(n) > 0 for n in nums)
    return has_init and has_positive


def _verify_list_archs(stdout: str, cwd: Path) -> bool:
    """Task 10: list_archs.py must print both 'llama' and 'mistral'."""
    if not (cwd / "list_archs.py").exists():
        return False
    out = _run("python list_archs.py", cwd).lower()
    return "llama" in out and "mistral" in out


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS: list[Task] = [
    # ── Easy ─────────────────────────────────────────────────────────────────
    Task(
        id=1,
        difficulty="easy",
        name="List top-level directories",
        prompt="List all top-level directories in this repository",
        verifier=lambda stdout, cwd: all(
            d in stdout for d in ["include", "src", "examples"]
        ),
    ),
    Task(
        id=2,
        difficulty="easy",
        name="Describe the project",
        prompt="Read README.md and give me a one-sentence description of what llama.cpp is",
        verifier=lambda stdout, cwd: (
            "llama" in stdout.lower()
            and any(w in stdout.lower() for w in ["inference", "model", "llm"])
        ),
    ),
    Task(
        id=3,
        difficulty="easy",
        name="Count examples",
        prompt=(
            "How many example programs are in the examples/ directory?"
            " Count the subdirectories and give me the number."
        ),
        verifier=lambda stdout, cwd: any(
            int(n) >= 10 for n in re.findall(r"\d+", stdout)
        ),
    ),
    # ── Medium ────────────────────────────────────────────────────────────────
    Task(
        id=4,
        difficulty="medium",
        name="Find the public API header",
        prompt=(
            "Where is the main public API header for this library?"
            " Find it and tell me its path."
        ),
        verifier=lambda stdout, cwd: "include/llama.h" in stdout,
    ),
    Task(
        id=5,
        difficulty="medium",
        name="Count LLAMA_API functions",
        prompt=(
            "How many public API functions are declared in include/llama.h?"
            " Count lines that contain LLAMA_API and give me the number."
        ),
        verifier=lambda stdout, cwd: (
            (lambda nums, gt: gt in nums)(
                re.findall(r"\b\d+\b", stdout),
                _run("grep -c 'LLAMA_API' include/llama.h", cwd),
            )
        ),
    ),
    Task(
        id=6,
        difficulty="medium",
        name="Explain llama_context",
        prompt=(
            "Read include/llama.h and explain what llama_context is"
            " and what it is used for"
        ),
        verifier=lambda stdout, cwd: (
            "llama_context" in stdout
            and any(
                w in stdout.lower()
                for w in ["context", "inference", "state", "session"]
            )
        ),
    ),
    # ── Hard ──────────────────────────────────────────────────────────────────
    Task(
        id=7,
        difficulty="hard",
        name="Script: extract API surface",
        prompt=(
            "Write a Python script api_surface.py that reads include/llama.h,"
            " extracts every function name declared with LLAMA_API, and prints"
            " them sorted alphabetically one per line. Then run it."
        ),
        verifier=_verify_api_surface,
        cleanup_files=["api_surface.py"],
    ),
    Task(
        id=8,
        difficulty="hard",
        name="Script: rank modules by size",
        prompt=(
            "Write a Python script module_sizes.py that counts the lines of code"
            " in every .cpp file under src/, sorts them largest-first, and prints"
            " the top 10 with their line counts. Then run it."
        ),
        verifier=_verify_module_sizes,
        cleanup_files=["module_sizes.py"],
    ),
    Task(
        id=9,
        difficulty="hard",
        name="Script: list sampler initializers",
        prompt=(
            "Read src/llama-sampler.cpp to understand the sampling API."
            " Write a Python script count_samplers.py that finds all sampler"
            " initializer functions defined in that file (names matching"
            " llama_sampler_init_*), prints each name, and prints the total"
            " count at the end. Then run it."
        ),
        verifier=_verify_count_samplers,
        cleanup_files=["count_samplers.py"],
    ),
    Task(
        id=10,
        difficulty="hard",
        name="Script: list supported architectures",
        prompt=(
            "Read src/llama-arch.cpp to understand how model architectures are"
            " registered. Write a Python script list_archs.py that parses that"
            " file and prints every architecture name it finds (look for string"
            " literals associated with architecture definitions). Then run it."
        ),
        verifier=_verify_list_archs,
        cleanup_files=["list_archs.py"],
    ),
]
