"""Benchmark runner for local-coding-assistant.

Usage:
    python benchmark/run.py --backend anthropic [--model claude-sonnet-4-6] [--task 1,2,3]
    python benchmark/run.py --backend local --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0

    # llama.cpp suite (requires a local clone of the repo)
    python benchmark/run.py --backend anthropic --suite llamacpp --working-dir /tmp/llama.cpp
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# Make sure the project src and benchmark/ dir are importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from local_coding_assistant.agent import Agent
from local_coding_assistant.config import Config
from local_coding_assistant.context import ContextManager
from local_coding_assistant.llm import LLMResponse, get_llm_client
from local_coding_assistant.server import start_local_server
from local_coding_assistant.tools import set_working_directory


# ── Metrics dataclass ─────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_id: int
    task_name: str
    difficulty: str
    passed: bool
    duration_s: float
    input_tokens: int
    output_tokens: int
    tool_calls: list[str]
    n_turns: int
    stdout: str
    error: str | None


# ── Instrumented LLM wrapper ──────────────────────────────────────────────────

class InstrumentedLLMClient:
    """Wraps any LLMClient to capture per-task token counts and call counts."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_turns = 0
        self.tool_calls: list[str] = []

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        response = self._inner.chat(messages=messages, tools=tools, system=system)
        self.n_turns += 1
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        for block in response.content:
            if block.get("type") == "tool_use":
                self.tool_calls.append(block["name"])
        return response

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_turns = 0
        self.tool_calls = []


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(
    task: Task,
    instrumented: InstrumentedLLMClient,
    config: Config,
    working_dir: Path,
) -> TaskResult:
    """Run one task in a fresh agent context; return collected metrics."""
    instrumented.reset()
    set_working_directory(str(working_dir))

    # Fresh agent per task (clean context window)
    agent = Agent(llm=instrumented, config=config)

    captured = io.StringIO()
    error: str | None = None

    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(captured):
            agent.run_turn(task.prompt)
    except Exception as exc:
        error = str(exc)
    duration = time.perf_counter() - start

    stdout = captured.getvalue()

    try:
        passed = task.verifier(stdout, working_dir)
    except Exception as exc:
        passed = False
        error = error or f"Verifier error: {exc}"

    # Clean up any files the agent was asked to create
    for fname in task.cleanup_files:
        p = working_dir / fname
        if p.exists():
            p.unlink()

    return TaskResult(
        task_id=task.id,
        task_name=task.name,
        difficulty=task.difficulty,
        passed=passed,
        duration_s=round(duration, 2),
        input_tokens=instrumented.input_tokens,
        output_tokens=instrumented.output_tokens,
        tool_calls=list(instrumented.tool_calls),
        n_turns=instrumented.n_turns,
        stdout=stdout,
        error=error,
    )


# ── Output helpers ────────────────────────────────────────────────────────────

def print_summary(
    results: list[TaskResult], backend: str, model: str, date: str
) -> None:
    print(f"\nModel : {model} ({backend})")
    print(f"Date  : {date}\n")

    col_task = 40
    header = (
        f"{'#':<4} {'Task':<{col_task}} {'Pass':<6} {'Time':>8}  "
        f"{'In/Out tokens':>18}  {'Turns':>5}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in results:
        sym = "\u2713" if r.passed else "\u2717"
        tokens = f"{r.input_tokens}/{r.output_tokens}"
        print(
            f"{r.task_id:<4} {r.task_name:<{col_task}} {sym:<6} "
            f"{r.duration_s:>7.1f}s  {tokens:>18}  {r.n_turns:>5}"
        )

    print(sep)
    n_passed = sum(1 for r in results if r.passed)
    total_tokens = sum(r.input_tokens + r.output_tokens for r in results)
    avg_time = sum(r.duration_s for r in results) / len(results) if results else 0.0
    print(
        f"\nScore: {n_passed}/{len(results)}  |  "
        f"Total tokens: {total_tokens}  |  "
        f"Avg time: {avg_time:.1f}s"
    )


def save_results(
    results: list[TaskResult], suite: str, backend: str, model: str, date: str
) -> Path:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = model.replace("/", "-").replace(":", "-").replace(" ", "_")
    out_file = results_dir / f"{timestamp}-{suite}-{backend}-{model_slug}.json"

    report = {
        "suite": suite,
        "backend": backend,
        "model": model,
        "date": date,
        "tasks": [asdict(r) for r in results],
    }
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    return out_file


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local-coding-assistant benchmark"
    )
    parser.add_argument(
        "--backend", required=True, choices=["anthropic", "local"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (Anthropic model ID or HuggingFace/GGUF path for local)",
    )
    parser.add_argument(
        "--task", default=None,
        help="Comma-separated task IDs to run (default: all 10)",
    )
    parser.add_argument(
        "--working-dir", default=None,
        help="Working directory for agent tool calls (default: project root)",
    )
    parser.add_argument(
        "--suite", default="default", choices=["default", "llamacpp"],
        help="Task suite to run: 'default' (local-coding-assistant) or 'llamacpp'",
    )
    args = parser.parse_args()

    # Load the chosen task suite
    if args.suite == "llamacpp":
        from tasks_llamacpp import TASKS
    else:
        from tasks import TASKS

    # Determine which tasks to run
    if args.task:
        wanted = {int(t.strip()) for t in args.task.split(",")}
        tasks_to_run = [t for t in TASKS if t.id in wanted]
        if not tasks_to_run:
            parser.error(f"No tasks found for IDs: {args.task}")
    else:
        tasks_to_run = TASKS

    # Build config
    config = Config()
    config.backend = args.backend  # type: ignore[assignment]
    if args.model:
        if config.backend == "anthropic":
            config.anthropic_model = args.model
        else:
            config.local_model = args.model

    model_name = (
        config.anthropic_model if config.backend == "anthropic" else config.local_model
    )

    working_dir = (
        Path(args.working_dir).resolve() if args.working_dir
        else (_PROJECT_ROOT if args.suite == "default" else Path.cwd())
    )

    # Start local server if needed
    server_proc: subprocess.Popen | None = None
    if config.backend == "local" and args.model:
        server_proc = start_local_server(config)

    try:
        # Build instrumented LLM client (shared; reset per task)
        raw_llm = get_llm_client(config)
        instrumented = InstrumentedLLMClient(raw_llm)

        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        print(
            f"Running {len(tasks_to_run)} task(s) | "
            f"suite={args.suite} | backend={args.backend} | model={model_name}"
        )
        print(f"Working dir: {working_dir}\n")

        results: list[TaskResult] = []
        for task in tasks_to_run:
            print(f"  [{task.id:>2}/10] {task.name} ...", end=" ", flush=True)
            result = run_task(task, instrumented, config, working_dir)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} ({result.duration_s:.1f}s)")
            results.append(result)

        print_summary(results, args.backend, model_name, date_str)

        out_file = save_results(results, args.suite, args.backend, model_name, date_str)
        print(f"\nResults saved to: {out_file}")

    finally:
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait()


if __name__ == "__main__":
    main()
