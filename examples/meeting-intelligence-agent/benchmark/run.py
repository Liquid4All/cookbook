"""Benchmark runner for the Meeting Intelligence Agent.

Usage:
    cd examples/meeting-intelligence-agent

    # Auto-start llama-server for the given model
    uv run benchmark/run.py --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0

    # Use an already-running llama-server
    uv run benchmark/run.py

    # Run a subset of tasks
    uv run benchmark/run.py --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0 --task 1,2,3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Make src/ importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from meeting_intelligence.agent import Agent
from meeting_intelligence.config import Config
from meeting_intelligence.llm import LLMResponse, get_llm_client
from meeting_intelligence.server import start_local_server

from tasks import Task, TASKS

_DATA_DIR = _PROJECT_ROOT / "data"


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


# ── Data-dir cleanup ──────────────────────────────────────────────────────────

def clean_data_dir(data_dir: Path) -> None:
    """Remove all agent-generated files before each task run."""
    tasks_file = data_dir / "tasks.json"
    emails_log = data_dir / "sent_emails.log"
    summaries_dir = data_dir / "summaries"

    if tasks_file.exists():
        tasks_file.unlink()
    if emails_log.exists():
        emails_log.unlink()
    if summaries_dir.exists():
        for f in summaries_dir.glob("*.md"):
            f.unlink()


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(
    task: Task,
    instrumented: InstrumentedLLMClient,
    config: Config,
    data_dir: Path,
) -> TaskResult:
    """Run one task in a fresh agent context; return collected metrics."""
    instrumented.reset()
    clean_data_dir(data_dir)

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
        passed = task.verifier(stdout, data_dir)
    except Exception as exc:
        passed = False
        error = error or f"Verifier error: {exc}"

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

def print_summary(results: list[TaskResult], model: str, date: str) -> None:
    print(f"\nModel : {model}")
    print(f"Date  : {date}\n")

    col_task = 44
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


def save_results(results: list[TaskResult], model: str, date: str) -> Path:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = model.replace("/", "-").replace(":", "-").replace(" ", "_")
    out_file = results_dir / f"{timestamp}-{model_slug}.json"

    report = {
        "suite": "meeting-intelligence",
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
        description="Run the Meeting Intelligence Agent benchmark"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name or HuggingFace/GGUF path (auto-starts llama-server)",
    )
    parser.add_argument(
        "--task", default=None,
        help="Comma-separated task IDs to run (default: all 10)",
    )
    args = parser.parse_args()

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
    if args.model:
        config.local_model = args.model

    model_name = config.local_model

    # Start local server if a model path was provided
    server_proc: subprocess.Popen | None = None
    if args.model:
        server_proc = start_local_server(config)

    try:
        raw_llm = get_llm_client(config)
        instrumented = InstrumentedLLMClient(raw_llm)
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        print(
            f"Running {len(tasks_to_run)} task(s) | "
            f"suite=meeting-intelligence | model={model_name}"
        )
        print(f"Data dir: {_DATA_DIR}\n")

        total = len(tasks_to_run)
        results: list[TaskResult] = []
        for i, task in enumerate(tasks_to_run, 1):
            print(f"  [{i:>2}/{total}] {task.name} ...", end=" ", flush=True)
            result = run_task(task, instrumented, config, _DATA_DIR)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status} ({result.duration_s:.1f}s)")
            results.append(result)

        print_summary(results, model_name, date_str)

        out_file = save_results(results, model_name, date_str)
        print(f"\nResults saved to: {out_file}")

    finally:
        if server_proc is not None:
            server_proc.terminate()
            server_proc.wait()
        # Clean up after the full run
        clean_data_dir(_DATA_DIR)


if __name__ == "__main__":
    main()
