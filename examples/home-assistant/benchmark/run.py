import sys
import time
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.agent import run_agent, get_model_name
from benchmark.tasks import TASKS


@dataclass
class TaskResult:
    task_id: int
    name: str
    difficulty: str
    passed: bool
    tool_called: str | None
    args_correct: bool
    duration_s: float


def run_task(task, backend: str = "local") -> TaskResult:
    tool_calls_seen = []

    def capture(name, args, result):
        tool_calls_seen.append({"name": name, "args": args})

    start = time.time()
    run_agent(task.prompt, history=task.history, backend=backend, on_tool_call=capture)
    duration = time.time() - start

    return task.verifier(tool_calls_seen, duration)


def format_results(results: list[TaskResult], backend: str, model_name: str) -> str:
    lines = []
    lines.append(f"{'#':<4} {'Task':<42} {'Diff':<8} {'Pass':<6} {'Tool':<24} {'Time':>6}")
    lines.append("-" * 96)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        tool = r.tool_called or "-"
        lines.append(f"{r.task_id:<4} {r.name:<42} {r.difficulty:<8} {status:<6} {tool:<24} {r.duration_s:>5.1f}s")
    passed = sum(r.passed for r in results)
    total = len(results)
    lines.append("-" * 96)
    lines.append(f"\nScore: {passed}/{total}  ({100 * passed / total:.0f}%)")
    return "\n".join(lines)


def save_results(results: list[TaskResult], backend: str, model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path("benchmark/results") / f"{timestamp}_{backend}.md"
    content = f"# Benchmark run: {timestamp}\n\nBackend: {backend} ({model_name})\n\n```\n{format_results(results, backend, model_name)}\n```\n"
    out_path.write_text(content)
    return out_path


def print_results(results: list[TaskResult], backend: str, model_name: str) -> None:
    print()
    print(format_results(results, backend, model_name))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run home assistant benchmark tasks.")
    parser.add_argument("--task", type=int, default=None, help="Run a single task by number (1-11)")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "openai"],
        help="Backend to use: 'local' (llama-server) or 'openai' (gpt-4o-mini). Default: local.",
    )
    args = parser.parse_args()

    tasks = [TASKS[args.task - 1]] if args.task else TASKS
    model_name = get_model_name(args.backend)
    print(f"Backend: {args.backend} ({model_name})")
    results = [run_task(t, backend=args.backend) for t in tasks]
    print_results(results, args.backend, model_name)
    out_path = save_results(results, args.backend, model_name)
    print(f"Results saved to {out_path}")
