import sys
import time
import argparse
from dataclasses import dataclass

sys.path.insert(0, "..")

from app.agent import run_agent
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


def run_task(task) -> TaskResult:
    tool_calls_seen = []

    def capture(name, args, result):
        tool_calls_seen.append({"name": name, "args": args})

    start = time.time()
    run_agent(task.prompt, on_tool_call=capture)
    duration = time.time() - start

    return task.verifier(tool_calls_seen, duration)


def print_results(results: list[TaskResult]) -> None:
    print()
    print(f"{'#':<4} {'Task':<42} {'Diff':<8} {'Pass':<6} {'Tool':<24} {'Time':>6}")
    print("-" * 96)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        tool = r.tool_called or "-"
        print(f"{r.task_id:<4} {r.name:<42} {r.difficulty:<8} {status:<6} {tool:<24} {r.duration_s:>5.1f}s")

    passed = sum(r.passed for r in results)
    total = len(results)
    print("-" * 96)
    print(f"\nScore: {passed}/{total}  ({100 * passed / total:.0f}%)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run home assistant benchmark tasks.")
    parser.add_argument("--task", type=int, default=None, help="Run a single task by number (1-11)")
    args = parser.parse_args()

    tasks = [TASKS[args.task - 1]] if args.task else TASKS
    results = [run_task(t) for t in tasks]
    print_results(results)
