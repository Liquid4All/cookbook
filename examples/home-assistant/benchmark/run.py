"""
benchmark/run.py
----------------
Runner for the taxonomy-driven benchmark suite.

  uv run python benchmark/run.py                          # local llama-server
  uv run python benchmark/run.py --backend openai         # gpt-4o-mini
  uv run python benchmark/run.py --task <id>              # single task
  uv run python benchmark/run.py --runs 3                 # multiple runs
  uv run python benchmark/run.py --no-reset               # keep state between tasks (debug)

Results are saved to benchmark/results/.
"""

import sys
import copy
import time
import signal
import argparse
import statistics
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.agent import run_agent, get_model_name
from app.state import home_state
from benchmark.tasks import TASKS, TaskResult

# Capture default state once at import time, before any task mutates it
_DEFAULT_STATE = copy.deepcopy(home_state)


@dataclass
class AggregatedResult:
    task_id: int
    name: str
    capability: str
    phrasing: str
    depth: str
    pass_rate: float        # 0.0 to 1.0
    std_dev: float
    mean_duration_s: float
    n_runs: int


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a deep copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def aggregate_results(task, results: list[TaskResult]) -> AggregatedResult:
    pass_rate = sum(r.passed for r in results) / len(results)
    mean_dur = sum(r.duration_s for r in results) / len(results)
    std_dev = statistics.stdev([1.0 if r.passed else 0.0 for r in results]) if len(results) > 1 else 0.0
    return AggregatedResult(
        task_id=task.id,
        name=results[0].name,
        capability=results[0].capability,
        phrasing=results[0].phrasing,
        depth=results[0].depth,
        pass_rate=pass_rate,
        std_dev=std_dev,
        mean_duration_s=mean_dur,
        n_runs=len(results),
    )


def _wait_for_server(timeout: int = 120) -> None:
    """Poll http://localhost:8080/v1/models until the server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen("http://localhost:8080/v1/models", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("llama-server did not become ready within timeout")


def start_llama_server(hf_repo: str, hf_file: str) -> subprocess.Popen:
    cmd = [
        "llama-server",
        "--hf-repo", hf_repo,
        "--hf-file", hf_file,
        "--port", "8080",
        "--ctx-size", "4096",
        "--n-gpu-layers", "99",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _wait_for_server(timeout=120)
    return proc


def run_task(task, backend: str = "local", n: int = 1, reset_state: bool = True, raw_tool_call_parsing: bool = False) -> list[TaskResult]:
    results = []
    for _ in range(n):
        if reset_state:
            # Apply task-specific initial_state on top of defaults
            if task.initial_state:
                merged = _deep_merge(_DEFAULT_STATE, task.initial_state)
            else:
                merged = copy.deepcopy(_DEFAULT_STATE)
            home_state.clear()
            home_state.update(merged)

        tool_calls_seen = []

        def capture(name, args, result):
            tool_calls_seen.append({"name": name, "args": args})

        start = time.time()
        run_agent(task.prompt, history=getattr(task, "history", []), backend=backend, on_tool_call=capture, raw_tool_call_parsing=raw_tool_call_parsing)
        duration = time.time() - start

        final_state = copy.deepcopy(home_state)
        results.append(task.verifier(tool_calls_seen, duration, final_state))
    return results


def format_results(agg_results: list[AggregatedResult], backend: str, model_name: str) -> str:
    lines = []
    n_runs = agg_results[0].n_runs if agg_results else 1
    multi = n_runs > 1

    if multi:
        lines.append(f"{'#':<4} {'Task':<46} {'Cap':<12} {'Phrasing':<12} {'Depth':<10} {'Pass%':<8} {'Std':<6} {'Time':>7}")
        lines.append("-" * 112)
        for r in agg_results:
            pct = f"{100 * r.pass_rate:.0f}%"
            std = f"{r.std_dev:.2f}"
            lines.append(
                f"{r.task_id:<4} {r.name:<46} {r.capability:<12} {r.phrasing:<12} "
                f"{r.depth:<10} {pct:<8} {std:<6} {r.mean_duration_s:>6.1f}s"
            )
    else:
        lines.append(f"{'#':<4} {'Task':<46} {'Cap':<12} {'Phrasing':<12} {'Depth':<10} {'Pass':<6} {'Time':>7}")
        lines.append("-" * 108)
        for r in agg_results:
            status = "PASS" if r.pass_rate == 1.0 else "FAIL"
            lines.append(
                f"{r.task_id:<4} {r.name:<46} {r.capability:<12} {r.phrasing:<12} "
                f"{r.depth:<10} {status:<6} {r.mean_duration_s:>6.1f}s"
            )

    passed = sum(1 for r in agg_results if r.pass_rate == 1.0)
    total = len(agg_results)
    if multi:
        avg_pass = sum(r.pass_rate for r in agg_results) / total if total else 0
        lines.append("-" * 112)
        lines.append(f"\nScore: {avg_pass * 100:.1f}% avg pass rate  ({n_runs} runs/task, {total} tasks)")
    else:
        lines.append("-" * 108)
        lines.append(f"\nScore: {passed}/{total}  ({100 * passed / total:.0f}%)")

    # --- Breakdown by capability ---
    cap_groups: dict[str, list[AggregatedResult]] = {}
    for r in agg_results:
        cap_groups.setdefault(r.capability, []).append(r)

    lines.append("\nBy capability:")
    for cap, group in sorted(cap_groups.items()):
        avg = sum(r.pass_rate for r in group) / len(group)
        lines.append(f"  {cap:<12} {avg * 100:>5.1f}%  ({len(group)} tasks)")

    # --- Breakdown by phrasing ---
    phrasing_groups: dict[str, list[AggregatedResult]] = {}
    for r in agg_results:
        phrasing_groups.setdefault(r.phrasing or "untagged", []).append(r)

    lines.append("\nBy phrasing:")
    for phrasing, group in sorted(phrasing_groups.items()):
        avg = sum(r.pass_rate for r in group) / len(group)
        lines.append(f"  {phrasing:<12} {avg * 100:>5.1f}%  ({len(group)} tasks)")

    # --- Breakdown by depth ---
    depth_groups: dict[str, list[AggregatedResult]] = {}
    for r in agg_results:
        depth_groups.setdefault(r.depth, []).append(r)

    lines.append("\nBy inference depth:")
    for depth in ("literal", "semantic", "boundary"):
        group = depth_groups.get(depth, [])
        if group:
            avg = sum(r.pass_rate for r in group) / len(group)
            lines.append(f"  {depth:<10} {avg * 100:>5.1f}%  ({len(group)} tasks)")

    return "\n".join(lines)


def save_results(agg_results: list[AggregatedResult], backend: str, model_name: str, n_runs: int) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = model_name.replace("/", "_")
    suffix = f"_n{n_runs}" if n_runs > 1 else ""
    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timestamp}_{slug}{suffix}.md"
    content = (
        f"# Benchmark2 run: {timestamp}\n\n"
        f"Backend: {backend} ({model_name})\n\n"
        f"```\n{format_results(agg_results, backend, model_name)}\n```\n"
    )
    out_path.write_text(content)
    return out_path


def print_results(agg_results: list[AggregatedResult], backend: str, model_name: str) -> None:
    print()
    print(format_results(agg_results, backend, model_name))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run taxonomy-driven home assistant benchmark.")
    parser.add_argument("--task", type=int, default=None, help="Run a single task by ID (1-100)")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "openai"],
        help="Backend to use: 'local' (llama-server) or 'openai' (gpt-4o-mini). Default: local.",
    )
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo ID for the GGUF model")
    parser.add_argument("--hf-file", default=None, help="GGUF filename within the repo")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per task for statistical reliability (default 1)")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip state reset between tasks (useful for debugging multi-turn chains)")
    parser.add_argument(
        "--raw-tool-call-parsing",
        action="store_true",
        help="Enable post-processing to parse tool calls from raw model output (for LFM2 text-format models).",
    )
    args = parser.parse_args()

    if bool(args.hf_repo) != bool(args.hf_file):
        parser.error("--hf-repo and --hf-file must be used together")

    server_proc = None
    try:
        if args.hf_repo and args.hf_file:
            if args.backend != "local":
                print(f"Warning: --hf-repo/--hf-file ignored for backend '{args.backend}'")
            else:
                print(f"Starting llama-server ({args.hf_file})...")
                server_proc = start_llama_server(args.hf_repo, args.hf_file)
                print("llama-server ready.")

        task_map = {t.id: t for t in TASKS}
        tasks = [task_map[args.task]] if args.task else TASKS
        model_name = get_model_name(args.backend)
        print(f"Backend: {args.backend} ({model_name})")
        all_agg = []
        for task in tasks:
            results = run_task(task, backend=args.backend, n=args.runs, reset_state=not args.no_reset, raw_tool_call_parsing=args.raw_tool_call_parsing)
            all_agg.append(aggregate_results(task, results))
        print_results(all_agg, args.backend, model_name)
        out_path = save_results(all_agg, args.backend, model_name, n_runs=args.runs)
        print(f"Results saved to {out_path}")

    finally:
        if server_proc is not None:
            print("Shutting down llama-server...")
            server_proc.send_signal(signal.SIGTERM)
            server_proc.wait()
