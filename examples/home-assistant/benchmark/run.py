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
    prompt: str
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
        prompt=task.prompt,
        capability=results[0].capability,
        phrasing=results[0].phrasing,
        depth=results[0].depth,
        pass_rate=pass_rate,
        std_dev=std_dev,
        mean_duration_s=mean_dur,
        n_runs=len(results),
    )


def _wait_for_server(timeout: int = 600, port: int = 8080) -> None:
    """Poll http://localhost:<port>/v1/models until the server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"llama-server did not become ready on port {port} within timeout")


def start_llama_server(
    hf_repo: str = None,
    hf_file: str = None,
    model_path: str = None,
    llama_server_bin: str = "llama-server",
    port: int = 8080,
) -> subprocess.Popen:
    import os
    env = os.environ.copy()
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if "HF_TOKEN" not in env and hf_token_path.exists():
        env["HF_TOKEN"] = hf_token_path.read_text().strip()
    if model_path:
        cmd = [llama_server_bin, "--model", model_path, "--port", str(port), "--ctx-size", "4096", "--n-gpu-layers", "99"]
    else:
        cmd = [llama_server_bin, "--hf-repo", hf_repo, "--hf-file", hf_file, "--port", str(port), "--ctx-size", "4096", "--n-gpu-layers", "99"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None, env=env)
    _wait_for_server(timeout=600, port=port)
    return proc


def run_task(task, backend: str = "local", n: int = 1, reset_state: bool = True, raw_tool_call_parsing: bool = False, port: int = 8080, debug: bool = False) -> tuple[list[TaskResult], list[dict] | None]:
    results = []
    last_assistant_turns = None
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

        messages_out = [] if debug else None
        start = time.time()
        run_agent(task.prompt, history=getattr(task, "history", []), backend=backend, on_tool_call=capture, messages_out=messages_out, raw_tool_call_parsing=raw_tool_call_parsing, port=port)
        duration = time.time() - start

        final_state = copy.deepcopy(home_state)
        results.append(task.verifier(tool_calls_seen, duration, final_state))

        if debug:
            last_assistant_turns = [m for m in messages_out if m.get("role") == "assistant"]

    return results, last_assistant_turns


def format_results(agg_results: list[AggregatedResult], backend: str, model_name: str) -> str:
    lines = []
    n_runs = agg_results[0].n_runs if agg_results else 1
    multi = n_runs > 1

    if multi:
        lines.append(f"{'#':<4} {'Query':<46} {'Cap':<12} {'Phrasing':<12} {'Depth':<10} {'Pass%':<8} {'Std':<6} {'Time':>7}")
        lines.append("-" * 112)
        for r in agg_results:
            pct = f"{100 * r.pass_rate:.0f}%"
            std = f"{r.std_dev:.2f}"
            query = r.prompt if len(r.prompt) <= 46 else r.prompt[:43] + "..."
            lines.append(
                f"{r.task_id:<4} {query:<46} {r.capability:<12} {r.phrasing:<12} "
                f"{r.depth:<10} {pct:<8} {std:<6} {r.mean_duration_s:>6.1f}s"
            )
    else:
        lines.append(f"{'#':<4} {'Query':<46} {'Cap':<12} {'Phrasing':<12} {'Depth':<10} {'Pass':<6} {'Time':>7}")
        lines.append("-" * 108)
        for r in agg_results:
            status = "PASS" if r.pass_rate == 1.0 else "FAIL"
            query = r.prompt if len(r.prompt) <= 46 else r.prompt[:43] + "..."
            lines.append(
                f"{r.task_id:<4} {query:<46} {r.capability:<12} {r.phrasing:<12} "
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


def get_llama_build_date(llama_server_bin: str) -> str:
    """Return the commit date (YYYY-MM-DD) for a llama-server binary.
    Extracts commit hash from --version output, queries GitHub API.
    Caches result in a .build_date file next to the binary.
    Returns 'unknown' on any failure.
    """
    import re
    import json

    bin_path = Path(llama_server_bin)
    cache_file = bin_path.parent / ".build_date"

    if cache_file.exists():
        return cache_file.read_text().strip()

    # Parse commit hash from --version
    try:
        result = subprocess.run(
            [llama_server_bin, "--version"],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout + result.stderr
        m = re.search(r"version:\s+\d+\s+\(([0-9a-f]+)\)", output)
        if not m:
            return "unknown"
        sha = m.group(1)
    except Exception:
        return "unknown"

    # Query GitHub Commits API
    try:
        url = f"https://api.github.com/repos/ggml-org/llama.cpp/commits/{sha}"
        req = urllib.request.Request(url, headers={"User-Agent": "benchmark/run.py"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        date = data["commit"]["author"]["date"][:10]  # YYYY-MM-DD
        cache_file.write_text(date)
        return date
    except Exception:
        return "unknown"


def save_results(agg_results: list[AggregatedResult], backend: str, model_name: str, n_runs: int, args) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = model_name.replace("/", "_")
    suffix = f"_n{n_runs}" if n_runs > 1 else ""
    out_dir = Path("benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timestamp}_{slug}{suffix}.md"

    # --- Parameters table ---
    params = [
        ("backend", backend),
        ("model", model_name),
        ("hf_repo", getattr(args, "hf_repo", None)),
        ("hf_file", getattr(args, "hf_file", None)),
        ("local_file", getattr(args, "local_file", None)),
        ("llama_build", getattr(args, "llama_build", None)),
        ("llama_build_date", (
            get_llama_build_date(str(Path.home() / ".local" / "llama-cpp" / args.llama_build / "bin" / "llama-server"))
            if getattr(args, "llama_build", None) else "N/A"
        )),
        ("port", getattr(args, "port", None)),
        ("runs", n_runs),
        ("no_reset", getattr(args, "no_reset", False)),
        ("raw_tool_call_parsing", getattr(args, "raw_tool_call_parsing", False)),
    ]
    param_rows = "\n".join(f"| {k:<22} | {str(v):<22} |" for k, v in params)
    params_section = (
        "## Run parameters\n\n"
        "| Parameter              | Value                  |\n"
        "|------------------------|------------------------|\n"
        f"{param_rows}"
    )

    # --- Score ---
    passed = sum(1 for r in agg_results if r.pass_rate == 1.0)
    total = len(agg_results)
    multi = n_runs > 1
    if multi:
        avg_pass = sum(r.pass_rate for r in agg_results) / total if total else 0
        score_section = f"## Score\n\n{avg_pass * 100:.1f}% avg pass rate ({n_runs} runs/task, {total} tasks)"
    else:
        score_section = f"## Score\n\n{passed}/{total} ({100 * passed / total:.0f}%)"

    # --- Breakdowns ---
    breakdown_lines = []

    cap_groups: dict[str, list[AggregatedResult]] = {}
    for r in agg_results:
        cap_groups.setdefault(r.capability, []).append(r)
    breakdown_lines.append("**By capability:**")
    for cap, group in sorted(cap_groups.items()):
        avg = sum(r.pass_rate for r in group) / len(group)
        breakdown_lines.append(f"  {cap:<12} {avg * 100:>5.1f}%  ({len(group)} tasks)")

    phrasing_groups: dict[str, list[AggregatedResult]] = {}
    for r in agg_results:
        phrasing_groups.setdefault(r.phrasing or "untagged", []).append(r)
    breakdown_lines.append("\n**By phrasing:**")
    for phrasing, group in sorted(phrasing_groups.items()):
        avg = sum(r.pass_rate for r in group) / len(group)
        breakdown_lines.append(f"  {phrasing:<12} {avg * 100:>5.1f}%  ({len(group)} tasks)")

    depth_groups: dict[str, list[AggregatedResult]] = {}
    for r in agg_results:
        depth_groups.setdefault(r.depth, []).append(r)
    breakdown_lines.append("\n**By inference depth:**")
    for depth in ("literal", "semantic", "boundary"):
        group = depth_groups.get(depth, [])
        if group:
            avg = sum(r.pass_rate for r in group) / len(group)
            breakdown_lines.append(f"  {depth:<10} {avg * 100:>5.1f}%  ({len(group)} tasks)")

    breakdown_section = "## Breakdown\n\n" + "\n".join(breakdown_lines)

    # --- Task table ---
    table_lines = []
    if multi:
        table_lines.append(f"{'#':<4} {'Query':<46} {'Cap':<12} {'Phrasing':<12} {'Depth':<10} {'Pass%':<8} {'Std':<6} {'Time':>7}")
        table_lines.append("-" * 112)
        for r in agg_results:
            pct = f"{100 * r.pass_rate:.0f}%"
            std = f"{r.std_dev:.2f}"
            query = r.prompt if len(r.prompt) <= 46 else r.prompt[:43] + "..."
            table_lines.append(
                f"{r.task_id:<4} {query:<46} {r.capability:<12} {r.phrasing:<12} "
                f"{r.depth:<10} {pct:<8} {std:<6} {r.mean_duration_s:>6.1f}s"
            )
    else:
        table_lines.append(f"{'#':<4} {'Query':<46} {'Cap':<12} {'Phrasing':<12} {'Depth':<10} {'Pass':<6} {'Time':>7}")
        table_lines.append("-" * 108)
        for r in agg_results:
            status = "PASS" if r.pass_rate == 1.0 else "FAIL"
            query = r.prompt if len(r.prompt) <= 46 else r.prompt[:43] + "..."
            table_lines.append(
                f"{r.task_id:<4} {query:<46} {r.capability:<12} {r.phrasing:<12} "
                f"{r.depth:<10} {status:<6} {r.mean_duration_s:>6.1f}s"
            )
    tasks_section = "## Tasks\n\n```\n" + "\n".join(table_lines) + "\n```"

    content = (
        f"# Benchmark run: {timestamp}\n\n"
        f"{params_section}\n\n"
        f"{score_section}\n\n"
        f"{breakdown_section}\n\n"
        f"{tasks_section}\n"
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
    parser.add_argument("--local-file", default=None, help="Path to a local GGUF file to serve with llama-server")
    parser.add_argument(
        "--llama-build",
        default=None,
        help=(
            "llama.cpp build version to use (e.g. 7930 or b8533). "
            "When set, uses ~/.local/llama-cpp/<version>/bin/llama-server "
            "instead of the llama-server resolved from PATH."
        ),
    )
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for llama-server (default 8080). Use different ports to run benchmarks in parallel.")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per task for statistical reliability (default 1)")
    parser.add_argument("--no-reset", action="store_true",
                        help="Skip state reset between tasks (useful for debugging multi-turn chains)")
    parser.add_argument(
        "--raw-tool-call-parsing",
        action="store_true",
        help="Enable post-processing to parse tool calls from raw model output (for LFM2 text-format models).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="For each failing task, print the assistant turns (tool calls and/or content) to stdout.",
    )
    args = parser.parse_args()

    if bool(args.hf_repo) != bool(args.hf_file):
        parser.error("--hf-repo and --hf-file must be used together")

    if args.backend == "local" and not args.hf_repo and not args.hf_file and not args.local_file:
        parser.error("When using the local backend you must specify either --hf-repo/--hf-file or --local-file.")

    llama_server_bin = "llama-server"
    if args.llama_build:
        resolved = Path.home() / ".local" / "llama-cpp" / args.llama_build / "bin" / "llama-server"
        if not resolved.exists():
            parser.error(
                f"--llama-build {args.llama_build!r}: binary not found at {resolved}"
            )
        llama_server_bin = str(resolved)

    server_proc = None
    model_name_override = None
    try:
        if args.local_file:
            if args.backend != "local":
                print(f"Warning: --local-file ignored for backend '{args.backend}'")
            else:
                print(f"Starting llama-server ({Path(args.local_file).name})...")
                server_proc = start_llama_server(model_path=args.local_file, llama_server_bin=llama_server_bin, port=args.port)
                model_name_override = Path(args.local_file).name
                print("llama-server ready.")
        elif args.hf_repo and args.hf_file:
            if args.backend != "local":
                print(f"Warning: --hf-repo/--hf-file ignored for backend '{args.backend}'")
            else:
                print(f"Starting llama-server ({args.hf_file})...")
                server_proc = start_llama_server(args.hf_repo, args.hf_file, llama_server_bin=llama_server_bin, port=args.port)
                model_name_override = args.hf_file
                print("llama-server ready.")

        task_map = {t.id: t for t in TASKS}
        tasks = [task_map[args.task]] if args.task else TASKS
        model_name = model_name_override or get_model_name(args.backend, port=args.port)
        print(f"Backend: {args.backend} ({model_name})")
        all_agg = []
        for task in tasks:
            results, assistant_turns = run_task(task, backend=args.backend, n=args.runs, reset_state=not args.no_reset, raw_tool_call_parsing=args.raw_tool_call_parsing, port=args.port, debug=args.debug)
            agg = aggregate_results(task, results)
            all_agg.append(agg)
            if args.debug and agg.pass_rate < 1.0 and assistant_turns:
                prompt_preview = task.prompt if len(task.prompt) <= 50 else task.prompt[:47] + "..."
                print(f"FAIL  {task.id:<4} {prompt_preview}")
                for turn in assistant_turns:
                    tool_calls = turn.get("tool_calls") or []
                    content = turn.get("content") or ""
                    if tool_calls:
                        calls_str = ", ".join(
                            f"{tc['function']['name']}({', '.join(f'{k}={v!r}' for k, v in tc['function'].get('arguments', {}).items())})"
                            if isinstance(tc.get('function', {}).get('arguments'), dict)
                            else f"{tc['function']['name']}({tc['function'].get('arguments', '')})"
                            for tc in tool_calls
                        )
                        print(f"      tool_calls: [{calls_str}]")
                    if content:
                        preview = content if len(content) <= 120 else content[:117] + "..."
                        print(f"      content:    {preview!r}")
        print_results(all_agg, args.backend, model_name)
        out_path = save_results(all_agg, args.backend, model_name, n_runs=args.runs, args=args)
        print(f"Results saved to {out_path}")

    finally:
        if server_proc is not None:
            print("Shutting down llama-server...")
            server_proc.send_signal(signal.SIGTERM)
            server_proc.wait()
