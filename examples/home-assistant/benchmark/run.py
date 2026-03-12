import sys
import copy
import time
import signal
import argparse
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

from app.agent import run_agent, get_model_name
from app.state import home_state
from benchmark.tasks import TASKS

# Capture default state once at import time, before any task mutates it
_DEFAULT_STATE = copy.deepcopy(home_state)


@dataclass
class TaskResult:
    task_id: int
    name: str
    difficulty: str
    passed: bool
    tool_called: str | None
    args_correct: bool
    duration_s: float


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


def run_task(task, backend: str = "local") -> TaskResult:
    # Reset home state to defaults before each task so results are order-independent
    home_state.clear()
    home_state.update(copy.deepcopy(_DEFAULT_STATE))

    tool_calls_seen = []

    def capture(name, args, result):
        tool_calls_seen.append({"name": name, "args": args})

    start = time.time()
    run_agent(task.prompt, history=task.history, backend=backend, on_tool_call=capture)
    duration = time.time() - start

    final_state = copy.deepcopy(home_state)
    return task.verifier(tool_calls_seen, duration, final_state)


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
    slug = model_name.replace("/", "_")
    out_path = Path("benchmark/results") / f"{timestamp}_{slug}.md"
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
    parser.add_argument("--hf-repo", default=None, help="HuggingFace repo ID for the GGUF model")
    parser.add_argument("--hf-file", default=None, help="GGUF filename within the repo")
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

        tasks = [TASKS[args.task - 1]] if args.task else TASKS
        model_name = get_model_name(args.backend)
        print(f"Backend: {args.backend} ({model_name})")
        results = [run_task(t, backend=args.backend) for t in tasks]
        print_results(results, args.backend, model_name)
        out_path = save_results(results, args.backend, model_name)
        print(f"Results saved to {out_path}")

    finally:
        if server_proc is not None:
            print("Shutting down llama-server...")
            server_proc.send_signal(signal.SIGTERM)
            server_proc.wait()
