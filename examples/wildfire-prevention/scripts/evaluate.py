"""Run an evaluation against a generated dataset.

Usage:
    uv run scripts/evaluate.py --dataset data/20260416_141946 --backend anthropic
    uv run scripts/evaluate.py --dataset data/20260416_141946 --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0
    uv run scripts/evaluate.py --dataset data/20260416_141946 --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0 --port 8081
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


from wildfire_prevention.evaluator import (
    EVAL_FIELDS,
    EvalSummary,
    SampleResult,
    anthropic_backend,
    evaluate_sample,
    llama_backend,
    model_name,
    render_report,
    start_llama_server,
    stop_server,
    wait_for_server,
)

EVALS_DIR = Path(__file__).parent.parent / "evals"


def load_sample(sample_dir: Path) -> tuple[bytes, bytes, dict[str, object]] | None:
    rgb_path = sample_dir / "rgb.png"
    swir_path = sample_dir / "swir.png"
    annotation_path = sample_dir / "annotation.json"

    if not (rgb_path.exists() and swir_path.exists() and annotation_path.exists()):
        return None

    ground_truth: dict[str, object] = json.loads(annotation_path.read_text(encoding="utf-8"))
    return rgb_path.read_bytes(), swir_path.read_bytes(), ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wildfire risk predictions.")
    parser.add_argument(
        "--dataset",
        required=True,
        metavar="PATH",
        help="Path to a generated dataset run, e.g. data/20260416_141946.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["anthropic", "local"],
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--model",
        metavar="REPO",
        default="",
        help="HuggingFace repo ID (required for --backend local, e.g. LiquidAI/LFM2.5-VL-450M-GGUF).",
    )
    parser.add_argument(
        "--quant",
        metavar="QUANT",
        default="",
        help="Quantization level within the repo (e.g. Q8_0). Appended as <repo>:<quant>.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="llama-server port (default: 8080, local backend only).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Parallel workers (default: 3 for anthropic, 1 for local).",
    )
    parser.add_argument(
        "--verbose-server",
        action="store_true",
        help="Show llama-server output (local backend only).",
    )
    args = parser.parse_args()

    if args.backend == "local" and not args.model:
        print("--model is required when using --backend local")
        sys.exit(1)

    if args.backend == "local" and not shutil.which("llama-server"):
        print("llama-server not found on PATH. Install llama.cpp and ensure llama-server is available.")
        sys.exit(1)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        print(f"Dataset not found: {dataset_dir}")
        sys.exit(1)

    sample_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not sample_dirs:
        print(f"No samples found in {dataset_dir}")
        sys.exit(1)

    concurrency = args.concurrency or (1 if args.backend == "local" else 3)
    eval_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        f"Eval: {eval_run_id}  |  dataset: {args.dataset}"
        f"  |  backend: {args.backend}  |  concurrency: {concurrency}"
    )

    # Start llama-server if needed.
    server_process = None
    if args.backend == "local":
        print(f"Starting llama-server with model {args.model} on port {args.port} ...")
        server_process = start_llama_server(
            args.model,
            quant=args.quant or None,
            port=args.port,
            verbose=args.verbose_server,
        )
        try:
            wait_for_server(port=args.port)
        except TimeoutError as exc:
            print(str(exc))
            stop_server(server_process)
            sys.exit(1)
        print("llama-server ready.")

    predict = anthropic_backend() if args.backend == "anthropic" else llama_backend(args.model, args.port)

    results: list[SampleResult] = []
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            for sample_dir in sample_dirs:
                loaded = load_sample(sample_dir)
                if loaded is None:
                    print(f"[{sample_dir.name}] SKIP: missing files")
                    continue
                rgb_bytes, swir_bytes, ground_truth = loaded
                future = pool.submit(
                    evaluate_sample,
                    sample_dir.name,
                    rgb_bytes,
                    swir_bytes,
                    ground_truth,
                    predict,
                )
                futures[future] = sample_dir.name

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                fm = result.field_matches
                status = " ".join(
                    f"{f[:4]}={'✓' if fm.get(f) else '✗'}" for f in EVAL_FIELDS
                )
                print(f"[{result.id}] {status}", flush=True)
    finally:
        if server_process is not None:
            stop_server(server_process)


    # Sort results to match dataset order.
    order = {p.name: i for i, p in enumerate(sample_dirs)}
    results.sort(key=lambda r: order.get(r.id, 999))

    summary = EvalSummary(results=results)
    mname = model_name(args.backend, args.model, args.quant)
    report = render_report(summary, args.dataset, args.backend, mname, eval_run_id)

    # Save report.
    eval_dir = EVALS_DIR / eval_run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "report.md").write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"Report saved to evals/{eval_run_id}/report.md")


if __name__ == "__main__":
    main()
