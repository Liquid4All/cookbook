"""Run an evaluation against a generated dataset.

--split is required: use 'test' to evaluate model quality, 'train' to check Opus self-consistency.

Usage:
    uv run scripts/evaluate.py --hf-dataset Paulescu/wildfire-prevention --backend anthropic --split test
    uv run scripts/evaluate.py --hf-dataset Paulescu/wildfire-prevention --backend anthropic --split train
    uv run scripts/evaluate.py --hf-dataset Paulescu/wildfire-prevention --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0 --split test
    uv run scripts/evaluate.py --dataset data/20260421_150039 --backend anthropic --split test
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

from huggingface_hub import snapshot_download

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
    transformers_backend,
    wait_for_server,
)

EVALS_DIR = Path(__file__).parent.parent / "evals"

# (sample_id, rgb_bytes, swir_bytes, ground_truth)
SampleData: TypeAlias = tuple[str, bytes, bytes, dict[str, object]]


def load_local_samples(dataset_dir: Path, split: str) -> list[SampleData]:
    """Load samples from a local run directory (train|test/{loc}/{tile}/ layout)."""
    split_dir = dataset_dir / split
    if not split_dir.is_dir():
        print(f"Split '{split}' not found in {dataset_dir}")
        sys.exit(1)

    samples: list[SampleData] = []
    for loc_dir in sorted(split_dir.iterdir()):
        if not loc_dir.is_dir():
            continue
        for tile_dir in sorted(loc_dir.iterdir()):
            if not tile_dir.is_dir():
                continue
            sample_id = f"{loc_dir.name}/{tile_dir.name}"
            rgb_path = tile_dir / "rgb.png"
            swir_path = tile_dir / "swir.png"
            annotation_path = tile_dir / "annotation.json"
            if not (rgb_path.exists() and swir_path.exists() and annotation_path.exists()):
                print(f"[{sample_id}] SKIP: missing files")
                continue
            ground_truth: dict[str, object] = json.loads(
                annotation_path.read_text(encoding="utf-8")
            )
            samples.append((
                sample_id,
                rgb_path.read_bytes(),
                swir_path.read_bytes(),
                ground_truth,
            ))
    return samples


def load_hf_samples(snapshot_dir: Path, split: str) -> list[SampleData]:
    """Load samples from a HF snapshot (parquet + flat images/ layout)."""
    from datasets import load_dataset

    ds = load_dataset(str(snapshot_dir), split=split)
    samples: list[SampleData] = []
    for row in ds:
        region    = str(row["region"])
        rgb_path  = snapshot_dir / str(row["rgb_path"])
        swir_path = snapshot_dir / str(row["swir_path"])
        # derive tile key from filename: e.g. attica_greece_s00_t00_rgb.png → s00_t00
        tile_key  = Path(str(row["rgb_path"])).stem.removesuffix("_rgb")[len(region) + 1:]
        sample_id = f"{region}/{tile_key}"
        ground_truth: dict[str, object] = json.loads(str(row["output"]))
        samples.append((
            sample_id,
            rgb_path.read_bytes(),
            swir_path.read_bytes(),
            ground_truth,
        ))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate wildfire risk predictions.")
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        help="Path to a local dataset run, e.g. data/20260421_150039.",
    )
    parser.add_argument(
        "--hf-dataset",
        metavar="REPO",
        help="Hugging Face dataset repo to evaluate against, e.g. Paulescu/wildfire-prevention. Downloads via snapshot_download and uses the cached copy.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["anthropic", "local", "hf"],
        help="Inference backend: 'anthropic' (Opus API), 'local' (llama-server GGUF), 'hf' (HuggingFace safetensors checkpoint).",
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
        "--split",
        required=True,
        choices=["train", "test"],
        help="Which data split to evaluate: 'train' checks Opus self-consistency, 'test' evaluates model quality.",
    )
    parser.add_argument(
        "--verbose-server",
        action="store_true",
        help="Show llama-server output (local backend only).",
    )
    args = parser.parse_args()

    if not args.dataset and not args.hf_dataset:
        print("Either --dataset or --hf-dataset is required.")
        sys.exit(1)
    if args.dataset and args.hf_dataset:
        print("--dataset and --hf-dataset are mutually exclusive.")
        sys.exit(1)

    if args.backend in ("local", "hf") and not args.model:
        print("--model is required when using --backend local or hf")
        sys.exit(1)

    if args.backend == "local" and not shutil.which("llama-server"):
        print("llama-server not found on PATH. Install llama.cpp and ensure llama-server is available.")
        sys.exit(1)

    if args.hf_dataset:
        print(f"Downloading dataset from Hugging Face: {args.hf_dataset} ...")
        snapshot_dir = Path(snapshot_download(repo_id=args.hf_dataset, repo_type="dataset"))
        print(f"Snapshot at {snapshot_dir}")
        samples = load_hf_samples(snapshot_dir, args.split)
        dataset_label = args.hf_dataset
    else:
        local_dir = Path(args.dataset)
        if not local_dir.is_dir():
            print(f"Dataset not found: {local_dir}")
            sys.exit(1)
        samples = load_local_samples(local_dir, args.split)
        dataset_label = args.dataset

    if not samples:
        print(f"No samples found for split '{args.split}'.")
        sys.exit(1)

    concurrency = args.concurrency or (1 if args.backend == "local" else 3)
    eval_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(
        f"Eval: {eval_run_id}  |  dataset: {dataset_label}  |  split: {args.split}"
        f"  |  samples: {len(samples)}  |  backend: {args.backend}"
        f"  |  concurrency: {concurrency}"
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

    if args.backend == "anthropic":
        predict = anthropic_backend()
    elif args.backend == "hf":
        print(f"Loading HuggingFace checkpoint from {args.model} ...")
        predict = transformers_backend(args.model)
        print("Model loaded.")
    else:
        predict = llama_backend(args.model, args.port)

    results: list[SampleResult] = []
    sample_order = {sid: i for i, (sid, *_) in enumerate(samples)}
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(evaluate_sample, sid, rgb, swir, gt, predict): sid
                for sid, rgb, swir, gt in samples
            }
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

    results.sort(key=lambda r: sample_order.get(r.id, 999))

    summary = EvalSummary(results=results)
    mname = model_name(args.backend, args.model, args.quant)
    report = render_report(summary, f"{dataset_label}/{args.split}", args.backend, mname, eval_run_id)

    eval_dir = EVALS_DIR / eval_run_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "report.md").write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"Report saved to evals/{eval_run_id}/report.md")


if __name__ == "__main__":
    main()
