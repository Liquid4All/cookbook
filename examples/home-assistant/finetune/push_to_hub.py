"""Upload the home-assistant SFT dataset to HuggingFace Hub.

Usage:
  uv run --group finetune finetune/push_to_hub.py \\
      --hub-repo USERNAME/home-assistant-sft \\
      [--dataset-path benchmark/datasets/TIMESTAMP_golden_dataset.jsonl]

The script auto-discovers the latest JSONL in benchmark/datasets/ when
--dataset-path is omitted.  It converts each line to a HF Dataset row and
pushes with a 90/10 train/test split as a private repo.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push SFT dataset to HuggingFace Hub")
    p.add_argument(
        "--hub-repo",
        required=True,
        help="HF Hub dataset repo, e.g. USERNAME/home-assistant-sft",
    )
    p.add_argument(
        "--dataset-path",
        default=None,
        help="Path to the golden JSONL file. Auto-detected when omitted.",
    )
    p.add_argument(
        "--public",
        action="store_true",
        help="Push as a public dataset (default: private)",
    )
    return p.parse_args()


def find_latest_dataset() -> Path:
    datasets_dir = Path(__file__).parent.parent / "benchmark" / "datasets"
    candidates = sorted(datasets_dir.glob("*_golden_dataset.jsonl"))
    if not candidates:
        sys.exit(
            f"No golden dataset found in {datasets_dir}. "
            "Run benchmark/generate_dataset.py first."
        )
    latest = candidates[-1]
    print(f"Auto-detected dataset: {latest}")
    return latest


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_path) if args.dataset_path else find_latest_dataset()

    print(f"Loading {dataset_path} ...")
    rows = load_jsonl(dataset_path)
    print(f"  {len(rows)} examples loaded")

    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        sys.exit("Run: uv sync --group finetune")

    ds = Dataset.from_list(rows)

    split = ds.train_test_split(test_size=0.1, seed=42)
    dataset_dict = DatasetDict({"train": split["train"], "test": split["test"]})

    print(
        f"  train: {len(dataset_dict['train'])} examples, "
        f"test: {len(dataset_dict['test'])} examples"
    )

    print(f"Pushing to {args.hub_repo} ...")
    dataset_dict.push_to_hub(args.hub_repo, private=not args.public)
    print("Done.")


if __name__ == "__main__":
    main()
