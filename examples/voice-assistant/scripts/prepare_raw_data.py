"""Fork LiquidAI/OHF-Voice-audio-20260504 to Paulescu/OHF-Voice-audio-20260504 with a deterministic
95/5 train/test split, stratified by function name.

This is an author-side one-off. Readers of the voice-assistant tutorial never run it. They consume
the published Paulescu/OHF-Voice-audio-20260504 dataset directly via load_dataset(..., split="train")
or load_dataset(..., split="test").

The motivation for forking is documented in docs/adr/0001-eval-methodology.md: the upstream dataset
ships only a `train` split, and reusing it for both training and evaluation leaks test data into
training. Carving the split once at the data layer (not at preprocess time) makes the disjointness
a property of the published artifact and removes any chance of drift across local environments.

Run once with HF_TOKEN set:

    HF_TOKEN=hf_... uv run python scripts/prepare_raw_data.py

Use --dry-run to compute and inspect the split without pushing.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()

SOURCE_REPO = "LiquidAI/OHF-Voice-audio-20260504"
TARGET_REPO = "Paulescu/OHF-Voice-audio-20260504"
TEST_SIZE = 0.05
SEED = 42


def extract_function_name(row: dict) -> dict[str, str]:
    """Return the function name (the substring before the first '|') from the assistant turn.

    Pulls from `text_chat` rather than `audio_chat` so the audio bytes never need to be decoded
    just to build the stratification key.
    """
    function_call = row["text_chat"][1]["content"][0]["text"]
    return {"_function_name": function_call.split("|")[0]}


def main(*, dry_run: bool) -> None:
    print(f"Loading {SOURCE_REPO} (train split)...")
    ds = load_dataset(SOURCE_REPO, split="train")
    print(f"  {len(ds):,} samples")

    print("Extracting function names (text_chat only, no audio decode)...")
    ds = ds.map(extract_function_name, desc="extracting function names")

    function_names: list[str] = ds["_function_name"]
    counts = Counter(function_names)
    print(f"  {len(counts)} distinct function names")
    print(f"  Most common: {counts.most_common(3)}")
    print(f"  Rarest:      {counts.most_common()[-3:]}")

    print(f"\nStratified split: {1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test (seed={SEED})")
    indices = np.arange(len(ds))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=function_names,
    )
    train_idx_sorted: list[int] = sorted(train_idx.tolist())
    test_idx_sorted: list[int] = sorted(test_idx.tolist())
    print(f"  train: {len(train_idx_sorted):,} samples")
    print(f"  test:  {len(test_idx_sorted):,} samples")

    train_counts = Counter(function_names[i] for i in train_idx_sorted)
    test_counts = Counter(function_names[i] for i in test_idx_sorted)
    print("\nPer-function counts (5 most common + 5 rarest):")
    print(f"  {'function':<45} {'total':>7} {'train':>7} {'test':>6}")
    sample_funcs = [f for f, _ in counts.most_common(5)] + [f for f, _ in counts.most_common()[-5:]]
    for func in sample_funcs:
        print(f"  {func:<45} {counts[func]:>7,} {train_counts[func]:>7,} {test_counts[func]:>6,}")

    print("\nMaterialising splits (dropping _function_name helper column)...")
    splits = DatasetDict(
        {
            "train": ds.select(train_idx_sorted).remove_columns("_function_name"),
            "test": ds.select(test_idx_sorted).remove_columns("_function_name"),
        }
    )

    if dry_run:
        print("\n[dry-run] Skipping push to hub.")
        return

    print(f"\nPushing to {TARGET_REPO}...")
    splits.push_to_hub(TARGET_REPO, private=False)
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the split and print stats without pushing to the hub.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dry_run=args.dry_run)
