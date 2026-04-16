#!/usr/bin/env python3
"""Aggregate all SFT JSONL files in this directory into sft_data_agg.jsonl."""

from pathlib import Path

DATASETS_DIR = Path(__file__).parent
OUTPUT_FILE = DATASETS_DIR / "sft_data_agg.jsonl"


def main():
    jsonl_files = sorted(
        p for p in DATASETS_DIR.rglob("*.jsonl")
        if p.resolve() != OUTPUT_FILE.resolve()
    )

    if not jsonl_files:
        print("No JSONL files found.")
        return

    total = 0
    with OUTPUT_FILE.open("w") as out:
        for path in jsonl_files:
            count = 0
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")
                        count += 1
            print(f"  {path.relative_to(DATASETS_DIR)}: {count} rows")
            total += count

    print(f"\nWrote {total} rows to {OUTPUT_FILE.relative_to(DATASETS_DIR)}")


if __name__ == "__main__":
    main()
