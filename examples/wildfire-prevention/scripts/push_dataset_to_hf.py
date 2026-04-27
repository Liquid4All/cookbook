"""Push a local generate_samples.py run directory to a HuggingFace dataset repo.

Usage:
    uv run scripts/push_dataset_to_hf.py --run-dir data/20260421_195018 --hf-dataset Paulescu/wildfire-prediction
"""

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_HF_DATASET = "Paulescu/wildfire-prediction"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push a local run directory to a HuggingFace dataset repo."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        metavar="DIR",
        help="Path to the run directory (e.g. data/20260421_195018).",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        metavar="REPO",
        help=f"HuggingFace dataset repo to push to (default: {DEFAULT_HF_DATASET}).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_fields = (
        "risk_level",
        "dry_vegetation_present",
        "urban_interface",
        "steep_terrain",
        "water_body_present",
        "image_quality_limited",
    )

    images_dir = run_dir / "images"
    images_dir.mkdir(exist_ok=True)

    rows: list[dict[str, str]] = []
    for ann_path in sorted(run_dir.glob("*/*/s*_t*/annotation.json")):
        tile_dir = ann_path.parent
        split = tile_dir.parent.parent.name       # train or test
        location = tile_dir.parent.name           # e.g. attica_greece
        tile = tile_dir.name                      # e.g. s00_t01

        ann = json.loads(ann_path.read_text())
        si = int(ann["spatial_index"])
        ti = int(ann["temporal_index"])
        tile_key = f"{location}_s{si:02d}_t{ti:02d}"

        rgb_name  = f"{tile_key}_rgb.png"
        swir_name = f"{tile_key}_swir.png"
        shutil.copy2(tile_dir / "rgb.png",  images_dir / rgb_name)
        shutil.copy2(tile_dir / "swir.png", images_dir / swir_name)

        rows.append({
            "region":    location,
            "timestamp": str(ann["timestamp"]),
            "split":     split,
            "rgb_path":  f"images/{rgb_name}",
            "swir_path": f"images/{swir_name}",
            "output":    json.dumps({k: ann[k] for k in output_fields}),
        })

    if not rows:
        raise RuntimeError(f"No annotation.json files found under {run_dir}")

    from datasets import Dataset, DatasetDict, Features, Value
    from huggingface_hub import HfApi

    features = Features({
        "region":    Value("string"),
        "timestamp": Value("string"),
        "split":     Value("string"),
        "rgb_path":  Value("string"),
        "swir_path": Value("string"),
        "output":    Value("string"),
    })

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows  = [r for r in rows if r["split"] == "test"]
    ds_dict: dict[str, Dataset] = {"train": Dataset.from_list(train_rows, features=features)}
    if test_rows:
        ds_dict["test"] = Dataset.from_list(test_rows, features=features)

    api = HfApi()
    api.create_repo(repo_id=args.hf_dataset, repo_type="dataset", exist_ok=True)

    print(f"Pushing parquet to {args.hf_dataset} (train: {len(train_rows)}, test: {len(test_rows)}) ...")
    DatasetDict(ds_dict).push_to_hub(args.hf_dataset)

    print(f"Uploading {len(rows) * 2} images ...")
    api.upload_folder(
        folder_path=str(images_dir),
        path_in_repo="images",
        repo_id=args.hf_dataset,
        repo_type="dataset",
    )

    print(f"\nDone. Dataset at https://huggingface.co/datasets/{args.hf_dataset}")


if __name__ == "__main__":
    main()
