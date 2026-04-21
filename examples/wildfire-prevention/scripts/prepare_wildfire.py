"""Convert the Paulescu/wildfire-prevention HF dataset to leap-finetune VLM SFT format.

Downloads the dataset from HuggingFace Hub and writes wildfire_train.jsonl
and wildfire_test.jsonl in the leap-finetune messages format. Images are
already in the snapshot and do not need to be copied.

The output directory will contain:
    wildfire_train.jsonl    -- training samples (messages format)
    wildfire_test.jsonl     -- evaluation samples (messages format)

The images/ directory lives inside the HF snapshot (resolved automatically by
huggingface_hub.snapshot_download). The leap-finetune config must set
image_root to <output_dir>/images/.

Usage:
    uv run scripts/prepare_wildfire.py
    uv run scripts/prepare_wildfire.py --dataset Paulescu/wildfire-prevention --output ./data/wildfire
    uv run scripts/prepare_wildfire.py --dataset Paulescu/wildfire-prevention --modal
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from wildfire_prevention.annotator import SYSTEM_PROMPT, USER_TEXT

DEFAULT_DATASET = "Paulescu/wildfire-prevention"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "wildfire"

# Modal configuration
MODAL_VOLUME_NAME = "wildfire-prevention"
MODAL_MOUNT_POINT = "/wildfire-prevention"
MODAL_OUTPUT_DIR = f"{MODAL_MOUNT_POINT}/data/wildfire"


def make_vlm_row(rgb_name: str, swir_name: str, output: str) -> dict[str, object]:
    """Build one leap-finetune VLM SFT row from image filenames and model output."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_name},
                    {"type": "image", "image": swir_name},
                    {"type": "text", "text": f"{SYSTEM_PROMPT.strip()}\n\n{USER_TEXT}"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": output}],
            },
        ]
    }


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"  Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert wildfire-prevention HF dataset to leap-finetune JSONL format."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        metavar="REPO",
        help=f"HuggingFace dataset repo (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="DIR",
        help=f"Directory to write JSONL files (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help=(
            f"Run data preparation on Modal (serverless cloud). "
            f"Writes output to the Modal volume '{MODAL_VOLUME_NAME}' at {MODAL_MOUNT_POINT}/. "
            f"Requires: pip install modal && modal setup"
        ),
    )
    args = parser.parse_args()

    if args.modal:
        _run_on_modal(args)
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading snapshot of {args.dataset} ...")
    snapshot_dir = Path(snapshot_download(repo_id=args.dataset, repo_type="dataset"))
    print(f"  Snapshot at: {snapshot_dir}")

    images_dir = snapshot_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"images/ directory not found in snapshot at {snapshot_dir}. "
            "Re-run generate_samples.py --hf-dataset to regenerate the dataset."
        )

    print(f"Loading dataset from {args.dataset} ...")
    ds = load_dataset(args.dataset)

    for split_name in ("train", "test"):
        if split_name not in ds:
            print(f"  Split '{split_name}' not found, skipping.")
            continue

        rows: list[dict[str, object]] = []
        for row in ds[split_name]:
            rgb_name  = Path(str(row["rgb_path"])).name
            swir_name = Path(str(row["swir_path"])).name
            output    = str(row["output"])
            rows.append(make_vlm_row(rgb_name, swir_name, output))

        write_jsonl(rows, output_dir / f"wildfire_{split_name}.jsonl")

    print(f"\nDone. Set image_root to: {images_dir}")
    print(f"Training config: uv run leap-finetune configs/wildfire_finetune.yaml")


def _run_on_modal(args: argparse.Namespace) -> None:
    """Run the data preparation pipeline on Modal (no local disk or bandwidth required)."""
    import modal

    app = modal.App("wildfire-prevention-data-prep")
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

    src_dir = Path(__file__).parent.parent / "src" / "wildfire_prevention"
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("datasets", "huggingface_hub")
        .add_local_file(__file__, "/app/prepare_wildfire.py", copy=True)
        .add_local_dir(str(src_dir), "/app/wildfire_prevention", copy=True)
    )

    @app.function(
        image=image,
        volumes={MODAL_MOUNT_POINT: volume},
        timeout=3600,
        serialized=True,
    )
    def prepare(dataset: str, output: str) -> None:
        cmd = [
            sys.executable,
            "/app/prepare_wildfire.py",
            "--dataset", dataset,
            "--output", output,
        ]
        env = {**os.environ, "PYTHONPATH": "/app"}
        subprocess.run(cmd, check=True, env=env)
        volume.commit()

    print(f"Preparing wildfire dataset on Modal (volume: '{MODAL_VOLUME_NAME}')...")
    with modal.enable_output():
        with app.run():
            prepare.remote(args.dataset, MODAL_OUTPUT_DIR)

    print(f"\nData ready in Modal volume '{MODAL_VOLUME_NAME}' at {MODAL_OUTPUT_DIR}.")
    print(f"Set image_root to: {MODAL_OUTPUT_DIR}/images/")
    print(f"Next step: uv run leap-finetune configs/wildfire_finetune_modal.yaml")


if __name__ == "__main__":
    main()
