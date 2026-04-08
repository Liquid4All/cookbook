"""
VRSBench -> leap-finetune VLM SFT format converter.

Downloads VRSBench from HuggingFace and converts to JSONL files
compatible with leap-finetune's vlm_sft training type.

Tasks (auto-detected from [tag] prefixes in the data):
  - vqa:        85K visual question answering samples
  - grounding:  36K visual grounding / object detection samples
  - captioning: 20K image captioning samples
  - all:        142K combined multi-task dataset (shuffled)

Usage:
    python prepare_vrsbench.py --task all --data-dir /path/to/vrsbench --output /path/to/train.jsonl
    python prepare_vrsbench.py --task vqa --limit 5000

Requires: huggingface_hub, tqdm
    pip install huggingface_hub tqdm
"""

import argparse
import json
import random
import re
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "xiang709/VRSBench"
REPO_TYPE = "dataset"

# Modal configuration
MODAL_VOLUME_NAME = "satellite-vlm"
MODAL_MOUNT_POINT = "/satellite-vlm"
MODAL_DATA_DIR = f"{MODAL_MOUNT_POINT}/data/vrsbench"

# Grounding prompt aligned with LFM VLM pretraining format
GROUNDING_PROMPT = (
    "Inspect the image and detect the {target}. "
    'Provide result as a valid JSON: [{{"label": str, "bbox": [x1,y1,x2,y2]}}, ...]. '
    "Coordinates must be normalized to 0-1."
)
CAPTIONING_PROMPT = "Describe this satellite image in detail."


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_vrsbench(data_dir: str) -> Path:
    """Download VRSBench from HuggingFace Hub."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    images_dir = data_path / "images"
    annotations_ready = (data_path / "VRSBench_train.json").exists()
    images_ready = images_dir.exists() and any(images_dir.iterdir())

    if annotations_ready and images_ready:
        print(f"VRSBench data already present at {data_path}")
        return data_path

    # Download annotation files
    json_files = [
        "VRSBench_train.json",
        "VRSBench_EVAL_vqa.json",
        "VRSBench_EVAL_referring.json",
        "VRSBench_EVAL_Cap.json",
    ]
    for fname in json_files:
        dest = data_path / fname
        if not dest.exists():
            print(f"Downloading {fname}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                repo_type=REPO_TYPE,
                local_dir=str(data_path),
            )

    # Download and extract image zips
    images_dir.mkdir(exist_ok=True)
    for zip_name in ["Images_train.zip", "Images_val.zip"]:
        zip_path = data_path / zip_name
        if not zip_path.exists():
            print(f"Downloading {zip_name} (this may take a while)...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=zip_name,
                repo_type=REPO_TYPE,
                local_dir=str(data_path),
            )
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # Flatten: strip parent dirs (e.g. Images_train/) so all images
                # end up directly in images_dir
                member.filename = Path(member.filename).name
                zf.extract(member, images_dir)
        zip_path.unlink()

    print(f"VRSBench ready at {data_path}")
    return data_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vlm_message(image_filename: str, user_text: str, assistant_text: str) -> dict:
    """Create a single VLM SFT training sample in leap-finetune format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_filename},
                    {"type": "text", "text": user_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
    }


def parse_bbox_tokens(text: str) -> list[float] | None:
    """Parse VRSBench bbox tokens like {<25><40><33><60>} to [0.25, 0.40, 0.33, 0.60]."""
    nums = re.findall(r"<(\d+)>", text)
    if len(nums) != 4:
        return None
    coords = [int(n) / 100.0 for n in nums]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(c, 4) for c in coords]


def extract_referring_expression(text: str) -> str:
    """Extract referring expression from <p>...</p> tags."""
    match = re.search(r"<p>(.*?)</p>", text)
    return match.group(1) if match else ""


def write_jsonl(samples: list[dict], output_path: str) -> None:
    """Write samples to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"  Wrote {len(samples):,} samples to {output_path}")


# ---------------------------------------------------------------------------
# Convert training data (ShareGPT format with [tag] prefixes)
# ---------------------------------------------------------------------------

def convert_train(data: list[dict], task_filter: str | None = None) -> dict[str, list[dict]]:
    """Convert VRSBench training data to VLM SFT format.

    Returns dict mapping task name -> list of samples.
    """
    results: dict[str, list[dict]] = {"vqa": [], "grounding": [], "captioning": []}

    for item in tqdm(data, desc="Converting training data"):
        image = item.get("image", "")
        conv = item.get("conversations", [])
        if len(conv) < 2 or not image:
            continue

        human_text = conv[0].get("value", "")
        gpt_text = conv[1].get("value", "")

        if "[vqa]" in human_text:
            if task_filter and task_filter != "vqa":
                continue
            # Extract question: strip "<image>\n[vqa] " prefix and " A short answer..." suffix
            question = human_text.replace("<image>\n", "").replace("[vqa] ", "")
            question = re.sub(r"\.\s*A short answer to the question is\s*$", "", question).strip()
            if question and gpt_text:
                results["vqa"].append(make_vlm_message(image, question, gpt_text))

        elif "[refer]" in human_text:
            if task_filter and task_filter != "grounding":
                continue
            ref_expr = extract_referring_expression(human_text)
            bbox = parse_bbox_tokens(gpt_text)
            if ref_expr and bbox:
                user_text = GROUNDING_PROMPT.format(target=ref_expr)
                bbox_json = json.dumps([{"label": ref_expr, "bbox": bbox}])
                results["grounding"].append(make_vlm_message(image, user_text, bbox_json))

        elif "[caption]" in human_text:
            if task_filter and task_filter != "captioning":
                continue
            if gpt_text:
                results["captioning"].append(make_vlm_message(image, CAPTIONING_PROMPT, gpt_text))

    return results


# ---------------------------------------------------------------------------
# Convert eval data (flat format with image_id, question, ground_truth)
# ---------------------------------------------------------------------------

def convert_eval_vqa(data: list[dict]) -> list[dict]:
    samples = []
    for item in data:
        image = item.get("image_id", "")
        question = item.get("question", "")
        answer = item.get("ground_truth", "")
        if image and question and answer:
            samples.append(make_vlm_message(image, question, answer))
    return samples


def convert_eval_grounding(data: list[dict]) -> list[dict]:
    samples = []
    for item in data:
        image = item.get("image_id", "")
        ref_expr = item.get("question", "")
        gt_text = item.get("ground_truth", "")
        obj_cls = item.get("obj_cls", ref_expr)

        bbox = parse_bbox_tokens(gt_text)
        if not bbox:
            # Fall back to obj_corner if token parsing fails
            corners = item.get("obj_corner", [])
            if len(corners) == 8:
                xs = [corners[i] for i in range(0, 8, 2)]
                ys = [corners[i] for i in range(1, 8, 2)]
                bbox = [
                    round(max(0, min(xs)), 4),
                    round(max(0, min(ys)), 4),
                    round(min(1, max(xs)), 4),
                    round(min(1, max(ys)), 4),
                ]
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

        if not image or not ref_expr or not bbox:
            continue

        user_text = GROUNDING_PROMPT.format(target=ref_expr)
        bbox_json = json.dumps([{"label": obj_cls, "bbox": bbox}])
        samples.append(make_vlm_message(image, user_text, bbox_json))
    return samples


def convert_eval_captioning(data: list[dict]) -> list[dict]:
    samples = []
    for item in data:
        image = item.get("image_id", "")
        caption = item.get("ground_truth", "")
        if image and caption:
            samples.append(make_vlm_message(image, CAPTIONING_PROMPT, caption))
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert VRSBench to leap-finetune VLM SFT format"
    )
    parser.add_argument(
        "--task",
        choices=["vqa", "grounding", "captioning", "all"],
        required=True,
        help="Task to convert (all = mixed multi-task training + separate per-task evals)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data/vrsbench",
        help="Directory to download/store VRSBench data (default: ./data/vrsbench)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path for training data",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per task (useful for quick testing)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading, assume data already exists",
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

    data_path = Path(args.data_dir)

    if not args.skip_download:
        download_vrsbench(args.data_dir)

    # --- Load and convert training data ---
    print("Loading training annotations...")
    with open(data_path / "VRSBench_train.json") as f:
        raw_train = json.load(f)
    print(f"Loaded {len(raw_train):,} training annotations")

    task_filter = None if args.task == "all" else args.task
    train_by_task = convert_train(raw_train, task_filter=task_filter)

    # Apply limit
    if args.limit:
        for t in train_by_task:
            train_by_task[t] = train_by_task[t][: args.limit]

    # Determine output directory
    if args.output:
        out_dir = str(Path(args.output).parent)
    else:
        out_dir = str(data_path)

    # Write training data
    if args.task == "all":
        all_train = []
        for t in train_by_task:
            all_train.extend(train_by_task[t])
        random.Random(42).shuffle(all_train)
        output = args.output or f"{out_dir}/vrsbench_multitask_train.jsonl"
        print(f"\nTraining data (mixed):")
        write_jsonl(all_train, output)
    else:
        output = args.output or f"{out_dir}/vrsbench_{args.task}_train.jsonl"
        print(f"\nTraining data:")
        write_jsonl(train_by_task[args.task], output)

    # --- Load and convert eval data (always separate per task) ---
    print("\nEval data (separate per task):")

    eval_converters = {
        "vqa": ("VRSBench_EVAL_vqa.json", convert_eval_vqa),
        "grounding": ("VRSBench_EVAL_referring.json", convert_eval_grounding),
        "captioning": ("VRSBench_EVAL_Cap.json", convert_eval_captioning),
    }

    tasks_to_eval = ["vqa", "grounding", "captioning"] if args.task == "all" else [args.task]

    for task in tasks_to_eval:
        eval_file, converter = eval_converters[task]
        eval_path = data_path / eval_file
        if eval_path.exists():
            with open(eval_path) as f:
                raw_eval = json.load(f)
            eval_samples = converter(raw_eval)
            if args.limit:
                eval_samples = eval_samples[: args.limit]
            write_jsonl(eval_samples, f"{out_dir}/vrsbench_{task}_eval.jsonl")
        else:
            print(f"  Warning: {eval_file} not found, skipping {task} eval")

    # --- Summary ---
    total_train = sum(len(v) for v in train_by_task.values())
    print(f"\n=== Summary ===")
    for t, samples in train_by_task.items():
        if samples:
            print(f"  {t}: {len(samples):,} training samples")
    print(f"  Total: {total_train:,} training samples")
    print(f"\nRun training:")
    print(f"  uv run leap-finetune cookbook/satellite-vlm/configs/vrsbench_multitask.yaml")


def _run_on_modal(args: argparse.Namespace) -> None:
    """Run the data preparation pipeline on Modal (no local GPU or large disk required)."""
    import modal

    app = modal.App("satellite-vlm-data-prep")
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("huggingface_hub", "tqdm")
        .add_local_file(__file__, "/app/prepare_vrsbench.py", copy=True)
    )

    @app.function(
        image=image,
        volumes={MODAL_MOUNT_POINT: volume},
        timeout=3600,
        serialized=True,
    )
    def prepare(task: str, limit: int | None, skip_download: bool) -> None:
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "/app/prepare_vrsbench.py",
            "--task", task,
            "--data-dir", "/satellite-vlm/data/vrsbench",
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        if skip_download:
            cmd.append("--skip-download")
        subprocess.run(cmd, check=True)
        volume.commit()

    print(f"Preparing VRSBench on Modal (volume: '{MODAL_VOLUME_NAME}')...")
    with modal.enable_output():
        with app.run():
            prepare.remote(args.task, args.limit, args.skip_download)

    print(f"\nData ready in Modal volume '{MODAL_VOLUME_NAME}'.")
    print(f"Next step: uv run leap-finetune cookbook/examples/satellite-vlm/configs/vrsbench_multitask_modal.yaml")


if __name__ == "__main__":
    main()
