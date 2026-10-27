"""Shared data loading, prompts, answer parsing and scoring for the wind turbine blade inspection example.

Every evaluation (an API model or a local Hugging Face model) writes one JSONL file of predictions with the
same schema, and the training notebooks use the same split, prompt and scorer, so all results compare directly.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATASET_URL = "https://ndownloader.figshare.com/files/53997578"  # WTBs2025.zip, CC BY 4.0, ~480 MB

# Class ids as in the dataset's data.yaml.
CLASSES = (
    "oil leakage",
    "paint cracks",
    "localized damage",
    "lightning strikes",
    "surface stains",
    "erosion",
    "coating detachment",
    "protective film damage",
    "pinholes",
)

# Fraction of image groups per class in each split. The split is a hash of the group id, so it is
# fixed without storing a file, and the same test images stay held out from fine-tuning later.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10

TASK = (
    "This photo of a wind turbine blade was taken during an inspection. "
    "Detect every surface defect on the blade. Use only these defect types: "
    + ", ".join(CLASSES)
    + ". Give each defect its own bounding box."
)

# All models answer on a 0-1000 grid, so a provider resizing the image can't shift the boxes.
# Gemini models are trained on [ymin, xmin, ymax, xmax] boxes, so they get their native order.
GEMINI_SUFFIX = (
    ' Answer in JSON as a list: [{"label": "<defect type>", "box_2d": [ymin, xmin, ymax, xmax]}, ...], '
    "with coordinates normalized to 0-1000. Answer [] if there is no defect."
)

# Every other model, LFM3-VL included, answers [x_min, y_min, x_max, y_max].
XYXY_SUFFIX = (
    ' Answer in JSON as a list: [{"label": "<defect type>", "bbox_2d": [x_min, y_min, x_max, y_max]}, ...], '
    "with coordinates normalized to 0-1000. Answer [] if there is no defect."
)

BOX_FORMATS = {"gemini_yxyx1000": GEMINI_SUFFIX, "xyxy1000": XYXY_SUFFIX}

IOU_THRESHOLD = 0.5


@dataclass
class Item:
    item_id: str  # "<class>/<file stem>"
    label: str  # the defect class of the folder; each image shows one defect type
    path: Path
    width: int
    height: int
    boxes: list[tuple[str, list[float]]]  # (label, [x1, y1, x2, y2]) normalized to 0-1

    def image(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


def download(data_dir: str | Path = "data") -> Path:
    """Download and unzip WTBs2025 into data_dir (skipped if already there). Returns the dataset root."""
    data_dir = Path(data_dir)
    root = data_dir / "WTBs2025"
    if (root / "data.yaml").exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "WTBs2025.zip"
    if not archive.exists():
        print("Downloading WTBs2025 (~480 MB) ...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return root


def group_id(label: str, stem: str) -> str:
    """Id shared by an original image and its augmented copies.

    Lightning strikes has 57 originals, each with GAN, brightness, flip and rotation copies
    ("12.jpg", "12GAN.png", "12be.png", ...). Oil leakage has 29 originals, each exported many times
    by Roboflow ("B-72_40_jpg.rf.<hash>"); photos of the same blade ("B-72_02", "B-72_40") share a group.
    Every other class has one file per image.
    """
    if label == "lightning strikes":
        return re.match(r"\d+", stem).group(0)
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    if label == "oil leakage":
        return re.sub(r"_\d+_jpg$", "", stem)
    return stem


def split_of(label: str, group: str) -> str:
    bucket = int(hashlib.sha1(f"{label}/{group}".encode()).hexdigest(), 16) % 1000 / 1000
    if bucket < TEST_FRACTION:
        return "test"
    if bucket < TEST_FRACTION + VAL_FRACTION:
        return "val"
    return "train"


def is_original(label: str, stem: str) -> bool:
    # Lightning strike originals are the plain "<n>.jpg" files; the others are augmented copies.
    return label != "lightning strikes" or stem.isdigit()


def read_labels(path: Path) -> list[tuple[str, list[float]]]:
    """YOLO labels (class cx cy w h, normalized) as (label, [x1, y1, x2, y2]).

    Malformed lines are skipped, and so are repeated lines: every coating detachment file lists each box twice.
    """
    boxes = []
    for line in dict.fromkeys(path.read_text().splitlines()):
        parts = line.split()
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        cx, cy, w, h = map(float, parts[1:])
        boxes.append((CLASSES[int(parts[0])], [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
    return boxes


@lru_cache(maxsize=2)
def cross_class_duplicate_groups(root: Path) -> frozenset[tuple[str, str]]:
    """Groups with a byte-identical image in another class folder.

    Exclude the whole group, including augmented copies of a matching original. The two
    folders annotate different defects on the same photo, so neither target is complete.
    """
    by_digest: dict[str, set[tuple[str, str]]] = {}
    for label in CLASSES:
        for path in (root / label / "images").iterdir():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            by_digest.setdefault(digest, set()).add((label, group_id(label, path.stem)))
    return frozenset(
        group
        for groups in by_digest.values()
        if len({label for label, _ in groups}) > 1
        for group in groups
    )


def load_items(
    root: str | Path,
    split: str = "test",
    n: int | None = None,
    seed: int = 0,
    exclude_cross_class_duplicates: bool = False,
) -> list[Item]:
    """Images of one split. In val and test, augmented copies are dropped and each group keeps one image.

    `n` takes that many images, spread evenly over the defect types (round-robin over a shuffled
    list per type, fixed by `seed`). Smaller samples are contained in larger ones, so a bigger run
    reuses the results of a smaller one.
    """
    root = Path(root).resolve()
    duplicate_groups = cross_class_duplicate_groups(root) if exclude_cross_class_duplicates else frozenset()
    rng = np.random.default_rng(seed)
    per_class = []
    for label in CLASSES:
        seen_groups = set()
        class_items = []
        for image_path in sorted((root / label / "images").iterdir()):
            stem = image_path.stem
            group = group_id(label, stem)
            if (label, group) in duplicate_groups:
                continue
            if split_of(label, group) != split:
                continue
            if split != "train":
                if not is_original(label, stem) or group in seen_groups:
                    continue
                seen_groups.add(group)
            label_path = root / label / "labels" / f"{stem}.txt"
            boxes = read_labels(label_path) if label_path.exists() else []
            if not boxes:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            class_items.append(Item(f"{label}/{stem}", label, image_path, width, height, boxes))
        per_class.append([class_items[i] for i in rng.permutation(len(class_items))])
    items = [queue[i] for i in range(max(map(len, per_class))) for queue in per_class if i < len(queue)]
    return items[:n] if n is not None else items


def build_prompt(box_format: str) -> str:
    if box_format not in BOX_FORMATS:
        raise ValueError(f"unknown box format {box_format!r}, expected one of {list(BOX_FORMATS)}")
    return TASK + BOX_FORMATS[box_format]


def format_boxes(boxes: list[tuple[str, list[float]]]) -> str:
    """Ground-truth boxes as a training target in the "xyxy1000" answer format, top to bottom, then left to right."""
    ordered = sorted(boxes, key=lambda box: (box[1][1], box[1][0]))
    return json.dumps(
        [
            {"label": label, "bbox_2d": [min(1000, max(0, round(v * 1000))) for v in coords]}
            for label, coords in ordered
        ]
    )


def normalize_label(text: str) -> str:
    """Map a predicted label to a class name; plural/singular and case differences are forgiven."""
    text = re.sub(r"[^a-z ]", " ", str(text).lower())
    text = " ".join(text.split())
    for name in CLASSES:
        if text == name or text.rstrip("s") == name.rstrip("s"):
            return name
    return text


def _json_list(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def parse_boxes(text: str, box_format: str) -> list[tuple[str, list[float]]] | None:
    """Parse an answer into (label, [x1, y1, x2, y2]) normalized to 0-1. Returns None if unparseable."""
    entries = _json_list(text)
    if entries is None:
        return None
    boxes = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        coords = entry.get("box_2d") or entry.get("bbox_2d") or entry.get("bbox")
        if not (isinstance(coords, list) and len(coords) == 4):
            continue
        try:
            a, b, c, d = map(float, coords)
        except (TypeError, ValueError):
            continue
        if box_format == "gemini_yxyx1000":
            a, b, c, d = b, a, d, c
        x1, y1, x2, y2 = a / 1000, b / 1000, c / 1000, d / 1000
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        boxes.append((normalize_label(entry.get("label", "")), [x1, y1, x2, y2]))
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, class_aware: bool = True, threshold: float = IOU_THRESHOLD) -> tuple[int, int, int]:
    """Greedy one-to-one matching by IoU. Returns (true positives, false positives, false negatives)."""
    matched = matched_pairs(preds, gts, class_aware, threshold)
    tp = len(matched)
    return tp, len(preds) - tp, len(gts) - tp


def matched_pairs(preds, gts, class_aware: bool = True, threshold: float = IOU_THRESHOLD) -> list[tuple[int, int]]:
    """Prediction and ground-truth indices matched by the scorer."""
    pairs = sorted(
        (
            (iou(p[1], g[1]), i, j)
            for i, p in enumerate(preds)
            for j, g in enumerate(gts)
            if not class_aware or p[0] == g[0]
        ),
        reverse=True,
    )
    used_p, used_g = set(), set()
    matched = []
    for overlap, i, j in pairs:
        if overlap < threshold:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matched.append((i, j))
    return matched


def predicted_type(preds) -> str | None:
    """The most frequent label among boxes; a few images carry boxes of a second type."""
    labels = [label for label, _ in preds]
    return max(set(labels), key=labels.count) if labels else None


def make_record(item: Item, model: str, answer: str, box_format: str, **extra) -> dict:
    preds = parse_boxes(answer, box_format)
    parsed = preds is not None
    preds = preds or []
    tp, fp, fn = match(preds, item.boxes, class_aware=True)
    tp_any, fp_any, fn_any = match(preds, item.boxes, class_aware=False)
    return {
        "item_id": item.item_id,
        "label": item.label,
        "model": model,
        "answer": answer,
        "parsed": parsed,
        "pred_boxes": preds,
        "gt_boxes": item.boxes,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_any": tp_any,
        "fp_any": fp_any,
        "fn_any": fn_any,
        "type_correct": predicted_type(preds) == predicted_type(item.boxes),
        **extra,
    }


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def done_keys(path: str | Path) -> set[str]:
    """item_ids already in a results file, so interrupted runs can resume."""
    return {r["item_id"] for r in read_jsonl(path)}


def _todo(items: list[Item], path: Path) -> list[Item]:
    done = done_keys(path)
    todo = [item for item in items if item.item_id not in done]
    print(f"{path.stem}: {len(todo)} images to run, {len(items) - len(todo)} already in {path}")
    return todo


def encode_image(item: Item, max_side: int) -> str:
    """The image as a base64 JPEG data URL, longest side capped at max_side."""
    import base64
    import io

    image = item.image()
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def api_request(item: Item, model: str, box_format: str, extra: dict, max_side: int = 1568) -> dict:
    """An OpenAI-format chat request, for any OpenAI-compatible endpoint.

    Temperature defaults to 0. `extra` overrides it, and a None value drops the setting, e.g.
    {"temperature": None} for reasoning models that only accept their default temperature.
    """
    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(item, max_side)}},
                    {"type": "text", "text": build_prompt(box_format)},
                ],
            }
        ],
        "temperature": 0,
        **extra,
    }
    return {key: value for key, value in request.items() if value is not None}


def run_api(
    client,
    items: list[Item],
    run_name: str,
    model: str,
    box_format: str,
    extra: dict | None = None,
    results_dir: str | Path = "results",
    max_workers: int = 8,
    attempts: int = 5,
) -> Path:
    """Call an API model on every image in parallel and append one record per image to results/<run_name>.jsonl.

    Errors are retried with exponential backoff. Images already in the file are skipped, so an
    interrupted run resumes.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    extra = extra or {}
    path = Path(results_dir) / f"{run_name}.jsonl"

    def run_item(item: Item) -> dict:
        request = api_request(item, model, box_format, extra)
        for attempt in range(attempts):
            try:
                start = time.perf_counter()
                response = client.chat.completions.create(**request)
                break
            except Exception as error:  # rate limits, timeouts, transient provider errors
                # A 4xx other than a rate limit (e.g. an unsupported parameter) fails the same way every time.
                status = getattr(error, "status_code", None)
                if attempt == attempts - 1 or (status is not None and 400 <= status < 500 and status != 429):
                    raise
                print(f"retrying in {2**attempt}s after: {error}")
                time.sleep(2**attempt)
        usage = response.usage
        return make_record(
            item,
            run_name,
            response.choices[0].message.content or "",
            box_format,
            latency_s=time.perf_counter() - start,
            provider_model=response.model,
            settings=extra,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_item, item) for item in _todo(items, path)]
        for future in as_completed(futures):
            try:
                append_jsonl(path, future.result())
            except Exception as error:
                print(f"{run_name}: image failed: {error}")
    return path


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> str:
    # On MPS, a fresh bfloat16 load of LFM3-VL breaks on long inputs (~1,500 tokens, the
    # 1280x1080 oil leakage images): it emits '[{"' and then only newlines. float32 works.
    return "bfloat16" if device == "cuda" else "float32"


def model_identity(model_id: str, revision: str | None = None) -> dict[str, str]:
    """Pin a Hub model to its commit, or fingerprint a local adapter's saved files."""
    path = Path(model_id)
    if path.is_dir():
        digest = hashlib.sha256()
        files = (
            sorted(path.glob("adapter_model*.safetensors"))
            + sorted(path.glob("adapter_model*.bin"))
            + sorted(path.glob("adapter_config.json"))
        )
        if not files:
            raise ValueError(f"No saved adapter files found in {path}")
        for file in files:
            digest.update(file.name.encode())
            with file.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
        return {"path": str(path.resolve()), "sha256": digest.hexdigest()}

    from huggingface_hub import HfApi

    return {"repo": model_id, "revision": revision or HfApi().model_info(model_id).sha}


def load_local_model(
    model_id: str,
    adapter_id: str | None = None,
    device: str | None = None,
    max_image_tokens: int | None = None,
    dtype: str | None = None,
    adapter_revision: str | None = None,
    model_revision: str | None = None,
):
    """Load a Hugging Face VLM, optionally with a LoRA adapter. Returns (model, processor, device).

    `max_image_tokens` caps the image resolution: each LFM3-VL image token covers a 32x32 pixel block.
    None keeps the processor default, which leaves the 640 px WTBs2025 images at full resolution.
    `dtype` is "bfloat16" or "float32"; None picks bfloat16 on CUDA and float32 elsewhere.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    processor = AutoProcessor.from_pretrained(model_id, revision=model_revision, trust_remote_code=True)
    if max_image_tokens:
        processor.image_processor.size["longest_edge"] = max_image_tokens * 32 * 32
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=model_revision, dtype=getattr(torch, dtype), trust_remote_code=True
    )
    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_id, revision=adapter_revision)
    return model.to(device).eval(), processor, device


def run_local(
    model,
    processor,
    items: list[Item],
    run_name: str,
    box_format: str = "xyxy1000",
    results_dir: str | Path = "results",
    max_new_tokens: int = 512,
    run_config: dict | None = None,
) -> Path:
    """Run a local VLM and save predictions under a configuration-specific name.

    Generation is greedy with thinking off. Existing records are reused only when their
    configuration matches; each row carries the full configuration for later audits.
    """
    import time

    import torch
    from tqdm.auto import tqdm

    if run_config is None or "adapter" not in run_config or "base_model" not in run_config:
        raise ValueError("run_config must identify the base_model and adapter (or None for the base model)")
    device = str(model.device)
    config = {
        **run_config,
        "device": device,
        "dtype": str(model.dtype).removeprefix("torch."),
        "box_format": box_format,
        "prompt_sha256": hashlib.sha256(build_prompt(box_format).encode()).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "decoding": "greedy, thinking off",
    }
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    model_name = f"{run_name}-{run_id}"
    path = Path(results_dir) / f"{model_name}.jsonl"
    if any(record.get("run_config") != config for record in read_jsonl(path)):
        raise ValueError(f"Existing records in {path} have a different run configuration")
    for item in tqdm(_todo(items, path), desc=run_name, unit="image"):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_prompt(box_format)}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        start = time.perf_counter()
        inputs = processor(text=[prompt], images=item.image(), return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        answer = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        append_jsonl(
            path,
            make_record(
                item,
                model_name,
                answer.strip(),
                box_format,
                latency_s=time.perf_counter() - start,
                device=device,
                dtype=str(model.dtype).removeprefix("torch."),
                prompt_tokens=int(inputs["input_ids"].shape[1]),
                completion_tokens=int(output.shape[1] - inputs["input_ids"].shape[1]),
                run_config=config,
            ),
        )
    return path


def _f1(tp, fp, fn) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def _bootstrap_f1(group, prefix: str = "", n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of micro F1, resampling images."""
    rng = np.random.default_rng(seed)
    counts = group[[f"tp{prefix}", f"fp{prefix}", f"fn{prefix}"]].to_numpy()
    idx = rng.integers(0, len(counts), size=(n_resamples, len(counts)))
    tp, fp, fn = counts[idx].sum(axis=1).T
    f1 = np.where(tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 1.0)
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def summarize(records: list[dict]) -> "pd.DataFrame":
    """One row per model: box F1 at IoU 0.5 with and without the defect type, defect-type accuracy, parse rate."""
    import pandas as pd

    records = pd.DataFrame(records)
    rows = []
    for model, group in records.groupby("model"):
        tp, fp, fn = group[["tp", "fp", "fn"]].sum()
        tp_any, fp_any, fn_any = group[["tp_any", "fp_any", "fn_any"]].sum()
        low, high = _bootstrap_f1(group)
        row = {
            "model": model,
            "n_images": len(group),
            "f1": _f1(tp, fp, fn),
            "f1_ci_low": low,
            "f1_ci_high": high,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1_box_only": _f1(tp_any, fp_any, fn_any),
            "type_accuracy": group["type_correct"].mean(),
            "parse_rate": group["parsed"].mean(),
        }
        if "latency_s" in group:
            row["latency_p50_s"] = group["latency_s"].quantile(0.5)
        rows.append(row)
    return pd.DataFrame(rows)


def per_class(records: list[dict]) -> "pd.DataFrame":
    """F1 by the image's annotated class, not standard detection F1 by predicted class."""
    import pandas as pd

    records = pd.DataFrame(records)
    sums = records.groupby(["label", "model"])[["tp", "fp", "fn"]].sum()
    f1 = sums.apply(lambda r: _f1(r["tp"], r["fp"], r["fn"]), axis=1)
    return f1.unstack("model")


def recall_by_size(records: list[dict]) -> "pd.DataFrame":
    """Ground-truth recall by box area, using the same one-to-one matches as F1."""
    import pandas as pd

    counts: dict[str, list[int]] = {"<0.1%": [0, 0, 0], "0.1–1%": [0, 0, 0], "≥1%": [0, 0, 0]}
    for record in records:
        preds, gts = record["pred_boxes"], record["gt_boxes"]
        typed = {j for _, j in matched_pairs(preds, gts)}
        box_only = {j for _, j in matched_pairs(preds, gts, class_aware=False)}
        for j, (_, (x1, y1, x2, y2)) in enumerate(gts):
            area = max(0, x2 - x1) * max(0, y2 - y1)
            bucket = "<0.1%" if area < 0.001 else "0.1–1%" if area < 0.01 else "≥1%"
            counts[bucket][0] += 1
            counts[bucket][1] += j in typed
            counts[bucket][2] += j in box_only
    return pd.DataFrame(
        [
            {"size": size, "n_defects": total, "recall": hits / total if total else float("nan"),
             "recall_box_only": hits_any / total if total else float("nan")}
            for size, (total, hits, hits_any) in counts.items()
        ]
    )


def draw(item: Item, record: dict | None = None) -> Image.Image:
    """The image with ground-truth boxes in green and predicted boxes in red."""
    image = item.image()
    canvas = ImageDraw.Draw(image)
    width, height = image.size
    line = max(2, width // 320)
    layers = [("lime", item.boxes)] + ([("red", record["pred_boxes"])] if record else [])
    for color, boxes in layers:
        for label, (x1, y1, x2, y2) in boxes:
            canvas.rectangle([x1 * width, y1 * height, x2 * width, y2 * height], outline=color, width=line)
            canvas.text((x1 * width + 3, y1 * height + 2), label, fill=color)
    return image
