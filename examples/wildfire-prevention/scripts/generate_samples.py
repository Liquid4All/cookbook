"""Generate wildfire risk annotations across a spatial and temporal tile grid.

Each run creates a timestamped folder under data/ with train/ and test/ splits,
e.g.:
    data/20260416_143052/train/attica_greece/s00_t00/rgb.png
    data/20260416_143052/train/attica_greece/s00_t00/swir.png
    data/20260416_143052/train/attica_greece/s00_t00/annotation.json
    data/20260416_143052/test/attica_greece/s00_t04/rgb.png
    ...

The split is determined by a temporal cutoff: all tiles whose timestamp falls
before the cutoff go to train/, the rest go to test/. This prevents near-
duplicate images (Sentinel-2 revisits every 5 days) from leaking across splits.

When --hf-dataset is passed the run directory is also pushed to Hugging Face Hub
as a dataset in leap-finetune VLM SFT format with train.jsonl and test.jsonl.

Usage:
    uv run scripts/generate_samples.py \\
        --start-date 2024-01-01 --end-date 2024-12-31 \\
        --n-temporal-tiles 12 --n-spatial-tiles 4 \\
        --test-ratio 0.2

    uv run scripts/generate_samples.py \\
        --start-date 2024-06-01 --end-date 2024-09-01 \\
        --n-temporal-tiles 6 \\
        --location attica_greece

    uv run scripts/generate_samples.py \\
        --start-date 2024-01-01 --end-date 2024-12-31 \\
        --n-temporal-tiles 12 --n-spatial-tiles 4 \\
        --test-ratio 0.2 \\
        --dry-run
"""

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

import requests
from tqdm import tqdm

from wildfire_prevention.annotator import SYSTEM_PROMPT, USER_TEXT, annotate
from wildfire_prevention.locations import LOCATIONS, LOCATIONS_BY_ID, Location
from wildfire_prevention.simsat import fetch_rgb, fetch_swir
from wildfire_prevention.tiles import (
    TileCoord,
    spatial_grid,
    temporal_timestamps,
    train_test_cutoff,
)

AnnotationResult: TypeAlias = dict[str, object]

DATA_DIR = Path(__file__).parent.parent / "data"

_RETRY_DELAYS = [5, 15, 30]  # seconds between retries on 429


@dataclass(frozen=True)
class TileTask:
    loc: Location
    spatial: TileCoord
    timestamp: str
    split: str          # "train" | "test"
    spatial_idx: int
    temporal_idx: int

    @property
    def label(self) -> str:
        return f"{self.loc.id}/s{self.spatial_idx:02d}_t{self.temporal_idx:02d}"

    @property
    def tile_key(self) -> str:
        return f"s{self.spatial_idx:02d}_t{self.temporal_idx:02d}"


def _annotate_with_retry(rgb_bytes: bytes, swir_bytes: bytes) -> dict[str, object]:
    """Call annotate with exponential backoff on Anthropic 429 rate-limit errors."""
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            return annotate(rgb_bytes, swir_bytes)
        except Exception as exc:
            if "429" in str(exc) and attempt <= len(_RETRY_DELAYS):
                tqdm.write(f"  rate-limited, retrying in {delay}s ...")
                time.sleep(delay)
            else:
                raise
    return annotate(rgb_bytes, swir_bytes)


def process_tile(
    task: TileTask,
    sample_dir: Path,
    size_km: float,
    dry_run: bool,
) -> AnnotationResult | None:
    sample_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"[{task.label}] fetching images ...")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rgb_future = pool.submit(
                fetch_rgb, task.spatial.lon, task.spatial.lat, task.timestamp, size_km
            )
            swir_future = pool.submit(
                fetch_swir, task.spatial.lon, task.spatial.lat, task.timestamp, size_km
            )
            rgb_bytes = rgb_future.result()
            swir_bytes = swir_future.result()
    except requests.HTTPError as exc:
        tqdm.write(f"[{task.label}] SKIP: SimSat returned {exc.response.status_code}")
        return None

    (sample_dir / "rgb.png").write_bytes(rgb_bytes)
    (sample_dir / "swir.png").write_bytes(swir_bytes)

    if dry_run:
        tqdm.write(f"[{task.label}] dry-run: images saved, skipping annotation")
        return None

    tqdm.write(f"[{task.label}] annotating ...")
    try:
        result = _annotate_with_retry(rgb_bytes, swir_bytes)
    except ValueError as exc:
        tqdm.write(f"[{task.label}] ERROR: {exc}")
        return None

    annotation: AnnotationResult = {
        "id": task.loc.id,
        "split": task.split,
        "spatial_index": task.spatial_idx,
        "temporal_index": task.temporal_idx,
        "lon": task.spatial.lon,
        "lat": task.spatial.lat,
        "timestamp": task.timestamp,
        "size_km": size_km,
        **result,
    }
    (sample_dir / "annotation.json").write_text(
        json.dumps(annotation, indent=2), encoding="utf-8"
    )
    tqdm.write(
        f"[{task.label}] done  split={task.split}  risk_level={result.get('risk_level', '?')}"
    )
    return annotation


def _build_tasks(
    locations: list[Location],
    timestamps: list[str],
    cutoff: datetime | None,
    n_spatial_tiles: int,
    size_km: float,
) -> list[tuple[TileTask, Path]]:
    """Build the full cross-product of (location, spatial tile, temporal tile)."""
    tasks: list[tuple[TileTask, Path]] = []
    for loc in locations:
        spatial_tiles = spatial_grid(loc.lon, loc.lat, n_spatial_tiles, size_km)
        for ti, ts in enumerate(timestamps):
            ts_dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            split = "test" if (cutoff is not None and ts_dt >= cutoff) else "train"
            for si, spatial in enumerate(spatial_tiles):
                task = TileTask(
                    loc=loc,
                    spatial=spatial,
                    timestamp=ts,
                    split=split,
                    spatial_idx=si,
                    temporal_idx=ti,
                )
                tasks.append((task, Path()))  # sample_dir filled in below
    return tasks


def push_to_hf(
    run_dir: Path,
    results: list[AnnotationResult],
    dataset_name: str,
) -> None:
    """Build leap-finetune VLM SFT datasets and push them to Hugging Face Hub.

    Creates train.jsonl and test.jsonl inside run_dir alongside a flat images/
    directory. Image paths in the JSONL are relative to the repo root so that
    leap-finetune can resolve them after snapshot_download.

    A README.md dataset card with `configs: []` is written to prevent HuggingFace
    from auto-converting the JSONL files to parquet (which embeds images as bytes
    and breaks the file-path references leap-finetune expects).

    Auth is handled via the HF_TOKEN environment variable.
    """
    from collections import defaultdict

    from huggingface_hub import HfApi

    images_dir = run_dir / "images"
    images_dir.mkdir(exist_ok=True)

    by_split: dict[str, list[AnnotationResult]] = defaultdict(list)
    for ann in results:
        by_split[str(ann["split"])].append(ann)

    for split, split_results in by_split.items():
        rows: list[str] = []
        for ann in split_results:
            loc_id = str(ann["id"])
            si = int(ann["spatial_index"])  # type: ignore[arg-type]
            ti = int(ann["temporal_index"])  # type: ignore[arg-type]
            tile_key = f"{loc_id}_s{si:02d}_t{ti:02d}"

            rgb_name = f"{tile_key}_rgb.png"
            swir_name = f"{tile_key}_swir.png"
            tile_dir = run_dir / split / loc_id / f"s{si:02d}_t{ti:02d}"
            shutil.copy2(tile_dir / "rgb.png", images_dir / rgb_name)
            shutil.copy2(tile_dir / "swir.png", images_dir / swir_name)

            model_output = {
                k: ann[k]
                for k in (
                    "risk_level",
                    "dry_vegetation_present",
                    "urban_interface",
                    "steep_terrain",
                    "water_body_present",
                    "image_quality_limited",
                )
            }
            row = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            # Paths are relative to the repo root so they resolve
                            # correctly after huggingface_hub.snapshot_download().
                            {"type": "image", "image": f"images/{rgb_name}"},
                            {"type": "image", "image": f"images/{swir_name}"},
                            {"type": "text", "text": f"{SYSTEM_PROMPT.strip()}\n\n{USER_TEXT}"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": json.dumps(model_output)}],
                    },
                ]
            }
            rows.append(json.dumps(row))

        (run_dir / f"{split}.jsonl").write_text("\n".join(rows), encoding="utf-8")
        print(f"  {split}.jsonl: {len(rows)} rows")

    # Prevent HuggingFace from auto-converting the JSONL files to parquet.
    # configs: [] tells the Hub there are no dataset configurations to detect,
    # so the raw files (JSONL + images/) are preserved as-is.
    (run_dir / "README.md").write_text(
        "---\nconfigs: []\n---\n",
        encoding="utf-8",
    )

    api = HfApi()
    api.create_repo(repo_id=dataset_name, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(run_dir),
        repo_id=dataset_name,
        repo_type="dataset",
    )
    print(f"Dataset pushed to https://huggingface.co/datasets/{dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate wildfire risk annotations across a spatial and temporal tile grid."
    )
    parser.add_argument(
        "--start-date",
        required=True,
        metavar="DATE",
        help="Start of the sampling window, ISO 8601 date, e.g. 2024-01-01.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        metavar="DATE",
        help="End of the sampling window, ISO 8601 date, e.g. 2024-12-31.",
    )
    parser.add_argument(
        "--n-temporal-tiles",
        type=int,
        default=1,
        metavar="N",
        help="Number of timestamps to sample per location within the window (default: 1).",
    )
    parser.add_argument(
        "--n-spatial-tiles",
        type=int,
        default=1,
        metavar="N",
        help="Number of spatial grid tiles per location per timestamp (default: 1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        metavar="RATIO",
        help=(
            "Fraction of the time window reserved for the test split, e.g. 0.2."
            " Uses a temporal cutoff so no near-duplicate images span the split boundary."
            " Default: 0.0 (all data goes to train)."
        ),
    )
    parser.add_argument(
        "--size-km",
        type=float,
        default=5.0,
        metavar="KM",
        help="Tile edge length in km, also used as the spatial grid spacing (default: 5.0).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        metavar="N",
        help="Number of tiles to annotate in parallel (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch images but skip the Opus annotation call.",
    )
    parser.add_argument(
        "--location",
        metavar="ID",
        help="Process a single location by its id (e.g. attica_greece).",
    )
    parser.add_argument(
        "--hf-dataset",
        metavar="REPO",
        default=None,
        help=(
            "Hugging Face dataset repo to push results to, e.g. username/wildfire-risk."
            " Requires HF_TOKEN env var. Skipped if not provided."
        ),
    )
    args = parser.parse_args()

    # Parse and validate dates.
    try:
        start_dt = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(args.end_date).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        print(f"Invalid date: {exc}")
        sys.exit(1)
    if end_dt <= start_dt:
        print("--end-date must be after --start-date")
        sys.exit(1)

    if args.test_ratio < 0.0 or args.test_ratio >= 1.0:
        print("--test-ratio must be in [0.0, 1.0)")
        sys.exit(1)

    # Resolve locations.
    locations = list(LOCATIONS)
    if args.location:
        if args.location not in LOCATIONS_BY_ID:
            ids = ", ".join(loc.id for loc in LOCATIONS)
            print(f"Unknown location id '{args.location}'. Available: {ids}")
            sys.exit(1)
        locations = [LOCATIONS_BY_ID[args.location]]

    # Build tile grid.
    timestamps = temporal_timestamps(start_dt, end_dt, args.n_temporal_tiles)
    cutoff = (
        train_test_cutoff(start_dt, end_dt, args.test_ratio)
        if args.test_ratio > 0.0
        else None
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(locations) * args.n_temporal_tiles * args.n_spatial_tiles
    cutoff_str = cutoff.isoformat() if cutoff else "none"
    print(
        f"Run: {run_id}"
        f"  |  locations: {len(locations)}"
        f"  |  temporal_tiles: {args.n_temporal_tiles}"
        f"  |  spatial_tiles: {args.n_spatial_tiles}"
        f"  |  total: {n_total}"
        f"  |  test_ratio: {args.test_ratio}"
        f"  |  cutoff: {cutoff_str}"
        f"  |  concurrency: {args.concurrency}"
    )

    # Build (task, sample_dir) pairs.
    task_pairs: list[tuple[TileTask, Path]] = []
    for loc in locations:
        spatial_tiles = spatial_grid(loc.lon, loc.lat, args.n_spatial_tiles, args.size_km)
        for ti, ts in enumerate(timestamps):
            ts_dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
            split = "test" if (cutoff is not None and ts_dt >= cutoff) else "train"
            for si, spatial in enumerate(spatial_tiles):
                task = TileTask(
                    loc=loc,
                    spatial=spatial,
                    timestamp=ts,
                    split=split,
                    spatial_idx=si,
                    temporal_idx=ti,
                )
                sample_dir = run_dir / split / loc.id / task.tile_key
                task_pairs.append((task, sample_dir))

    # Run in parallel.
    annotations: list[AnnotationResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_tile, task, sample_dir, args.size_km, args.dry_run): task
            for task, sample_dir in task_pairs
        }
        with tqdm(total=len(task_pairs), desc="tiles", unit="tile") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                exc = future.exception()
                if exc:
                    tqdm.write(f"[{task.label}] UNEXPECTED ERROR: {exc}")
                else:
                    result = future.result()
                    if result is not None:
                        annotations.append(result)
                        pbar.set_postfix(
                            split=task.split,
                            risk=result.get("risk_level", "?"),
                        )
                pbar.update(1)

    train_count = sum(1 for a in annotations if a["split"] == "train")
    test_count = sum(1 for a in annotations if a["split"] == "test")
    print(f"\nDone: {len(annotations)} annotations  (train={train_count}  test={test_count})")

    if args.hf_dataset:
        if not annotations:
            print("No annotations produced; skipping Hugging Face push.")
        else:
            print(f"Pushing {len(annotations)} samples to {args.hf_dataset} ...")
            push_to_hf(run_dir, annotations, args.hf_dataset)


if __name__ == "__main__":
    main()
