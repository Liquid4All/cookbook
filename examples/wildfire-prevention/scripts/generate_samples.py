"""Generate wildfire risk annotations for all locations.

Each run creates a timestamped folder under data/, e.g.:
    data/20260416_143052/angeles_nf_ca/rgb.png
    data/20260416_143052/angeles_nf_ca/swir.png
    data/20260416_143052/angeles_nf_ca/annotation.json

Usage:
    uv run scripts/generate_samples.py
    uv run scripts/generate_samples.py --size-km 5.0
    uv run scripts/generate_samples.py --concurrency 5
    uv run scripts/generate_samples.py --dry-run
    uv run scripts/generate_samples.py --location angeles_nf_ca
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildfire_prevention.annotator import annotate
from wildfire_prevention.locations import LOCATIONS, Location
from wildfire_prevention.simsat import fetch_rgb, fetch_swir

DATA_DIR = Path(__file__).parent.parent / "data"

_RETRY_DELAYS = [5, 15, 30]  # seconds between retries on 429


def _annotate_with_retry(rgb_bytes: bytes, swir_bytes: bytes) -> dict[str, object]:
    """Call annotate with exponential backoff on Anthropic 429 rate-limit errors."""
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        try:
            return annotate(rgb_bytes, swir_bytes)
        except Exception as exc:
            # anthropic raises APIStatusError; check the message for 429.
            if "429" in str(exc) and attempt <= len(_RETRY_DELAYS):
                print(f"  rate-limited, retrying in {delay}s ...", flush=True)
                time.sleep(delay)
            else:
                raise
    return annotate(rgb_bytes, swir_bytes)


def process_location(loc: Location, run_dir: Path, size_km: float, dry_run: bool) -> None:
    sample_dir = run_dir / loc.id
    sample_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{loc.id}] fetching images ...", flush=True)
    try:
        # Fetch RGB and SWIR in parallel — they are independent SimSat calls.
        with ThreadPoolExecutor(max_workers=2) as pool:
            rgb_future = pool.submit(fetch_rgb, loc.lon, loc.lat, loc.timestamp, size_km)
            swir_future = pool.submit(fetch_swir, loc.lon, loc.lat, loc.timestamp, size_km)
            rgb_bytes = rgb_future.result()
            swir_bytes = swir_future.result()
    except requests.HTTPError as exc:
        print(f"[{loc.id}] SKIP: SimSat returned {exc.response.status_code}")
        return

    (sample_dir / "rgb.png").write_bytes(rgb_bytes)
    (sample_dir / "swir.png").write_bytes(swir_bytes)

    if dry_run:
        print(f"[{loc.id}] dry-run: images saved, skipping annotation")
        return

    print(f"[{loc.id}] annotating ...", flush=True)
    try:
        result = _annotate_with_retry(rgb_bytes, swir_bytes)
    except ValueError as exc:
        print(f"[{loc.id}] ERROR: {exc}")
        return

    annotation = {
        "id": loc.id,
        "lon": loc.lon,
        "lat": loc.lat,
        "timestamp": loc.timestamp,
        "size_km": size_km,
        **result,
    }
    (sample_dir / "annotation.json").write_text(
        json.dumps(annotation, indent=2), encoding="utf-8"
    )
    print(f"[{loc.id}] done — risk_level: {result.get('risk_level', '?')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wildfire risk annotations.")
    parser.add_argument(
        "--size-km",
        type=float,
        default=10.0,
        metavar="KM",
        help="Tile edge length in km passed to SimSat (default: 10.0).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        metavar="N",
        help="Number of locations to annotate in parallel (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch images but skip the Opus annotation call.",
    )
    parser.add_argument(
        "--location",
        metavar="ID",
        help="Process a single location by its id (e.g. angeles_nf_ca).",
    )
    args = parser.parse_args()

    locations = LOCATIONS
    if args.location:
        locations = [loc for loc in LOCATIONS if loc.id == args.location]
        if not locations:
            ids = ", ".join(loc.id for loc in LOCATIONS)
            print(f"Unknown location id '{args.location}'. Available: {ids}")
            sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Run: {run_id}  |  size_km: {args.size_km}"
        f"  |  locations: {len(locations)}  |  concurrency: {args.concurrency}"
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_location, loc, run_dir, args.size_km, args.dry_run): loc
            for loc in locations
        }
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                loc = futures[future]
                print(f"[{loc.id}] UNEXPECTED ERROR: {exc}")


if __name__ == "__main__":
    main()
