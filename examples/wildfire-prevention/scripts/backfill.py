"""One-shot backfill: run predictions for evaluation locations over the last N days.

Fetches historical Sentinel-2 images from SimSat for each location and each day
in the requested range, runs inference, and saves everything to the DB. Use this
to seed the database before starting the live watch loop, or to build a seasonal
time-series.

Without --location all 22 evaluation locations are backfilled. With --location
only that specific location is backfilled.

Usage:
    uv run scripts/backfill.py --backend anthropic --days 7
    uv run scripts/backfill.py --backend anthropic --days 7 --location attica_greece
    uv run scripts/backfill.py --backend anthropic --days 90 --location angeles_nf_ca
    uv run scripts/backfill.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0 --days 7
"""

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from wildfire_prevention.db import init_db, insert_prediction
from wildfire_prevention.evaluator import (
    PredictFn,
    anthropic_backend,
    llama_backend,
    model_name,
    start_llama_server,
    stop_server,
    wait_for_server,
)
from wildfire_prevention.locations import LOCATIONS, LOCATIONS_BY_ID, Location
from wildfire_prevention.simsat import fetch_rgb, fetch_swir

DB_PATH = Path(__file__).parent.parent / "wildfire.db"
DB_IMAGES_DIR = Path(__file__).parent.parent / "db_images"


def _save_images(row_id: int, rgb_bytes: bytes, swir_bytes: bytes) -> tuple[str, str]:
    tile_dir = DB_IMAGES_DIR / str(row_id)
    tile_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = tile_dir / "rgb.png"
    swir_path = tile_dir / "swir.png"
    rgb_path.write_bytes(rgb_bytes)
    swir_path.write_bytes(swir_bytes)
    return str(rgb_path), str(swir_path)


def _start_server_if_needed(
    args: argparse.Namespace,
) -> subprocess.Popen[bytes] | None:
    if args.backend != "local":
        return None
    if not args.model:
        print("--model is required when using --backend local")
        sys.exit(1)
    if not shutil.which("llama-server"):
        print("llama-server not found on PATH.")
        sys.exit(1)
    print(f"Starting llama-server with {args.model} on port {args.port} ...")
    proc = start_llama_server(args.model, quant=args.quant or None, port=args.port)
    try:
        wait_for_server(port=args.port)
    except TimeoutError as exc:
        print(str(exc))
        stop_server(proc)
        sys.exit(1)
    print("llama-server ready.")
    return proc


def _build_location_list(location_arg: str | None) -> list[Location]:
    if location_arg:
        return [LOCATIONS_BY_ID[location_arg]]
    return list(LOCATIONS)


def run_backfill(args: argparse.Namespace, predict: PredictFn, mname: str) -> None:
    conn = init_db(DB_PATH)

    locations = _build_location_list(args.location)

    today = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
    dates = [today - timedelta(days=d) for d in range(args.days)]

    tasks: list[tuple[Location, str]] = [
        (loc, ts.isoformat())
        for loc in locations
        for ts in dates
    ]

    label = args.location if args.location else f"all ({len(locations)} locations)"
    print(
        f"Backfill: location={label}  days={args.days}"
        f"  tasks={len(tasks)}  backend={args.backend}  concurrency={args.concurrency}"
    )

    def _process(loc: Location, timestamp: str) -> str:
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                rgb_f = pool.submit(fetch_rgb, loc.lon, loc.lat, timestamp, args.size_km)
                swir_f = pool.submit(fetch_swir, loc.lon, loc.lat, timestamp, args.size_km)
                rgb_bytes = rgb_f.result()
                swir_bytes = swir_f.result()
        except requests.HTTPError as exc:
            return f"SKIP ({exc.response.status_code})"

        prediction = predict(rgb_bytes, swir_bytes)
        row_id = insert_prediction(
            conn, loc.lon, loc.lat, timestamp, args.size_km,
            source="backfill",
            rgb_path=None, swir_path=None,
            prediction=prediction, model=mname,
            region_id=loc.id,
        )
        rgb_path, swir_path = _save_images(row_id, rgb_bytes, swir_bytes)
        conn.execute(
            "UPDATE predictions SET rgb_path=?, swir_path=? WHERE id=?",
            (rgb_path, swir_path, row_id),
        )
        conn.commit()
        return f"risk={prediction.get('risk_level', '?')}"

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(_process, loc, ts): (loc.id, ts[:10])
            for loc, ts in tasks
        }
        for future in as_completed(futures):
            loc_id, date = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                status = f"ERROR: {exc}"
            print(f"[{loc_id} {date}] {status}", flush=True)

    print("Backfill complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the DB with historical predictions for evaluation locations."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["anthropic", "local"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--location",
        default=None,
        choices=list(LOCATIONS_BY_ID),
        metavar="LOCATION",
        help=(
            f"Location to backfill. Available: {', '.join(LOCATIONS_BY_ID)}."
            " When omitted all evaluation locations are backfilled."
        ),
    )
    parser.add_argument(
        "--model",
        metavar="REPO",
        default="",
        help="HuggingFace repo ID (required for --backend local).",
    )
    parser.add_argument(
        "--quant",
        metavar="QUANT",
        default="",
        help="Quantization level, e.g. Q8_0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="llama-server port (default: 8080, local backend only).",
    )
    parser.add_argument(
        "--size-km",
        type=float,
        default=5.0,
        metavar="KM",
        help="Tile edge length in km (default: 5.0).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of past days to cover per location (default: 7).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        metavar="N",
        help="Parallel workers (default: 3).",
    )
    args = parser.parse_args()

    server = _start_server_if_needed(args)
    predict = (
        anthropic_backend()
        if args.backend == "anthropic"
        else llama_backend(args.model, args.port)
    )
    mname = model_name(args.backend, args.model, args.quant)

    try:
        run_backfill(args, predict, mname)
    finally:
        if server is not None:
            stop_server(server)


if __name__ == "__main__":
    main()
