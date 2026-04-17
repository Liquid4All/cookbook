"""Continuous watch loop: poll the current satellite position, run inference, save to DB.

Runs until interrupted with Ctrl+C. Without --region, every satellite position
is scored as it moves across the Earth's surface.

With --region, the loop only scores tiles that belong to the named region. When
the satellite is outside the region it logs a skip line so the operator can
confirm the loop is alive.

Usage:
    uv run scripts/predict.py --backend anthropic
    uv run scripts/predict.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0
    uv run scripts/predict.py --backend anthropic --region collserola
    uv run scripts/predict.py --backend anthropic --region garraf --interval 10

Available regions: collserola, garraf, montseny, donana, sierra_nevada
"""

import argparse
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
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
from wildfire_prevention.live import _fetch_current_image, get_current_position
from wildfire_prevention.regions import REGIONS, find_tile, generate_tile_grid

DB_PATH = Path(__file__).parent.parent / "wildfire.db"
DB_IMAGES_DIR = Path(__file__).parent.parent / "db_images"


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


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


def watch_loop(
    args: argparse.Namespace,
    predict: PredictFn,
    mname: str,
) -> None:
    conn = init_db(DB_PATH)
    last_lon: float | None = None
    last_lat: float | None = None
    processed = 0

    # Pre-build tile grid when a region is specified.
    region_id: str | None = None
    tiles: list[tuple[float, float]] = []
    if args.region:
        region = REGIONS[args.region]
        region_id = region.id
        tiles = generate_tile_grid(region, args.size_km)
        print(
            f"Region mode: {region.name}  |  {len(tiles)} tiles at {args.size_km} km"
        )

    print(
        f"Watching  |  backend: {args.backend}  |  interval: {args.interval}s"
        f"  |  min-distance: {args.min_distance_km}km  |  Ctrl+C to stop"
    )

    while True:
        try:
            sat_lon, sat_lat = get_current_position()
        except requests.ConnectionError:
            print("[position] Cannot connect to SimSat. Is `docker compose up` running?", flush=True)
            time.sleep(args.interval)
            continue
        except Exception as exc:
            print(f"[position] ERROR: {exc}", flush=True)
            time.sleep(args.interval)
            continue

        timestamp = datetime.now(timezone.utc).isoformat()

        # Region gate: when --region is set, only process if the satellite is
        # over one of the region's tiles. Use the tile center as canonical
        # coordinates so live and backfill predictions are colocated in the DB.
        if tiles:
            hit = find_tile(sat_lon, sat_lat, tiles, args.size_km)
            if hit is None:
                print(
                    f"[{timestamp[:19]}] lon={sat_lon:.4f}  lat={sat_lat:.4f}"
                    f"  outside region {region_id}, skipping",
                    flush=True,
                )
                time.sleep(args.interval)
                continue
            lon, lat = hit
        else:
            lon, lat = sat_lon, sat_lat

        if last_lon is not None and last_lat is not None:
            dist = _haversine_km(last_lon, last_lat, lon, lat)
            if dist < args.min_distance_km:
                time.sleep(args.interval)
                continue

        print(f"[{timestamp[:19]}] lon={lon:.4f}  lat={lat:.4f}  fetching ...", flush=True)

        try:
            rgb_bytes = _fetch_current_image(["red", "green", "blue"], args.size_km)
            swir_bytes = _fetch_current_image(["swir16", "nir08", "red"], args.size_km)
        except requests.HTTPError as exc:
            print(f"  SimSat {exc.response.status_code}: no coverage, skipping", flush=True)
            time.sleep(args.interval)
            continue
        except (requests.ConnectionError, requests.Timeout):
            print("  SimSat unreachable or timed out, skipping", flush=True)
            time.sleep(args.interval)
            continue

        if not rgb_bytes or not swir_bytes:
            print("  Empty image returned by SimSat (no coverage), skipping", flush=True)
            time.sleep(args.interval)
            continue

        prediction = predict(rgb_bytes, swir_bytes)
        row_id = insert_prediction(
            conn, lon, lat, timestamp, args.size_km,
            source="live",
            rgb_path=None, swir_path=None,
            prediction=prediction, model=mname,
            region_id=region_id,
        )
        rgb_path, swir_path = _save_images(row_id, rgb_bytes, swir_bytes)
        conn.execute(
            "UPDATE predictions SET rgb_path=?, swir_path=? WHERE id=?",
            (rgb_path, swir_path, row_id),
        )
        conn.commit()

        last_lon, last_lat = lon, lat
        processed += 1
        print(
            f"  risk={prediction.get('risk_level')}  id={row_id}  total={processed}",
            flush=True,
        )
        time.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously predict wildfire risk from the live satellite position."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["anthropic", "local"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--region",
        default=None,
        choices=list(REGIONS),
        metavar="REGION",
        help=(
            f"Only score tiles within this region. Available: {', '.join(REGIONS)}."
            " When omitted every satellite position is scored."
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
        "--interval",
        type=int,
        default=30,
        metavar="SEC",
        help="Seconds between position polls (default: 30).",
    )
    parser.add_argument(
        "--min-distance-km",
        type=float,
        default=None,
        metavar="KM",
        help=(
            "Skip if satellite has moved less than this distance from the last"
            " processed tile (default: same as --size-km)."
        ),
    )
    args = parser.parse_args()

    if args.min_distance_km is None:
        args.min_distance_km = args.size_km

    server = _start_server_if_needed(args)
    predict = (
        anthropic_backend()
        if args.backend == "anthropic"
        else llama_backend(args.model, args.port)
    )
    mname = model_name(args.backend, args.model, args.quant)

    try:
        watch_loop(args, predict, mname)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if server is not None:
            stop_server(server)


if __name__ == "__main__":
    main()
