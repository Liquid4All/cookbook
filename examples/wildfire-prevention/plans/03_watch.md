# Intent: predict.py as a continuous watch loop

Replace the one-shot `live` and `backfill` subcommands with a single persistent process that:

1. Polls the current satellite position from SimSat at a fixed interval
2. Skips the position if it is too close to the last processed tile (avoids re-running on the same ground)
3. Fetches RGB + SWIR, runs inference, saves to DB
4. Repeats until interrupted (Ctrl+C)

As time passes and the simulated satellite moves, the DB accumulates predictions covering progressively more of the Earth's surface. The Streamlit app visualises the growing map in real time.

---

## Design decisions

### No subcommands

Drop `live` and `backfill`. The script has one job: watch the satellite and predict continuously. Invocation becomes:

```
uv run scripts/predict.py --backend anthropic
uv run scripts/predict.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0
```

### Deduplication via distance threshold

After each successful prediction, record the processed (lon, lat). On the next poll, compute the great-circle distance to the last processed position. If it is less than `--min-distance-km` (default: same as `--size-km`), skip and sleep. This is the simplest approach that avoids redundant inference without needing a spatial index.

Tradeoff: if the satellite orbits back over a previously processed tile, it will be re-processed. That is acceptable for a demo. A grid-cell approach would avoid this but adds complexity.

### Poll interval

Controlled by `--interval` (seconds, default: 30). SimSat compresses orbital time, so the right value depends on how fast the simulation moves. Users can tune this down to 5s for a more responsive demo or up to 300s to reduce API costs.

### Graceful shutdown

Catch `KeyboardInterrupt`. Print a summary (tiles processed, elapsed time), stop llama-server if running, close DB connection, exit cleanly.

### SimSat errors

If SimSat returns 404 (no Sentinel coverage at the current position, e.g. ocean or cloud gap), skip silently and sleep. Print a single-line status. Do not crash.

---

## Implementation plan

### Step 1: rewrite `scripts/predict.py`

Remove `cmd_live`, `cmd_backfill`, and the subparser. Replace with a single `watch_loop` function:

```python
def watch_loop(args: argparse.Namespace, predict: PredictFn, conn: sqlite3.Connection) -> None:
    last_lon: float | None = None
    last_lat: float | None = None
    processed = 0

    while True:
        try:
            lon, lat = get_current_position()
        except Exception as exc:
            print(f"[position] ERROR: {exc}", flush=True)
            time.sleep(args.interval)
            continue

        # Deduplication: skip if too close to last processed tile.
        if last_lon is not None and last_lat is not None:
            dist = _haversine_km(last_lon, last_lat, lon, lat)
            if dist < args.min_distance_km:
                time.sleep(args.interval)
                continue

        timestamp = datetime.now(timezone.utc).isoformat()
        print(f"[{timestamp[:19]}] lon={lon:.4f}  lat={lat:.4f}  fetching ...", flush=True)

        try:
            rgb_bytes = _fetch_current_image(["red", "green", "blue"], args.size_km)
            swir_bytes = _fetch_current_image(["swir16", "nir08", "red"], args.size_km)
        except requests.HTTPError as exc:
            print(f"  SimSat {exc.response.status_code}: skipping", flush=True)
            time.sleep(args.interval)
            continue

        prediction = predict(rgb_bytes, swir_bytes)
        row_id = insert_prediction(conn, lon, lat, timestamp, args.size_km,
                                   source="live", rgb_path=None, swir_path=None,
                                   prediction=prediction, model=mname)
        rgb_path, swir_path = _save_images(row_id, rgb_bytes, swir_bytes)
        conn.execute("UPDATE predictions SET rgb_path=?, swir_path=? WHERE id=?",
                     (rgb_path, swir_path, row_id))
        conn.commit()

        processed += 1
        last_lon, last_lat = lon, lat
        print(f"  risk={prediction.get('risk_level')}  id={row_id}  total={processed}", flush=True)
        time.sleep(args.interval)
```

### Step 2: add `_haversine_km`

A small pure-Python helper. No new dependencies:

```python
import math

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))
```

### Step 3: CLI flags

```
--backend        required, choices=[anthropic, local]
--model          HF repo ID (local only)
--quant          quantization, e.g. Q8_0 (local only)
--port           default 8080 (local only)
--size-km        default 5.0
--interval       seconds between polls, default 30
--min-distance-km  skip if within this distance of last tile, default = size_km
```

### Step 4: update `app/app.py`

Replace the manual "Fetch live prediction" sidebar button with an **auto-refresh toggle**. When enabled, the app calls `st.rerun()` after a configurable delay (e.g. 30s) using `time.sleep` in a `st.empty()` spinner. This way the map updates automatically as `predict.py` writes new rows to the DB.

Keep the manual button as well for one-off fetches.

### Step 5: update README

Replace the "Proof of concept" section to reflect the new watch-mode design.

---

## What to remove

- `cmd_live` and `cmd_backfill` functions
- The argparse subparser setup
- `--save` flag (watch mode always saves)
- `--days` and `--concurrency` flags
- References to `LOCATIONS` in `predict.py` (no longer iterated — the satellite position drives coverage)
