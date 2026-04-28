# Intent: create a CLI and webapp that generate predictions in real-time

A basic CLI that uses the live data from SimSat, fetches the necessary image bands (as we do already when we generate_samples.py) and print prediction to console.

Add support also to save to a db, so data persists between runs.

Add also option to backfill this db, by pointing the app to historical data for the last N days.

After this I was thinking of having a nicer streamlit app that shows both the live image. and the generated prediction, but also some way to visualize the already stored data in the db. These are tiles and predictions. I need some advice on the best way to visualize this data on the frontend.

---

## Implementation plan

### Visualization decision

Use `pydeck` (via `st.pydeck_chart`) rather than `streamlit-folium` or `plotly`. The key reason: pydeck supports a `BitmapLayer` that renders the actual PNG at its geographic bounding box. Since each tile has a known center (lon/lat) and size_km, the bounding box is trivially computed. This shows the real satellite imagery on the map in geographic context, which is the core value of the demo. `folium` only supports markers, not raster imagery.

Layout:
- `BitmapLayer`: renders `rgb.png` at the tile's bounding box
- `ScatterplotLayer`: colored circle on each tile center, color = risk level (green/orange/red for low/medium/high)
- Sidebar: shows the selected tile's images (RGB + SWIR side by side) and prediction JSON

---

### New files

```
src/wildfire_prevention/db.py       — SQLite database layer
src/wildfire_prevention/live.py     — fetches current satellite position + images from SimSat
scripts/predict.py                  — CLI: live prediction + backfill
app/app.py                          — Streamlit frontend
```

---

### Step 1: `db.py` — SQLite persistence

Single table `predictions`. Store images on disk under `db_images/{id}/rgb.png` and `db_images/{id}/swir.png` (same layout as `data/`), and keep paths in the DB.

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lon         REAL    NOT NULL,
    lat         REAL    NOT NULL,
    timestamp   TEXT    NOT NULL,
    size_km     REAL    NOT NULL,
    source      TEXT    NOT NULL,   -- "live" or "historical"
    rgb_path    TEXT,
    swir_path   TEXT,
    risk_level  TEXT,
    dry_vegetation_present  INTEGER,
    urban_interface         INTEGER,
    steep_terrain           INTEGER,
    water_body_present      INTEGER,
    image_quality_limited   INTEGER,
    model       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
)
```

Public API:

```python
def init_db(path: Path) -> sqlite3.Connection: ...
def insert_prediction(conn, lon, lat, timestamp, size_km, source, rgb_path, swir_path, prediction, model) -> int: ...
def fetch_all(conn) -> list[dict]: ...
def fetch_recent(conn, hours: int) -> list[dict]: ...
```

---

### Step 2: `live.py` — current satellite position

Wraps the SimSat `/data/current/position` endpoint to get the satellite's current lon/lat, then calls `fetch_rgb` and `fetch_swir` with those coordinates.

```python
def get_current_position(base_url: str = SIMSAT_BASE_URL) -> tuple[float, float]:
    """Return (lon, lat) of the satellite's current simulated position."""

def fetch_live_images(size_km: float = 5.0) -> tuple[float, float, bytes, bytes]:
    """Return (lon, lat, rgb_bytes, swir_bytes) at the current satellite position."""
```

---

### Step 3: `scripts/predict.py` — CLI

Two subcommands:

**`live`** — fetch current satellite image, run prediction, print to console, optionally save to DB.

```
uv run scripts/predict.py live --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0
uv run scripts/predict.py live --backend anthropic --save
```

Flags: `--backend`, `--model`, `--quant`, `--port`, `--size-km`, `--save` (persist to DB).

**`backfill`** — iterate over the last N days using the historical SimSat endpoint for a set of locations (uses `LOCATIONS` from `locations.py`), run predictions, save all to DB.

```
uv run scripts/predict.py backfill --days 7 --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0
```

Flags: `--days`, `--backend`, `--model`, `--quant`, `--port`, `--size-km`, `--concurrency`.

Both subcommands share the same prediction logic already in `evaluator.py` (reuse `anthropic_backend` / `llama_backend`).

---

### Step 4: `app/app.py` — Streamlit frontend

**Layout:**

```
[sidebar]                  [main]
  filters                    pydeck map
    risk level                 BitmapLayer (rgb.png per tile)
    date range                 ScatterplotLayer (risk color)
    model
                           [below map or right panel]
                             selected tile:
                               rgb | swir images side by side
                               prediction table
```

**Map layer configuration:**

Each tile has center (lon, lat) and size_km = 5. Convert to bounding box:
- `delta_deg ≈ size_km / 111` (degrees per km at equator, good enough for a demo)
- `bounds = [[lon - d, lat - d], [lon + d, lat + d]]`

Risk level color map:
- `low`: green `[0, 180, 0]`
- `medium`: orange `[255, 165, 0]`
- `high`: red `[220, 0, 0]`

**Sidebar filters:** risk level multiselect, date range slider, model selector — all filter the `fetch_all()` result before rendering.

**Live refresh button:** "Fetch live prediction" button triggers a `live` prediction and inserts into DB, then reruns the app.

---

### Step 5: dependencies to add to `pyproject.toml`

```
streamlit>=1.35
pydeck>=0.9
```

---

### Open questions / tradeoffs

- The `BitmapLayer` renders images from local file paths via a base64 data URL. Streamlit serves files from the project directory, so paths must be relative to the app working directory or converted to data URLs at render time.
- The `/data/current/position` endpoint returns the satellite's simulated current position — this may be over ocean or a location with no Sentinel coverage. The live CLI should handle SimSat 404/500 gracefully and print a clear message rather than crashing.
- Backfill uses `LOCATIONS` from `locations.py` as the set of tiles to fetch. A future improvement could let the user specify arbitrary coordinates.