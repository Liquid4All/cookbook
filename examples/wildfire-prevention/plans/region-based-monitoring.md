# Plan: Region-Based Tile Monitoring

## Goal

Replace the ad-hoc location list and unconstrained satellite-following loop with a structured regional monitoring system. The user picks a named region (a Mediterranean or Spanish natural park), the system generates a grid of tiles covering it, and both the live watch loop (`predict.py`) and the historical backfill (`backfill.py`) use that grid as their unit of work. The Streamlit app renders a spatial risk map and a time-series chart (per-tile lines + region average) so seasonal wildfire patterns become visible.

---

## Regions

Replace the existing US-centric and Alentejo locations with five Mediterranean/Spanish regions that are well-known wildfire areas.

| id | Name | Approx bbox | Why |
|----|------|-------------|-----|
| `collserola` | Parc de Collserola, Barcelona | lon -2.15 to 2.22, lat 41.40 to 41.50 | Literally on Barcelona's doorstep; burns most summers |
| `garraf` | Parc del Garraf, Barcelona | lon 1.70 to 1.95, lat 41.18 to 41.38 | Coastal limestone scrubland south of Barcelona; very high fire frequency |
| `montseny` | Parc Natural del Montseny | lon 2.25 to 2.55, lat 41.70 to 41.85 | UNESCO Biosphere Reserve, 50 km NE of Barcelona |
| `donana` | Parque Nacional de Doñana, Huelva | lon -6.60 to -6.20, lat 36.90 to 37.20 | Spain's most famous national park; major fire in 2017 |
| `sierra_nevada` | Parque Nacional Sierra Nevada, Granada | lon -3.60 to -2.90, lat 36.90 to 37.25 | Mediterranean mountain ecosystem; fire risk in summer |

All five are either in Spain or adjacent Mediterranean territory, giving useful contrast: two Catalan coastal parks, one inland Catalan biosphere reserve, one Atlantic-coast lowland wetland, one high mountain range.

---

## Step 1: `src/wildfire_prevention/regions.py` (new file)

Define a `Region` dataclass and a `REGIONS` registry dict.

```python
@dataclass(frozen=True)
class Region:
    id: str
    name: str
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float
    description: str
```

Add a `generate_tile_grid(region: Region, size_km: float) -> list[tuple[float, float]]` function that returns `(lon, lat)` tile centers covering the bounding box. The latitude step is `size_km / 111.0`; the longitude step is corrected for the latitude midpoint with `/ cos(lat_mid_rad)` so tiles are approximately square on the ground.

Add a helper `tile_contains(lon: float, lat: float, tile_lon: float, tile_lat: float, size_km: float) -> bool` that checks whether a point falls within a tile's bounding box. This is used by `predict.py` to gate on-satellite-over-region.

---

## Step 2: `src/wildfire_prevention/db.py`

Add a nullable `region_id TEXT` column to `predictions`.

Migration: inside `init_db`, after creating the table, check `PRAGMA table_info(predictions)`. If `region_id` is absent, run `ALTER TABLE predictions ADD COLUMN region_id TEXT`. This is safe to call on existing databases.

Update `insert_prediction` to accept and store `region_id: str | None`.

---

## Step 3: `backfill.py` absorbs the region-scan use case

`backfill.py` already has the right shape: it iterates over `(lon, lat, timestamp)` triples, fetches images from SimSat's historical API, runs inference, and inserts to DB. The only thing that changes with regions is *how the tile list is produced*.

Make `--region` a required argument. Remove `--locations` and the `LOCATIONS` import entirely from `backfill.py`. The `LOCATIONS` list stays in `locations.py` because `generate_samples.py` and `check_samples.py` still use it for training-data generation - that is a separate workflow and needs no changes.

The updated `run_backfill`:
- Resolves the `Region` from `REGIONS`.
- Calls `generate_tile_grid(region, size_km)` to get tile centers.
- Builds the task list as `[(lon, lat, timestamp, tile_id) for tile in grid for timestamp in dates]` where `tile_id = f"{region.id}_{lon:.3f}_{lat:.3f}"`.
- Passes `region_id=region.id` to `insert_prediction`.

The `_process` inner function already takes `(lon, lat, timestamp, loc_id)`. The only required change is threading `region_id` through it. No duplication of logic; no new script.

Updated usage:

```bash
uv run scripts/backfill.py --backend anthropic --days 90 --region collserola
uv run scripts/backfill.py --backend anthropic --days 365 --region garraf
```

---

## Step 4: `predict.py` gains a `--region` flag

The polling loop is unchanged. What changes is the gate logic after fetching the satellite's current position.

Without `--region` (current behavior): process every position.

With `--region`:
1. At startup, call `generate_tile_grid(region, size_km)` to build the tile grid once.
2. On each poll, after getting `(lon, lat)` from SimSat, check whether it falls inside any tile with `tile_contains(lon, lat, tile_lon, tile_lat, size_km)`.
3. If the satellite is not over any tile: log `[position] outside region, skipping` and sleep.
4. If the satellite is over a tile: use the *tile center* `(tile_lon, tile_lat)` as the canonical coordinates (not the raw satellite position), so predictions from the live loop and the backfill are colocated in the DB and comparable on the map.
5. Pass `region_id=region.id` to `insert_prediction`.

Using the tile center instead of the raw satellite position is the key design decision: it means live predictions land on the same grid points as backfill predictions, making time-series grouping in the app trivial (group by `(round(lon, 3), round(lat, 3))`).

Updated usage:

```bash
uv run scripts/predict.py --backend anthropic
uv run scripts/predict.py --backend anthropic --region collserola
uv run scripts/predict.py --backend anthropic --region garraf --size-km 10
```

---

## Step 5: `app/app.py` updates

### Region filter and map zoom

Add a "Region" selector in the sidebar. Options: "All" plus each region id. When a region is selected:
- Filter rows to `region_id == selected`.
- Compute `ViewState` center and zoom from the region's bbox (center = midpoint of bbox; zoom derived from the larger of lat/lon span).

When "All" is selected, keep the current behavior (center on the first row).

### Time-series chart

Below the map, add a "Risk over time" section, visible when any rows are present.

Group rows by tile, using `(round(lon, 3), round(lat, 3))` as the tile key. Map `risk_level` to a numeric score (low=1, medium=2, high=3) for plotting.

Render two charts side by side (or stacked):

1. **Per-tile lines**: one line per tile, X = `timestamp` (date of Sentinel-2 image), Y = risk score. Each line is colored by the tile's mean risk (green/orange/red). If more than 20 tiles are present, show only the top-10 highest-risk tiles by mean score to keep the chart readable.

2. **Region average**: a single bold line showing the mean risk score across all tiles at each timestamp, plus a shaded band between min and max. This is always shown regardless of tile count.

Use `st.line_chart` if a simple chart suffices, or switch to `plotly` (already available as a transitive dependency via Streamlit) for better control over colors and the shaded band.

---

## File change summary

| File | Type | Change |
|------|------|--------|
| `src/wildfire_prevention/regions.py` | New | `Region` dataclass, `REGIONS` dict, `generate_tile_grid`, `tile_contains` |
| `src/wildfire_prevention/db.py` | Modify | Add `region_id` column + migration; update `insert_prediction` signature |
| `scripts/backfill.py` | Modify | Replace `--locations` with required `--region`; remove `LOCATIONS` import; thread `region_id` through `_process` |
| `scripts/predict.py` | Modify | Add `--region` flag; add tile-gate logic in watch loop |
| `app/app.py` | Modify | Region filter, map zoom, per-tile + region-average time-series chart |

No new scripts. `scan_region.py` does not exist.

---

## Remaining open questions

1. **`predict.py` with region: what if the satellite never passes over the region during a session?** Expected. Each polling cycle where the satellite is outside the region prints a timestamped log line, e.g. `[2026-04-17T10:23:01] lon=12.4321 lat=48.2100 outside region collserola, skipping`, so the operator can confirm the loop is alive. The watch loop is best used for opportunistic live capture; `backfill.py --region` is the reliable way to fill a complete grid.
