# Intent: create a data pipeline for our wildfire risk level problem

## Goal

Python source code that downloads a "good" sample of Sentinel images from [SimSat](https://github.com/DPhi-Space/SimSat) and annotates each image using `claude-opus-4-6` to produce a labeled dataset for evaluating and fine-tuning `LFM2.5-VL-450M`.

---

## Band selection rationale

Sentinel-2 has 13 bands. For a VLM doing wildfire risk assessment, the most informative combinations are:

**Primary: SWIR composite (B12-B8-B4)**
- B12 (SWIR2, 2190 nm): highly sensitive to vegetation moisture stress and active fire heat
- B8 (NIR, 842 nm): strong contrast between healthy vs. stressed vegetation
- B4 (Red, 665 nm): separates bare soil from vegetation
- In composite: burned areas appear dark red/black, stressed vegetation orange, healthy vegetation green. Penetrates haze and thin smoke. Best for risk differentiation.

**Secondary: RGB (B4-B3-B2)**
- Natural color, intuitive for a VLM trained on human-visible imagery
- Captures terrain features, roads, buildings, open corridors
- Worse at vegetation stress, but gives the model familiar scene context

**Decision: fetch both composites per location.** Pass both images to Opus 4.6 in the same request (multi-image prompt). SWIR drives vegetation/dryness signals; RGB drives structural/terrain signals. Together they cover all five fields of the output schema.

Bands not needed: B1 (aerosol correction), B9/B10 (cirrus), B5/B6/B7 (red-edge) are redundant with B8 for this task, and raw index images (NDVI, NBR) are derived products a VLM cannot natively interpret as well as composites.

---

## SimSat API

Base URL: `http://localhost:9005` (must be running via `docker compose up` in the SimSat repo).

```
GET /data/image/sentinel
  lon            float   longitude of tile center
  lat            float   latitude of tile center
  timestamp      str     ISO 8601 datetime, e.g. "2024-07-15T12:00:00"
  spectral_bands str     comma-separated band names, e.g. "swir2,nir,red"
  size_km        float   tile edge length in km (default 5.0)
  return_type    str     "png" (default) or "array"
```

Band name strings (as SimSat expects them): `red`, `green`, `blue`, `nir`, `swir1`, `swir2`, `rededge1`, `rededge2`, `rededge3`, `nir08`, `nir09`, `swir16`, `swir22`.

---

## Location set

Pick ~20 locations that span the four risk levels, so the dataset is balanced. Diversity axes: continent, climate zone, season (pick timestamps where the region is in its dry/fire season).

| Location | Lon | Lat | Timestamp | Expected risk |
|---|---|---|---|---|
| Angeles NF, California | -118.1 | 34.3 | 2024-08-10 | critical |
| Santa Barbara hills, CA | -119.7 | 34.5 | 2024-09-01 | high |
| Napa Valley, CA | -122.3 | 38.5 | 2024-07-20 | high |
| Sierra Nevada foothills | -120.5 | 37.5 | 2024-08-15 | medium |
| Portuguese interior (Alentejo) | -7.9 | 38.5 | 2024-07-25 | high |
| Attica, Greece | 23.7 | 38.1 | 2024-08-01 | critical |
| Cerrado, Brazil | -47.9 | -15.8 | 2024-08-20 | high |
| Patagonia steppe, Argentina | -69.0 | -40.5 | 2024-01-15 | medium |
| Black Forest, Germany | 8.1 | 48.2 | 2024-06-15 | low |
| Scottish Highlands | -4.5 | 57.0 | 2024-05-10 | low |
| Tropical rainforest, Borneo | 114.0 | 1.5 | 2024-03-01 | low |
| Wet savanna, Tanzania | 35.0 | -5.0 | 2024-04-01 | low |
| Australian outback (NSW) | 145.0 | -32.0 | 2024-01-20 | critical |
| Victorian Alpine, Australia | 147.0 | -37.0 | 2024-02-01 | high |
| Kalahari, Botswana | 24.0 | -22.0 | 2024-09-01 | medium |
| Zagros mountains, Iran | 47.0 | 33.5 | 2024-07-01 | medium |
| Israeli Negev | 34.8 | 30.8 | 2024-06-01 | medium |
| Alpine meadow, Switzerland | 8.2 | 46.8 | 2024-06-20 | low |
| Amazon, Brazil | -60.0 | -3.0 | 2024-08-15 | low |
| Congo basin, DRC | 23.0 | -2.0 | 2024-03-15 | low |

---

## Output schema (per sample)

Each sample is saved to `data/samples/<id>/`:
- `rgb.png`: RGB composite (B4-B3-B2)
- `swir.png`: SWIR composite (B12-B8-B4)
- `annotation.json`: Opus 4.6 output

```json
{
  "id": "angeles_nf_ca",
  "lon": -118.1,
  "lat": 34.3,
  "timestamp": "2024-08-10T12:00:00",
  "risk_level": "low | medium | high | critical",
  "dry_vegetation_present": true,
  "urban_interface": false,
  "steep_terrain": true,
  "water_body_present": false,
  "image_quality_limited": false
}
```

---

## Implementation plan

### Step 1: scaffold the package

Files to create:
- `src/wildfire_prevention/simsat.py`: thin client for the SimSat HTTP API
- `src/wildfire_prevention/annotator.py`: calls Opus 4.6 with both images and returns the parsed JSON
- `src/wildfire_prevention/locations.py`: the hardcoded location list as a dataclass list
- `scripts/generate_samples.py`: CLI entry point that iterates locations, fetches images, calls assessor, saves outputs

### Step 2: `simsat.py`

```python
def fetch_image(
    lon: float,
    lat: float,
    timestamp: str,
    bands: list[str],
    size_km: float = 10.0,
    base_url: str = "http://localhost:9005",
) -> bytes:
    """Return raw PNG bytes for the requested band composite."""
```

Two convenience wrappers over `fetch_image`:
- `fetch_rgb(...)` passing `["red", "green", "blue"]`
- `fetch_swir(...)` passing `["swir2", "nir", "red"]`

### Step 3: `annotator.py`

Prompt strategy: send both images in a single `claude-opus-4-6` message.
- System prompt: describe the task, output schema, and that it is looking at Sentinel-2 RGB and SWIR composites.
- User message: two `image` content blocks (RGB first, SWIR second) followed by the JSON schema and instruction to return only valid JSON.
- Parse response with `json.loads`; raise on parse failure.

Use `anthropic` SDK with `max_tokens=1024`. Model: `claude-opus-4-6`.

### Step 4: `locations.py`

```python
@dataclass
class Location:
    id: str
    lon: float
    lat: float
    timestamp: str  # ISO 8601
    expected_risk: str  # for sanity checking only, not passed to the model
```

Hardcode the 20 locations from the table above.

### Step 5: `scripts/generate_samples.py`

```
for each location:
    fetch rgb.jpg and swir.jpg via SimSat
    call assessor
    write data/samples/<id>/rgb.jpg
    write data/samples/<id>/swir.jpg
    write data/samples/<id>/annotation.json
    print progress line with id and returned risk_level
```

Add `--dry-run` flag that fetches images but skips the Opus call (useful for checking SimSat connectivity).
Add `--location` flag to run a single location by id.

### Step 6: validation

After generation, run a quick sanity check script (`scripts/check_samples.py`) that:
- Counts samples per risk level (goal: roughly balanced, at least 3 per level)
- Prints a table: `id | expected_risk | model_risk | match?`
- Flags any JSON parse failures or missing files

---

## Open questions / tradeoffs

- `size_km`: 10 km tiles give more scene context (terrain, infrastructure) but reduce spatial resolution. 5 km is sharper but may miss regional context. Start with 10 km.
- If SimSat returns HTTP 404 for a (lon, lat, timestamp) combination (no Sentinel pass in the archive), log and skip rather than crash.
- Opus 4.6 cost: 20 locations x 2 images each = 40 image API calls. At current pricing, roughly $0.30-0.50 total. Acceptable.
