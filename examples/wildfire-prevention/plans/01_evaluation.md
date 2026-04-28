# Intent: build an evaluation pipeline

We need an evaluation pipeline we can use either with

- Anthropic models, to double check the labels the annotator produced are fully reproducible

- LFM2.5-VL-450 and other LFM2.5 VL models served with llama.cpp. Here's an example on how I would like to do inference: https://github.com/Liquid4All/cookbook/tree/main/examples/invoice-parser

The evaluation pipeline accepts as input a name for a dataset (like data/20260416_141946) and it runs an evaluation.

Out of all the fields in the output

```json
{
  "id": "alentejo_portugal",
  "lon": -7.9,
  "lat": 38.5,
  "timestamp": "2024-07-25T12:00:00",
  "size_km": 5.0,
  "risk_level": "medium",
  "dry_vegetation_present": true,
  "urban_interface": true,
  "steep_terrain": false,
  "water_body_present": false,
  "image_quality_limited": false
}
```

the script evaluates:

- is the output correctly formatted as JSON
- does it contain all required fields: `risk_level`, `dry_vegetation_present`, `urban_interface`, `steep_terrain`, `water_body_present`, `image_quality_limited`
- does each field match the ground truth (exact match; booleans and the 3-level categorical)

---

## Architecture

Two new files:
- `src/wildfire_prevention/evaluator.py`: shared inference interface + metrics
- `scripts/evaluate.py`: CLI entry point

---

## Step 1: `evaluator.py` — inference interface

Define a common `predict(rgb_bytes, swir_bytes) -> dict` protocol with two backends:

**Anthropic backend** — reuses `annotate()` from `annotator.py` directly. No new code, just a thin wrapper.

**llama.cpp backend** — OpenAI-compatible client pointing to `http://127.0.0.1:{port}/v1`, following the invoice-parser pattern:
- Images passed as base64 data URLs in `image_url` content blocks (one per image)
- Same system prompt as `annotator.py`
- `response_format` with JSON schema for structured output (avoids the markdown-fence stripping hack)
- `temperature=0.0`

The two backends differ only in client and image encoding. The prompt is identical, ensuring a fair comparison.

**JSON schema for structured output** (used by llama.cpp backend):

```python
SCHEMA = {
    "type": "object",
    "properties": {
        "risk_level":             {"type": "string", "enum": ["low", "medium", "high"]},
        "dry_vegetation_present": {"type": "boolean"},
        "urban_interface":        {"type": "boolean"},
        "steep_terrain":          {"type": "boolean"},
        "water_body_present":     {"type": "boolean"},
        "image_quality_limited":  {"type": "boolean"},
    },
    "required": [
        "risk_level", "dry_vegetation_present", "urban_interface",
        "steep_terrain", "water_body_present", "image_quality_limited"
    ],
}
```

---

## Step 2: `evaluator.py` — metrics

```python
EVAL_FIELDS = [
    "risk_level",
    "dry_vegetation_present",
    "urban_interface",
    "steep_terrain",
    "water_body_present",
    "image_quality_limited",
]
```

Per sample:
- `valid_json`: bool — did the model return parseable JSON?
- `fields_present`: bool — are all 6 fields present?
- Per-field match: bool — predicted value == ground truth value

Aggregate across all samples:
- Per-field accuracy: `correct / total`
- Overall accuracy: `sum of all field matches / (total samples * num fields)`

---

## Step 3: `scripts/evaluate.py` — CLI

```
uv run scripts/evaluate.py --dataset data/20260416_141946 --backend anthropic
uv run scripts/evaluate.py --dataset data/20260416_141946 --backend llama --model LFM2.5-VL-450M --port 8080
```

For each sample:
1. Load `rgb.png`, `swir.png`, `annotation.json` from `{dataset}/{location_id}/`
2. Run `predict()` with the chosen backend
3. Compute per-sample metrics

---

## Step 4: output

Each run is saved to `evals/{eval_run_id}/` where `eval_run_id` is a timestamp (`YYYYMMDD_HHMMSS`), the same convention used by `generate_samples.py`.

Each run folder contains one file: `report.md`.

### `report.md` structure

```markdown
# Wildfire Risk Eval — {eval_run_id}

**Dataset:** data/20260416_141946
**Backend:** anthropic | llama
**Model:** claude-opus-4-6 | LFM2.5-VL-450M

## Per-sample results

| id | valid_json | fields_present | risk_level | dry_veg | urban | terrain | water | quality |
|---|---|---|---|---|---|---|---|---|
| angeles_nf_ca | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| ...

## Accuracy summary

| field | accuracy |
|---|---|
| risk_level | 0.82 |
| dry_vegetation_present | 0.91 |
| urban_interface | 0.77 |
| steep_terrain | 0.86 |
| water_body_present | 0.95 |
| image_quality_limited | 1.00 |
| **overall** | **0.88** |
```

The report is also printed to stdout as the script runs.

---

## Open questions / tradeoffs

- `--backend llama` needs `llama-server` on `PATH`. Fail fast with a clear error if not found.
- Should the eval support `--concurrency`? Yes: default `3` for Anthropic, default `1` for llama (single-threaded server).
