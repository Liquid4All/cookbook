# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Generate sample images and Opus 4.6 annotations
uv run scripts/generate_samples.py --size-km 5.0 --concurrency 3

# Validate a generated dataset run
uv run scripts/check_samples.py                  # most recent run
uv run scripts/check_samples.py 20260416_143052  # specific run

# Continuous watch loop (polls satellite position, runs inference, writes to SQLite)
uv run scripts/predict.py --backend anthropic
uv run scripts/predict.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0

# Backfill historical predictions
uv run scripts/backfill.py --backend anthropic --days 7

# Evaluate a model against a generated dataset
uv run scripts/evaluate.py --dataset data/20260416_141946 --backend anthropic

# Launch the Streamlit map app
uv run streamlit run app/app.py

# Type checking
pyright
```

## Architecture

This is a VLM demo that uses `claude-opus-4-6` to assess wildfire risk from Sentinel-2 satellite imagery. The workflow:

1. **Data source:** [SimSat](https://github.com/DPhi-Space/SimSat), a local Docker service wrapping the Sentinel-2 STAC catalog. Must be running (`docker compose up` from its repo root) before fetching images.
2. **Input:** Two Sentinel-2 composites per land tile, passed together in a single model call: RGB (B4-B3-B2) for terrain/infrastructure, and SWIR (B12-B8-B4) for vegetation moisture stress.
3. **Output:** Structured JSON risk assessments. `generate_samples.py` writes timestamped folders under `data/` (one `rgb.png` + `swir.png` + `annotation.json` per location). `predict.py` writes predictions to a local SQLite database.
4. **Goal:** Use Opus 4.6 outputs as distillation data to fine-tune `LFM2.5-VL-450M`.

The VLM output schema is:
```json
{
  "risk_level": "low | medium | high",
  "dry_vegetation_present": true,
  "urban_interface": false,
  "steep_terrain": true,
  "water_body_present": false,
  "image_quality_limited": false
}
```

Source code lives in `src/wildfire_prevention/`. Requires `ANTHROPIC_API_KEY` set in the environment.
