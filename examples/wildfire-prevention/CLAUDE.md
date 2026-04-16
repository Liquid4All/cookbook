# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the sample generation script
uv run generate_samples.py

# Type checking
pyright
```

## Architecture

This is a VLM demo that uses `claude-opus-4-6` to assess wildfire risk from Sentinel-2 satellite imagery. The workflow:

1. **Data source:** [SimSat](https://github.com/DPhi-Space/SimSat), a local Docker service wrapping the Sentinel-2 STAC catalog. Must be running (`docker compose up` from its repo root) before fetching images.
2. **Input:** Single Sentinel-2 RGB or false-color infrared images per land tile.
3. **Output:** Structured JSON risk assessments written to `data/samples/` (one `.jpg` + one `.json` per location).
4. **Goal:** Use Opus 4.6 outputs as distillation data to fine-tune `LFM2.5-VL-450M`.

The VLM output schema is:
```json
{
  "risk_level": "low | medium | high | critical",
  "vegetation_type": "...",
  "dryness_indicators": [...],
  "risk_factors": [...],
  "reasoning": "..."
}
```

Source code lives in `src/wildfire_prevention/`. Requires `ANTHROPIC_API_KEY` set in the environment.
