# Let's build an wildfire prevention system

In this example you will learn how to

- frame the problem
- get high quality labeled data
- evaluate LFM2.5-VL-450M on this task.
- fine-tune it to boost performance.

## Goal

In the hackathon we have access to Sentinel-2 images. Images for the same tile of land are typically available at a 5-10 day frequency.
Wildfires spread way faster than that, so we cannot use Sentinel-2 images to build a system that early detects smoke and alerts.
We need to build something that works at a lower frequency and focus on prevention.

Moreover, the focus of this tutorial is on VLMs, so we want to pick a task and dataset for which pixel-by-pixel measures are not enough, and a more holistic scene understanding is necessary to assess risk. In other words, we want to justify the added complexity of using a VLM by framing a task that really requires a VLM.

The output this VLM produces does not need to be very complex. This is a demo, not a full project. The goal is to guide hackathon participants through a working example, and let them focus on details, or alternative paths we encounter along the way.

## Problem framing

**What do we want our VLM to detect?**

We want the VLM to assess the **wildfire risk level** of a land tile based on scene-level understanding. Specifically, it should identify:
- Whether dry, burnable vegetation is present
- Whether human infrastructure sits at the wildland-urban interface
- Whether terrain features (slopes, ridges) would accelerate fire spread
- Whether natural firebreaks (rivers, reservoirs) are present

This task requires holistic scene understanding, not pixel-level statistics. A pixel-level index like NDVI can tell you that vegetation is stressed, but it cannot tell you that it is dense dry chaparral sitting uphill from a residential neighborhood in an open wind corridor. That contextual judgment is exactly what a VLM is for.

**What images should we provide the VLM?**

Two Sentinel-2 composites per land tile, passed together in a single model call:
- RGB (bands B4-B3-B2): natural color, useful for terrain and infrastructure
- SWIR (bands B12-B8-B4): highlights vegetation moisture stress and dryness

**What is the output format?**

A compact JSON object with one categorical field and five boolean flags:

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

The boolean schema is intentionally simple: each flag is a direct, observable binary signal from the images, making model evaluation straightforward.

## Generate sample data

Images are fetched via [SimSat](https://github.com/DPhi-Space/SimSat), a local Docker service that wraps the Sentinel-2 STAC catalog on AWS Element84.

The `SimSat` service exposes 2 endpoints;

1. live data (`GET /data/current/image/sentinel`). This data is not actually "live", but simulated as if it were live. `SimSat` simulates the actual orbital position of the satellite at the current time, then queries the Sentinel-2 L2A STAC catalog for the most recent acquisition within a 10-day lookback window over that ground position. So "live" means: the newest real Sentinel-2 image captured in the last 10 days at the satellite's simulated current location.

2. historical data (`GET /data/image/sentinel`). This data is the actual historical data fetched from the Sentinel-2 STAC database.

Images are fetched at 5 km tiles (`--size-km 5.0`), which keeps images at or below 512x512 px — the native resolution of LFM2.5-VL-450M — avoiding tiling overhead at inference time.

To generate sample data, first start the `SimSat` service:

```bash
# 1. Start SimSat (from the SimSat repo root, keep it running in a separate terminal)
docker compose up

# 2. Install Python dependencies
uv sync

# 3. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-...

# 4. Generate sample images and Opus 4.6 annotations
uv run scripts/generate_samples.py --size-km 5.0
```

Each run creates a timestamped folder under `data/`, e.g.:

```
data/
  20260416_143052/
    angeles_nf_ca/
      rgb.png
      swir.png
      annotation.json
    alentejo_portugal/
    ...
```

To validate a run:

```bash
uv run scripts/check_samples.py                  # most recent run
uv run scripts/check_samples.py 20260416_143052  # specific run
```

## Tasks

- [x] Clearly define the problem we are solving
- [ ] Generate a sample of images using some location/time that makes sense and check output produced by Opus 4.6. I want to check a frontier model produces sensible outputs, so I can use it to generate distillation data for my VLM.

