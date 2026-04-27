# Let's build a wildfire prevention system using a compact Vision-Language Modell and Sentinel-2 satellite images

In this example you will learn how to build a basic wildfire prevention system using:

- Sentinel-2 satellite images
- [LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M), a compact Vision-Language Model running directly on the satellite, so inference happens in orbit and only a lightweight JSON payload is downlinked to Earth.

```mermaid
flowchart LR
    subgraph satellite["Satellite"]
        SimSat["SimSat<br/>(Docker service)"]
        predict["predict.py<br/>(watch loop)"]
        LFM["LFM2.5-VL-450M<br/>(llama-server)"]

        SimSat -->|"satellite position<br/>(lon, lat)"| predict
        SimSat -->|"RGB + SWIR images"| predict
        predict -->|"RGB + SWIR bytes"| LFM
        LFM -->|"risk JSON"| predict
    end

    subgraph earth["Earth"]
        DB[("wildfire.db<br/>(SQLite)")]
    end

    predict -->|"INSERT prediction row"| DB
```

We will cover all the stages of the journey:

## Steps
- [Problem framing](#1-problem-framing)
- [System design](#2-system-design)
- [Data collection and labeling](#3-data-collection-and-labeling-pipeline)
- [Evaluation](#4-evaluation)
- [Fine-tuning](#5-fine-tuning)


## 1. Problem framing

We want to reduce the number of wildfires by identifying areas with high-risk from Sentinel-2 images, and providing actionable feedback to local authorities like firefighters so they can act before the fire has even started.

![](./assets/wildfire_stages.gif)

> **What is Sentinel-2?**
>
> Sentinel-2 is a European Space Agency (ESA) satellite mission that captures high-resolution optical imagery of Earth's surface. It's part of the EU's Copernicus programme.
>
> It consists of 3 satellites (Sentinel-2A, 2B and 2C) which orbit in tandem:
>
> - Revisiting the same location **every 5 days** at the equator (more frequently at higher latitudes).
> - Capturing **multispectral** images. Instead of capturing a single photograph, they measure reflected light across **13 discrete wavelength ranges** simultaneously. Each range is called a band, and each band carries some information about vegetation health, water content, soil moisture or atmospheric conditions that is not visible to the naked eye.

In this repository we will use two different images for a given location:

- **RGB (B4-B3-B2):** natural color. Useful for reading urban texture, terrain shape from shadows, and water bodies.
- **SWIR (B12-B8-B4):** shortwave infrared. Highlights vegetation moisture stress and dryness, the primary fuel indicator.

Using this input, we can extract early signs of vegeatation distress, or urban risk, and alert local authorities

Let's go through an example:

### Example

1. A Sentinel-2 satellite flies over *Attica (Greece)* on 2024-08-01, and takes these 2 pictures.

    | RGB | SWIR |
    |-----|------|
    | ![RGB](assets/attica_greece_rgb.png) | ![SWIR](assets/attica_greece_swir.png) |
    | *Attica, Greece. 2024-08-01* | *Attica, Greece. 2024-08-01* |

2. This image pair is passed to the Vision-Language Model, which has holistic scene understanding, not just pixel-level statistics, and the model extracts the following risk profile.

    ```json
    {
      # Primary signal for prioritization
      # Either "low", "medium" or "high"
      "risk_level": "high",

      # The most direct fuel indicator
      "dry_vegetation_present": true,
      
      # Fire next to urban areas elevates the stakes
      "urban_interface": true,
      
      # Fire spreads faster uphill
      "steep_terrain": true,

      # Natural firebreaks than can limit spread and inform
      # suppression strategy
      "water_body_present": false,

      # When true take the risk scores with lower confidence 
      "image_quality_limited": false
    }
    ```

3. This payload is downlinked to ground control on Earth. As the image tile has high risk, the system sends an alert to local fire services. These can then take precautionary measures like
    - ground patrol deployment or
    - controlled burns to reduce available fuel.


## 2. System design

### Design rationale

You could point a frontier model (GPT-5, Gemini 2.0 Flash, or Claude 3.6 Sonnet) at satellite images and it would do a good job. So why bother using a smaller one that needs fine-tuning?

The bottleneck is not capability. It is **data transmission**.

A frontier model runs on a server on Earth. To use it:

- The satellite downlinks raw images to a ground station.
- The ground station feeds the model.
- The model produces the output on Earth.

Images are high-dimensional: large matrices of pixel values per band, per frame. Multiply that by the number of captures per orbit, and you have a serious bandwidth problem.

```mermaid
flowchart LR
    subgraph satellite["Satellite"]
        A["📷 Camera"]
    end

    subgraph earth["Earth"]
        B["🖥️ Ground Station"]
        C["🌍 Frontier Model<br/>on Earth"]
        D["📦 Payload Output"]
        B --> C --> D
    end

    A =="|⚠️ raw image downlink<br/>hundreds of MB per frame|"==> B
    linkStyle 2 stroke:#ff0000,stroke-width:3px
```

A small model removes that bottleneck entirely. At 450M parameters, LFM2.5-VL-450M is compact enough to run directly on the satellite:

- The satellite captures the image and runs inference on-board.
- The local model produces the payload output in orbit.
- Only the lightweight output is downlinked to the ground station.

```mermaid
flowchart LR
    subgraph satellite["🛰️ Satellite"]
        A["📷 Capture Image"] --> B["Local Model<br/>LFM2.5-VL-450M"] --> C["📦 Payload Output"]
    end

    C -->|"lightweight output only"| D["🖥️ Ground Station"]
    linkStyle 2 stroke:#008000,stroke-width:2px
```

### Proof of Concept (PoC)

Rather than building a full satellite stack, we simulate the on-board pipeline locally using three components:

- **[SimSat](https://github.com/DPhi-Space/SimSat):** a local Docker service that simulates a satellite orbit and serves real Sentinel-2 imagery from the AWS Element84 STAC catalog. It provides the satellite's current position and the corresponding RGB and SWIR images.
- **`predict.py`:** a lightweight Python watch loop that polls SimSat for the current position, fetches the images, and drives the inference pipeline.
- **LFM2.5-VL-450M:** the local model running via `llama-server`, playing the role of the on-board VLM.

```mermaid
flowchart LR
    subgraph satellite["Satellite"]
        SimSat["SimSat<br/>(Docker service)"]
        predict["predict.py<br/>(watch loop)"]
        LFM["LFM2.5-VL-450M<br/>(llama-server)"]

        SimSat -->|"satellite position<br/>(lon, lat)"| predict
        SimSat -->|"RGB + SWIR images"| predict
        predict -->|"RGB + SWIR bytes"| LFM
        LFM -->|"risk JSON"| predict
    end

    subgraph earth["Earth"]
        DB[("wildfire.db<br/>(SQLite)")]
    end

    predict -->|"INSERT prediction row"| DB
```

The system monitors 22 fixed locations. Each location is a single 5 km tile centered on a known fire-prone coordinate. One prediction is produced per location per satellite pass. These are the locations we monitor:

| id | Location |
|----|----------|
| `angeles_nf_ca` | Angeles National Forest, California |
| `santa_barbara_ca` | Santa Barbara, California |
| `napa_valley_ca` | Napa Valley, California |
| `sierra_nevada_ca` | Sierra Nevada, California |
| `alentejo_portugal` | Alentejo, Portugal |
| `attica_greece` | Attica, Greece |
| `cerrado_brazil` | Cerrado, Brazil |
| `patagonia_argentina` | Patagonia, Argentina |
| `black_forest_germany` | Black Forest, Germany |
| `scottish_highlands` | Scottish Highlands |
| `borneo_rainforest` | Borneo Rainforest |
| `tanzania_savanna` | Tanzania Savanna |
| `outback_nsw_australia` | Outback NSW, Australia |
| `victorian_alpine_au` | Victorian Alps, Australia |
| `kalahari_botswana` | Kalahari, Botswana |
| `zagros_iran` | Zagros Mountains, Iran |
| `negev_israel` | Negev Desert, Israel |
| `alpine_switzerland` | Swiss Alps |
| `amazon_brazil` | Amazon, Brazil |
| `congo_basin_drc` | Congo Basin, DRC |
| `lahaina_maui_hi` | Lahaina, Maui, Hawaii |
| `mati_attica_gr` | Mati, Attica, Greece |

### Quickstart

1. Clone the SimSat repository:

    ```bash
    git clone https://github.com/DPhi-Space/SimSat.git
    cd SimSat
    ```

2. Start SimSat (keep it running in a separate terminal):

    ```bash
    docker compose up
    ```

3. Open the SimSat dashboard at [http://localhost:8000](http://localhost:8000), click **Start**, and verify the satellite position is moving.

4. Install Python dependencies:

    ```bash
    uv sync
    ```

5. Start the watch loop:

    ```bash
    # Watch all 22 locations
    uv run scripts/predict.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0

    # Watch a single location
    uv run scripts/predict.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0 --location attica_greece
    ```

6. Optionally, backfill historical predictions to seed the database before the live loop. For each location and day, backfill writes the RGB and SWIR images to `db_images/{row_id}/` on disk and inserts the prediction into `wildfire.db`, linked by row ID.

    ```bash
    # All locations, last 7 days
    uv run scripts/backfill.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0 --days 7

    # Single location, last 90 days (builds a seasonal dataset)
    uv run scripts/backfill.py --backend local --model LiquidAI/LFM2.5-VL-450M-GGUF --quant Q8_0 --days 90 --location attica_greece
    ```

7. Once the database has predictions, launch the app:

    ```bash
    uv run streamlit run app/app.py
    ```

## 3. Data collection and labeling pipeline

We use `claude-opus-4-6` to label a dataset of satellite image pairs.

![](./assets/input_output.jpg)

The dataset is built from a cross-product of locations, spatial tiles, and temporal tiles, then split into train and test by a temporal cutoff.

![](./assets/spatial_temporal_tiles.gif)

To run the data collection and labeling pipeline you will need an Anthropic API key.

```bash
export ANTHROPIC_API_KEY=sk-...

# All 22 locations, push final dataset to Hugging Face
uv run scripts/generate_samples.py \
  --start-date 2024-01-01 --end-date 2025-12-31 \
  --n-temporal-tiles 12 --n-spatial-tiles 4 \
  --test-ratio 0.2 --concurrency 3 \
  --hf-dataset your-username/wildfire-risk

# Only 1 region, Attica, Greece
uv run scripts/generate_samples.py \
    --start-date 2024-01-01 --end-date 2024-12-31 \
    --n-temporal-tiles 12 --n-spatial-tiles 4 \
    --test-ratio 0.2 --concurrency 3 \
    --location attica_greece
```

For each `(location, spatial tile, timestamp)` triple, `generate_samples.py` does the following:

1. Creates a timestamped run directory under `data/` (e.g., `data/20260416_143052/`).

2. Samples `--n-temporal-tiles` timestamps evenly spaced within `[--start-date, --end-date]` using bin-center placement, so timestamps are always in the interior of the window.

    ```
    Jan 2024                                       Dec 2024
    │                                                    │
    │   t00    t01    t02    t03    t04    t05           │
    │    ●      ●      ●      ●      ●      ●            │
    ├────┴──────┴──────┴──────┴──────┼──────┴────────────┤
    │                                │                   │
    │           train                │       test        │
    │         (80% of window)        │   (20% of window) │
    └────────────────────────────────┴───────────────────┘
                                    ↑
                                    cutoff
    ```

3. Builds a centered square grid of `--n-spatial-tiles` tiles around each location center, spaced `--size-km` apart.

    ```
    ┌─────────┬─────────┬─────────┐
    │   s00   │   s01   │   s02   │
    ├─────────┼─────────┼─────────┤
    │   s03   │   s04   │   s05   │   ← s04 = location center
    ├─────────┼─────────┼─────────┤
    │   s06   │   s07   │   s08   │
    └─────────┴─────────┴─────────┘
            ←  5 km  →
    ```

4. Fetches the RGB and SWIR images in parallel from SimSat for each `(spatial tile, timestamp)` pair.

5. Saves `rgb.png` and `swir.png` to the tile subfolder.

6. Sends both images to `claude-opus-4-6` for risk annotation, with automatic retry on rate-limit errors.

7. Saves the structured JSON output as `annotation.json`.

8. Assigns the tile to `train/` or `test/` based on a temporal cutoff: `cutoff = start + (1 - test_ratio) × duration`. All tiles with `timestamp < cutoff` go to `train/`, the rest go to `test/`. This prevents near-duplicate images (Sentinel-2 revisits every 5 days) from appearing on both sides of the split. Tiles are indexed row-major from the top-left of the grid. For a 3×3 grid (`n_spatial_tiles=9`), `s04` is the location center. `t00` is always the earliest timestamp. Indices are zero-padded.

    ```
    data/20260416_143052/
      train/
        attica_greece/
          s00_t00/        <- spatial tile 0, temporal tile 0 (earliest timestamp)
            rgb.png
            swir.png
            annotation.json
          s00_t01/
          s01_t00/
          ...
      test/
        attica_greece/
          s00_t09/        <- same spatial tiles, later timestamps
          ...
    ```

9. Optionally packages the run in [leap-finetune](https://github.com/LiquidAI/leap-finetune) VLM SFT format and pushes `train.jsonl` and `test.jsonl` to Hugging Face Hub (`--hf-dataset`).

To validate a run:

```bash
uv run scripts/check_samples.py                  # most recent run
uv run scripts/check_samples.py 20260416_143052  # specific run
```

The dataset used in the rest of this guide is [Paulescu/wildfire-prevention](https://huggingface.co/datasets/Paulescu/wildfire-prevention). To reproduce it exactly you just need to run, and replace `Paulescu/wildfire-detection` with `YOUR_HF_USER_NAME/DATASET_NAME`

```bash
uv run scripts/generate_samples.py \
  --start-date 2024-01-01 --end-date 2025-12-31 \
  --n-temporal-tiles 12 \
  --n-spatial-tiles 4 \
  --test-ratio 0.2 \
  --concurrency 4 \
  --hf-dataset Paulescu/wildfire-prevention
```

## 4. Evaluation

The evaluation pipeline runs a model against a generated dataset and measures how closely its predictions match the Opus-generated ground truth annotations.

```bash
# Evaluate Claude Opus 4.6 (sanity check)
uv run scripts/evaluate.py \
    --hf-dataset Paulescu/wildfire-prevention \
    --backend anthropic 
    --split test

# Evaluate LFM2.5-VL-450M-GGUF at q8_0 quantization
uv run scripts/evaluate.py \
  --hf-dataset Paulescu/wildfire-prevention \
  --backend local \
  --model LiquidAI/LFM2.5-VL-450M-GGUF \
  --quant Q8_0 \
  --split test
```

Each run saves three files to `evals/{timestamp}/`:
- `report.md`: human-readable accuracy table
- `results.json`: per-sample records with the model's actual predictions, ground truth, and per-field match results
- `meta.json`: run metadata (model, dataset, backend, split)

Once you have two or more eval runs, launch the comparison app to explore results visually:

```bash
uv run streamlit run app/eval_compare.py
```

![](./assets/eval_app_demo.gif)

### Results

Evaluated on 22 locations ([Paulescu/wildfire-prevention](https://huggingface.co/datasets/Paulescu/wildfire-prevention)), ground truth from `claude-opus-4-6`, these are the results:

- `claude-opus-4-6` scores 0.99 overall, near-perfect across all fields. This is no surprise, as this is the model we used to label the data in the first place. The result are not perfect though (99% rather than 100%) due to non-determinism in the token sampling during generation.

- The base LFM2.5-VL-450M scores 0.38 overall: it produces valid JSON reliably but struggles with field accuracy, especially `risk_level` (0.08) and `urban_interface` (0.25). This is expected for a zero-shot compact model on a specialized task. Fine-tuning addresses this gap.

| field | claude-opus-4-6 | LFM2.5-VL-450M Q8_0 |
|---|---|---|
| valid_json | 1.00 | 1.00 |
| fields_present | 1.00 | 1.00 |
| risk_level | 0.99 | 0.08 |
| dry_vegetation_present | 0.99 | 0.48 |
| urban_interface | 0.98 | 0.25 |
| steep_terrain | 0.99 | 0.45 |
| water_body_present | 0.99 | 0.74 |
| image_quality_limited | 1.00 | 0.28 |
| **overall** | **0.99** | **0.38** |
| **avg latency (s)** | **2.91** | **0.72** |

## 5. Fine-tuning

We use [leap-finetune](https://github.com/Liquid4All/leap-finetune) to fine-tune `LFM2.5-VL-450M` on the Opus-labeled dataset via Modal's serverless H100 infrastructure.

### Step 1. Install leap-finetune

Let's clone [leap-finetune](https://github.com/Liquid4All/leap-finetune) inside this project directory and install its dependencies:

```bash
git clone https://github.com/LiquidAI/leap-finetune.git
cd leap-finetune && uv sync && cd ..
```

Authenticate with Hugging Face (necessary if you plan to pull or push private datasets or models) and Modal, so leap-fine tune can run the workload on Modal's serverless GPU platform.

```bash
cd leap-finetune
uv run huggingface-cli login   # needed to pull the model and dataset
uv run python -m modal setup   # needed to launch the training job
cd ..
```

### Step 2. Prepare the dataset

Prepare the dataset and push it to a Modal volume:

```bash
uv run scripts/prepare_wildfire.py --dataset Paulescu/wildfire-prevention --modal
```

The `--modal` flag spins up a Modal container, downloads the dataset from HuggingFace, converts it to JSONL, and writes everything to a Modal volume named `wildfire-prevention`. The volume is then used directly by the training job in the next step.

```mermaid
sequenceDiagram
    participant User as Local machine
    participant Modal
    participant HF as HuggingFace Hub
    participant Vol as Modal volume<br/>wildfire-prevention

    User->>Modal: prepare_wildfire.py --modal
    Modal->>HF: snapshot_download(Paulescu/wildfire-prevention)
    HF-->>Modal: images + dataset splits
    Modal->>Vol: wildfire_train.jsonl, wildfire_test.jsonl
    Modal-->>User: Done. Data ready in volume.
```

### Step 3. Prepare the configuration file

This YAML file is the only file you need to pass to leap-finetune. You can find plenty of examples for different tasks in the [leap-finetune repository](https://github.com/Liquid4All/leap-finetune/tree/main/job_configs).

This is what ours looks like:

```yaml
project_name: "wildfire-prevention"
model_name: "lfm2.5-VL-450M"
training_type: "vlm_sft"

dataset:
  ...

training_config:
  ...

peft_config:
  extends: "DEFAULT_VLM_LORA"
  use_peft: false

benchmarks:
  ...

modal:
  app_name: "wildfire-prevention"
  gpu: "H100:1"
  timeout: 7200
  output_volume: "wildfire-prevention"
  output_dir: "/outputs"
  detach: false

```

Two important observations:

- **Full fine-tuning, not LoRA (`use_peft: false`):** we update both the multimodal projector and the full language model backbone. Satellite imagery is severely underrepresented in standard VLM pretraining data, so the projector needs to genuinely re-learn how to map multispectral patches into meaningful tokens. At 450M parameters, full fine-tuning fits on a single H100 without the memory pressure that motivates LoRA on larger models.

- **Modal section:** the `modal` block tells leap-finetune to run the training job on Modal's serverless GPU platform rather than locally. It specifies the GPU type (`H100:1`), a timeout, and the Modal volume where the prepared dataset lives and where checkpoints are written.


### Step 4. Kick off the fine-tuning

Once the configuration YAML file is ready, fine-tuning is as easy as running:

```bash
cd leap-finetune && uv run leap-finetune ../configs/wildfire_finetune_modal.yaml
```

```mermaid
sequenceDiagram
    participant User as Local machine
    participant Modal
    participant HF as HuggingFace Hub
    participant Vol as Modal volume<br/>wildfire-prevention
    participant Space as HF Space<br/>(training dashboard)

    User->>Modal: leap-finetune wildfire_finetune_modal.yaml
    Modal->>HF: download base model (LFM2.5-VL-450M)
    HF-->>Modal: model weights
    Modal->>Vol: read wildfire_train.jsonl + images
    Vol-->>Modal: training data
    loop H100 training (3 epochs)
        Modal->>Space: stream loss / metrics (trackio)
    end
    Modal->>Vol: save checkpoint per epoch
    Modal-->>User: Done. Checkpoint saved to /outputs/.
```

Training progress is visible at [https://huggingface.co/spaces/Paulescu/wildfire-prevention-finetune](https://huggingface.co/spaces/Paulescu/wildfire-prevention-finetune).


### Step 5. Retrieve the checkpoint

```bash
uv run modal volume ls wildfire-prevention /outputs/
uv run modal volume get wildfire-prevention /outputs/<run-name> ./outputs
```

### Step 6. Quantize the model to GGUF

Running inference with a VLM requires two GGUF files. The following script produces both from a single command:

```bash
uv run scripts/quantize.py \
    --checkpoint ./outputs/<run-name>/<checkpoint> \
    --output ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf
```

- **`--output`** sets the backbone path (`lfm2.5-vl-wildfire-Q8_0.gguf`): the language model weights, quantized to Q8_0 by default.

- The mmproj (`mmproj-lfm2.5-vl-wildfire-Q8_0.gguf`) is written automatically to the same directory, with `mmproj-` prepended to the backbone filename. It contains the vision tower and multimodal projector weights (always F16), the component that encodes satellite images into visual tokens.

To use a different quantization level for the backbone, pass `--quant Q4_K_M` (or `Q4_0`, `Q5_K_M`, `Q6_K`, `F16`). The mmproj is always F16 regardless of `--quant`.

### Step 7. (Optional) Push the GGUF pair to HuggingFace


```bash
uv run scripts/push_gguf_to_hf.py \
    --backbone ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf \
    --mmproj ./outputs/mmproj-lfm2.5-vl-wildfire-Q8_0.gguf \
    --repo <your-hf-username>/wildfire-risk-detector
```

### Step 8. Evaluate the fine-tuned model

```bash
# From local artifacts
uv run scripts/evaluate.py \
    --hf-dataset Paulescu/wildfire-prevention \
    --backend local \
    --model ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf \
    --mmproj ./outputs/mmproj-lfm2.5-vl-wildfire-Q8_0.gguf \
    --split test

# From HF
uv run scripts/evaluate.py \
    --hf-dataset Paulescu/wildfire-prevention \
    --backend local \
    --model Paulescu/wildfire-risk-detector \
    --quant Q8_0 \
    --split test
```

### Results

Evaluated on 172 test samples ([Paulescu/wildfire-prevention](https://huggingface.co/datasets/Paulescu/wildfire-prevention)), ground truth from `claude-opus-4-6`.

| field | claude-opus-4-6 | LFM2.5-VL-450M Q8_0 (base) | LFM2.5-VL-450M Q8_0 (fine-tuned) |
|---|---|---|---|
| valid_json | 1.00 | 1.00 | 1.00 |
| fields_present | 1.00 | 1.00 | 1.00 |
| risk_level | 0.99 | 0.08 | 0.76 |
| dry_vegetation_present | 0.99 | 0.48 | 0.83 |
| urban_interface | 0.98 | 0.25 | 0.93 |
| steep_terrain | 0.99 | 0.45 | 0.81 |
| water_body_present | 0.99 | 0.74 | 0.87 |
| image_quality_limited | 1.00 | 0.28 | 0.86 |
| **overall** | **0.99** | **0.38** | **0.84** |
| **avg latency (s)** | **2.91** | **0.72** | **0.59** |

Fine-tuning takes the model from 0.38 to 0.84 overall accuracy, more than doubling performance. The largest gains are on `risk_level` (0.08 → 0.76), `urban_interface` (0.25 → 0.93), and `image_quality_limited` (0.28 → 0.86).

## Tired of talking to bots?

I am.

This is why I spend a few hours every week talking, helping, and discussing all things AI in the Liquid AI Community. Got questions about this example, about fine-tuning compact VLMs, or about satellite ML in general? Come say hi.

99% humans. 1% bots.

See you on the other side :-)

[![Join the Liquid AI Community](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20the%20Liquid%20AI%20Community&logo=discord&logoColor=white)](https://discord.com/invite/DFU3WQeaYD)