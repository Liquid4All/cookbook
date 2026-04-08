# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This example fine-tunes [LiquidAI/lfm2.5-VL-450M](https://huggingface.co/LiquidAI/lfm2.5-VL-450M) on satellite imagery using the [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench) dataset. It is part of the `leap-finetune` monorepo and lives under `cookbook/examples/satellite-vlm/`.

Three tasks are supported, each auto-detected from `[tag]` prefixes in VRSBench's ShareGPT-style training data:
- **VQA** (`[vqa]`): visual question answering, 85K train samples
- **Grounding** (`[refer]`): object detection with normalized bounding boxes, 36K train samples
- **Captioning** (`[caption]`): image description, 20K train samples

## Setup

Install dependencies from the leap-finetune root:

```bash
cd /path/to/leap-finetune
uv sync
```

## Common Commands

**Prepare data** (run from `cookbook/examples/satellite-vlm/`):

```bash
uv run python prepare_vrsbench.py --task vqa           # single task (~12 GB download)
uv run python prepare_vrsbench.py --task all           # all tasks combined (multitask)
uv run python prepare_vrsbench.py --task vqa --limit 500  # quick test with 500 samples
uv run python prepare_vrsbench.py --task vqa --skip-download  # skip download if data exists
```

Output: `./data/vrsbench_{task}_train.jsonl` and `./data/vrsbench_{task}_eval.jsonl`

**Train** (run from leap-finetune root):

```bash
uv run leap-finetune cookbook/examples/satellite-vlm/configs/vrsbench_multitask.yaml
```

**Full standalone evaluation** (no training, requires a checkpoint path in the config):

```bash
uv run leap-finetune cookbook/examples/satellite-vlm/configs/vrsbench_full_eval.yaml
```

**Tests and linting** (run from leap-finetune root):

```bash
uv run pytest                              # all tests
uv run pytest tests/test_config_parser.py  # single test file
uv run pytest -m vlm                       # VLM end-to-end tests only
uv run pre-commit run --all-files          # ruff + prettier
```

## Architecture

### Data Preparation (`prepare_vrsbench.py`)

Single script that downloads, extracts, and converts VRSBench to the leap-finetune VLM SFT JSONL format:

```json
{
  "messages": [
    {"role": "user", "content": [{"type": "image", "image": "filename"}, {"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]
}
```

Key functions:
- `download_vrsbench()`: downloads annotation JSONs and image ZIPs from HuggingFace, flattens images into `./data/vrsbench/images/`
- `convert_train()`: routes each ShareGPT conversation to the correct task by `[tag]`
- `convert_eval_*()`: separate converters per task for flat eval JSON format
- `parse_bbox_tokens()`: converts VRSBench's `{<25><40><33><60>}` token format to normalized `[0.25, 0.40, 0.33, 0.60]`; skips invalid boxes where `x2 <= x1` or `y2 <= y1`

When `--task all`, all three tasks are combined and shuffled (seed 42) into a single `vrsbench_multitask_train.jsonl`; eval files are always written separately per task.

### Training Configs (`configs/`)

YAML configs consumed by the leap-finetune CLI:

- `vrsbench_multitask.yaml`: main training config; extends `DEFAULT_VLM_SFT` and `DEFAULT_VLM_LORA` from leap-finetune defaults; LoRA disabled by default; evaluates 4 benchmarks every 1000 steps with 500-sample limits for speed
- `vrsbench_full_eval.yaml`: evaluation-only config; requires a fine-tuned checkpoint path; no sample limits; runs all 5 benchmarks (adds ROUGE-L for captioning)

### Grounding Data Format

Bounding boxes use JSON with 0-1 normalized coordinates, matching LFM VLM pretraining format:

```
User:      Inspect the image and detect the large white ship.
           Provide result as a valid JSON:
           [{"label": str, "bbox": [x1,y1,x2,y2]}, ...].
           Coordinates must be normalized to 0-1.
Assistant: [{"label": "ship", "bbox": [0.37, 0.00, 0.80, 0.99]}]
```

### Evaluation Metrics

| Task | Metric | Notes |
|------|--------|-------|
| VQA | `short_answer` | Case-insensitive substring match |
| Grounding | `grounding_iou` | IoU@0.5 (default) and IoU@0.25 |
| Captioning | `bleu` / `cider` / `rouge_l` | BLEU during training, all three for full eval |

### Experiment Tracking

WandB tracking is disabled by default. To enable, uncomment `tracker: "wandb"` in the YAML config.
