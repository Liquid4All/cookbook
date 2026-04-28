"""Push a fine-tuned GGUF model pair to a HuggingFace model repository.

Uploads the backbone GGUF and mmproj GGUF produced by quantize.py, along with
a model card, so end users can reproduce eval results without fine-tuning.

Usage:
    uv run scripts/push_gguf_to_hf.py \\
        --backbone ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf \\
        --mmproj   ./outputs/mmproj-lfm2.5-vl-wildfire-Q8_0.gguf \\
        --repo     Paulescu/LFM2.5-VL-450M-wildfire-GGUF

    # Private repo
    uv run scripts/push_gguf_to_hf.py \\
        --backbone ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf \\
        --mmproj   ./outputs/mmproj-lfm2.5-vl-wildfire-Q8_0.gguf \\
        --repo     Paulescu/LFM2.5-VL-450M-wildfire-GGUF \\
        --private
"""

import argparse
import sys
from pathlib import Path


MODEL_CARD_TEMPLATE = """\
---
base_model: LiquidAI/LFM2.5-VL-450M
language:
- en
license: other
tags:
- gguf
- vlm
- wildfire
- satellite
- sentinel-2
---

# LFM2.5-VL-450M wildfire risk (GGUF)

Fine-tuned from [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) \
on Sentinel-2 satellite imagery to assess wildfire risk. \
Part of the [Liquid Cookbook wildfire-prevention example](https://github.com/Liquid4All/cookbook/tree/main/examples/wildfire-prevention).

Given an RGB and SWIR Sentinel-2 image pair, the model outputs a structured JSON risk assessment:

```json
{{
  "risk_level": "low | medium | high",
  "dry_vegetation_present": true,
  "urban_interface": false,
  "steep_terrain": true,
  "water_body_present": false,
  "image_quality_limited": false
}}
```

## Eval results

Evaluated on 172 test samples from \
[Paulescu/wildfire-prevention](https://huggingface.co/datasets/Paulescu/wildfire-prevention), \
ground truth from `claude-opus-4-6`.

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

## Files

Running inference with a VLM in llama.cpp requires two GGUF files:

| file | description |
|---|---|
| `{backbone_name}` | Language model backbone (Q8_0) |
| `{mmproj_name}` | Vision tower + multimodal projector (F16) |

## Usage

### llama-server

```bash
llama-server \\
    -m {backbone_name} \\
    --mmproj {mmproj_name} \\
    --jinja --port 8080
```

### Reproduce eval results

Clone the [Liquid Cookbook](https://github.com/Liquid4All/cookbook), then:

```bash
cd examples/wildfire-prevention
uv sync
uv run scripts/evaluate.py \\
    --hf-dataset Paulescu/wildfire-prevention \\
    --backend local \\
    --model {backbone_name} \\
    --mmproj {mmproj_name} \\
    --split test
```
"""


def make_model_card(backbone_name: str, mmproj_name: str) -> str:
    return MODEL_CARD_TEMPLATE.format(
        backbone_name=backbone_name,
        mmproj_name=mmproj_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push a fine-tuned GGUF model pair to HuggingFace."
    )
    parser.add_argument(
        "--backbone",
        required=True,
        metavar="PATH",
        help="Path to the backbone GGUF file.",
    )
    parser.add_argument(
        "--mmproj",
        required=True,
        metavar="PATH",
        help="Path to the mmproj GGUF file.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        metavar="REPO_ID",
        help="HuggingFace model repo to create/push to (e.g. Paulescu/LFM2.5-VL-450M-wildfire-GGUF).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public).",
    )
    args = parser.parse_args()

    backbone = Path(args.backbone)
    mmproj = Path(args.mmproj)

    for path in (backbone, mmproj):
        if not path.is_file():
            print(f"File not found: {path}")
            sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Creating repo: {args.repo} ...")
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading backbone: {backbone.name} ...")
    api.upload_file(
        path_or_fileobj=str(backbone),
        path_in_repo=backbone.name,
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"Uploading mmproj: {mmproj.name} ...")
    api.upload_file(
        path_or_fileobj=str(mmproj),
        path_in_repo=mmproj.name,
        repo_id=args.repo,
        repo_type="model",
    )

    print("Uploading model card ...")
    card = make_model_card(backbone.name, mmproj.name)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )

    print()
    print(f"Done. Model at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
