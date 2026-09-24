"""Merge LoRA weights from a fine-tuned checkpoint back into the base model,
producing a clean model.safetensors with standard Linear layers that
scripts/quantize.py can consume.

Usage:
    uv run --group finetune python scripts/merge_lora_checkpoint.py \\
        --checkpoint outputs/ohf_voice/ohf-voice-20260513-130007/final/model.safetensors \\
        --output outputs/checkpoint
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

from liquid_audio import LFM2AudioModel
from liquid_audio.moshi.modules.lora import (
    replace_all_linear_with_lora,
    replace_lora_with_linear,
)

load_dotenv()

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B"


def merge(
    checkpoint_path: Path,
    output_dir: Path,
    lora_rank: int,
    lora_scaling: float,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model ({MODEL_ID}) on {device} ...", flush=True)
    model = LFM2AudioModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)

    print(f"Applying LoRA (rank={lora_rank}, scaling={lora_scaling}) ...", flush=True)
    replace_all_linear_with_lora(model.lfm, rank=lora_rank, scaling=lora_scaling)

    print(f"Loading fine-tuned weights from {checkpoint_path} ...", flush=True)
    state_dict = {}
    from safetensors import safe_open
    with safe_open(str(checkpoint_path), framework="pt", device=str(device)) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
        for k in missing:
            print(f"  {k}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"  {k}")
    print("Done loading weights.", flush=True)

    print("Merging LoRA weights back into Linear layers ...", flush=True)
    replace_lora_with_linear(model.lfm)
    print("LoRA merge complete.", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.safetensors"
    print(f"Saving merged checkpoint to {output_path} ...", flush=True)

    model = model.cpu()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    merged_sd = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    del model
    from safetensors.torch import save_file
    save_file(merged_sd, str(output_path))
    print(f"Saved merged checkpoint ({output_path.stat().st_size / 1024**3:.2f} GB)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the LoRA fine-tuned model.safetensors",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/checkpoint"),
        help="Output directory for the merged checkpoint (default: outputs/checkpoint)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank used during training (default: 16)",
    )
    parser.add_argument(
        "--lora-scaling",
        type=float,
        default=2.0,
        help="LoRA scaling used during training (default: 2.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_scaling=args.lora_scaling,
    )
