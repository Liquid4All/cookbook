"""Merge a LoRA adapter into the base model and save the result.

Usage:
    uv run --group export python finetune/export.py \\
        --lora-path finetune/output/350M-lora \\
        --output-path finetune/output/350M-merged
"""

import argparse
from pathlib import Path

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--lora-path", required=True, type=Path, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output-path", required=True, type=Path, help="Path to save the merged model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading LoRA adapter from {args.lora_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(str(args.lora_path))
    tokenizer = AutoTokenizer.from_pretrained(str(args.lora_path))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_path))
    tokenizer.save_pretrained(str(args.output_path))

    print("Done.")


if __name__ == "__main__":
    main()
