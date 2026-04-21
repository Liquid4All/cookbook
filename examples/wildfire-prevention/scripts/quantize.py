"""Merge a LoRA checkpoint into the base model and export to quantized GGUF.

Step 6 in the fine-tuning pipeline. Run after retrieving the checkpoint from
Modal (step 5) and before evaluating with --backend local (step 7).

No external tools required. Conversion uses the vendored
finetune/convert_hf_to_gguf.py (a copy of llama.cpp's converter). The gguf
Python package handles all quantization inline, so no llama-quantize binary
is needed.

Note: the vendored converter was validated against text-only LFM models
(home-assistant example). LFM2.5-VL-450M is a VLM, so verify the output
GGUF works with llama-server before deploying.

Usage:
    # Merge LoRA adapter, convert to GGUF, and quantize to Q8_0
    uv run scripts/quantize.py \\
        --checkpoint ./outputs/<run-name> \\
        --output ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf

    # Different quantization level
    uv run scripts/quantize.py \\
        --checkpoint ./outputs/<run-name> \\
        --output ./outputs/lfm2.5-vl-wildfire-Q4_K_M.gguf \\
        --quant Q4_K_M

    # Skip LoRA merge if --checkpoint is already a full merged model
    uv run scripts/quantize.py \\
        --checkpoint ./outputs/<run-name> \\
        --output ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf \\
        --skip-merge
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
VALID_QUANTS = ["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

CONVERT_SCRIPT = Path(__file__).parent.parent / "finetune" / "convert_hf_to_gguf.py"


def merge_lora(checkpoint: Path, base_model: str, output_dir: Path) -> None:
    """Merge a LoRA adapter into the base model and save the merged weights."""
    try:
        import torch
        from peft import PeftModel  # type: ignore[import-untyped]
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-untyped]
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: uv sync")
        sys.exit(1)

    print(f"Loading base model: {base_model} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter from: {checkpoint} ...")
    model = PeftModel.from_pretrained(model, str(checkpoint))

    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir} ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print("Merge complete.")


def convert_to_gguf(model_dir: Path, output: Path, quant: str) -> None:
    """Convert a HuggingFace model directory to quantized GGUF in one pass."""
    if not CONVERT_SCRIPT.exists():
        print(f"Converter not found at: {CONVERT_SCRIPT}")
        sys.exit(1)

    # NO_LOCAL_GGUF forces the converter to use the installed gguf package
    # instead of looking for a local gguf-py directory.
    env = {**os.environ, "NO_LOCAL_GGUF": "1"}

    print(f"Converting to GGUF ({quant}): {output} ...")
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERT_SCRIPT),
            str(model_dir),
            "--outtype", quant.lower(),
            "--outfile", str(output),
        ],
        env=env,
    )
    if result.returncode != 0:
        print("GGUF conversion failed.")
        sys.exit(result.returncode)
    print(f"Conversion complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and export to quantized GGUF."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the LoRA checkpoint directory (from: modal volume get wildfire-prevention /outputs/<run>).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for the GGUF file (e.g. ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf).",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        metavar="REPO",
        help=f"Base model HuggingFace repo ID (default: {BASE_MODEL}).",
    )
    parser.add_argument(
        "--quant",
        default="Q8_0",
        choices=VALID_QUANTS,
        help="Quantization type (default: Q8_0). Q8_0 preserves accuracy; Q4_K_M halves the size.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip the LoRA merge step if --checkpoint already contains a full merged model.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)

    if not checkpoint.is_dir():
        print(f"Checkpoint directory not found: {checkpoint}")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wildfire-quantize-") as tmp:
        if args.skip_merge:
            model_dir = checkpoint
        else:
            merged_dir = Path(tmp) / "merged"
            merge_lora(checkpoint, args.base_model, merged_dir)
            model_dir = merged_dir

        convert_to_gguf(model_dir, output, args.quant)

    print()
    print(f"Done. Quantized model: {output}")
    print()
    print("To evaluate with the local backend:")
    print(f"  uv run scripts/evaluate.py --backend local --model {output} --split test")
    print()
    print("To run the prediction loop:")
    print(f"  uv run scripts/predict.py --backend local --model {output}")


if __name__ == "__main__":
    main()
