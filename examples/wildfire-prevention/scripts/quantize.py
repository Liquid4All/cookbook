"""Convert a fine-tuned LFM2.5-VL-450M checkpoint to quantized GGUF.

Produces two artifacts required by llama-server for VLM inference:
  1. Quantized backbone GGUF  (language model)
  2. mmproj GGUF              (vision tower + multimodal projector)

Handles the full setup automatically: clones llama.cpp if not present and
builds the llama-quantize binary.

Prerequisites (cannot be automated):
  - git
  - cmake
  - A C++ compiler (macOS: run `xcode-select --install` if missing)

Usage:
    uv run scripts/quantize.py \\
        --checkpoint ./leap-finetune/outputs/<run>/<checkpoint> \\
        --output ./outputs/lfm2.5-vl-wildfire-Q8_0.gguf

    # Different quantization level
    uv run scripts/quantize.py \\
        --checkpoint ./leap-finetune/outputs/<run>/<checkpoint> \\
        --output ./outputs/lfm2.5-vl-wildfire-Q4_K_M.gguf \\
        --quant Q4_K_M
"""

import argparse
import subprocess
import sys
from pathlib import Path


LLAMA_CPP_DIR = Path(__file__).parent.parent / "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"
VALID_QUANTS = ["Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def check_build_tools() -> None:
    for tool in ("git", "cmake", "c++"):
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            print(f"Missing required tool: {tool}")
            if tool in ("cmake", "c++"):
                print("  On macOS: run `xcode-select --install` and `brew install cmake`")
            sys.exit(1)


def setup_llama_cpp() -> None:
    if not LLAMA_CPP_DIR.exists():
        print(f"Cloning llama.cpp into {LLAMA_CPP_DIR} ...")
        run(["git", "clone", "--depth=1", LLAMA_CPP_REPO, str(LLAMA_CPP_DIR)])

    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("Building llama-quantize ...")
        run(["cmake", "-B", "build"], cwd=LLAMA_CPP_DIR)
        run(["cmake", "--build", "build", "--config", "Release", "-t", "llama-quantize"], cwd=LLAMA_CPP_DIR)

    # The project venv already provides all packages needed by convert_hf_to_gguf.py
    # (gguf, transformers, torch, numpy, sentencepiece, protobuf), so no separate
    # install step is required.


def convert_to_f16(checkpoint: Path, f16_output: Path) -> None:
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    print(f"Converting backbone to F16 GGUF: {f16_output} ...")
    run([
        sys.executable,
        str(convert_script),
        str(checkpoint),
        "--outtype", "f16",
        "--outfile", str(f16_output),
    ])


def convert_to_mmproj(checkpoint: Path, mmproj_output: Path) -> None:
    """Extract vision tower + projector into a mmproj GGUF using --mmproj mode."""
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    print(f"Converting vision components to mmproj GGUF: {mmproj_output} ...")
    run([
        sys.executable,
        str(convert_script),
        str(checkpoint),
        "--mmproj",
        "--outfile", str(mmproj_output),
    ])


def quantize(f16_path: Path, output: Path, quant: str) -> None:
    if quant == "F16":
        f16_path.rename(output)
        return
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    print(f"Quantizing to {quant}: {output} ...")
    run([str(quantize_bin), str(f16_path), str(output), quant])


def mmproj_path(output: Path) -> Path:
    """Derive mmproj output path from the backbone output path."""
    return output.parent / f"mmproj-{output.stem.replace('-' + output.stem.split('-')[-1], '')}.gguf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned LFM2.5-VL-450M checkpoint to quantized GGUF."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to the HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for the backbone GGUF (e.g. ./outputs/model-Q8_0.gguf). "
             "The mmproj is written alongside it as mmproj-<stem>.gguf.",
    )
    parser.add_argument(
        "--quant",
        default="Q8_0",
        choices=VALID_QUANTS,
        help="Quantization type for the backbone (default: Q8_0). The mmproj is always F16.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output)

    if not checkpoint.is_dir():
        print(f"Checkpoint directory not found: {checkpoint}")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    f16_path = output.parent / (output.stem + "-F16.gguf")
    mmproj = output.parent / f"mmproj-{output.stem}.gguf"

    check_build_tools()
    setup_llama_cpp()

    # Backbone: HF checkpoint → F16 GGUF → quantized GGUF
    convert_to_f16(checkpoint, f16_path)
    try:
        quantize(f16_path, output, args.quant)
    finally:
        if args.quant != "F16" and f16_path.exists():
            f16_path.unlink()

    # mmproj: vision tower + projector → F16 GGUF (no further quantization needed)
    convert_to_mmproj(checkpoint, mmproj)

    print()
    print(f"Done.")
    print(f"  Backbone : {output}")
    print(f"  mmproj   : {mmproj}")
    print()
    print("To evaluate with the local backend:")
    print(f"  uv run scripts/evaluate.py --backend local --model {output} --mmproj {mmproj} --split test")
    print()
    print("To run the prediction loop:")
    print(f"  uv run scripts/predict.py --backend local --model {output} --mmproj {mmproj}")


if __name__ == "__main__":
    main()
