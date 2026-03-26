"""Merge a LoRA adapter into the base model and save the result.

Usage:
    # Merge only
    uv run --group export python finetune/export.py \\
        --lora-path finetune/output/350M-lora \\
        --output-path finetune/output/350M-merged

    # Merge + convert to GGUF + push to HuggingFace
    uv run --group export python finetune/export.py \\
        --lora-path finetune/output/350M-lora \\
        --output-path finetune/output/350M-merged \\
        --push-to-hub \\
        --llama-cpp-path ~/llama.cpp \\
        --quant-type q8_0
"""

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, whoami
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--lora-path", required=True, type=Path, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output-path", required=True, type=Path, help="Path to save the merged model.")
    parser.add_argument("--push-to-hub", action="store_true", help="Convert merged model to GGUF and push to HuggingFace.")
    parser.add_argument("--llama-cpp-path", type=Path, help="Path to llama.cpp source directory (required when --push-to-hub is set).")
    parser.add_argument("--quant-type", type=str, default="q8_0", help="GGUF quantization type (default: q8_0).")
    return parser.parse_args()


def push_to_hub(output_path: Path, llama_cpp_path: Path, quant_type: str) -> None:
    # Step 1: derive model name from config.json
    config_path = output_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    name_or_path = config["_name_or_path"]
    model_name = name_or_path.split("/")[-1]

    # Step 2: convert to GGUF
    gguf_filename = f"{model_name}-{quant_type}.gguf"
    gguf_path = output_path / gguf_filename
    convert_script = Path(llama_cpp_path).expanduser() / "convert_hf_to_gguf.py"
    print(f"Converting merged model to GGUF ({quant_type})...")
    subprocess.run(
        ["python", str(convert_script), str(output_path), "--outtype", quant_type, "--outfile", str(gguf_path)],
        check=True,
    )
    print(f"GGUF saved to {gguf_path}")

    # Step 3: create HF repo if it does not exist
    hf_username = whoami()["name"]
    repo_name = f"home-assistant-{model_name}-GGUF"
    repo_id = f"{hf_username}/{repo_name}"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"HuggingFace repo: https://huggingface.co/{repo_id}")

    # Step 4: upload GGUF file
    print(f"Uploading {gguf_filename} to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_filename,
        repo_id=repo_id,
        repo_type="model",
    )

    print("\nPush complete.")
    print(f"  HF repo: https://huggingface.co/{repo_id}")
    print(f"\nUse these flags in your benchmark command:")
    print(f"  --hf-repo {repo_id} --hf-file {gguf_filename}")


def main() -> None:
    args = parse_args()

    if args.push_to_hub and args.llama_cpp_path is None:
        raise ValueError("--llama-cpp-path is required when --push-to-hub is set.")

    print(f"Loading LoRA adapter from {args.lora_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(str(args.lora_path))
    tokenizer = AutoTokenizer.from_pretrained(str(args.lora_path))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_path))
    tokenizer.save_pretrained(str(args.output_path))

    if args.push_to_hub:
        push_to_hub(args.output_path, args.llama_cpp_path, args.quant_type)

    print("Done.")


if __name__ == "__main__":
    main()
