"""Merge a LoRA adapter into the base model and save the result.

--lora-path can point to either:
- A run directory containing epoch checkpoint subdirectories: the script scans all
  epoch checkpoints, prints their eval losses, and selects the one with the lowest
  eval loss automatically.
- A specific epoch checkpoint directory: used as-is (backward compatible).

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
        --quant-type q8_0
"""

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, whoami
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def find_best_epoch_checkpoint(lora_path: Path) -> Path:
    """Return the epoch checkpoint subdirectory with the lowest eval loss.

    If lora_path itself contains adapter_config.json, it is already a specific
    checkpoint — return it unchanged (backward compatibility).
    """
    if (lora_path / "adapter_config.json").exists():
        return lora_path

    candidates = [
        d for d in sorted(lora_path.iterdir())
        if d.is_dir()
        and (d / "adapter_config.json").exists()
        and (d / "trainer_state.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoints found under {lora_path}")

    best_path, best_loss = None, float("inf")
    for ckpt in candidates:
        with open(ckpt / "trainer_state.json") as f:
            state = json.load(f)
        eval_entries = [e for e in state["log_history"] if "eval_loss" in e]
        if not eval_entries:
            continue
        loss = eval_entries[-1]["eval_loss"]
        epoch = eval_entries[-1]["epoch"]
        print(f"  {ckpt.name}: eval_loss={loss:.4f} (epoch {epoch:.0f})")
        if loss < best_loss:
            best_loss = loss
            best_path = ckpt

    print(f"\nSelected: {best_path.name}  (eval_loss={best_loss:.4f})")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--lora-path", required=True, type=Path, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output-path", required=True, type=Path, help="Path to save the merged model.")
    parser.add_argument("--push-to-hub", action="store_true", help="Convert merged model to GGUF and push to HuggingFace.")
    parser.add_argument("--quant-type", type=str, default="q8_0", help="GGUF quantization type (default: q8_0).")
    return parser.parse_args()


def push_to_hub(output_path: Path, quant_type: str, lora_path: Path | None = None) -> None:
    # Step 1: derive model name from config.json or adapter_config.json
    config_path = output_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    name_or_path = config.get("_name_or_path")
    if not name_or_path and lora_path is not None:
        adapter_config_path = lora_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            name_or_path = adapter_config.get("base_model_name_or_path")
    if not name_or_path:
        raise ValueError("Could not determine model name from config.json or adapter_config.json")
    model_name = name_or_path.split("/")[-1]

    # Step 2: convert to GGUF
    gguf_filename = f"{model_name}-{quant_type}.gguf"
    gguf_path = output_path / gguf_filename
    convert_script = Path(__file__).parent / "convert_hf_to_gguf.py"
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

    print(f"Finding best checkpoint under {args.lora_path}...")
    lora_path = find_best_epoch_checkpoint(args.lora_path)
    print(f"Loading LoRA adapter from {lora_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(str(lora_path))
    tokenizer = AutoTokenizer.from_pretrained(str(lora_path))

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_path))
    tokenizer.save_pretrained(str(args.output_path))

    if args.push_to_hub:
        push_to_hub(args.output_path, args.quant_type, lora_path=lora_path)

    print("Done.")


if __name__ == "__main__":
    main()
