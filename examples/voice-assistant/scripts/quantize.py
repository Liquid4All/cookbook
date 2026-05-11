"""Quantize a fine-tuned LFM2.5-Audio checkpoint and publish the resulting
GGUF set to a HuggingFace model repo so `scripts/eval.py` (or any other
llama-liquid-audio-server consumer) can load it.

Pipeline:
  1. snapshot_download the fine-tuned safetensors checkpoint from `--source-repo`.
  2. Clone llama.cpp PR #18641 (audio-mtmd support is WIP and not yet on main).
  3. Run `convert_hf_to_gguf.py` twice on the checkpoint: once for the LM
     backbone, once with `--mmproj` for the audio encoder + projector.
  4. If `--quant` is not F16, run llama-quantize on the LM to that target.
  5. Pull the upstream `vocoder-` and `tokenizer-` GGUFs and the `runners/`
     folder from `LiquidAI/LFM2.5-Audio-1.5B-GGUF` (these don't change with
     fine-tuning) so the published repo is self-contained.
  6. Push the four GGUFs + the runners to `--target-repo` with a model card.

Prerequisites that can't be automated: git, cmake, a C++ compiler. On macOS:
`xcode-select --install` and `brew install cmake`.

Usage:
    uv run --group finetune python scripts/quantize.py \\
        --source-repo Paulescu/LFM2.5-Audio-1.5B-OHF-Voice \\
        --target-repo Paulescu/LFM2.5-Audio-1.5B-OHF-Voice-GGUF \\
        --quant F16
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

UPSTREAM_GGUF_REPO = "LiquidAI/LFM2.5-Audio-1.5B-GGUF"
UPSTREAM_MODEL_STEM = "LFM2.5-Audio-1.5B"

# llama.cpp branch where LFM2.5-Audio multimodal support lives. Once the PR
# merges to main, this can be retargeted to a stable tag.
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_PR_BRANCH = "audio-mtmd"  # PR #18641 branch name
LLAMA_CPP_DIR = Path(__file__).parent.parent / "llama.cpp"

VALID_QUANTS = ["F16", "Q8_0", "Q4_0"]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def check_build_tools() -> None:
    for tool in ("git", "cmake", "c++"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            print(f"Missing required tool: {tool}", file=sys.stderr)
            if tool in ("cmake", "c++"):
                print(
                    "  On macOS: run `xcode-select --install` and `brew install cmake`",
                    file=sys.stderr,
                )
            sys.exit(1)


def setup_llama_cpp() -> Path:
    """Clone llama.cpp at the audio-support PR branch and build llama-quantize.

    Returns the path to the convert_hf_to_gguf.py script.
    """
    if not LLAMA_CPP_DIR.exists():
        print(f"Cloning {LLAMA_CPP_REPO} into {LLAMA_CPP_DIR} ...", flush=True)
        run(
            [
                "git",
                "clone",
                "--depth=1",
                "--branch",
                LLAMA_CPP_PR_BRANCH,
                LLAMA_CPP_REPO,
                str(LLAMA_CPP_DIR),
            ]
        )

    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        print("Building llama-quantize ...", flush=True)
        run(["cmake", "-B", "build"], cwd=LLAMA_CPP_DIR)
        run(
            ["cmake", "--build", "build", "--config", "Release", "-t", "llama-quantize"],
            cwd=LLAMA_CPP_DIR,
        )
    return LLAMA_CPP_DIR / "convert_hf_to_gguf.py"


def convert_lm(convert_script: Path, checkpoint: Path, out: Path) -> None:
    print(f"Converting LM backbone to F16 GGUF: {out.name}", flush=True)
    run(
        [
            sys.executable,
            str(convert_script),
            str(checkpoint),
            "--outtype",
            "f16",
            "--outfile",
            str(out),
        ]
    )


def convert_mmproj(convert_script: Path, checkpoint: Path, out: Path) -> None:
    print(f"Converting mmproj to F16 GGUF: {out.name}", flush=True)
    run(
        [
            sys.executable,
            str(convert_script),
            str(checkpoint),
            "--mmproj",
            "--outfile",
            str(out),
        ]
    )


def quantize_lm(f16_path: Path, out: Path, quant: str) -> None:
    if quant == "F16":
        if f16_path != out:
            shutil.move(str(f16_path), str(out))
        return
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
    print(f"Quantizing LM to {quant}: {out.name}", flush=True)
    run([str(quantize_bin), str(f16_path), str(out), quant])


def copy_unchanged_artifacts(
    target_dir: Path,
    target_stem: str,
    quant: str,
) -> tuple[Path, Path]:
    """Download upstream vocoder + tokenizer GGUFs (these don't change with
    fine-tuning), rename to the target stem, write into target_dir. Returns
    (vocoder_path, tokenizer_path).
    """
    print(
        f"Downloading upstream vocoder + tokenizer ({quant}) from {UPSTREAM_GGUF_REPO} ...",
        flush=True,
    )
    upstream_dir = Path(
        snapshot_download(
            repo_id=UPSTREAM_GGUF_REPO,
            allow_patterns=[
                f"vocoder-{UPSTREAM_MODEL_STEM}-{quant}.gguf",
                f"tokenizer-{UPSTREAM_MODEL_STEM}-{quant}.gguf",
            ],
        )
    )
    vocoder_src = upstream_dir / f"vocoder-{UPSTREAM_MODEL_STEM}-{quant}.gguf"
    tokenizer_src = upstream_dir / f"tokenizer-{UPSTREAM_MODEL_STEM}-{quant}.gguf"
    vocoder_dst = target_dir / f"vocoder-{target_stem}-{quant}.gguf"
    tokenizer_dst = target_dir / f"tokenizer-{target_stem}-{quant}.gguf"
    shutil.copy2(vocoder_src, vocoder_dst)
    shutil.copy2(tokenizer_src, tokenizer_dst)
    return vocoder_dst, tokenizer_dst


def download_upstream_runners(target_dir: Path) -> Path:
    """Pull the entire upstream runners/ directory; we re-publish it as-is so
    that consumers don't need to dual-source GGUFs and binaries.
    """
    print(f"Downloading upstream runners/ from {UPSTREAM_GGUF_REPO} ...", flush=True)
    upstream_dir = Path(
        snapshot_download(repo_id=UPSTREAM_GGUF_REPO, allow_patterns=["runners/*"])
    )
    runners_src = upstream_dir / "runners"
    runners_dst = target_dir / "runners"
    if runners_dst.exists():
        shutil.rmtree(runners_dst)
    shutil.copytree(runners_src, runners_dst)
    return runners_dst


MODEL_CARD = """\
---
base_model: LiquidAI/LFM2.5-Audio-1.5B
language:
- en
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/blob/main/LICENSE
tags:
- liquid
- lfm2.5
- edge
- llama.cpp
- audio
- speech
- gguf
- home-assistant
- function-calling
---

# {target_stem}

Fine-tuned from [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) \
on [Paulescu/OHF-Voice-audio-20260504](https://huggingface.co/datasets/Paulescu/OHF-Voice-audio-20260504) \
to map spoken Home Assistant voice commands directly to function calls. \
Part of the [Liquid Cookbook voice-assistant example](https://github.com/Liquid4All/cookbook/tree/main/examples/voice-assistant).

Output format (no system prompt; the function set is baked into the weights):

```
HassStartTimer|$minutes=5|$name=oven
HassLightSet|$area=bedroom|$brightness=70
HassGetCurrentTime
```

## Files

llama-liquid-audio-server requires four GGUFs to run inference:

| file | description | source |
|---|---|---|
| `{target_stem}-{quant}.gguf` | Language model backbone | fine-tuned in this repo |
| `mmproj-{target_stem}-{quant}.gguf` | Audio encoder + projector | fine-tuned in this repo |
| `vocoder-{target_stem}-{quant}.gguf` | Audio decoder (unused for function-calling) | copied from upstream |
| `tokenizer-{target_stem}-{quant}.gguf` | Tokenizer / speaker file | copied from upstream |

The `runners/` folder bundles `llama-liquid-audio-server` and `llama-liquid-audio-cli` binaries \
for macos-arm64, ubuntu-x64, ubuntu-arm64, and android-arm64, built from \
[llama.cpp PR #18641](https://github.com/ggml-org/llama.cpp/pull/18641).

## Reproduce the eval

```bash
git clone https://github.com/Liquid4All/cookbook
cd cookbook/examples/voice-assistant
uv sync
# point configs/finetuned.yaml at this repo and run:
uv run python scripts/eval.py --config configs/finetuned.yaml
```
"""


def make_model_card(target_stem: str, quant: str) -> str:
    return MODEL_CARD.format(target_stem=target_stem, quant=quant)


def push_to_hub(target_dir: Path, target_repo: str, target_stem: str, quant: str, private: bool) -> None:
    api = HfApi()
    print(f"Creating repo: {target_repo} ...", flush=True)
    api.create_repo(repo_id=target_repo, repo_type="model", private=private, exist_ok=True)
    print(f"Uploading folder {target_dir} ...", flush=True)
    api.upload_folder(
        folder_path=str(target_dir),
        repo_id=target_repo,
        repo_type="model",
    )
    print("Uploading model card ...", flush=True)
    api.upload_file(
        path_or_fileobj=make_model_card(target_stem, quant).encode(),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="model",
    )
    print(f"Done. Model at https://huggingface.co/{target_repo}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        required=True,
        metavar="REPO",
        help="HF repo with the fine-tuned safetensors checkpoint.",
    )
    parser.add_argument(
        "--target-repo",
        required=True,
        metavar="REPO",
        help="HF repo to publish the GGUF set to (will be created if needed).",
    )
    parser.add_argument(
        "--quant",
        default="F16",
        choices=VALID_QUANTS,
        help="Quantization target for the LM backbone. mmproj is always F16. (default: F16)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gguf"),
        help="Local staging directory (default: outputs/gguf).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repo as private.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Stop after producing local files; skip the HF upload.",
    )
    args = parser.parse_args()

    check_build_tools()
    convert_script = setup_llama_cpp()

    target_stem = args.target_repo.split("/")[-1].removesuffix("-GGUF")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading source checkpoint: {args.source_repo} ...", flush=True)
    src_dir = Path(snapshot_download(repo_id=args.source_repo))

    lm_f16 = args.output_dir / f"{target_stem}-F16.gguf"
    convert_lm(convert_script, src_dir, lm_f16)

    mmproj_out = args.output_dir / f"mmproj-{target_stem}-{args.quant}.gguf"
    convert_mmproj(convert_script, src_dir, mmproj_out)

    lm_out = args.output_dir / f"{target_stem}-{args.quant}.gguf"
    quantize_lm(lm_f16, lm_out, args.quant)

    vocoder, tokenizer = copy_unchanged_artifacts(args.output_dir, target_stem, args.quant)
    runners = download_upstream_runners(args.output_dir)

    print()
    print("Local artifacts:")
    print(f"  LM      : {lm_out}")
    print(f"  mmproj  : {mmproj_out}")
    print(f"  vocoder : {vocoder}")
    print(f"  tokenizer: {tokenizer}")
    print(f"  runners : {runners}")

    if args.skip_push:
        print("\n[--skip-push] Not uploading.")
        return

    push_to_hub(args.output_dir, args.target_repo, target_stem, args.quant, args.private)


if __name__ == "__main__":
    main()
