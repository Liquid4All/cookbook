"""Quantize a fine-tuned LFM2.5-Audio checkpoint and publish the resulting
GGUF set to a HuggingFace model repo so `scripts/eval.py` (or any other
llama-liquid-audio-server consumer) can load it.

Pipeline:
  1. Materialise the source HF-shaped checkpoint directory:
     - `--source-repo` snapshot-downloads a full fine-tuned HF repo, OR
     - `--source-checkpoint` overlays a local fine-tuned `model.safetensors`
       on top of `--base-repo`'s configs (Trainer only saves weights, so we
       supply the surrounding config.json / tokenizer files from upstream).
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

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()

UPSTREAM_GGUF_REPO = "LiquidAI/LFM2.5-Audio-1.5B-GGUF"
UPSTREAM_MODEL_STEM = "LFM2.5-Audio-1.5B"
UPSTREAM_BASE_REPO = "LiquidAI/LFM2.5-Audio-1.5B"  # safetensors repo with configs we overlay

# llama.cpp PR where LFM2.5-Audio multimodal support lives. Once the PR
# merges to main, this can be retargeted to a stable tag.
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_PR_NUMBER = 18641
LLAMA_CPP_PR_LOCAL_BRANCH = "audio-mtmd"  # local checkout name for the PR
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
        print(f"Cloning {LLAMA_CPP_REPO} (depth=1, main only) ...", flush=True)
        run(["git", "clone", "--depth=1", LLAMA_CPP_REPO, str(LLAMA_CPP_DIR)])
        # The PR head ref is the universal way to grab a GitHub PR's tip
        # without needing to know which fork / branch it came from.
        pr_refspec = f"pull/{LLAMA_CPP_PR_NUMBER}/head:{LLAMA_CPP_PR_LOCAL_BRANCH}"
        print(f"Fetching PR #{LLAMA_CPP_PR_NUMBER} ({pr_refspec}) ...", flush=True)
        run(["git", "fetch", "--depth=1", "origin", pr_refspec], cwd=LLAMA_CPP_DIR)
        run(["git", "checkout", LLAMA_CPP_PR_LOCAL_BRANCH], cwd=LLAMA_CPP_DIR)

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
    """Pull the upstream runner zips (one per platform) so the published repo
    is self-contained: consumers can resolve their platform's binaries from
    the same place as the GGUFs.

    We copy only the `*.zip` files, NOT the entire `runners/` snapshot dir.
    `scripts/eval.py` extracts the zip in-place on first run, so a cached
    snapshot of this repo may also contain the extracted platform subfolders
    (50+ files of dylibs, headers, licenses). Republishing those would bloat
    the target repo without adding anything consumers can't extract themselves.
    """
    print(f"Downloading upstream runner zips from {UPSTREAM_GGUF_REPO} ...", flush=True)
    upstream_dir = Path(
        snapshot_download(repo_id=UPSTREAM_GGUF_REPO, allow_patterns=["runners/*.zip"])
    )
    runners_src = upstream_dir / "runners"
    runners_dst = target_dir / "runners"
    if runners_dst.exists():
        shutil.rmtree(runners_dst)
    runners_dst.mkdir(parents=True)
    for zip_file in runners_src.glob("*.zip"):
        shutil.copy2(zip_file, runners_dst / zip_file.name)
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
# point configs/finetuned-{quant}.yaml at this repo and run:
uv run python scripts/eval.py --config configs/finetuned-q8.yaml
```
"""


def make_model_card(target_stem: str, quant: str) -> str:
    return MODEL_CARD.format(target_stem=target_stem, quant=quant)


def push_to_hub(target_dir: Path, target_repo: str, target_stem: str, quant: str, private: bool) -> None:
    api = HfApi()
    print(f"Creating repo: {target_repo} ...", flush=True)
    api.create_repo(repo_id=target_repo, repo_type="model", private=private, exist_ok=True)
    print(f"Uploading folder {target_dir} (GGUFs + runner zips only) ...", flush=True)
    # Defensive enumeration: even though we control which files land under
    # target_dir, allow_patterns guarantees we never accidentally republish
    # staging dirs (e.g. _staging/merged/) or unzipped runner contents that
    # a future change might leak into the output folder.
    api.upload_folder(
        folder_path=str(target_dir),
        repo_id=target_repo,
        repo_type="model",
        allow_patterns=["*.gguf", "runners/*.zip"],
    )
    print("Uploading model card ...", flush=True)
    api.upload_file(
        path_or_fileobj=make_model_card(target_stem, quant).encode(),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="model",
    )
    print(f"Done. Model at https://huggingface.co/{target_repo}", flush=True)


def overlay_checkpoint_on_base(
    checkpoint_path: Path, base_repo: str, output_dir: Path
) -> Path:
    """Build a local HF-shaped directory by overlaying a single fine-tuned
    `model.safetensors` on top of `base_repo`'s configs.

    The Trainer in liquid-audio v1.2.0 saves only `model.safetensors` (via
    `accelerator.save_state` and a final `accelerator.save_model`), without
    the surrounding config.json / tokenizer files that `convert_hf_to_gguf.py`
    needs. We snapshot the upstream base, copy its non-weight files into a
    staging dir, then drop our checkpoint on top.

    The merged dir lives under `output_dir/_staging/` (rather than directly in
    `output_dir`) so the eventual `upload_folder(folder_path=output_dir,
    allow_patterns=...)` call leaves it on local disk. Republishing a 3 GB
    `model.safetensors` we already encode as the LM GGUF would just bloat
    the target repo.

    Returns the path to the merged directory.
    """
    print(f"Snapshotting base model configs from {base_repo} ...", flush=True)
    # `*.jinja` is critical: LFM2.5-Audio's chat template lives in
    # chat_template.jinja, and convert_hf_to_gguf.py embeds it as the
    # tokenizer.chat_template GGUF key. Without it the server can't format
    # incoming messages and rejects audio inputs with "audio input is not
    # supported", even though the mmproj is loaded correctly.
    base_dir = Path(
        snapshot_download(
            repo_id=base_repo,
            allow_patterns=["*.json", "*.txt", "*.jinja", "tokenizer*", "*.model", "*.py"],
        )
    )
    merged = output_dir / "_staging" / "merged"
    if merged.exists():
        shutil.rmtree(merged)
    merged.mkdir(parents=True)
    for f in base_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, merged / f.name)
    print(f"Copying fine-tuned weights from {checkpoint_path} ...", flush=True)
    shutil.copy2(checkpoint_path, merged / "model.safetensors")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--source-repo",
        metavar="REPO",
        help="HF repo with the fine-tuned safetensors checkpoint (full HF directory).",
    )
    src_group.add_argument(
        "--source-checkpoint",
        type=Path,
        metavar="PATH",
        help=(
            "Local fine-tuned model.safetensors (weights only). The script "
            "overlays it on --base-repo configs before conversion."
        ),
    )
    parser.add_argument(
        "--base-repo",
        default=UPSTREAM_BASE_REPO,
        metavar="REPO",
        help=(
            "Base model whose configs to overlay our weights on. Only used "
            f"with --source-checkpoint. (default: {UPSTREAM_BASE_REPO})"
        ),
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

    if args.source_repo:
        print(f"Downloading source checkpoint: {args.source_repo} ...", flush=True)
        src_dir = Path(snapshot_download(repo_id=args.source_repo))
    else:
        if not args.source_checkpoint.is_file():
            raise FileNotFoundError(
                f"--source-checkpoint not found or not a file: {args.source_checkpoint}"
            )
        src_dir = overlay_checkpoint_on_base(
            checkpoint_path=args.source_checkpoint,
            base_repo=args.base_repo,
            output_dir=args.output_dir,
        )

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
