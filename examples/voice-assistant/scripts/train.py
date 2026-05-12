"""Fine-tune `LiquidAI/LFM2.5-Audio-1.5B` on the preprocessed OHF-Voice train
split, optionally on a Modal A100-80GB.

Expects `scripts/preprocess_ohf_voice.py` to have produced an on-disk dataset
under `--data` (locally) or `--modal-data-path` (on Modal's ohf-voice-data
volume). A `--val-split-ratio` slice is carved off the train split for the
training-loop validation metric. The held-out evaluation **Test set** lives in
the `test` split of `Paulescu/OHF-Voice-audio-20260504` and is never seen here.

Loss and val-loss stream to stdout (visible in Modal logs). W&B integration is
deferred until public liquid-audio adds tracker args to its Trainer; see
`.scratch/wandb-integration/issues/01-tracker-args.md`.

Run locally on a single GPU:
    uv run --group finetune python scripts/train.py --data data/ohf_voice/train

Run on Modal with an A100-80GB:
    HF_TOKEN=hf_... uv run --group finetune python scripts/train.py --modal

Vendored from liquid-audio-staging on 2026-05-11.
  source : examples/train.py
  branch : examples/audio-to-function-calling
  commit : 376b06a10386b0887b320122b13d2d99378c19ea

Adaptations from upstream:
  - argparse defaults retuned to match the OHF-Voice recommended config
    documented in liquid-audio-staging/FINETUNING_EXAMPLE.md (batch 32,
    max-steps 10000, lr 5e-5, context-length 512, val-split 0.05). Upstream
    defaults were generic Jenny-TTS values (batch 64, max-steps 5000, lr 1e-4).
  - `--data` default points at `data/ohf_voice/train` instead of
    `data/jenny_tts/train`.
  - `.add_local_python_source("liquid_audio")` removed from the Modal image
    build: this cookbook installs `liquid-audio` from PyPI rather than from a
    local editable workspace, so there is no local source to add.
  - W&B tracker integration dropped: staging's Trainer accepted `log_with` /
    `tracker_project_name` / `tracker_run_name`, but those args aren't
    available in public liquid-audio v1.2.0. Tracked in
    .scratch/wandb-integration/issues/01-tracker-args.md.
  - Per-run output subfolder: instead of bare `/checkpoints` (staging) we
    write to `/checkpoints/{run_id}` where `run_id` is a CLI flag with a
    timestamped default. Accelerate's `automatic_checkpoint_naming=True`
    refuses to overwrite `checkpoint_N`, so without this each re-run would
    collide with the previous run's saves; per-run folders make collisions
    structurally impossible.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.trainer import Trainer

load_dotenv()


def run_training(
    model_id: str,
    data: str,
    context_length: int,
    batch_size: int,
    max_steps: int,
    warmup_steps: int,
    lr: float,
    num_workers: int,
    output_dir: str,
    val_split_ratio: float = 0.0,
    seed: int = 42,
) -> None:
    dataset_path = Path(data)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {dataset_path}. "
            "Run scripts/preprocess_ohf_voice.py before training."
        )

    train_data = LFM2DataLoader(dataset_path=str(dataset_path), context_length=context_length)

    val_data: LFM2DataLoader | None = None
    if val_split_ratio > 0:
        splits = train_data.dataset.train_test_split(test_size=val_split_ratio, seed=seed)
        train_data.dataset = splits["train"]
        val_data = LFM2DataLoader(dataset_path=str(dataset_path), context_length=context_length)
        val_data.dataset = splits["test"]

    # W&B / Accelerate tracker integration lives in liquid-audio-staging's
    # Trainer but hasn't landed in public liquid-audio v1.2.0 yet (no
    # `log_with`, `tracker_project_name`, or `tracker_run_name` kwargs). For
    # this cookbook iteration the run's loss/val-loss/checkpoints stream to
    # stdout (visible in Modal logs); revisit when v1.2.1+ ships tracker
    # support. See `.scratch/wandb-integration/issues/01-tracker-args.md`.
    trainer = Trainer(
        model_id=model_id,
        train_data=train_data,
        val_data=val_data,
        lr=lr,
        batch_size=batch_size,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        dataloader_num_workers=num_workers,
        logging_interval=10,
        save_interval=500,
        val_interval=100,
        output_dir=output_dir,
    )
    trainer.train()


def run_on_modal(args: argparse.Namespace) -> None:
    import modal

    app = modal.App("lfm2-audio-train")

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg")
        .pip_install_from_pyproject("pyproject.toml")
    )

    data_vol = modal.Volume.from_name(args.modal_volume)
    output_vol = modal.Volume.from_name(args.modal_output_volume, create_if_missing=True)
    secrets = [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}),
    ]

    @app.function(
        gpu=args.modal_gpu,
        image=image,
        volumes={"/data": data_vol, "/checkpoints": output_vol},
        secrets=secrets,
        timeout=args.modal_timeout,
        serialized=True,
    )
    def remote_train() -> None:
        run_training(
            model_id=args.model_id,
            data=args.modal_data_path,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            lr=args.lr,
            num_workers=args.num_workers,
            output_dir=f"/checkpoints/{args.run_id}",
            val_split_ratio=args.val_split_ratio,
            seed=args.seed,
        )

    with app.run():
        remote_train.remote()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--data", default="data/ohf_voice/train")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/ohf_voice")
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("ohf-voice-%Y%m%d-%H%M%S"),
        help=(
            "Per-run subfolder under the output volume / output dir. Defaults "
            "to a timestamped id so re-runs never collide with each other's "
            "checkpoint_N directories."
        ),
    )

    # Train/val split (val is carved from the train split, NOT from the held-out test set)
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=0.05,
        help="Fraction of the preprocessed train split held out for in-loop validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the in-loop val split.")

    # Modal arguments
    parser.add_argument("--modal", action="store_true", help="Run training on Modal with a GPU.")
    parser.add_argument("--modal-gpu", default="A100-80GB", help="GPU type on Modal.")
    parser.add_argument(
        "--modal-volume",
        default="ohf-voice-data",
        help="Modal volume containing the preprocessed data.",
    )
    parser.add_argument(
        "--modal-data-path",
        default="/data/train",
        help="Path to the preprocessed data inside the Modal volume.",
    )
    parser.add_argument(
        "--modal-output-volume",
        default="lfm2-training-output",
        help="Modal volume for checkpoints.",
    )
    parser.add_argument(
        "--modal-timeout",
        type=int,
        default=21600,
        help="Max job duration in seconds (default: 6h).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.modal:
        run_on_modal(args)
    else:
        run_training(
            model_id=args.model_id,
            data=args.data,
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            lr=args.lr,
            num_workers=args.num_workers,
            output_dir=f"{args.output_dir}/{args.run_id}",
            val_split_ratio=args.val_split_ratio,
            seed=args.seed,
        )
