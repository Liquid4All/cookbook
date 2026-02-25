#!/usr/bin/env python3
"""
Fine-Tune LFM2.5-1.2B-Instruct for LocalCowork Tool Routing

Full fine-tune (not QLoRA) on H100 80GB. The 1.2B model fits entirely in
memory with optimizer states (~8-10 GB total), making full fine-tune the
right choice over LoRA for this model size.

Base model: LFM2.5-1.2B-Instruct (0.880 agent score on tool-calling
benchmark — tied #1 among 21 small models, lowest latency of top tier).
Upgraded from LFM2-1.2B-Tool which scored 78% on our internal benchmarks.

Usage:
    # On H100 (after rsync of training data):
    python fine-tune-router.py \
        --data-dir /home/ubuntu/localcowork-finetune/training-data \
        --output-dir /home/ubuntu/localcowork-finetune/output \
        --epochs 5 \
        --batch-size 16 \
        --lr 2e-5

Requirements:
    pip install torch transformers trl datasets accelerate

Hardware: NVIDIA H100 80GB HBM3
Expected training time: 45-90 minutes for ~1,000 examples x 5 epochs
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────

# LFM2.5-1.2B-Instruct from HuggingFace — best-in-class tool calling at 1.2B
# (0.880 agent score, 49.12 BFCLv3, tied #1 among 21 small LLMs)
DEFAULT_BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

# Maximum sequence length for training. Tool calls are short (~400 tokens avg),
# but system prompts with K=15 tools can reach ~1500 tokens.
MAX_SEQ_LENGTH = 2048


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LFM2.5-1.2B-Instruct for LocalCowork")
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="HuggingFace model name or local path to base model",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing train.jsonl, eval.jsonl, test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save fine-tuned model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint directory")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="Early stopping patience")
    return parser.parse_args()


def load_jsonl_dataset(filepath: str) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records: list[dict[str, Any]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)

    logger.info("Loaded %d records from %s", len(records), filepath)
    return Dataset.from_list(records)


def format_to_chatml(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Format a training example to ChatML using the tokenizer's chat template.

    Falls back to manual ChatML formatting if no template is available.
    """
    messages = example["messages"]

    # Try using the tokenizer's built-in chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Manual ChatML fallback
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        text = "\n".join(parts)

    return {"text": text}


def verify_gpu() -> None:
    """Verify CUDA is available and log GPU info."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Check nvidia-smi and NVreg_NvLinkDisable=1 fix.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("GPU: %s (%.1f GB)", device_name, device_memory)

    if device_memory < 40:
        logger.warning(
            "GPU memory (%.1f GB) may be insufficient for full fine-tune. "
            "Consider QLoRA instead.",
            device_memory,
        )


def main() -> None:
    """Main training loop."""
    args = parse_args()

    logger.info("=== LocalCowork Router Fine-Tuning ===")
    logger.info("Base model: %s", args.base_model)
    logger.info("Data dir: %s", args.data_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Epochs: %d, Batch size: %d, LR: %s", args.epochs, args.batch_size, args.lr)

    # Verify GPU
    verify_gpu()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model (full precision for full fine-tune)
    logger.info("Loading model (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Use "flash_attention_2" if available
    )

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model loaded: %.1fB total params, %.1fB trainable (full fine-tune)",
        total_params / 1e9,
        trainable_params / 1e9,
    )

    # Load datasets
    train_data_path = os.path.join(args.data_dir, "train.jsonl")
    eval_data_path = os.path.join(args.data_dir, "eval.jsonl")

    if not os.path.exists(train_data_path):
        logger.error("Train data not found at %s", train_data_path)
        sys.exit(1)
    if not os.path.exists(eval_data_path):
        logger.error("Eval data not found at %s", eval_data_path)
        sys.exit(1)

    train_dataset = load_jsonl_dataset(train_data_path)
    eval_dataset = load_jsonl_dataset(eval_data_path)

    # Format datasets to ChatML text
    logger.info("Formatting datasets to ChatML...")
    train_dataset = train_dataset.map(
        lambda ex: format_to_chatml(ex, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda ex: format_to_chatml(ex, tokenizer),
        remove_columns=eval_dataset.column_names,
    )

    logger.info("Train examples: %d", len(train_dataset))
    logger.info("Eval examples: %d", len(eval_dataset))

    # Log a sample
    sample_text = train_dataset[0]["text"]
    logger.info("Sample training text (first 500 chars):\n%s", sample_text[:500])

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Epochs & batch size
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,

        # Learning rate schedule
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        # Precision & optimization (H100-optimized)
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_torch_fused",
        max_grad_norm=1.0,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to="none",

        # Data
        dataloader_num_workers=4,
        remove_unused_columns=True,

        # SFT-specific
        max_seq_length=args.max_seq_length,
        packing=True,
        dataset_text_field="text",

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
    )

    # Callbacks
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final model
    best_dir = os.path.join(args.output_dir, "best")
    logger.info("Saving best model to %s", best_dir)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Log final metrics
    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", json.dumps(metrics, indent=2))

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_seq_length": args.max_seq_length,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
                "final_eval_loss": metrics.get("eval_loss"),
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            f,
            indent=2,
        )

    logger.info("Training complete! Model saved to %s", best_dir)
    logger.info("Next steps:")
    logger.info("  1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py %s", best_dir)
    logger.info("  2. Quantize: llama-quantize <F16.gguf> <Q8_0.gguf> Q8_0")
    logger.info("  3. Benchmark: npx tsx benchmark-lfm.ts --endpoint <new-model-endpoint>")


if __name__ == "__main__":
    main()
