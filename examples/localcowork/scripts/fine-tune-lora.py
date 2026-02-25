#!/usr/bin/env python3
"""
LoRA Fine-Tune LFM2.5-1.2B-Instruct for LocalCowork Tool Routing (v2)

LoRA fine-tuning on H100 (or any GPU with >= 16 GB VRAM). The full fine-tune
approach (fine-tune-router.py) failed on LFM2 architecture — loss exploded
and produced NaN outputs. LoRA with packing disabled works perfectly.

Base model: LFM2.5-1.2B-Instruct (0.880 agent score on tool-calling
benchmark — tied #1 among 21 small models, lowest latency of top tier).

v1 results (H100 80GB, 841 train / 110 eval, r=32, alpha=64, K=15 only):
  - Token accuracy: 99.2% (train), 98.9% (eval)
  - Live accuracy: 83% at K=15, 38% at K=24+
  - Training time: ~2 minutes

v2 changes (83 tools across 15 servers, ~4000 train / ~400 eval):
  - LoRA rank: 32 → 64 (more capacity for 83-tool vocabulary)
  - LoRA alpha: 64 → 128 (matched 2x rank)
  - Max seq length: 2048 → 4096 (K=83 system prompts reach ~3500 tokens)
  - Epochs: 5 → 3 (larger dataset, less overfitting risk)
  - Learning rate: 2e-4 → 1e-4 (slightly lower for stability)
  - Grad accumulation: 4 → 8 (effective batch 32)
  - Target: >90% at K=15, >70% at K=25+

Critical notes:
  - packing=False is REQUIRED for LFM2 architecture (conv layers, not standard transformer)
  - Full fine-tune destroys LFM2 weights — always use LoRA
  - TRL 0.28.0+ uses `max_length` (not `max_seq_length`) and `processing_class` (not `tokenizer`)
  - PyTorch 2.10+ uses `dtype` (not `torch_dtype`) and `total_memory` (not `total_mem`)

Usage:
    # v2 on H100 (after rsync of training data):
    python fine-tune-lora.py \\
        --data-dir /home/ubuntu/localcowork-finetune/training-data-v2 \\
        --output-dir /home/ubuntu/localcowork-finetune/output-v2

    # Override defaults for experimentation:
    python fine-tune-lora.py \\
        --data-dir /home/ubuntu/localcowork-finetune/training-data-v2 \\
        --output-dir /home/ubuntu/localcowork-finetune/output-v2 \\
        --lora-r 64 --lora-alpha 128 \\
        --epochs 3 --lr 1e-4 \\
        --max-seq-length 4096 \\
        --gradient-accumulation-steps 8

Requirements:
    pip install torch transformers trl datasets accelerate peft

Hardware: Any GPU with >= 16 GB VRAM (tested on NVIDIA H100 80GB)
Expected training time: ~3-4 minutes for ~4000 examples x 3 epochs on H100
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
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# --- Constants ----------------------------------------------------------------

# LFM2.5-1.2B-Instruct from HuggingFace — best-in-class tool calling at 1.2B
DEFAULT_BASE_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

# Maximum sequence length for training. Tool calls are short (~400 tokens avg),
# but system prompts with K=83 tools can reach ~3500 tokens.
MAX_SEQ_LENGTH = 4096

# LoRA target modules for LFM2 architecture.
# Includes attention projections (q/k/v/o) + MLP layers (w1/w2/w3) + special LFM layers.
LFM2_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "out_proj",                                 # Output projection (LFM-specific)
    "w1", "w2", "w3",                          # MLP / MoE gates
    "in_proj",                                  # Input projection (LFM-specific)
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune LFM2.5-1.2B-Instruct for LocalCowork tool routing"
    )
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
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help="Max sequence length"
    )
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint dir")
    parser.add_argument("--eval-steps", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument(
        "--early-stopping-patience", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps"
    )
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

    try:
        text: str = tokenizer.apply_chat_template(
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
        logger.error("CUDA not available! Fine-tuning requires a GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("GPU: %s (%.1f GB)", device_name, device_memory)

    if device_memory < 16:
        logger.warning(
            "GPU memory (%.1f GB) is below recommended 16 GB. "
            "Training may be slow or fail. Reduce batch size if needed.",
            device_memory,
        )


def main() -> None:
    """Main LoRA training loop."""
    args = parse_args()

    logger.info("=== LocalCowork Router LoRA Fine-Tuning ===")
    logger.info("Base model: %s", args.base_model)
    logger.info("Data dir: %s", args.data_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info(
        "LoRA: r=%d, alpha=%d, dropout=%.2f",
        args.lora_r, args.lora_alpha, args.lora_dropout,
    )
    logger.info(
        "Training: epochs=%d, batch=%d, grad_accum=%d, lr=%s",
        args.epochs, args.batch_size, args.gradient_accumulation_steps, args.lr,
    )

    # Verify GPU
    verify_gpu()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model in bfloat16
    logger.info("Loading base model (BF16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # Scaled dot-product attention (memory efficient)
    )

    # Log base model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Base model: %.3fB params", total_params / 1e9)

    # Detect available LoRA target modules
    available_targets: list[str] = []
    all_module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    for target in LFM2_LORA_TARGETS:
        if target in all_module_names:
            available_targets.append(target)
        else:
            logger.info("LoRA target '%s' not found in model — skipping", target)

    if not available_targets:
        logger.error("No LoRA target modules found! Check model architecture.")
        sys.exit(1)

    logger.info("LoRA target modules: %s", available_targets)

    # Create LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=available_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    pct_trainable = 100.0 * trainable_params / total_params
    logger.info(
        "LoRA applied: %.2fM trainable (%.1f%%), %.3fB frozen",
        trainable_params / 1e6,
        pct_trainable,
        frozen_params / 1e9,
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

    # Format to ChatML
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

    # SFT training config
    # CRITICAL: packing=False — LFM2 architecture breaks with packing enabled
    training_args = SFTConfig(
        output_dir=args.output_dir,

        # Epochs & batch size
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Learning rate schedule
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        # Precision (BF16 preferred on H100/A100, FP16 fallback)
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

        # SFT-specific — NO PACKING (critical for LFM2 architecture)
        max_length=args.max_seq_length,
        packing=False,
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

    # Trainer — TRL 0.28.0+ uses `processing_class` instead of `tokenizer`
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting LoRA training...")
    if args.resume_from:
        logger.info("Resuming from checkpoint: %s", args.resume_from)
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Merge LoRA weights back into base model and save
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    best_dir = os.path.join(args.output_dir, "best")
    logger.info("Saving merged model to %s", best_dir)
    merged_model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Final eval
    metrics = trainer.evaluate()
    logger.info("Final eval metrics: %s", json.dumps(metrics, indent=2))

    # Save training metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "method": "lora",
                "base_model": args.base_model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_targets": available_targets,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.lr,
                "max_seq_length": args.max_seq_length,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "pct_trainable": pct_trainable,
                "final_eval_loss": metrics.get("eval_loss"),
                "final_eval_token_accuracy": metrics.get("eval_mean_token_accuracy"),
            },
            f,
            indent=2,
        )

    logger.info("Training complete! Merged model saved to %s", best_dir)
    logger.info("Next steps:")
    logger.info("  1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py %s", best_dir)
    logger.info("  2. Quantize: llama-quantize <F16.gguf> <Q8_0.gguf> Q8_0")
    logger.info("  3. Copy to _models/: cp <Q8_0.gguf> ~/Projects/localCoWork/_models/")
    logger.info("  4. Benchmark: npx tsx benchmark-lfm.ts --endpoint <new-model-endpoint>")


if __name__ == "__main__":
    main()
