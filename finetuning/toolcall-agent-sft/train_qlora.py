#!/usr/bin/env python3
"""QLoRA (NF4, fp32 compute) SFT for LFM3-1.7B on AMD RDNA1 8 GB.

Verified on RX 5700 XT / gfx1010: 64 examples / 5 steps / max_length=640 peaked at
3.77 GiB. Peak VRAM depends heavily on sequence length; see README for other runs.
bf16 and fp16 are intentionally disabled: RDNA1 fp16 AMP yields NaN gradients.
"""
from __future__ import annotations

import argparse
import os
import json

# Required for ROCm on gfx1010. Set before importing torch.
os.environ.setdefault("AMDGCN_USE_BUFFER_OPS", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "LiquidAI/LFM3-1.7B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/sft_dataset.jsonl")
    ap.add_argument("--out", default="./out/agent-sft")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    dataset = Dataset.from_list(rows)
    print(f"[data] {len(dataset)} examples", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def formatting(example):
        return {"text": tok.apply_chat_template(
            example["messages"], tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # must match inference service
        )}

    dataset = dataset.map(formatting, remove_columns=dataset.column_names)
    print(f"[data] formatted; sample length {len(dataset[0]['text'])}", flush=True)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        device_map="auto", quantization_config=quant,
    )

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    train_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        max_length=args.max_length,
        bf16=False, fp16=False,           # required on RDNA1
        optim="adamw_torch",
        report_to="none",
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=20,
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model, processing_class=tok,
        train_dataset=dataset, peft_config=peft_config, args=train_args,
    )
    print("[train] start", flush=True)
    trainer.train()
    trainer.save_model(f"{args.out}/adapter")
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[train] done. peak VRAM={peak:.2f} GiB", flush=True)
    print(f"[train] adapter saved to {args.out}/adapter", flush=True)


if __name__ == "__main__":
    main()
