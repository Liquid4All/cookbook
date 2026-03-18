# /// script
# dependencies = ["unsloth", "trl>=0.15.0", "datasets>=2.0", "trackio", "huggingface_hub"]
# ///
"""LoRA fine-tune LFM2.5-1.2B-Instruct on the home-assistant SFT dataset.

Submitted to HuggingFace Jobs via:
  hf jobs uv run finetune/train.py \\
      --flavor l4-x1 \\
      --secrets HF_TOKEN \\
      --timeout 4h \\
      --dataset-repo USERNAME/home-assistant-sft \\
      --model        LiquidAI/LFM2.5-1.2B-Instruct \\
      --output-repo  USERNAME/LFM2.5-1.2B-home-assistant-sft

LoRA config mirrors the mobile-actions fine-tune (same base model, same hardware):
  rank=16, alpha=16, dropout=0, max_seq_length=2048, epochs=3, lr=2e-4
"""

import argparse

import unsloth  # noqa: F401 — must be imported before trl/transformers to apply patches


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune LFM2 on home-assistant SFT dataset")
    p.add_argument(
        "--dataset-repo",
        required=True,
        help="HF Hub dataset path, e.g. USERNAME/home-assistant-sft",
    )
    p.add_argument(
        "--output-repo",
        default=None,
        help="HF Hub model path for the fine-tuned model, e.g. USERNAME/LFM2.5-1.2B-home-assistant-sft. Omit to skip push-to-hub.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after N optimizer steps. -1 means run all epochs (default: -1).",
    )
    p.add_argument(
        "--model",
        default="LiquidAI/LFM2.5-1.2B-Instruct",
        help="Base model on HF Hub (default: LiquidAI/LFM2.5-1.2B-Instruct)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank and alpha (default: 16)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    # 1. Load model + fast Unsloth kernels, 4-bit quantization
    print(f"Loading base model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        max_seq_length=2048,
    )

    # Unsloth may reset eos_token to '<EOS_TOKEN>' which doesn't exist in LFM2's
    # vocabulary. Reset it to the actual end-of-turn token used in LFM2's chat template.
    tokenizer.eos_token = "<|im_end|>"

    # 2. Attach LoRA adapters (same target modules as mobile-actions fine-tune)
    print(f"Attaching LoRA adapters (rank={args.rank})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. Load dataset from HF Hub, apply chat template with tools
    print(f"Loading dataset: {args.dataset_repo}")
    dataset = load_dataset(args.dataset_repo)

    def format_example(ex: dict) -> dict:
        text = tokenizer.apply_chat_template(
            ex["messages"],
            tools=ex["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_example, desc="Applying chat template")
    # Remove raw columns so SFTTrainer uses the pre-formatted "text" field
    # and doesn't try to re-apply the chat template internally.
    dataset = dataset.remove_columns(["messages", "tools"])

    # 4. SFTTrainer
    print("Initialising SFTTrainer ...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=SFTConfig(
            output_dir="./output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            warmup_steps=min(10, max(1, args.max_steps)) if args.max_steps > 0 else 10,
            dataset_text_field="text",
            eos_token="<|im_end|>",
            bf16=True,
            logging_steps=1,
            eval_strategy="no" if args.max_steps > 0 else "epoch",
            save_strategy="no" if args.max_steps > 0 else "epoch",
            load_best_model_at_end=False if args.max_steps > 0 else True,
            push_to_hub=args.output_repo is not None,
            hub_model_id=args.output_repo,
            report_to="trackio" if args.output_repo else "none",
        ),
    )

    # 5. Mask loss to assistant turns only.
    #    System, user, and tool-result tokens are excluded from the loss.
    #    Both tool_calls turns and final text turns remain supervised.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Training ...")
    trainer.train()

    if args.output_repo:
        print(f"Pushing fine-tuned model to {args.output_repo} ...")
        trainer.push_to_hub()
    print("Done.")


if __name__ == "__main__":
    main()
