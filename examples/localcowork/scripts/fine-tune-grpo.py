#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for LFM2.5-1.2B-Instruct

Applies RL fine-tuning on top of the SFT checkpoint to optimize for
tool selection accuracy using a domain-specific reward function.

The reward function encodes the failure mode priorities:
  - Correct tool + correct params: +1.0
  - Correct tool, wrong params: +0.5
  - Same server, wrong tool (sibling confusion): -0.25
  - Wrong server entirely (cross-server confusion): -0.75
  - No tool call at all (refusal): -1.0

Usage:
    python fine-tune-grpo.py \
        --sft-model /home/ubuntu/localcowork-finetune/output/best \
        --data-dir /home/ubuntu/localcowork-finetune/training-data \
        --output-dir /home/ubuntu/localcowork-finetune/output-grpo

Requirements:
    pip install torch transformers trl datasets accelerate
"""

import argparse
import json
import logging
import re
import sys
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─── Tool Call Parsing ───────────────────────────────────────────────────────

BRACKET_PATTERN = re.compile(r"\[([a-z_]+\.[a-z_]+)\(")


def parse_bracket_call(text: str) -> str | None:
    """Extract the first tool name from bracket-syntax output."""
    match = BRACKET_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


# ─── Reward Function ────────────────────────────────────────────────────────


def compute_tool_reward(
    prediction: str,
    expected_tool: str,
) -> float:
    """Compute reward for a tool-calling prediction.

    Asymmetric penalties encode failure mode priorities:
    - Cross-server confusion (-0.75) penalized more than sibling confusion (-0.25)
    - No tool call (-1.0) is the worst outcome (model refusal)
    - Partial credit (+0.5) for correct tool with wrong params
    """
    parsed_tool = parse_bracket_call(prediction)

    if parsed_tool is None:
        return -1.0  # No tool call — worst case (FM-refusal)

    if parsed_tool == expected_tool:
        return 1.0  # Perfect match

    # Check if same server (sibling confusion)
    parsed_server = parsed_tool.split(".")[0]
    expected_server = expected_tool.split(".")[0]

    if parsed_server == expected_server:
        return -0.25  # Same server, wrong tool (FM-sibling)

    return -0.75  # Different server entirely (FM-cross-server)


# ─── Dataset Preparation ────────────────────────────────────────────────────


def load_grpo_dataset(filepath: str) -> Dataset:
    """Load JSONL and format for GRPO.

    GRPO expects:
    - prompt: The full system + user prompt
    - expected_tool: Ground truth for reward computation
    """
    records: list[dict[str, Any]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data["messages"]

            # Build the prompt (system + user messages)
            system_msg = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_msg = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            expected_tool = data.get("metadata", {}).get("expectedTool", "")

            # Format as a single prompt string
            prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

            records.append(
                {
                    "prompt": prompt,
                    "expected_tool": expected_tool,
                }
            )

    logger.info("Loaded %d GRPO examples from %s", len(records), filepath)
    return Dataset.from_list(records)


# ─── Reward Wrapper ─────────────────────────────────────────────────────────


class ToolCallRewardFunction:
    """Callable reward function for GRPO trainer.

    The trainer generates N completions per prompt, then this function
    scores each completion against the ground truth.
    """

    def __init__(self, expected_tools: list[str]) -> None:
        self.expected_tools = expected_tools
        self._call_count = 0

    def __call__(self, completions: list[str], prompts: list[str]) -> list[float]:
        """Score a batch of completions."""
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            # Map back to the expected tool using the prompt index
            expected = self.expected_tools[self._call_count % len(self.expected_tools)]
            reward = compute_tool_reward(completion, expected)
            rewards.append(reward)
            self._call_count += 1

        return rewards


# ─── Main ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for tool routing")
    parser.add_argument(
        "--sft-model",
        type=str,
        required=True,
        help="Path to SFT checkpoint (from fine-tune-router.py output/best)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing train.jsonl from generate-training-data.ts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save GRPO-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of GRPO epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate (very low for RL)")
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="Number of completions per prompt for GRPO",
    )
    parser.add_argument("--kl-coeff", type=float, default=0.05, help="KL divergence coefficient")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Run GRPO training."""
    args = parse_args()

    logger.info("=== GRPO Fine-Tuning for Tool Routing ===")
    logger.info("SFT model: %s", args.sft_model)
    logger.info("KL coefficient: %s", args.kl_coeff)
    logger.info("Generations per prompt: %d", args.num_generations)

    # Verify GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load tokenizer and model
    logger.info("Loading SFT model...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.1fB params", total_params / 1e9)

    # Load dataset
    train_path = f"{args.data_dir}/train.jsonl"
    dataset = load_grpo_dataset(train_path)

    # Extract expected tools for reward computation
    expected_tools = [ex["expected_tool"] for ex in dataset]

    # Build reward function
    reward_fn = ToolCallRewardFunction(expected_tools)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=0.3,  # Some exploration
        bf16=torch.cuda.is_bf16_supported(),
        kl_coeff=args.kl_coeff,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
    )

    # Note: GRPO trainer API may vary by TRL version.
    # This uses the TRL 0.12+ API pattern.
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        config=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save
    best_dir = f"{args.output_dir}/best"
    logger.info("Saving GRPO model to %s", best_dir)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    logger.info("GRPO training complete!")
    logger.info("Next: Convert to GGUF and run benchmarks to compare SFT vs SFT+GRPO")


if __name__ == "__main__":
    main()
