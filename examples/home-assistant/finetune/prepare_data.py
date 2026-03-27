"""Prepare SFT data for fine-tuning LFM2 or LFM2.5 models.

Loads sft_data.jsonl, converts assistant tool_calls to the target model's text format,
splits 80/20 stratified by capability, saves local copies, and uploads to HF Hub.

Usage:
    uv run --group finetune python finetune/prepare_data.py                      # LFM2 format (default)
    uv run --group finetune python finetune/prepare_data.py --model-family lfm25 # LFM2.5 Hermes format
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict


REPO_ID_LFM2  = "Paulescu/home-assistant-sft"
REPO_ID_LFM25 = "Paulescu/home-assistant-sft-lfm25"
SEED = 42

SFT_DATA_PATH = Path(__file__).parent.parent / "benchmark" / "datasets" / "sft_data.jsonl"
OUTPUT_DIR = Path(__file__).parent / "data"


def convert_assistant_message(msg: dict) -> dict:
    """Convert an assistant message from OpenAI format to LFM2 text format.

    OpenAI format:
        {"role": "assistant", "content": null, "tool_calls": [...]}

    LFM2 text format (JSON variant):
        {"role": "assistant", "content": "<|tool_call_start|>[{...}]<|tool_call_end|>"}
    """
    if msg.get("role") != "assistant":
        return msg
    if not msg.get("tool_calls"):
        return {"role": "assistant", "content": msg.get("content") or ""}

    calls = []
    for tc in msg["tool_calls"]:
        fn = tc["function"]
        arguments = fn["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append({"name": fn["name"], "arguments": arguments})

    content = f"<|tool_call_start|>{json.dumps(calls)}<|tool_call_end|>"
    return {"role": "assistant", "content": content}


def convert_assistant_message_lfm25(msg: dict) -> dict:
    """Convert an assistant message from OpenAI format to Hermes 2 Pro format.

    OpenAI format:
        {"role": "assistant", "content": null, "tool_calls": [...]}

    Hermes 2 Pro format (natively parsed by llama-server into structured tool_calls):
        {"role": "assistant", "content": "<tool_call>\\n{\"name\": \"...\", \"arguments\": {...}}\\n</tool_call>"}
    """
    if msg.get("role") != "assistant":
        return msg
    if not msg.get("tool_calls"):
        return {"role": "assistant", "content": msg.get("content") or ""}

    blocks = []
    for tc in msg["tool_calls"]:
        fn = tc["function"]
        args = fn["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        blocks.append(f"<tool_call>\n{json.dumps({'name': fn['name'], 'arguments': args})}\n</tool_call>")
    return {"role": "assistant", "content": "\n".join(blocks)}


def convert_messages(messages: list[dict], model_family: str) -> list[dict]:
    result = []
    if model_family == "lfm25":
        for msg in messages:
            converted = convert_assistant_message_lfm25(msg)
            result.append(converted)
            # leap-finetune requires a role='tool' response after every tool call
            if converted.get("role") == "assistant" and "<tool_call>" in (converted.get("content") or ""):
                result.append({"role": "tool", "content": "OK"})
    else:
        for msg in messages:
            converted = convert_assistant_message(msg)
            result.append(converted)
            # leap-finetune requires a role='tool' response after every tool call
            if converted.get("role") == "assistant" and "<|tool_call_start|>" in (converted.get("content") or ""):
                result.append({"role": "tool", "content": "OK"})
    return result


def load_examples(model_family: str) -> list[dict]:
    examples = []
    with open(SFT_DATA_PATH) as f:
        for line in f:
            raw = json.loads(line)
            capability = raw["_meta"]["capability"]
            examples.append(
                {
                    "messages": convert_messages(raw["messages"], model_family),
                    "tools": raw["tools"],
                    "_capability": capability,
                }
            )
    return examples


def stratified_split(
    examples: list[dict], train_ratio: float = 0.8, seed: int = SEED
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_cap: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_cap[ex["_capability"]].append(ex)

    train, eval_ = [], []
    for cap, items in sorted(by_cap.items()):
        rng.shuffle(items)
        n_train = round(len(items) * train_ratio)
        train.extend(items[:n_train])
        eval_.extend(items[n_train:])
        print(f"  {cap}: {len(items)} total -> {n_train} train / {len(items) - n_train} eval")

    return train, eval_


def strip_meta(examples: list[dict]) -> list[dict]:
    return [{"messages": ex["messages"], "tools": ex["tools"]} for ex in examples]


def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-family",
        choices=["lfm2", "lfm25"],
        default="lfm2",
        help="Target model family: 'lfm2' (default) or 'lfm25' (Hermes 2 Pro format)",
    )
    args = parser.parse_args()

    repo_id = REPO_ID_LFM25 if args.model_family == "lfm25" else REPO_ID_LFM2

    print(f"Model family: {args.model_family}")
    print(f"Target dataset: {repo_id}")
    print("\nLoading and converting examples...")
    examples = load_examples(args.model_family)
    print(f"Loaded {len(examples)} examples")

    print("\nSplitting by capability (80/20):")
    train_raw, eval_raw = stratified_split(examples)
    print(f"\nTotal: {len(train_raw)} train / {len(eval_raw)} eval")

    train = strip_meta(train_raw)
    eval_ = strip_meta(eval_raw)

    print("\nSaving local copies...")
    save_jsonl(train, OUTPUT_DIR / "train.jsonl")
    save_jsonl(eval_, OUTPUT_DIR / "eval.jsonl")

    print(f"\nUploading to HuggingFace Hub as '{repo_id}'...")
    ds = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "test": Dataset.from_list(eval_),
        }
    )
    ds.push_to_hub(repo_id, private=False)
    print(f"Done. Dataset at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
