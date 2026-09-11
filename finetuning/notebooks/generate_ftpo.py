#!/usr/bin/env python3
"""
Generate FTPO training pairs from LFM2.5-1.2B-Base via llama.cpp.

Saves progress to a checkpoint file so you can Ctrl+C and resume later.

Usage:
    # Start llama.cpp server with parallel slots:
    llama-server -m models/lfm-base/LFM2.5-1.2B-Base-BF16.gguf --port 8080 -ngl 99 -np 4 --temp 0.01 --top-k 50 --top-p 1.0 --min-p 0.01

    # Generate 1000 FTPO pairs with 4 parallel requests (Ctrl+C to pause, re-run to resume):
    python generate_ftpo.py --target 1000 --parallel 4

    # Push results to Hugging Face Hub:
    python generate_ftpo.py --push iamleonie/antidoom-test
"""

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer


# ── Doom loop detection (from antidoom/repetition.py) ────────────────────

@dataclass(frozen=True)
class RepeatHit:
    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self):
        return self.start + self.period


def _verify_repetition_at(text, start_pos, period, min_repeats, min_total_repeated):
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return False, None
    pattern = text[start_pos : start_pos + period]
    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos : pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos
    pos = start_pos - period
    while pos >= 0 and text[pos : pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period
    total = reps * period
    if reps >= min_repeats and total >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return True, RepeatHit(start_pos, end_pos, period, reps, snippet)
    return False, None


def find_inner_repetition(
    text,
    min_repeats=4,
    max_period=1024,
    min_period=1,
    min_total_repeated=60,
    sample_len=16,
    sample_interval=128,
):
    if not text or len(text) < min_total_repeated:
        return False, None
    n = len(text)
    for sample_pos in range(0, n - sample_len, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]
        for other_pos in [
            text.find(fingerprint, sample_pos + sample_len),
            text.rfind(fingerprint, 0, sample_pos),
        ]:
            if other_pos == -1:
                continue
            candidate_period = abs(other_pos - sample_pos)
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(
                    text, min(sample_pos, other_pos), candidate_period,
                    min_repeats=min_repeats, min_total_repeated=min_total_repeated,
                )
                if found:
                    return True, hit
    return False, None


# ── FTPO extraction ──────────────────────────────────────────────────────

def find_rejected_token_index(generated_ids, hit, tokenizer):
    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    char_pos = hit.repeat_start
    while char_pos < len(gen_text) and gen_text[char_pos].isspace():
        char_pos += 1
    if char_pos >= len(gen_text):
        return None
    for i in range(len(generated_ids)):
        prefix = tokenizer.decode(generated_ids[: i + 1], skip_special_tokens=True)
        if len(prefix) > char_pos:
            return i
    return None


def extract_ftpo_pair(result, tokenizer, llama_url):
    generated_ids = tokenizer.encode(result["generated_text"], add_special_tokens=False)
    reject_idx = find_rejected_token_index(generated_ids, result["hit"], tokenizer)
    if reject_idx is None or reject_idx == 0:
        return None

    input_ids = tokenizer.encode(result["full_prompt"], add_special_tokens=False)
    prompt_ids = input_ids + generated_ids[:reject_idx]
    rejected_id = generated_ids[reject_idx]

    context_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    resp = requests.post(llama_url, json={
        "prompt": context_text,
        "n_predict": 1,
        "n_probs": 50,
        "temperature": 0.0,
    })
    data = resp.json()

    if "completion_probabilities" not in data or not data["completion_probabilities"]:
        return None

    top_tokens = data["completion_probabilities"][0]["top_logprobs"]
    chosen_ids = []
    for t in top_tokens:
        tid = t["id"]
        if tid == rejected_id:
            continue
        if len(tokenizer.decode([tid]).strip()) >= 1:
            chosen_ids.append(tid)
        if len(chosen_ids) >= 20:
            break

    if not chosen_ids:
        return None

    return {
        "full_prompt": result["full_prompt"],
        "prompt_ids": prompt_ids,
        "chosen_ids": chosen_ids,
        "rejected_token_id": rejected_id,
    }


# ── Checkpoint management ────────────────────────────────────────────────

def load_checkpoint(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "prompt_offset": 0,
        "pairs_count": 0,
        "doom_count": 0,
        "clean_count": 0,
        "elapsed_seconds": 0.0,
    }


def save_checkpoint(state, path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.rename(path)


# ── Main ─────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "{prompt}\n\n"
    'Think through the problem step by step, then respond with your final answer as "Answer: <your answer>".'
)

running = True

def handle_interrupt(signum, frame):
    global running
    print("\n\n⏸️  Stopping after current batch... (progress will be saved)")
    running = False

signal.signal(signal.SIGINT, handle_interrupt)


def generate_one(client, formatted, max_tokens, temperature):
    response = client.completions.create(
        model="default",
        prompt=formatted,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        extra_body={"min_p": 0.01, "top_k": 50},
    )
    return response.choices[0].text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=1000, help="Target number of FTPO pairs (default: 1000)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent requests — match llama-server -np (default: 1)")
    parser.add_argument("--max-new-tokens", type=int, default=1000, help="Max tokens per generation (default: 1000)")
    parser.add_argument("--temperature", type=float, default=0.01, help="Sampling temperature (default: 0.01)")
    parser.add_argument("--server-url", type=str, default="http://localhost:8080", help="llama.cpp server URL")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output files")
    parser.add_argument("--push", type=str, default=None, help="Push results to HF Hub repo (e.g. iamleonie/antidoom-test)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "ftpo_checkpoint.json"
    output_path = output_dir / "ftpo_pairs.jsonl"

    state = load_checkpoint(checkpoint_path)

    if args.push:
        load_dotenv()
        from huggingface_hub import login
        from datasets import Dataset as HFDataset
        login(token=os.environ["HF_TOKEN"])

        pairs = []
        with open(output_path) as f:
            for line in f:
                pairs.append(json.loads(line))
        print(f"📤 Pushing {len(pairs)} pairs to {args.push}...")
        HFDataset.from_list(pairs).push_to_hub(args.push)
        print(f"✅ Done! Dataset at https://huggingface.co/datasets/{args.push}")
        return

    # Setup
    load_dotenv()
    print("📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-1.2B-Base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🔗 Connecting to llama.cpp server...")
    client = OpenAI(base_url=f"{args.server_url}/v1", api_key="not-needed")
    try:
        models = client.models.list()
        print(f"   Connected: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"❌ Cannot connect to server at {args.server_url}: {e}")
        print("   Start it with: llama-server -m models/lfm-base/LFM2.5-1.2B-Base-BF16.gguf --port 8080 -ngl 99 -np 4")
        sys.exit(1)

    llama_url = f"{args.server_url}/completion"

    print("📥 Loading prompt dataset...")
    dataset = load_dataset("LiquidAI/antidoom-mix-v1.0", split="train")
    dataset = dataset.shuffle(seed=42)

    offset = state["prompt_offset"]
    remaining = args.target - state["pairs_count"]
    parallel = args.parallel

    if state["pairs_count"] > 0:
        total = state["doom_count"] + state["clean_count"]
        doom_rate = state["doom_count"] / total if total > 0 else 0
        print(f"\n🔄 Resuming from checkpoint:")
        print(f"   Pairs: {state['pairs_count']}/{args.target}")
        print(f"   Completions: {total} ({doom_rate:.0%} doom rate)")
        print(f"   Time so far: {state['elapsed_seconds']/60:.1f} min")
        print(f"   Next prompt index: {offset}")
    else:
        print(f"\n🆕 Starting fresh — target: {args.target} pairs")

    if remaining <= 0:
        print(f"\n✅ Already have {state['pairs_count']} pairs (target: {args.target}). Use --push to upload.")
        return

    print(f"   Remaining: {remaining} pairs to generate")
    print(f"   Parallel requests: {parallel}")
    print(f"   Press Ctrl+C to pause (progress is saved)\n")

    start_time = time.time()
    session_pairs = 0
    session_completions = 0

    while state["pairs_count"] < args.target and running:
        batch_size = min(parallel, len(dataset) - offset)
        if batch_size <= 0:
            print("⚠️  Exhausted all prompts in the dataset!")
            break

        # Build batch of formatted prompts
        batch = []
        for i in range(batch_size):
            sample = dataset[int(offset + i)]
            prompt_text = sample["conversations"][0]["value"]
            batch.append(PROMPT_TEMPLATE.format(prompt=prompt_text))

        # Fire generation requests in parallel threads
        generated = [None] * batch_size
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(generate_one, client, prompt, args.max_new_tokens, args.temperature): i
                for i, prompt in enumerate(batch)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    generated[idx] = future.result()
                except Exception as e:
                    print(f"\n⚠️  Generation failed for prompt {offset + idx}: {e}")

        # Process results
        for i, gen_text in enumerate(generated):
            if gen_text is None:
                state["clean_count"] += 1
                continue

            session_completions += 1
            found, hit = find_inner_repetition(gen_text)

            if found:
                state["doom_count"] += 1
                result = {
                    "full_prompt": batch[i],
                    "generated_text": gen_text,
                    "hit": hit,
                }
                pair = extract_ftpo_pair(result, tokenizer, llama_url)
                if pair is not None:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(pair) + "\n")
                    state["pairs_count"] += 1
                    session_pairs += 1
            else:
                state["clean_count"] += 1

        offset += batch_size
        state["prompt_offset"] = offset

        # Progress update
        total = state["doom_count"] + state["clean_count"]
        elapsed = state["elapsed_seconds"] + (time.time() - start_time)
        rate = total / elapsed if elapsed > 0 else 0
        doom_rate = state["doom_count"] / total if total > 0 else 0
        eta_pairs = args.target - state["pairs_count"]
        pairs_per_sec = state["pairs_count"] / elapsed if elapsed > 0 else 0
        eta_sec = eta_pairs / pairs_per_sec if pairs_per_sec > 0 else float("inf")
        print(
            f"  [{total:,} gen | {state['pairs_count']}/{args.target} pairs | "
            f"{doom_rate:.0%} doom | {rate:.1f} gen/s | "
            f"ETA: {eta_sec/60:.0f} min]"
        )
        save_checkpoint(state, checkpoint_path)

    # Final save
    state["elapsed_seconds"] += time.time() - start_time
    state["prompt_offset"] = offset
    save_checkpoint(state, checkpoint_path)

    total = state["doom_count"] + state["clean_count"]
    doom_rate = state["doom_count"] / total if total > 0 else 0
    elapsed = state["elapsed_seconds"]

    print(f"\n{'=' * 60}")
    print(f"📊 Session: +{session_pairs} pairs from {session_completions} completions")
    print(f"📊 Total:   {state['pairs_count']}/{args.target} pairs from {total} completions")
    print(f"   Doom rate: {doom_rate:.1%}")
    if total > 0:
        print(f"   Total time: {elapsed/60:.1f} min ({elapsed/total:.1f}s per completion)")

    if state["pairs_count"] >= args.target:
        print(f"\n🎉 Target reached! Push to Hub with:")
        print(f"   python generate_ftpo.py --push iamleonie/antidoom-test")
    else:
        print(f"\n⏸️  Paused. Re-run to continue ({args.target - state['pairs_count']} pairs remaining)")


if __name__ == "__main__":
    main()
