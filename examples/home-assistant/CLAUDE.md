# Home Assistant: Claude Code Context

## Project overview

Fine-tuning pipeline for a local home assistant powered by LFM models running via llama.cpp.
The pipeline covers: proof-of-concept app, benchmark, synthetic data generation, LoRA fine-tuning on Modal, and GGUF export/push to HuggingFace.

All commands are run from the `home-assistant/` directory unless stated otherwise.

## Key files

| File | Purpose |
|------|---------|
| `app/server.py` | FastAPI server, starts llama-server on model selection |
| `app/agent.py` | Agent loop: inference, tool dispatch, LFM2 fallback parser |
| `benchmark/run.py` | Runs the 100-task benchmark suite |
| `benchmark/datasets/generate.py` | Generates synthetic SFT data via GPT-4o-mini |
| `benchmark/datasets/sft_data.jsonl` | Generated SFT data in OpenAI format (gitignored) |
| `finetune/prepare_data.py` | Converts OpenAI format to LFM2 text format, uploads to HF |
| `finetune/export.py` | Merges LoRA adapter, converts to GGUF, pushes to HuggingFace |
| `finetune/configs/LFM2-350M.yaml` | Training config for LFM2-350M |
| `FINETUNE-FINDINGS.md` | Detailed results from every fine-tuning run |

## llama.cpp builds

Two builds are installed side-by-side under `~/.local/llama-cpp/`. Select the build per benchmark run with `--llama-build`.

| Version | Build | Location |
|---|---|---|
| 7930 | Homebrew snapshot (Feb 2026) | `~/.local/llama-cpp/7930/bin/` |
| b8533 | Source build (Mar 2026) | `~/.local/llama-cpp/b8533/bin/` |

## Benchmark

```bash
# Run against a HuggingFace-hosted GGUF
uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
    --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf \
    --llama-build b8533

# Run against a local GGUF file
uv run python benchmark/run.py --local-file /path/to/model.gguf --llama-build b8533

# Run against GPT-4o-mini (requires OPENAI_API_KEY in .env)
uv run python benchmark/run.py --backend openai

# Run a single task by number (1-101)
uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
    --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf \
    --llama-build b8533 \
    --task 5

# Run two builds in parallel (use separate ports and terminals)
uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
    --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf \
    --llama-build 7930 --port 8080

uv run python benchmark/run.py \
    --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
    --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf \
    --llama-build b8533 --port 8081
```

Results are saved to `benchmark/results/`.

### Current benchmark results

| Model | Score | Notes |
|-------|-------|-------|
| gpt-4o-mini | 93/100 | Cloud API, reference ceiling |
| LFM2.5-1.2B-Instruct Q4_0 | 71/100 | Baseline |
| LFM2-350M Q8_0 | 28/100 | Baseline |
| LFM2-350M fine-tuned Q8_0 | 47/100 | +19 points, correct format |

## Generate synthetic data

```bash
# Dry run: show generation plan without calling the API
uv run python benchmark/datasets/generate.py --dry-run

# Generate 500 examples (default)
uv run python benchmark/datasets/generate.py

# Custom count and output path
uv run python benchmark/datasets/generate.py \
    --count 500 \
    --output benchmark/datasets/sft_data.jsonl
```

Output: `benchmark/datasets/sft_data.jsonl` (OpenAI format, gitignored).

## Fine-tune

### One-time setup

```bash
git clone https://github.com/Liquid4All/leap-finetune
cd leap-finetune && uv sync && cd -

huggingface-cli login
modal setup
```

### Step 1: Prepare and upload data

```bash
uv run --group finetune python finetune/prepare_data.py
```

HF dataset produced: `Paulescu/home-assistant-sft`

### Step 2: Train on Modal (run from `leap-finetune/` subdirectory)

```bash
cd leap-finetune

# LFM2-350M (5 epochs, batch 4, ~$1.50)
uv run leap-finetune ../finetune/configs/LFM2-350M.yaml
```

### Step 3: Download checkpoints (run from `leap-finetune/` subdirectory)

```bash
# Check actual volume paths first
uv run modal volume ls leap-finetune

uv run modal volume get leap-finetune /outputs/home-assistant-350M ../finetune/output/350M-lora
```

### Step 4: Merge, convert to GGUF, and push to HuggingFace

```bash
uv run --group export python finetune/export.py \
    --lora-path finetune/output/350M-lora \
    --output-path finetune/output/350M-merged \
    --push-to-hub \
    --llama-cpp-path ~/llama.cpp \
    --quant-type q8_0
```

The script prints the exact `--hf-repo` and `--hf-file` flags to use in the benchmark.
HF repo pattern: `<hf-username>/home-assistant-<model-name>-GGUF`

### Step 5: Re-run benchmark

```bash
# Use the flags printed by export.py. Pattern:
uv run python benchmark/run.py \
    --hf-repo <your-hf-username>/home-assistant-LFM2-350M-GGUF \
    --hf-file LFM2-350M-q8_0.gguf
```

## LFM2 tool-call format

`sft_data.jsonl` stores data in OpenAI format (`tool_calls` objects, `"content": null`).
`prepare_data.py` converts this to LFM2 text format before uploading:

```
<|tool_call_start|>[{"name": "toggle_lights", "arguments": {...}}]<|tool_call_end|>
```

The agent in `app/agent.py` has an `_extract_lfm2_tool_calls()` fallback that parses these text-format tool calls from the `content` field when `tool_calls` is null. This is needed because llama-server returns LFM2 tool calls as plain text content, not structured tool_calls.

## HuggingFace repos (Paulescu account)

| Repo | Contents |
|------|---------|
| `Paulescu/home-assistant-sft` | SFT data in LFM2 format |
| `Paulescu/home-assistant-LFM2-350M-GGUF` | Fine-tuned LFM2-350M GGUF |
| `Paulescu/home-assistant-finetune` | Trackio training dashboard |
