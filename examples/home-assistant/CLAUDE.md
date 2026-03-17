# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies**
```bash
uv sync
```

**Start the app server**
```bash
uv run uvicorn app.server:app --port 5173 --reload
```

The app auto-starts `llama-server` in the background when you select a model in the UI.
No need to start `llama-server` manually before running the app.

**Run the full benchmark against the local model** (requires model server running on port 8080)
```bash
uv run python benchmark/run.py
```

**Run the full benchmark against OpenAI gpt-4o-mini** (requires `OPENAI_API_KEY` in `.env`)
```bash
uv run python benchmark/run.py --backend openai
```

**Run a single benchmark task**
```bash
uv run python benchmark/run.py --task <1-10>
uv run python benchmark/run.py --task <1-10> --backend openai
```

**Generate a golden SFT dataset** (requires `OPENAI_API_KEY` in `.env`)
```bash
# Full run: 20 runs x ~4 prompts x 15 tasks ≈ 1 200 examples
uv run python benchmark/generate_dataset.py

# Quick sanity check (1 run per prompt)
uv run python benchmark/generate_dataset.py --runs 1

# Use the local model instead
uv run python benchmark/generate_dataset.py --backend local
```

Output is a timestamped JSONL file in `benchmark/datasets/`. Each line is a verified,
correct conversation trace (system + user + tool calls + tool results + assistant reply)
paired with the full `TOOL_SCHEMAS`. Only runs that pass the task verifier are kept.
Paraphrases for all 15 tasks are defined in `benchmark/generate_dataset.py`.

**Run a full GPU fine-tune on HF Jobs** (requires `HF_TOKEN` secret, ~4h on a10g-small)
```bash
hf jobs run \
    --flavor a10g-small \
    --secrets HF_TOKEN \
    --timeout 4h \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    bash -c 'pip install -q huggingface_hub "trl>=0.15.0" datasets trackio unsloth && \
             python3 -c "import os; from huggingface_hub import hf_hub_download; \
             hf_hub_download(\"Paulescu/hf-cli-jobs-uv-run-scripts\", \"train.py\", \
             repo_type=\"dataset\", local_dir=\"/tmp/s\", token=os.environ.get(\"HF_TOKEN\",\"\"))" && \
             python3 /tmp/s/train.py \
               --dataset-repo Paulescu/home-assistant-sft \
               --model LiquidAI/LFM2.5-1.2B-Instruct \
               --output-repo Paulescu/LFM2.5-1.2B-home-assistant-sft'
```

Use `--max-steps 5` (without `--output-repo`) for a quick debug smoke-test.

Note: use `hf jobs run` (Docker), NOT `hf jobs uv run`. The `uv run` variant uses a plain
Debian image with no CUDA drivers regardless of `--flavor`, so GPU is never available.
Before running, ensure the latest `finetune/train.py` is uploaded to the scripts repo:
```bash
uv run --group finetune python3 -c "
from huggingface_hub import HfApi; import os; from dotenv import load_dotenv; load_dotenv()
HfApi(token=os.getenv('HF_TOKEN')).upload_file(
    path_or_fileobj='finetune/train.py', path_in_repo='train.py',
    repo_id='Paulescu/hf-cli-jobs-uv-run-scripts', repo_type='dataset')
"
```

## Architecture

```
Browser (index.html + assets/)
  POST /chat  -->  app/server.py  (FastAPI)
  GET  /state -->       |
                        v
                   app/agent.py        <-- also imported by benchmark/run.py
                        |
                   app/tools/schemas.py   (TOOL_SCHEMAS list)
                   app/tools/handlers.py  (TOOL_HANDLERS dict)
                   app/state.py           (home_state dict, in-memory)
                        |
                   OpenAI SDK (base_url=http://localhost:8080/v1)
                        |
                   llama-server (GGUF model, OpenAI-compatible API)
```

### Key design decisions

**`app/agent.py` is the core** and is shared verbatim between the demo server and the benchmark. Any change to the agent loop (system prompt, guards, tool dispatch) is immediately reflected in both benchmark scores and live demo behavior.

**Tool calling via OpenAI SDK.** `llama-server` handles LFM's special tokens internally and returns structured `message.tool_calls` objects. No custom token parsing is needed in Python.

**In-memory state.** `app/state.py` exports a single `home_state` dict that tool handlers mutate directly. State resets on server restart.

**Adding a new tool** requires two changes only:
1. Add a JSON schema entry to `TOOL_SCHEMAS` in `app/tools/schemas.py`
2. Add a handler function to `TOOL_HANDLERS` in `app/tools/handlers.py`

### Agent loop guards (in `app/agent.py`)

- `max_iter = 5`: prevents infinite loops
- `seen_calls` set: breaks early when the model repeats an identical `(tool, args)` pair, avoiding a spin-to-max-iter failure mode common with small models

## Findings

Document all noteworthy observations about LFM tool calling in `FINDINGS.md` at the project root. This is a standing instruction for all Claude Code sessions in this project.
