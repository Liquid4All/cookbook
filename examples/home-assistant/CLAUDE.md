# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies**
```bash
uv sync
```

**Start the model server** (required before running the app or benchmarks)
```bash
llama-server \
  --hf-repo LiquidAI/LFM2.5-1.2B-Instruct-GGUF \
  --hf-file LFM2.5-1.2B-Instruct-Q4_0.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 99
```

**Start the app server**
```bash
uv run uvicorn app.server:app --port 5173 --reload
```

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
