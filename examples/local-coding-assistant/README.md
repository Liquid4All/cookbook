# Local Coding Assistant

A minimal agentic coding assistant — a "Hello, World!" for AI agents.

It mirrors the core design of Claude Code at a readable scale: a model-independent agentic loop, a small set of coding tools, simple context management, and a CLI interface.

## How it works

The assistant runs a loop:

1. You type a request
2. The model decides which tools to call
3. Tools are executed and results are fed back to the model
4. The loop continues until the model has nothing left to do
5. The final response is printed and you can type the next request

The four tools available to the model are:

| Tool | What it does |
|---|---|
| `read_file` | Read the contents of a file |
| `write_file` | Create or overwrite a file |
| `list_directory` | List files in a directory |
| `run_bash` | Run any shell command (grep, git, python, tests, …) |

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/)

```bash
# Clone and enter the project
cd examples/local-coding-assistant

# Install dependencies
uv sync

# Copy the example env file and add your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

## Running

```bash
uv run lca
```

You'll see a prompt like this:

```
╔══════════════════════════════════════╗
║      Local Coding Assistant          ║
║  Type your request and press Enter.  ║
║  Ctrl+C or 'exit' to quit.           ║
╚══════════════════════════════════════╝

  Backend : anthropic
  Model   : claude-sonnet-4-6
  Work dir: .

>
```

Type your request and press Enter. Use `exit` or `Ctrl+C` to quit.

## Example interactions

```
> List the files in this directory

> Read pyproject.toml and summarize it

> Create a file hello.py that prints Hello World, then run it

> Find all .py files and count their lines of code

> What does the agent.py file do?
```

## Switching to a local model

The assistant supports [llama.cpp](https://github.com/ggerganov/llama.cpp) as a local backend via its OpenAI-compatible server.

Pass `--backend local` and `--model` to select the model. The llama-server is started and stopped automatically.

**From a HuggingFace repo** (downloaded and cached on first run):

```bash
uv run lca --backend local --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0
uv run lca --backend local --model unsloth/Qwen3.5-27B-GGUF:Q4_0
```

If the repo requires authentication, set `HF_TOKEN` in your environment or `.env` file.

**From a local GGUF file:**

```bash
uv run lca --backend local --model /path/to/model.gguf
```

**Non-interactive mode** works the same way:

```bash
uv run lca --backend local --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0 -p "What does agent.py do?"
```

If you prefer to manage the llama-server yourself, omit `--model` and the assistant will connect to the already-running server:

```bash
# terminal 1 — start server manually
llama-server -hf LiquidAI/LFM2-24B-A2B-GGUF:Q4_0 --port 8080

# terminal 2 — connect without auto-start
uv run lca --backend local
```

## Configuration

All settings are controlled via environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `LCA_BACKEND` | `anthropic` | Backend to use: `anthropic` or `local` |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `LCA_ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic model name |
| `LCA_LOCAL_BASE_URL` | `http://localhost:8080/v1` | llama.cpp server URL |
| `LCA_LOCAL_MODEL` | `local` | Model passed to the server (HF path or file path) |
| `LCA_LOCAL_CTX_SIZE` | `8192` | Context window size for the local server |
| `LCA_LOCAL_GPU_LAYERS` | `99` | Number of layers to offload to GPU |
| `LCA_MAX_TOKENS` | `8192` | Max tokens per response |
| `LCA_WORKING_DIR` | `.` | Working directory for bash commands |
| `HF_TOKEN` | — | HuggingFace token (required for gated models) |

CLI flags override env vars:

```bash
uv run lca --backend local --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0 --working-dir /path/to/my/project
```

## Benchmarking

The `benchmark/` directory contains two task suites, each with 10 tasks of increasing difficulty (easy → hard) and automated verifiers. Use them to compare how different models perform on real coding tasks.

### Default suite — this project

Tasks range from reading `pyproject.toml` to multi-file code analysis:

```bash
# Quick smoke test (tasks 1–3)
uv run python benchmark/run.py --backend anthropic --task 1,2,3

# Full benchmark
uv run python benchmark/run.py --backend anthropic

# Full benchmark against a local model
uv run python benchmark/run.py --backend local --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0
```

### llama.cpp suite — real-world C++ codebase

Tasks operate on a large open-source C++ project. Hard tasks require the agent to
write and run Python scripts that parse and analyse the codebase:

```bash
# Clone the target repo once
git clone https://github.com/ggerganov/llama.cpp /tmp/llama.cpp

# Quick smoke test (tasks 1–3)
uv run python benchmark/run.py --backend anthropic \
    --suite llamacpp --working-dir /tmp/llama.cpp --task 1,2,3

# Full benchmark
uv run python benchmark/run.py --backend anthropic \
    --suite llamacpp --working-dir /tmp/llama.cpp

# Full benchmark against a local model
uv run python benchmark/run.py --backend local \
    --model LiquidAI/LFM2-24B-A2B-GGUF:Q4_0 \
    --suite llamacpp --working-dir /tmp/llama.cpp
```

### Output

Results are saved to `benchmark/results/<timestamp>-<suite>-<backend>-<model>.json`
and a summary table is printed:

```
Model : claude-sonnet-4-6 (anthropic)
Date  : 2026-02-27 12:55

#    Task                                     Pass       Time       In/Out tokens  Turns
----------------------------------------------------------------------------------------
1    List directory                           ✓          4.8s            2191/267      2
...
10   Compare LLM backends                     ✓         47.5s          10594/2419      4
----------------------------------------------------------------------------------------

Score: 10/10  |  Total tokens: 75702  |  Avg time: 13.5s
```

### Flags

| Flag | Description |
|---|---|
| `--backend` | `anthropic` or `local` (required) |
| `--suite` | `default` (this project) or `llamacpp` (default: `default`) |
| `--model` | Override the model (Anthropic model ID or HF/GGUF path for local) |
| `--task` | Comma-separated task IDs to run, e.g. `1,2,3` |
| `--working-dir` | Directory the agent operates in (default: project root for `default` suite) |

## Project structure

```
src/local_coding_assistant/
├── main.py          # CLI entry point
├── agent.py         # The agentic loop
├── tools.py         # Tool implementations
├── context.py       # Conversation history + compaction
├── config.py        # Configuration
└── llm/
    ├── base.py              # LLMClient protocol
    ├── anthropic_client.py  # Anthropic backend
    ├── llama_client.py      # llama.cpp backend
    └── __init__.py          # Backend factory

benchmark/
├── run.py               # CLI runner (metrics, reporting, server management)
├── tasks.py             # Default suite: 10 tasks on this project
├── tasks_llamacpp.py    # llama.cpp suite: 10 tasks on a real-world C++ repo
└── results/             # Timestamped JSON output (gitignored)
```
