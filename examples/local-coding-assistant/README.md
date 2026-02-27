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

**Start the llama.cpp server:**

```bash
./llama-server \
  --model /path/to/LFM2-24B-A2B-Q4_0.gguf \
  --ctx-size 8192 \
  --port 8080
```

**Run the assistant against the local model:**

```bash
LCA_BACKEND=llama uv run lca
```

Or set it permanently in your `.env`:

```
LCA_BACKEND=llama
LCA_LLAMA_BASE_URL=http://localhost:8080/v1
```

## Configuration

All settings are controlled via environment variables (or a `.env` file):

| Variable | Default | Description |
|---|---|---|
| `LCA_BACKEND` | `anthropic` | Backend to use: `anthropic` or `llama` |
| `ANTHROPIC_API_KEY` | — | Your Anthropic API key |
| `LCA_ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic model name |
| `LCA_LLAMA_BASE_URL` | `http://localhost:8080/v1` | llama.cpp server URL |
| `LCA_LLAMA_MODEL` | `local` | Model name reported by the server |
| `LCA_MAX_TOKENS` | `8192` | Max tokens per response |
| `LCA_WORKING_DIR` | `.` | Working directory for bash commands |

CLI flags override env vars:

```bash
uv run lca --backend llama --working-dir /path/to/my/project
```

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
```
