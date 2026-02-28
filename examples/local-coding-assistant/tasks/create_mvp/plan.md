# Local Coding Assistant - Implementation Plan

## Overview

This is a minimal agentic coding assistant built as a "Hello, World!" tutorial for software engineers new to agentic AI. The design mirrors Claude Code's architecture at a readable scale: a model-independent agentic loop, a small set of coding tools, simple context management, and a clean CLI interface.

---

## 1. Minimal Tool Set - Justification

Based on Claude Code's architecture, the tool categories that matter for a coding agent are: file operations, search, and execution. Web access is optional for an MVP.

The minimal viable tool set is:

| Tool | Category | Why it is essential |
|---|---|---|
| `read_file` | File ops | The agent must read source files to understand context before making changes |
| `write_file` | File ops | The agent must create and overwrite files to implement changes |
| `list_directory` | File ops / Search | Needed to explore project structure without reading every file |
| `run_bash` | Execution | Replaces grep, find, git, test runners — a single general-purpose execution tool covers all search and execution needs |

**Why `run_bash` unifies search and execution:** A coding agent needs to "run the test suite," "search for relevant files," and "use git." All of these are bash commands. One `run_bash` tool keeps the system simple while retaining full power.

**Why NOT to add more tools initially:**
- No web search: adds external dependencies and distraction for a tutorial
- No patch/diff tool: `write_file` + `read_file` is sufficient for the MVP
- No code intelligence: LSP integration is complex; `run_bash` covers linting/type-checking
- No subagents: out of scope for a tutorial

---

## 2. Project Module Structure

```
local-coding-assistant/
├── pyproject.toml                          # dependencies and entry point
├── .python-version                         # 3.13
├── README.md
└── src/
    └── local_coding_assistant/
        ├── __init__.py                     # package marker (already exists)
        ├── py.typed                        # PEP 561 marker (already exists)
        ├── main.py                         # CLI entry point
        ├── agent.py                        # the agentic loop
        ├── tools.py                        # all tool implementations
        ├── context.py                      # context/history management
        ├── config.py                       # configuration dataclass + loader
        └── llm/
            ├── __init__.py                 # exports get_llm_client()
            ├── base.py                     # LLMClient protocol + LLMResponse
            ├── anthropic_client.py         # Anthropic API backend
            └── llama_client.py             # llama.cpp / OpenAI-compat backend
```

Every file has one clear job.

---

## 3. Key Classes and Responsibilities

### `config.py` - Configuration

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Config:
    # Which backend to use
    backend: Literal["anthropic", "llama"] = "anthropic"

    # Anthropic settings
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""            # reads from ANTHROPIC_API_KEY env var

    # llama.cpp server settings
    llama_base_url: str = "http://localhost:8080/v1"
    llama_model: str = "local"
    llama_api_key: str = "sk-no-key"       # llama.cpp server ignores this

    # Agent behavior
    max_tokens: int = 8192
    max_context_messages: int = 40         # before compaction triggers
    working_directory: str = "."

def load_config() -> Config:
    """Load config from environment variables."""
    ...
```

### `llm/base.py` - Abstract Backend Protocol

```python
from typing import Protocol
from dataclasses import dataclass

class LLMClient(Protocol):
    """
    Minimal protocol that both Anthropic and llama.cpp backends implement.
    The agentic loop only talks to this interface.
    """
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> "LLMResponse": ...

@dataclass
class LLMResponse:
    stop_reason: str       # "end_turn" | "tool_use"
    content: list[dict]    # mix of text blocks and tool_use blocks
    input_tokens: int
    output_tokens: int
```

The key design decision: both backends normalize their responses into the same `LLMResponse` shape. The agentic loop in `agent.py` never imports `anthropic` or `openai` directly.

### `llm/anthropic_client.py` - Anthropic Backend

Translates the neutral tool schema (using `parameters`) into Anthropic's format (using `input_schema`), calls the Anthropic API, and normalizes the response into `LLMResponse`.

### `llm/llama_client.py` - llama.cpp Backend

Uses the `openai` Python library pointed at the local llama.cpp server (OpenAI-compatible REST API). Tool schemas use the OpenAI `parameters` format (no translation needed).

### `llm/__init__.py` - Factory

```python
def get_llm_client(config: Config) -> LLMClient:
    if config.backend == "anthropic":
        return AnthropicClient(config)
    elif config.backend == "llama":
        return LlamaClient(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")
```

This is the **only** place that imports backend-specific libraries.

### `tools.py` - Tool Implementations

```python
# Each tool is a plain Python function.
# TOOLS is the list of JSON Schema dicts sent to the model.
# TOOL_FUNCTIONS maps names to callables.

def read_file(path: str) -> str: ...
def write_file(path: str, content: str) -> str: ...
def list_directory(path: str = ".") -> str: ...
def run_bash(command: str, timeout: int = 30) -> str: ...

def execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call and return the string result."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"[error] Unknown tool: {name}"
    try:
        return fn(**inputs)
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
```

### `context.py` - Context Management

```python
class ContextManager:
    """
    Manages conversation history passed to the model.

    Simple strategy: when message count exceeds the limit,
    keep the first 2 messages (original task context) and the
    most recent N messages. Insert a notice where messages were dropped.
    """
    def __init__(self, max_messages: int = 40): ...
    def add(self, message: dict) -> None: ...
    def get_messages(self) -> list[dict]: ...
    def should_compact(self) -> bool: ...
    def compact(self) -> None: ...
```

### `agent.py` - The Agentic Loop

```python
class Agent:
    def __init__(self, llm: LLMClient, config: Config): ...

    def run_turn(self, user_input: str) -> None:
        """Process one user message, running the inner loop until end_turn."""
        self._context.add({"role": "user", "content": user_input})

        while True:
            if self._context.should_compact():
                self._context.compact()

            response = self._llm.chat(
                messages=self._context.get_messages(),
                tools=TOOLS,
                system=SYSTEM_PROMPT,
            )

            self._context.add({"role": "assistant", "content": response.content})

            tool_calls = [b for b in response.content if b["type"] == "tool_use"]

            if not tool_calls:
                # Print final text response and return to CLI
                for block in response.content:
                    if block["type"] == "text":
                        print(block["text"])
                break

            # Execute all tool calls, collect results
            tool_results = []
            for call in tool_calls:
                print(f"  [tool] {call['name']}({call['input']})")
                result = execute_tool(call["name"], call["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result,
                })

            # Feed results back as a user message and loop
            self._context.add({"role": "user", "content": tool_results})
```

### `main.py` - CLI Entry Point

Simple REPL using `input()` with `readline` for arrow-key history. Prints a banner with backend/model info. Handles `Ctrl+C` and `exit`/`quit` gracefully.

---

## 4. The Agentic Loop in Detail

```
User types a message
        │
        ▼
  Add to context
        │
        ▼
  should_compact? ──yes──► compact() ──►┐
        │ no                             │
        ◄────────────────────────────────┘
        │
        ▼
  Call LLM (AnthropicClient or LlamaClient)
        │
        ▼
  Add assistant response to context
        │
        ▼
  Response has tool_use blocks?
        │
     yes│                    no│
        ▼                      ▼
  Execute each tool      Print text response
  Print tool name              │
  Collect results              ▼
        │               Wait for next user input
        ▼
  Add tool_results as
  "user" message to context
        │
        └─────────────► (loop back to Call LLM)
```

**Key design decisions:**

1. **Tool results go back as a user message** — standard pattern for both Anthropic (`tool_result` blocks) and OpenAI (`tool` role message). Each backend normalizes this.
2. **The loop continues until `stop_reason == "end_turn"`** — the model decides when it's done.
3. **All tool calls in a response are executed before the next LLM call** — the model may request multiple tools in parallel; execute all, then send all results together.
4. **`KeyboardInterrupt` during `run_turn` returns control to the CLI prompt** — clean interruption at any time.

---

## 5. Model Switching via Configuration

```bash
# Use Anthropic (validation)
export LCA_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
lca

# Use local llama.cpp
export LCA_BACKEND=llama
export LCA_LLAMA_BASE_URL=http://localhost:8080/v1
lca
```

The agentic loop calls `self._llm.chat(...)` and never knows which backend is running.

---

## 6. Running the Local Model (llama.cpp)

```bash
# Start the llama.cpp server with LFM2-24B-A2B Q4_0
./llama-server \
  --model /path/to/LFM2-24B-A2B-Q4_0.gguf \
  --ctx-size 8192 \
  --port 8080
```

The `LlamaClient` uses the `openai` Python library with `base_url` pointing at the local server — no custom HTTP client needed.

---

## 7. Dependencies (`pyproject.toml`)

```toml
[project]
name = "local-coding-assistant"
version = "0.1.0"
description = "A minimal agentic coding assistant"
requires-python = ">=3.13"
dependencies = [
    "anthropic>=0.40.0",       # Anthropic API backend
    "openai>=1.50.0",           # llama.cpp backend (OpenAI-compatible)
    "python-dotenv>=1.0.0",     # Load .env for API keys
    "click>=8.0.0",             # CLI argument parsing
]

[project.scripts]
lca = "local_coding_assistant.main:main"
```

`rich` is optional — adds colored output for minimal extra code.

---

## 8. System Prompt

```
You are a local coding assistant running in a terminal.
You help users understand, create, and modify code.

You have access to these tools:
- read_file: read the contents of any file
- write_file: create or overwrite a file with new content
- list_directory: list files in a directory
- run_bash: run any shell command (git, grep, python, tests, etc.)

Guidelines:
- Before making changes, read the relevant files first
- After making changes, verify by reading the file back or running tests
- Use run_bash for searching (grep, find), running tests, and git operations
- Be concise — show your work through tool use, not long explanations
- When writing files, always write the complete file content, not just the changed parts
```

---

## 9. Step-by-Step Implementation Order

Each step produces something runnable.

| Step | Task | File(s) |
|---|---|---|
| 1 | Update `pyproject.toml`, run `uv sync` | `pyproject.toml` |
| 2 | Implement `Config` + `load_config()` | `config.py` |
| 3 | Implement all 4 tools + `execute_tool` | `tools.py` |
| 4 | Implement `AnthropicClient` + `LLMResponse` | `llm/base.py`, `llm/anthropic_client.py` |
| 5 | Implement `ContextManager` + `Agent.run_turn()` | `context.py`, `agent.py` |
| 6 | Implement CLI REPL | `main.py` |
| 7 | Validate with Anthropic on real coding tasks | — |
| 8 | Implement `LlamaClient` | `llm/llama_client.py` |
| 9 | Switch to local model, repeat validation | — |

**Suggested validation tasks (Step 7 & 9):**
- "List the files in this directory"
- "Read `pyproject.toml` and summarize it"
- "Create a file `hello.py` that prints Hello World and run it"
- "Find all `.py` files and count lines of code"

---

## 10. Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Tool set | 4 tools | Minimal but covers file ops, search, and execution |
| Search tool | `run_bash` | Avoids a separate search tool; bash covers grep/find/git |
| LLM abstraction | Protocol class + factory | Zero coupling; the loop never imports a backend library |
| OpenAI library for llama.cpp | Yes | llama.cpp server is OpenAI-compatible; no custom HTTP code |
| Context compaction | Simple head+tail truncation | Sufficient for a tutorial; avoids LLM-based summarization |
| CLI | `click` + `input()` REPL | Simple, supports flags |
| Config | Env vars + dataclass | Standard practice; no config file parsing complexity |
| Tool results | String only | Both backends accept string content; avoids serialization edge cases |
