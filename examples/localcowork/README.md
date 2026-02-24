# LocalCowork

**Tool-calling that actually feels instant on a laptop.**

Building a local AI agent sounds great until you try to use one all day. The hard part isn't getting a model to understand you -- it's getting it to choose the right tool and do it fast enough that the experience feels interactive. This is where [LFM2-24B-A2B](https://huggingface.co/LiquidAI/LFM2-24B-A2B-Preview) shines: it's designed for tool dispatch on consumer hardware, where latency and memory aren't abstract constraints -- they decide whether your agent is a product or a demo.

LocalCowork is a desktop AI agent that runs entirely on-device. No cloud APIs, no data leaving your machine. The model calls pre-built tools via the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), and every tool execution is logged to a local audit trail.

## What It Does

LocalCowork ships with **75 tools across 14 MCP servers** covering filesystem operations, document processing, OCR, security scanning, email drafting, task management, and more ([full tool registry](docs/mcp-tool-registry.yaml)). For the demo, we run a curated set of **20 tools across 6 servers** -- every tool scoring 80%+ single-step accuracy with proven multi-step chain participation.

### Demo 1: Scan for Leaked Secrets

Every developer has `.env` files with API keys scattered across old projects. You'd never upload your filesystem to a cloud model to find them. That defeats the purpose.

```
You:   "Scan my Projects folder for exposed API keys"
Agent: security.scan_for_secrets → found 3 secrets in 2 files (420ms)

You:   "Encrypt the ones you found"
Agent: security.encrypt_file → encrypted .env and config.yaml (380ms)

You:   "Show me the audit trail"
Agent: audit.get_tool_log → 3 tool calls, all succeeded (12ms)
```

Three tools, under 2 seconds total. The scan, encryption, and audit trail all happen locally.

### Demo 2: Compare Contracts Without a Cloud API

A freelancer gets a revised NDA. They need to know what changed. These are confidential documents that should never leave the machine.

```
You:   "Compare these two contract versions"
Agent: document.extract_text (v1) → 2,400 words (350ms)
       document.extract_text (v2) → 2,600 words (340ms)
       document.diff_documents → 12 changes found (180ms)
       document.create_pdf → diff_report.pdf generated (420ms)
```

Four tools, under 2 seconds. The extraction, diff, and PDF generation never touch a network.

### Demo 3: File Search

The simplest test of an agent: can it answer a direct question in one tool call without going off-script?

```
You:   "List what's in my Downloads folder"
Agent: filesystem.list_dir → 26 files found (9ms)
       "Here are the files: DEMO CARD styles.png, benchmark_results.csv,
        Liquid AI Notes.pdf, and 23 others."
```

One tool, one answer. No unnecessary follow-up calls, no asking what to do next.

## Architecture

```
Presentation    Tauri 2.0 (Rust) + React/TypeScript
                    |
Agent Core      Rust — ConversationManager, ToolRouter, MCP Client,
                       Orchestrator, ToolPreFilter, Audit
                    |
Inference       OpenAI-compatible API @ localhost (llama.cpp / Ollama / vLLM)
                    |
MCP Servers     14 servers, 75 tools (8 TypeScript + 6 Python)
```

The agent core communicates with the inference layer via the OpenAI chat completions API. Changing the model is a config change, not a code change. MCP servers are auto-discovered at startup by scanning `mcp-servers/`.

### MCP Servers

| Server | Lang | Tools | What It Does |
|--------|------|-------|-------------|
| **filesystem** | TS | 9 | File CRUD, search, watch (sandboxed) |
| **document** | Py | 8 | Text extraction, conversion, diff, PDF generation |
| **ocr** | Py | 4 | LFM Vision primary, Tesseract fallback |
| **knowledge** | Py | 5 | SQLite-vec RAG pipeline, semantic search |
| **meeting** | Py | 4 | Whisper.cpp transcription + diarization |
| **security** | Py | 6 | PII/secrets scanning + encryption |
| **calendar** | TS | 4 | .ics parsing + system calendar API |
| **email** | TS | 5 | MBOX/Maildir parsing + SMTP |
| **task** | TS | 5 | Local SQLite task database |
| **data** | TS | 5 | CSV + SQLite operations |
| **audit** | TS | 4 | Audit log reader + compliance reports |
| **clipboard** | TS | 3 | OS clipboard (Tauri bridge) |
| **system** | TS | 10 | OS APIs -- sysinfo, processes, screenshots |
| **screenshot-pipeline** | Py | 3 | Capture, UI elements, action suggestion |

### Human-in-the-Loop

Every tool execution is logged to a local audit trail. The confirmation system (ToolRouter, PermissionStore, frontend ConfirmationDialog) is built but not yet wired into the agent loop. Today, tools execute immediately after the model selects them. Integrating the confirmation flow is a future workstream.

Once live, write actions will show a preview and require confirmation. Destructive actions will require typed confirmation. That turns 80% model accuracy into near-100% effective accuracy: the user sees what the agent wants to do before it does it.

## Benchmarks

We tested 6 models against 67 tools on Apple M4 Max. LFM2-24B-A2B (24B total, ~2B active per token) delivers 80% tool accuracy at 390ms. That's 94% of the best dense model's accuracy at 3% of its latency.

| Model | Active Params | Accuracy | Latency | Multi-Step |
|-------|-------------|----------|---------|-----------|
| **LFM2-24B-A2B** | **~2B (MoE)** | **80%** | **390ms** | **26%** |
| Gemma 3 27B | 27B (dense) | 91% | 24,088ms | 48% |
| Mistral-Small-24B | 24B (dense) | 85% | 1,239ms | 66% |
| Qwen3 32B | 32B (dense) | ~70% | 28,385ms | -- |
| GPT-OSS-20B | ~3.6B (MoE) | 51% | 2,303ms | 0% |
| Qwen3-30B-A3B | ~3B (MoE) | 44% | 5,938ms | 4% |

The speed comes from the combination of the hybrid conv+attention design and MoE sparsity. Every model we tested fails at cross-server transitions. That's the universal barrier, not a model-specific gap. UX is designed around single-turn tool calls with human confirmation to compensate.

Full study with 8 models, 150+ scenarios, and 12 failure modes: [`docs/model-analysis/`](docs/model-analysis/).

## Quick Start

```bash
# 1. Clone and set up
git clone <repo-url> && cd localCoWork
./scripts/setup-dev.sh

# 2. Download LFM2-24B-A2B (~14 GB, requires HuggingFace access)
#    Request access: https://huggingface.co/LiquidAI/LFM2-24B-A2B-Preview
pip install huggingface-hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('LiquidAI/LFM2-24B-A2B-GGUF',
                'LFM2-24B-A2B-Q4_K_M.gguf',
                local_dir='$HOME/Projects/_models/')
"

# 3. Start the model server
./scripts/start-model.sh

# 4. Launch the app (in another terminal)
cargo tauri dev
```

MCP servers start automatically. The app auto-discovers them by scanning `mcp-servers/` at startup.

## Customizing the Tool Surface

Out of the box, the app starts **6 servers with 20 curated tools** -- the set that scores 80%+ accuracy. Two settings in `_models/config.yaml` control this:

```yaml
# Which servers to start (comment out to start ALL 14 servers)
enabled_servers:
  - filesystem
  - document
  - security
  - audit
  - system
  - clipboard

# Which tools the model sees (comment out to expose all ~75 tools)
enabled_tools:
  - filesystem.list_dir
  - filesystem.read_file
  - document.extract_text
  - security.scan_for_secrets
  # ... 20 tools total (see config.yaml for the full list)
```

Fewer tools means less context window usage and higher selection accuracy. The system prompt and tool definitions sent to the model update automatically when you change these lists -- no manual prompt editing required.

**Enabling all tools:**

Comment out both `enabled_servers` and `enabled_tools` in `config.yaml`. The app will start all 14 discovered servers and expose all ~75 tools to the model. You get the full capability set -- OCR, knowledge base RAG, meeting transcription, calendar, email, and more.

**Scaling up: the dual-model orchestrator:**

With 20 tools, a single model handles selection well. At 40+ tools, accuracy drops as the model has more options to confuse. The dual-model orchestrator ([ADR-009](docs/architecture-decisions/009-dual-model-orchestrator.md)) solves this with a plan-execute-synthesize pipeline:

1. **Plan** -- LFM2-24B-A2B decomposes the request into self-contained steps (no tool definitions sent, just natural language).
2. **Execute** -- A fine-tuned 1.2B router model selects one tool per step from a RAG pre-filtered set of K=15 candidates.
3. **Synthesize** -- LFM2-24B-A2B streams a user-facing summary from the accumulated results.

Enable it in `config.yaml`:

```yaml
orchestrator:
  enabled: true
  planner_model: lfm2-24b-a2b
  router_model: lfm25-1.2b-router-ft
  router_top_k: 15
```

Requires ~14.5 GB VRAM (planner ~13 GB + router ~1.5 GB). If orchestration fails at any phase, it falls back to the single-model loop automatically.

**Adding a tool to an existing server:**

1. Create a new tool file in the server's `src/tools/` directory (one tool per file).
2. Register it in the server's `src/index.ts` (TS) or `src/server.py` (Python).
3. Add your tool name to `enabled_tools` in `config.yaml`.
4. Restart the app.

**Adding a new MCP server:**

1. Create a directory under `mcp-servers/` with a `package.json` (TypeScript) or `pyproject.toml` (Python).
2. Implement tools following the pattern in [`docs/patterns/mcp-server-pattern.md`](docs/patterns/mcp-server-pattern.md).
3. Restart the app -- the server is auto-discovered.
4. Add your server to `enabled_servers` and your tools to `enabled_tools`.

Tool definitions live in [`docs/mcp-tool-registry.yaml`](docs/mcp-tool-registry.yaml) -- the machine-readable source of truth for all 75 tools.

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Node.js | 20+ | TypeScript MCP servers, React frontend |
| Python | 3.11+ | Python MCP servers (document, OCR, security, etc.) |
| Rust | 1.77+ | Tauri backend, Agent Core |
| llama.cpp | latest | Serves LFM2 models (`brew install llama.cpp`) |

Optional: [Ollama](https://ollama.ai) (alternative runtime), [Tesseract](https://github.com/tesseract-ocr/tesseract) (fallback OCR).

## Tests

853 tests across multiple tiers, all passing without a live model:

```bash
npm test                          # TypeScript server unit tests
source .venv/bin/activate && \
  for s in document ocr knowledge meeting security screenshot-pipeline; do
    (cd "mcp-servers/$s" && pytest); done   # Python server unit tests
npm run test:integration          # UC-1 through UC-10 end-to-end
npm run test:model-behavior       # 180 prompt-to-tool definitions
cd src-tauri && cargo test        # 357 Rust tests (agent core, MCP client, inference)
```

See the [testing section](docs/PRD.md) in the PRD for the full strategy.

## Project Structure

```
localCoWork/
+-- src-tauri/src/              Rust backend (Tauri 2.0)
|   +-- agent_core/             ConversationManager, ToolRouter, Audit
|   +-- mcp_client/             JSON-RPC stdio transport
|   +-- inference/              OpenAI-compat API client
|   +-- commands/               Tauri IPC (chat, session, settings)
+-- src/                        React + TypeScript frontend
+-- mcp-servers/                14 MCP servers (8 TS + 6 Python)
+-- tests/                      Unit, integration, model-behavior, cross-platform
+-- _models/                    Local model registry (binaries gitignored)
+-- docs/                       PRD, tool registry, ADRs, benchmarks
+-- scripts/                    setup-dev.sh, start-model.sh, smoke-test.sh
```

## Documentation

| Document | What's In It |
|----------|-------------|
| [`docs/PRD.md`](docs/PRD.md) | Full product requirements |
| [`docs/mcp-tool-registry.yaml`](docs/mcp-tool-registry.yaml) | Machine-readable definitions for all 75 tools |
| [`docs/model-analysis/`](docs/model-analysis/) | Benchmark study: 8 models, 67 tools, failure taxonomy |
| [`docs/demo/lfm2-24b-demo.md`](docs/demo/lfm2-24b-demo.md) | Demo workflows with exact prompts and expected tool calls |
| [`docs/architecture-decisions/`](docs/architecture-decisions/) | ADRs for orchestrator, pre-filter, sampling, etc. |
| [`docs/patterns/`](docs/patterns/) | Implementation patterns (MCP servers, HITL, error handling) |

## Known Limitations

- **1-2 step workflows are reliable.** 4+ step chains degrade as conversation history grows -- multi-step completion is 26% across all tools. The curated 20-tool demo set is selected for proven chain participation.
- **Batch operations process partial results** -- the model may handle 2 items from a set of 10. Iteration prompting is a known gap.
- **Cross-server transitions are the universal barrier.** Every model tested fails at these. UX is designed around human confirmation to compensate.

These limits are documented because they're instructive. See the [failure taxonomy](docs/model-analysis/) for all 12 failure modes with evidence.

## License

MIT
