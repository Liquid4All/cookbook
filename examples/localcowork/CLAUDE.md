# CLAUDE.md — LocalCowork

This file provides guidance to Claude Code when working on the LocalCowork project.

## Project Overview

LocalCowork is a desktop AI agent that runs entirely on-device. It delivers a Claude Cowork-style experience powered by a locally-hosted LLM (dev: GPT-OSS-20B via Ollama, production: LFM2-24B-A2B via llama.cpp). The model calls pre-built tools via MCP — it never writes code. The user confirms every mutable action.

**Source of truth:** `docs/PRD.md` (the full product requirements document).
**Tool contract:** `docs/mcp-tool-registry.yaml` (machine-readable tool definitions extracted from the PRD Appendix).
**Progress checkpoint:** `PROGRESS.yaml` (session-to-session state — read this FIRST every session).

## Architecture (Three Layers)

```
Presentation → Tauri 2.0 (Rust) + React/TypeScript
Agent Core   → Rust backend: ConversationManager, ToolRouter, ContextWindowManager, MCP Client,
               Orchestrator (ADR-009), ToolPreFilter (ADR-010), ResponseAnalysis
Inference    → OpenAI-compatible API at localhost (llama.cpp / MLX / vLLM / LEAP SDK)
```

The Agent Core communicates with the inference layer exclusively via the OpenAI chat completions API. This is the model abstraction layer — changing the model is a config change, not a code change.

### Dual-Model Orchestrator (ADR-009)

When enabled, a plan-execute-synthesize pipeline runs two models cooperatively:

1. **Plan** — LFM2-24B-A2B decomposes requests into self-contained steps (bracket-format plans, no tool defs sent)
2. **Execute** — LFM2.5-1.2B-Router-FT-v2 selects one tool per step with RAG pre-filtered tools (K=15)
3. **Synthesize** — LFM2-24B-A2B streams a user-facing summary from accumulated results

The orchestrator is opt-in (`_models/config.yaml` → `orchestrator.enabled`). If it fails at any phase, control falls through to the single-model agent loop. Requires ~14.5 GB VRAM (planner ~13 GB + router ~1.5 GB). See `docs/architecture-decisions/009-dual-model-orchestrator.md`.

## Key Paths

```
docs/PRD.md                        # Product requirements (SOURCE OF TRUTH)
docs/mcp-tool-registry.yaml        # Tool specifications (machine-readable)
docs/architecture-decisions/       # ADRs for significant choices
docs/model-analysis/              # Model benchmarks and analysis (6 models, 67 tools)
docs/patterns/                     # Implementation patterns

src-tauri/src/                     # Rust backend
├── agent_core/                    # Conversation, routing, context, audit
│   ├── orchestrator.rs            # Dual-model plan-execute-synthesize (ADR-009)
│   ├── tool_prefilter.rs          # RAG embedding index for tool selection (ADR-010)
│   └── response_analysis.rs       # Deflection/incomplete/completion detection
├── mcp_client/                    # MCP client (JSON-RPC over stdio)
├── inference/                     # LLM API client (OpenAI-compat)
└── commands/                      # Tauri IPC commands

src/                               # React + TypeScript frontend
├── components/                    # Chat, FileBrowser, Confirmation, Settings
├── stores/                        # Zustand stores
├── hooks/                         # Custom hooks
└── types/                         # Shared TypeScript types

mcp-servers/                       # All 14 MCP server implementations
├── _shared/                       # Shared base classes (TS + Python)
├── filesystem/                    # TypeScript — file CRUD, watch, search
├── document/                      # Python — extraction, conversion, diff, PDF
├── ocr/                           # Python — LFM Vision + Tesseract fallback
├── knowledge/                     # Python — SQLite-vec RAG pipeline
├── meeting/                       # Python — Whisper.cpp + diarization
├── security/                      # Python — PII/secrets scan + encryption
├── screenshot-pipeline/           # Python — capture, OCR, action suggestion
├── calendar/                      # TypeScript — .ics + system calendar API
├── email/                         # TypeScript — MBOX/Maildir + SMTP
├── task/                          # TypeScript — local SQLite task DB
├── data/                          # TypeScript — CSV + SQLite operations
├── audit/                         # TypeScript — audit log reader + reports
├── clipboard/                     # TypeScript — OS clipboard (Tauri bridge)
└── system/                        # TypeScript — OS APIs (Tauri bridge)

tests/
├── unit/                          # Per-tool unit tests (mirrored in each server)
├── integration/                   # UC-1 through UC-10 end-to-end tests
├── model-behavior/                # Tool-calling accuracy + orchestrator benchmarks
│   ├── benchmark-lfm.ts           # Single-step tool selection (with RAG pre-filter)
│   ├── benchmark-multi-step.ts    # Multi-step chain completion
│   └── benchmark-orchestrator.ts  # Dual-model orchestrator end-to-end
└── fixtures/                      # Sample receipts, contracts, audio, etc.

_shared/services/                  # Shared services (model-gateway, logger, etc.)
_models/                           # Local model files (gitignored)
scripts/                           # Dev tooling scripts
```

## Shared Services Contract

All components must use these shared services — never implement alternatives:

| Service | Import (Python) | Import (TypeScript) | Purpose |
|---------|----------------|--------------------| --------|
| Model Gateway | `from _shared.services.model_gateway import ModelGateway` | `import { ModelGateway } from '@shared/model-gateway'` | All LLM calls. Streaming, fallback chains, timeout. |
| State Manager | `from _shared.services.state_manager import StateManager` | `import { StateManager } from '@shared/state-manager'` | Checkpointing. Atomic writes, scoped restore. |
| Logger | `from _shared.services.logger import Logger` | `import { Logger } from '@shared/logger'` | Structured JSON logging. No `print()` or `console.log()` ever. |
| Config Loader | `from _shared.services.config_loader import ConfigLoader` | `import { ConfigLoader } from '@shared/config-loader'` | YAML configs with env var interpolation. |

## MCP Server Development

Every MCP server follows the canonical pattern in `docs/patterns/mcp-server-pattern.md`.

### Creating a new server
```bash
/new-mcp-server <server-name> <language: ts|py>
```

### Server requirements
1. **Declare tools at startup** — JSON-RPC `initialize` response lists all tools with params, returns, and metadata.
2. **One tool per file** — e.g., `mcp-servers/filesystem/src/tools/list_dir.ts`.
3. **Typed params and returns** — zod schemas for TS, pydantic models for Python. Must match `docs/mcp-tool-registry.yaml` exactly.
4. **Confirmation metadata** — each tool declares `confirmation_required: boolean` and `undo_supported: boolean` per the registry.
5. **Structured errors** — return MCP-standard error objects, never throw raw exceptions.
6. **Unit test per tool** — every tool has a test verifying the input/output contract.

### Validating a server
```bash
/validate-server <server-name>
```

### Language selection rationale

| Server | Language | Why |
|--------|----------|-----|
| filesystem, calendar, email, task, data, audit, clipboard, system | TypeScript | I/O-centric, Node.js libs, Tauri bridge |
| document, ocr, knowledge, meeting, security | Python | ML libs (PaddleOCR, Whisper, sentence-transformers, pyannote) |

## Coding Standards

### Rust (src-tauri/)
- Edition 2021. Clippy clean: `cargo clippy -- -D warnings`.
- All public functions have doc comments.
- Error handling: `thiserror` for custom errors, `anyhow` for application errors.
- Async runtime: `tokio`.
- Max 300 lines per file. Extract to submodules when approaching.

### TypeScript (src/ + mcp-servers/*/src/)
- Strict mode: `"strict": true` in all tsconfig files.
- No `any` types without a justification comment.
- `interface` over `type` for object shapes.
- Formatting: prettier. Linting: eslint with recommended + typescript rules.
- Max 300 lines per file.

### Python (mcp-servers/*/src/)
- Python 3.11+. Type hints on all public functions.
- `mypy --strict`. `ruff check`. `black --line-length=100`.
- No `Any` without justification. No bare `except:`.
- Max 300 lines per file.

### All Languages
- Composition over inheritance.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- Test coverage: 80% minimum (85% for `_shared/` and `agent_core/`).
- No hardcoded paths — use config-loader and environment variables.
- No `TODO` without a linked issue or ADR reference.


## Human-in-the-Loop Pattern

Every tool execution follows: Intent → Plan → Preview → Confirm → Execute → Undo Option.

- **Non-destructive** (read, list, search, extract): execute immediately, show results.
- **Mutable** (rename, move, create, write): show preview, require confirmation.
- **Destructive** (delete, overwrite): explicit warning + typed confirmation.
- **All mutable/destructive**: push to undo stack (original state stored in SQLite).

See `docs/patterns/human-in-the-loop.md` for the full specification.

## Context Window Management

The LLM has a 32k token context window. Budget allocation:

| Component | Token Budget | Notes |
|-----------|-------------|-------|
| System prompt + persona | ~500 | Static, loaded once |
| Tool definitions (all 13 servers) | ~2000 | Auto-generated from registry |
| Conversation history | ~20000 | Rolling window with eviction |
| Active file/document content | ~9500 | Dynamically managed per request |

See `docs/patterns/context-window-management.md` for the eviction strategy and priority rules.

## Commands

### Build & Test
```bash
# Full build (Rust + React + MCP servers)
cargo tauri build

# Dev mode (hot reload frontend, restart Rust on change)
cargo tauri dev

# Smoke tests (fast regression suite — runs before every push)
./scripts/smoke-test.sh              # Full suite
./scripts/smoke-test.sh --contract   # Contract validation only
./scripts/smoke-test.sh --server filesystem  # One server only

# Validate all MCP servers against PRD spec
./scripts/validate-mcp-servers.sh

# Test specific server
cd mcp-servers/filesystem && npm test
cd mcp-servers/document && pytest

# Integration tests (requires running model via Ollama)
npm run test:integration

# Model behavior tests (requires running model via Ollama)
npm run test:model-behavior

# Lint everything
ruff check mcp-servers/*/src/ && mypy --strict mcp-servers/*/src/
npx tsc --noEmit && npx eslint src/ mcp-servers/*/src/
cargo clippy -- -D warnings

# Doc health (cross-references, staleness, code-doc drift)
./scripts/doc-health.sh              # All checks
./scripts/doc-health.sh --refs       # Cross-reference integrity only
./scripts/doc-health.sh --staleness  # Staleness detection only
./scripts/doc-health.sh --drift      # Code-doc drift only
./scripts/doc-health.sh --fix        # Show suggested fix commands
```

### Slash Commands
- `/new-mcp-server` — Scaffold a new MCP server from template
- `/validate-server` — Validate server implementation against PRD tool registry
- `/add-smoke-test` — Scaffold smoke tests for a server or individual tool
- `/test-usecase` — Run a specific use case integration test (UC-1 through UC-10)
- `/build-check` — Full cross-language build + lint + test
- `/model-test` — Test tool-calling accuracy with local LLM
- `/progress` — Show development progress across all workstreams
- `/session-end` — Checkpoint current session: update PROGRESS.yaml, doc-sync audit, commit, summarize

### Skills
- `mcp-server-dev` — MCP server development pattern and checklist
- `tauri-dev` — Tauri 2.0 + Rust backend development
- `tool-chain-test` — Integration tests composing multiple MCP tools (UC-1 through UC-10)
- `feature-dev` — Doc-sync checklist for any feature change (read before coding, verify after)

## Testing Strategy

Four tiers, each building on the last:

1. **Smoke tests (pre-push gate):** Fast (<30s) regression checks run before every push. Auto-discovered by naming convention: `*.smoke.test.ts` / `*_smoke_test.py`. Includes contract validation against the tool registry and server health checks. Run with `./scripts/smoke-test.sh`.

2. **Unit tests (per tool):** Input/output contract tests against `docs/mcp-tool-registry.yaml`. Mock filesystem for destructive operations. Snapshot tests for document generation. Run with `npm test` or `pytest` per server.

3. **Integration tests (per UC):** End-to-end use case tests composing multiple MCP tools in the PRD-defined sequence. One test file per use case: `tests/integration/uc<N>_*.test.ts`. Require all servers for that UC to be running.

4. **Model behavior tests:** 100 prompts → expected tool calls (selection accuracy). 50 multi-step scenarios → expected tool chains (completion rate). Edge cases: ambiguous requests, missing files, permission errors. Require a running local model (Qwen2.5-32B via Ollama for dev). Results stored in `tests/model-behavior/.results/` for regression tracking.

### Smoke Test Suite

The smoke test runner (`./scripts/smoke-test.sh`) has three phases:

- **Contract validation** — reads `docs/mcp-tool-registry.yaml` and checks every implemented tool file has a matching schema, correct metadata, and a test file. This grows automatically as servers are built.
- **Server health** — starts each implemented server and sends a JSON-RPC `initialize` request. Verifies the server responds with capabilities.
- **Per-tool smoke tests** — runs all `*.smoke.test.ts` and `*_smoke_test.py` files discovered across the codebase.

The pre-push git hook runs the smoke suite and blocks the push if any test fails. To scaffold smoke tests for a server, use `/add-smoke-test <server-name>`.

## Breaking Changes

Any change to the following requires an ADR in `docs/architecture-decisions/`:
- Shared services interfaces (`_shared/services/`)
- MCP server tool signatures (params, returns, confirmation, undo metadata)
- Model abstraction layer (the OpenAI-compatible API contract)
- Human-in-the-loop confirmation flow
- Audit log schema
- Context window management strategy

## Development Phases

| Phase | Weeks | Focus | Key Workstreams |
|-------|-------|-------|----------------|
| Foundation | 1 | Repo scaffold, shared base classes, Tauri shell | WS-0A through WS-0D |
| Core Servers | 2–4 | filesystem, document, ocr, data, audit servers | WS-1A through WS-1E |
| Agent Core | 3–5 | MCP Client, Inference Client, ConversationManager, ToolRouter | WS-2A through WS-2E |
| Frontend | 4–6 | Chat UI, ToolTrace, FileBrowser, Confirmation, Settings | WS-3A through WS-3E |
| Advanced Servers | 5–8 | knowledge, security, task, calendar, email servers | WS-4A through WS-4E |
| ML Servers | 7–10 | meeting (Whisper), clipboard, system, screenshot pipeline | WS-5A through WS-5D |
| Integration | 9–12 | UC tests, model behavior tests, model swap, cross-platform, onboarding | WS-6A through WS-6E |

See `PROGRESS.yaml` for detailed workstream statuses and dependencies.

## Session Protocol

Claude Code has no memory between sessions. `PROGRESS.yaml` is the persistent state file that bridges this gap. Follow this protocol **every session**.

### Session Start (MANDATORY — do this before any work)
1. **Read `PROGRESS.yaml`** — understand current phase, what's complete, what's next.
2. **Read any blockers** — check `blockers:` and `pending_decisions:` sections.
3. **Identify the target workstream** — pick the next `not_started` or `in_progress` workstream whose dependencies are `complete`. State it clearly to the user.
4. **Read the relevant skill** — load the appropriate `.claude/skills/*/SKILL.md` for the workstream type.

### During the Session
- Work on **one workstream at a time**. Finish or explicitly pause before switching.
- After each significant milestone (tool implemented, test passing, module complete), mentally note it for the checkpoint.
- If you hit a blocker, add it to `blockers:` in PROGRESS.yaml immediately — don't wait for session end.
- **Before any feature work**, read `.claude/skills/feature-dev/SKILL.md` for the doc-sync checklist. This tells you which docs to update for each type of change.

### Session End (MANDATORY — do this before the session closes)
Run `/session-end` or manually perform these steps:
1. **Update `PROGRESS.yaml`** — set workstream statuses, update `last_updated`, `last_session_id`, `current_phase`.
2. **Append a session log entry** — add to the `sessions:` list with: id, date, focus, completed items, artifacts created, next recommendation.
3. **Doc-sync audit** — verify registry, tests, pattern docs, and ADRs are in sync with code changes.
4. **Commit with a conventional commit** — `chore: checkpoint session-NNN — <summary>`.
4. **State what the next session should do** — this goes in the session log entry's `next_recommended` field.

### Session Compacting
When a session runs long and context is compacted, always preserve:
- List of files modified in the current session
- Current test results (pass/fail counts)
- Which MCP server is being worked on
- Which use cases are currently passing
- Current development phase (Foundation/Core/Agent Core/Frontend/Advanced/ML/Integration)
- Any open blockers or pending ADR decisions
- The current workstream ID (e.g., WS-1A)

### Key Paths for Session Management
```
PROGRESS.yaml                      # Persistent progress state (the "JIRA board")
.claude/commands/session-end.md    # Slash command to checkpoint a session
.git/hooks/pre-commit              # Guardrail: warns if PROGRESS.yaml not updated
```
