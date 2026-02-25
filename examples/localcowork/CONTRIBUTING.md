# Contributing to LocalCowork

Thanks for your interest in contributing. LocalCowork is a reference implementation for building local AI agents with MCP tools — contributions that improve the architecture, tools, or benchmark data are especially welcome.

## Quick Start

```bash
# 1. Fork and clone
git clone <your-fork-url> && cd localCoWork

# 2. Run setup (installs all deps, creates venvs, checks prereqs)
./scripts/setup-dev.sh

# 3. Verify everything works
./scripts/smoke-test.sh

# 4. Start the app in dev mode
cargo tauri dev
```

## What to Contribute

### Adding a tool to an existing MCP server

This is the most common and highest-impact contribution. Each tool lives in its own file:

1. Check `docs/mcp-tool-registry.yaml` for the full tool catalog and which server owns it
2. Create the tool file: `mcp-servers/<server>/src/tools/<tool_name>.ts` (or `.py`)
3. Define typed input/output schemas (zod for TypeScript, pydantic for Python)
4. Implement the tool handler
5. Set confirmation metadata: `confirmation_required` and `undo_supported`
6. Add a unit test: `mcp-servers/<server>/tests/<tool_name>.test.ts`
7. Register the tool in the server's tool list
8. Validate: `./scripts/smoke-test.sh --server <server>`

See [`docs/patterns/mcp-server-pattern.md`](docs/patterns/mcp-server-pattern.md) for the full specification.

### Adding a new MCP server

New servers should follow the canonical pattern:

- **TypeScript** for I/O-centric servers (filesystem, calendar, email, task, data, audit, clipboard, system)
- **Python** for ML-dependent servers (document, OCR, knowledge, meeting, security)

See [`docs/architecture-decisions/002-mcp-server-language-selection.md`](docs/architecture-decisions/002-mcp-server-language-selection.md) for the rationale.

### Improving model accuracy

The benchmark suite in `tests/model-behavior/` tests tool selection against all 67 tools. If you have access to a local model:

```bash
# Run with a live model (requires Ollama or llama-server)
LOCALCOWORK_MODEL_ENDPOINT=http://localhost:11434/v1 npm run test:model-behavior
```

Contributions that improve prompt engineering, tool filtering, or error recovery in the agent loop (`src-tauri/src/agent_core/`) are valuable. See [`docs/model-analysis/`](docs/model-analysis/) for the current benchmark data and failure taxonomy.

### Improving documentation

The model analysis directory (`docs/model-analysis/`) contains benchmark findings that are useful beyond this project. Contributions that add new model benchmarks, improve the failure taxonomy, or document workarounds are welcome.

## Code Standards

- **300 lines max** per file — split into modules when approaching the limit
- **Strict types everywhere** — `mypy --strict` (Python), `tsc --noEmit` with strict mode (TypeScript), `cargo clippy` (Rust)
- **80% test coverage** minimum (85% for `_shared/` and `agent_core/`)
- **Conventional commits** — `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
- **Structured JSON logging** — no `print()` or `console.log()`, use the shared Logger
- **No hardcoded paths** — use config-loader and environment variables

## Testing Before Submitting a PR

```bash
# TypeScript: type check + lint + tests
npx tsc --noEmit && npx eslint src/ mcp-servers/*/src/ --ext .ts,.tsx && npm test

# Python: lint + type check + tests (activate venv first)
source .venv/bin/activate
ruff check mcp-servers/*/src/ && mypy --strict mcp-servers/*/src/ && pytest

# Rust: lint + check
cd src-tauri && cargo clippy -- -D warnings && cargo check && cd ..

# Smoke tests (contract validation against tool registry)
./scripts/smoke-test.sh
```

## Architecture Decisions

Changes to shared interfaces, tool signatures, the agent loop, or the model abstraction layer require an Architecture Decision Record (ADR) in `docs/architecture-decisions/`. See existing ADRs for the format. This includes:

- MCP server tool signatures (params, returns, confirmation metadata)
- Shared services interfaces (`_shared/services/`)
- Human-in-the-loop confirmation flow
- Context window management strategy
- Audit log schema

## Human-in-the-Loop Pattern

Every tool execution follows: Intent → Plan → Preview → Confirm → Execute → Undo Option.

- **Non-destructive** (read, list, search, extract): execute immediately
- **Mutable** (rename, move, create, write): show preview, require confirmation
- **Destructive** (delete, overwrite): explicit warning + typed confirmation

See [`docs/patterns/human-in-the-loop.md`](docs/patterns/human-in-the-loop.md) for the full specification.

## Questions?

Open an issue. For questions about the model benchmarks or architecture decisions, reference the specific document in `docs/` — the maintainers can respond with context.
