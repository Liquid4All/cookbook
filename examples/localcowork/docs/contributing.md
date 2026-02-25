# Contributing to LocalCowork

LocalCowork is built by [Liquid AI](https://liquid.ai) and welcomes contributions from both team members and the open-source community. This guide covers how to get set up, what to work on, and how to submit changes.

## Getting Started

```bash
git clone <repo-url> && cd localCoWork
./scripts/setup-dev.sh
```

This installs all dependencies (Rust, Node.js, Python), creates a virtual environment, and configures git hooks. See the README for prerequisites.

If you plan to run model behavior tests locally, you'll also need [Ollama](https://ollama.ai) with a compatible model pulled (`ollama pull gpt-oss:20b`).

## Project Layout

LocalCowork has three layers — **Presentation** (Tauri + React), **Agent Core** (Rust), and **Inference** (OpenAI-compatible localhost API) — plus **13 MCP servers** that expose tools the model calls. The servers split across two languages: TypeScript for I/O-centric servers, Python for ML-heavy ones. See [ADR-002](architecture-decisions/002-mcp-server-language-selection.md) for why.

Key references before you start writing code:

- **`docs/PRD.md`** — the product spec. Every feature traces back to this document.
- **`docs/mcp-tool-registry.yaml`** — machine-readable definitions for all 68 tools.
- **`docs/patterns/mcp-server-pattern.md`** — canonical pattern every MCP server follows.
- **`docs/patterns/human-in-the-loop.md`** — the confirmation and undo flow.

## What to Work On

### For Liquid AI Team Members

Check `PROGRESS.yaml` at the repo root. It tracks every workstream (WS-0A through WS-6E), their status, and their dependencies. Pick a workstream whose status is `not_started` and whose dependencies are all `complete`. The `next_recommended` field in the latest session log entry is a good starting point.

### For Community Contributors

Good first contributions:

- **Bug fixes** — if you find a bug in an MCP server tool, fix it and add a regression test.
- **Test coverage** — several servers have room for more edge-case tests. Run `npm test` or `pytest` in a server directory to see current coverage.
- **Documentation** — improvements to doc clarity, fixing stale references, adding examples.
- **New smoke tests** — add `*.smoke.test.ts` or `*_smoke_test.py` files for tools that don't have them yet. See the naming convention below.

Larger contributions (new tools, new servers, architectural changes) should start with an issue or discussion before writing code.

## Development Workflow

### 1. Branch

```bash
git checkout -b <type>/<short-description>
# Examples:
#   feat/filesystem-watch-recursive
#   fix/ocr-confidence-threshold
#   test/calendar-free-slots-edge-cases
```

### 2. Write Code

Follow the standards for the language you're working in:

**TypeScript** — strict mode, zod for schemas, no `any` without a comment, prettier + eslint, max 300 lines per file.

**Python** — type hints on all public functions, pydantic for schemas, `mypy --strict`, `ruff check`, `black --line-length=100`, max 300 lines per file.

**Rust** — edition 2021, clippy clean (`-D warnings`), `thiserror` + `anyhow` for errors, doc comments on public functions, max 300 lines per file.

If you're adding or modifying an MCP server tool, the tool signature (params, returns, `confirmation_required`, `undo_supported`) must match `docs/mcp-tool-registry.yaml` exactly.

### 3. Write Tests

Every tool needs a **unit test** and a **smoke test**:

- Unit test: thorough — param validation, happy path, error paths, sandbox enforcement, metadata check.
- Smoke test: fast (<500ms) — can the tool be imported, does it accept valid params, does it reject garbage, does the metadata match? Naming: `*.smoke.test.ts` or `*_smoke_test.py`.

The smoke test runner auto-discovers tests by naming convention. No registration needed.

### 4. Run Checks Locally

```bash
# Smoke tests (what the pre-push hook runs)
./scripts/smoke-test.sh

# Full lint + type check
npx tsc --noEmit && npx eslint src/ mcp-servers/*/src/ --ext .ts,.tsx
source .venv/bin/activate && ruff check mcp-servers/*/src/ && mypy --strict mcp-servers/*/src/
cd src-tauri && cargo clippy -- -D warnings && cd ..

# Server-specific tests
cd mcp-servers/<server> && npm test    # TypeScript
cd mcp-servers/<server> && pytest      # Python
```

The pre-push git hook runs the smoke suite and blocks the push if anything fails. Don't bypass it with `--no-verify` unless you have a good reason.

### 5. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add recursive option to filesystem.watch_folder
fix: handle empty CSV in data.parse_csv
test: add edge cases for calendar.find_free_slots
refactor: extract validation helpers in security server
docs: clarify LFM2.5 tool-call normalization in ADR-003
chore: update Rust dependencies in Cargo.lock
```

Keep messages concise.

### 6. Open a Pull Request

- PR title: under 70 characters, same style as commit messages.
- PR body: describe *what* changed and *why*. Include test results if relevant.
- Tag relevant reviewers. For MCP server changes, tag someone who knows the server's language.

CI runs the full smoke suite and lint checks. PRs that fail CI won't be merged.

## Architecture Decisions

Any change to the following requires a new Architecture Decision Record (ADR) in `docs/architecture-decisions/`:

- Shared services interfaces (`_shared/services/`)
- MCP tool signatures (params, returns, confirmation, undo metadata)
- The OpenAI-compatible API contract
- Human-in-the-loop confirmation flow
- Audit log schema
- Context window management strategy

ADR format: see existing ADRs in that directory for the template (Status, Context, Decision, Rationale, Trade-offs).

## MCP Server Contribution Guide

Adding a tool to an existing server:

1. Read the tool's definition in `docs/mcp-tool-registry.yaml`.
2. Create one file in `mcp-servers/<server>/src/tools/<tool_name>.ts` (or `.py`).
3. Define typed params (zod for TS, pydantic for Python) matching the registry exactly.
4. Implement the logic. Use the shared Logger — never `console.log()` or `print()`.
5. Set `confirmationRequired` and `undoSupported` per the registry.
6. Write a unit test in `mcp-servers/<server>/tests/<tool_name>.test.ts` (or `_test.py`).
7. Write a smoke test: `<tool_name>.smoke.test.ts` (or `_smoke_test.py`).
8. Lint and type-check: `npx tsc --noEmit` or `mypy --strict src/`.
9. Run `./scripts/smoke-test.sh --server <server>` to verify.

Adding a new server is a larger effort — open an issue first to discuss scope and language selection.

## Code Review Expectations

What reviewers look for:

- **Correctness** — does it do what the PRD says?
- **Tests** — unit test + smoke test for every tool. Edge cases covered.
- **Types** — no untyped params, no `any`/`Any` without justification.
- **Error handling** — structured `MCPError`, no bare exceptions, no swallowed errors.
- **Security** — sandbox checks on all filesystem paths, no hardcoded secrets.
- **Logging** — shared Logger, not print/console.log.
- **Size** — files under 300 lines, commits focused on one concern.

## Reporting Issues

File an issue with:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS, model, and runtime versions (include output of `ollama --version` and `rustc --version` if relevant)

For security vulnerabilities, email security@liquid.ai instead of filing a public issue.

## License

Contributions are made under the project's MIT license. By submitting a PR, you agree that your contribution is licensed under the same terms.
