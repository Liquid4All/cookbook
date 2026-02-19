# /build-check

Full cross-language build, lint, and test cycle for the entire project.

**Usage:** `/build-check`

## Steps

Run these checks in order. Stop and report on first failure category, but try to complete all checks within a category before reporting.

### 1. Rust Backend

```bash
cd src-tauri
cargo clippy -- -D warnings
cargo test
cargo build
```

Report: compile status, clippy warnings (should be 0), test results.

### 2. TypeScript Frontend

```bash
npx tsc --noEmit
npx eslint src/ --ext .ts,.tsx
npx vitest run src/
```

Report: type check status, lint issues, test results.

### 3. TypeScript MCP Servers

For each TypeScript server (filesystem, calendar, email, task, data, audit, clipboard, system):

```bash
cd mcp-servers/<server>
npx tsc --noEmit
npx eslint src/ --ext .ts
npx vitest run tests/
```

Report per server: type check, lint, test results.

### 4. Python MCP Servers

For each Python server (document, ocr, knowledge, meeting, security):

```bash
cd mcp-servers/<server>
ruff check src/
mypy --strict src/
black --check --line-length=100 src/
pytest tests/
```

Report per server: ruff, mypy, black, test results.

### 5. Integration Tests (if model is available)

Check if Ollama or llama.cpp is running:
```bash
curl -s http://localhost:11434/api/tags > /dev/null 2>&1
```

If available:
```bash
npx vitest run tests/integration/
```

If not available, report: "⏭️ Integration tests skipped (no model running). Start Ollama with `ollama serve` and load Qwen2.5-32B."

### 6. Summary

```
=== Build Check Summary ===

Rust Backend:     ✅ build OK, 0 clippy warnings, 12/12 tests pass
TS Frontend:      ✅ types OK, 0 lint issues, 8/8 tests pass
TS MCP Servers:   ✅ 8/8 servers pass (filesystem ✅, calendar ✅, ...)
PY MCP Servers:   ⚠️ 4/5 servers pass (ocr ❌: 2 mypy errors)
Integration:      ⏭️ skipped (no model)

Overall: ⚠️ 1 issue to resolve
```
