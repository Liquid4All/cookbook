# LocalCowork Quickstart Issues

Issues found while following the Quick Start section of README.md (tested on Linux, 2026-02-25).

| # | Step | Issue | Severity |
|---|------|-------|----------|
| 1 | Step 2 & 3 | `setup-dev.sh` + `start-model.sh` use old model filename `*-Preview-Q4_K_M.gguf` — README was fixed but scripts weren't | **Blocking** for fresh installs |
| 2 | Step 1 | `setup-dev.sh` runs `cargo check` before `npm run build`, so it always fails with a misleading "Cargo.toml may need setup" warning | Medium |
| 3 | Step 1 | Node.js 18 installed, project requires 20+. Setup script doesn't version-check, just spams EBADENGINE warnings | Medium |
| 4 | Step 1 | Git hooks silently not installed in monorepo layout (`.git` at repo root, not in `localcowork/`) | Low |
| 5 | `npm test` | UC-10 test seeds DB with old column names (`status`, `file_path`, `params_json`) — schema changed but test wasn't updated → `SQLITE_ERROR` | **Test blocker** |
| 6 | `npm test` | UC-3 fixture has 6 files, test asserts 3 — encrypted output files were added but assertion wasn't updated | **Test blocker** |
| 7 | Step 4 | `cargo tauri dev` fails with GDK errors on headless Linux (no `$DISPLAY`). Not documented in prerequisites | Env-specific |
| 8 | Step 3 | `start-model.sh` passes `--flash-attn` as a bare flag; newer llama.cpp requires an explicit value (`--flash-attn on`). Server exits immediately with a usage error | **Blocking** |

---

## Issue 1: `setup-dev.sh` and `start-model.sh` still reference old HuggingFace repo/filename

**Step:** Step 2 (Download LFM2-24B-A2B) and Step 3 (Start model server)

**Problem:** Commit 492a80c updated the repo name and filename in `README.md`, but the same stale values remain in two scripts:

| File | Line | Stale value | Correct value |
|------|------|-------------|---------------|
| `scripts/setup-dev.sh` | 190 | `LFM2-24B-A2B-Preview-Q4_K_M.gguf` | `LFM2-24B-A2B-Q4_K_M.gguf` |
| `scripts/setup-dev.sh` | 199, 202 | `LiquidAI/LFM2-24B-A2B-Preview` | `LiquidAI/LFM2-24B-A2B-GGUF` |
| `scripts/setup-dev.sh` | 203 | `LFM2-24B-A2B-Preview-Q4_K_M.gguf` | `LFM2-24B-A2B-Q4_K_M.gguf` |
| `scripts/start-model.sh` | 18 | `LFM2-24B-A2B-Preview-Q4_K_M.gguf` | `LFM2-24B-A2B-Q4_K_M.gguf` |
| `scripts/start-model.sh` | 93–95 | `LiquidAI/LFM2-24B-A2B-Preview` / `LFM2-24B-A2B-Preview-Q4_K_M.gguf` | `LiquidAI/LFM2-24B-A2B-GGUF` / `LFM2-24B-A2B-Q4_K_M.gguf` |

**Impact:** A fresh user following the README downloads `LFM2-24B-A2B-Q4_K_M.gguf` (correct name), but `setup-dev.sh` reports it as missing and shows wrong download instructions. `start-model.sh` will fail to find the file and exit early.

---

## Issue 2: `setup-dev.sh` runs `cargo check` before the frontend is built

**Step:** Step 1 (`./scripts/setup-dev.sh`)

**Problem:** `setup-dev.sh` calls `cargo check` inside `src-tauri/` before `npm run build` has been run. Tauri's proc macro validates that `frontendDist` (`../dist`) exists at compile time, so `cargo check` always fails on a fresh clone:

```
error: proc macro panicked
   --> src/lib.rs:622:14
    |
622 |         .run(tauri::generate_context!())
    |              ^^^^^^^^^^^^^^^^^^^^^^^^^^
    |
    = help: message: The `frontendDist` configuration is set to `"../dist"` but this path doesn't exist
```

The script logs `⚠️  Cargo check failed — Cargo.toml may need setup` and continues, but this makes it look like a Cargo.toml problem rather than a missing frontend build step.

**Fix:** Run `npm run build` before `cargo check` in the setup script, or change the `cargo check` step to `cargo check --no-default-features` / skip it entirely (Rust compilation happens as part of `cargo tauri dev` anyway).

---

## Issue 3: Node.js version mismatch (18 installed, 20+ required)

**Step:** Step 1 (`./scripts/setup-dev.sh`)

**Problem:** The project's `package.json` requires `node >= 20.0.0`. The system has Node.js 18.19.1. The setup script checks that `node` is present but does not validate the version. Every `npm install` step produces:

```
npm WARN EBADENGINE Unsupported engine {
  package: 'localcowork@0.1.0',
  required: { node: '>=20.0.0' },
  current: { node: 'v18.19.1', npm: '9.2.0' }
}
npm WARN EBADENGINE Unsupported engine {
  package: '@vitejs/plugin-react@5.1.4',
  required: { node: '^20.19.0 || >=22.12.0' },
  current: { node: 'v18.19.1', npm: '9.2.0' }
}
```

This happens for the root package and all 8 TypeScript MCP servers (9 times total).

**Fix:** Add a Node.js version check to `setup-dev.sh` after detecting `node`, e.g.:
```bash
NODE_MAJOR=$(node -e "process.stdout.write(process.versions.node.split('.')[0])")
if [ "$NODE_MAJOR" -lt 20 ]; then
  echo "❌ Node.js 20+ required, found $(node --version). Install from https://nodejs.org"
  exit 1
fi
```

---

## Issue 4: Git hooks not installed when running from a monorepo

**Step:** Step 1 (`./scripts/setup-dev.sh`)

**Problem:** `setup-dev.sh` checks for a `.git` directory in the current working directory (`localcowork/`). When the project lives inside a monorepo (e.g., `cookbook/examples/localcowork/`), the `.git` directory is at the repo root, not here. The script prints:

```
⚠️  Not a git repo yet — run 'git init' first, then 'git config core.hooksPath .git-hooks'
```

This is misleading — the repo exists, but the hook is never installed. The pre-push smoke-test gate described in the README will not run.

**Fix:** Use `git rev-parse --git-dir` instead of checking for `.git` in the current directory, then set `core.hooksPath` relative to the actual git directory.

---

## Issue 5: UC-10 integration test fails — schema mismatch in test seed data

**Step:** `npm test` (mentioned in the Tests section of the README)

**Problem:** `tests/integration/uc10_compliance_pack.test.ts` seeds an in-memory database using column names from an old schema:

```ts
INSERT INTO audit_log (session_id, timestamp, tool_name, status, file_path, params_json, duration_ms)
```

The current schema in `mcp-servers/audit/src/db.ts` uses different column names:

| Column in test | Actual column in schema |
|----------------|------------------------|
| `status` | `result_status` |
| `file_path` | _(not present)_ |
| `params_json` | _(not present — args stored as `arguments`)_ |
| `duration_ms` | `execution_time_ms` |

This causes a `SQLITE_ERROR` in `beforeAll` and the entire test suite for UC-10 is skipped.

---

## Issue 6: UC-3 integration test fails — fixture directory has extra files

**Step:** `npm test`

**Problem:** `tests/integration/uc3_security_steward.test.ts` asserts `files.length === 3` for `tests/fixtures/uc3/sample_files/`. But the directory contains 6 files:

```
clean_file.txt
has_api_key.enc       ← added after test was written
has_api_key.env
has_ssn.txt
has_ssn.txt.enc       ← added after test was written
has_ssn_encrypted.enc ← added after test was written
```

The `.enc` encrypted output files were added to the fixture directory but the test expectation was not updated, causing an assertion failure: `expected 6 to be 3`.

---

## Issue 7: `cargo tauri dev` fails silently on headless Linux

**Step:** Step 4 (`cargo tauri dev`)

**Problem:** On Linux without an active display server (no `$DISPLAY` or `$WAYLAND_DISPLAY`), the Tauri window fails to open with GDK_CRITICAL errors:

```
(localcowork:PID): Gdk-CRITICAL **: gdk_monitor_get_scale_factor: assertion 'GDK_IS_MONITOR (monitor)' failed
```

The README's Prerequisites section lists macOS-oriented instructions (`brew install llama.cpp`) and mentions Linux only for Tesseract. It does not mention that a display server is required on Linux, or how to run the app with an existing display (e.g., `DISPLAY=:0 cargo tauri dev`).

**Note:** The Vite dev server and Rust compilation succeed; only the window display fails.

---

## Issue 8: `start-model.sh` passes `--flash-attn` without a value

**Step:** Step 3 (`./scripts/start-model.sh`)

**Problem:** The script launches `llama-server` with `--flash-attn` as a bare boolean flag. A newer version of llama.cpp changed this to a required-value argument (`--flash-attn [on|off|auto]`). The server exits immediately with:

```
error while handling argument "--flash-attn": expected value for argument

usage:
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
```

The health-check loop then times out after 60 seconds and the script exits with an error.

**Fix:** Change `--flash-attn` to `--flash-attn on` in `scripts/start-model.sh`.

---

## Test results summary (after `./scripts/setup-dev.sh`)

```
npm test:    591 passed, 1 failed (UC-3 fixture mismatch), 2 test files failed
             UC-10 fails entirely due to SQLITE_ERROR in beforeAll
```
