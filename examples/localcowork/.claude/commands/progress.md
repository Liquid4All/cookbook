# /progress

Show current development progress against the PRD phasing and workstream plan.

**Usage:** `/progress`

## Steps

### 1. MCP Server Status

For each of the 13 MCP servers, check:
- Directory exists in `mcp-servers/<name>/`
- Has `src/index.ts` or `src/__init__.py` (server entry point)
- Count of tool files in `src/tools/` vs tools defined in `docs/mcp-tool-registry.yaml`
- Unit tests exist and pass (run `npm test` or `pytest` — use `--quiet` flag)
- Any TODO markers remaining in source files

### 2. Agent Core Status

Check `src-tauri/src/` for:
- `agent_core/conversation.rs` exists and compiles
- `agent_core/tool_router.rs` exists and compiles
- `agent_core/context_window.rs` exists and compiles
- `agent_core/audit.rs` exists and compiles
- `mcp_client/` modules exist and compile
- `inference/client.rs` exists and compiles
- Run `cargo check` to verify

### 3. Frontend Status

Check `src/components/` for:
- Chat/ components (ChatPanel, MessageBubble, StreamingIndicator, ToolTrace)
- FileBrowser/ components (FileBrowser, FileTree, FilePreview)
- Confirmation/ components (ConfirmDialog, UndoBar, PreviewTable)
- Settings/ components (ModelSettings, MCPServerManager, PermissionsPanel)
- Stores and hooks implemented
- Run `npx tsc --noEmit` to verify

### 4. Integration Test Status

For each UC (1-10):
- Test file exists in `tests/integration/`
- Test fixtures available in `tests/fixtures/`
- Last test result (pass/fail/not run)

### 5. Phase Assessment

Based on the above, determine current phase:

| Phase | Criteria |
|-------|----------|
| Foundation | Repo exists, shared base classes done, Tauri shell launches |
| Core Servers | filesystem + document + ocr + data + audit servers passing |
| Agent Core | MCP Client + Inference Client + ConversationManager + ToolRouter built |
| Frontend | Chat UI + ToolTrace + FileBrowser + Confirmation working |
| Advanced Servers | knowledge + security + task + calendar + email servers passing |
| ML Servers | meeting + clipboard + system servers passing |
| Integration | UC integration tests passing, model behavior tests passing |

### 6. Report Format

```
═══════════════════════════════════════════════════
  LocalCowork Development Progress
  Date: 2026-02-12 | Phase: Core Servers (Weeks 2-4)
═══════════════════════════════════════════════════

MCP SERVERS                           Tools    Tests
──────────────────────────────────────────────────
  ✅ filesystem    (TypeScript)       9/9      ✅ pass
  🔨 document      (Python)          5/8      ⚠️ 3 stubs
  🔨 ocr           (Python)          2/4      ✅ pass
  ❌ knowledge     (Python)          0/5      — none
  ❌ meeting       (Python)          0/5      — none
  ❌ security      (Python)          0/6      — none
  ❌ calendar      (TypeScript)      0/4      — none
  ❌ email         (TypeScript)      0/5      — none
  🔨 task          (TypeScript)      3/5      ✅ pass
  ✅ data          (TypeScript)      5/5      ✅ pass
  ✅ audit         (TypeScript)      4/4      ✅ pass
  ❌ clipboard     (TypeScript)      0/3      — none
  ❌ system        (TypeScript)      0/5      — none

  Total: 28/68 tools (41%)  |  3/13 servers complete

AGENT CORE (Rust)
──────────────────────────────────────────────────
  ✅ MCP Client         compiles, tests pass
  🔨 Inference Client   compiles, streaming TODO
  🔨 ConversationManager compiles, eviction TODO
  ❌ ToolRouter         not started

FRONTEND (React)
──────────────────────────────────────────────────
  🔨 ChatPanel          functional, needs polish
  ❌ ToolTrace          not started
  ❌ FileBrowser        not started
  ❌ ConfirmDialog      not started
  ❌ Settings           not started

USE CASE INTEGRATION TESTS
──────────────────────────────────────────────────
  ❌ UC-1  Receipt Reconciliation    — servers incomplete
  ❌ UC-2  Contract Copilot          — servers incomplete
  ...
  ❌ UC-10 Compliance Pack           — servers incomplete

  Total: 0/10 passing

NEXT PRIORITIES
──────────────────────────────────────────────────
  1. Complete document server (3 remaining tools)
  2. Complete ocr server (2 remaining tools)
  3. Start ToolRouter in Agent Core
  4. Set up UC-1 integration test (first end-to-end validation)
```
