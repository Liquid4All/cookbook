# Cross-Platform Audit Report — LocalCowork v0.1.0

**Date**: 2026-02-14
**Scope**: Full codebase audit for Windows + macOS compatibility and App Store readiness
**Methodology**: Systematic review of Tauri config, Rust backend, React/TS frontend, MCP servers, onboarding flow, model changing flow, and assistant execution flow

---

## Executive Summary

LocalCowork is well-architected for macOS development but has **27 cross-platform issues** that must be resolved before Windows support and App Store submission. The most critical finding is an **architectural incompatibility** between MCP server child-process spawning and the Mac App Store sandbox. A distribution strategy decision is required before fixes can be prioritized.

**Severity breakdown**: 5 Critical, 9 High, 8 Medium, 5 Low

---

## Critical Issues (Store Submission Blockers)

### C-1: MCP Child Process Spawning Incompatible with Mac App Store Sandbox

**Files**: `src-tauri/src/mcp_client/lifecycle.rs` (line 94)

The entire MCP architecture spawns child processes via `tokio::process::Command::new()`. The Mac App Store requires App Sandbox entitlement (`com.apple.security.app-sandbox`), which **prohibits spawning arbitrary child processes** unless they are:
- XPC Services (macOS-only)
- Embedded helper apps signed with the same team ID
- Login items

All 13 MCP servers (TS via `npx tsx`, Python via `python -m`) are spawned as unrestricted child processes. This is a fundamental architectural conflict.

**Options**:
1. **Direct download + notarization** (skip Mac App Store) — minimal code changes, broad distribution via website
2. **Compile MCP tools into the Rust binary** — eliminate child processes entirely, major rearchitecture
3. **macOS XPC Services** — macOS-only solution, doesn't help Windows Store (MSIX)
4. **Distribute via Homebrew/winget** — developer-friendly, no store review

**Recommendation**: Option 1 (direct download + notarize) for initial release. This unblocks shipping. Revisit store submission as a v2 goal.

---

### C-2: Data Directory Bypasses Platform Conventions ✅ FIXED (session-027)

**Files**: `src-tauri/src/lib.rs` (lines 22-24, 48-50), `src-tauri/src/agent_core/permissions.rs`, `src-tauri/src/commands/model_download.rs`

All application data is stored at `~/.localcowork/` via `dirs::home_dir()`. This is:
- **Non-standard on macOS**: Should be `~/Library/Application Support/com.localcowork.app/`
- **Non-standard on Windows**: Should be `C:\Users\<user>\AppData\Roaming\localcowork\` (or `Local`)
- **Sandbox-incompatible**: Mac App Store sandbox restricts access to `~/Library/Containers/<bundle-id>/`

Tauri provides `app_data_dir()` which resolves to the correct platform-standard location automatically.

**Affected paths**:
| Current | Should Be |
|---------|-----------|
| `~/.localcowork/agent.db` | `{app_data_dir}/agent.db` |
| `~/.localcowork/agent.log` | `{app_data_dir}/agent.log` |
| `~/.localcowork/permissions.json` | `{app_data_dir}/permissions.json` |
| `~/.localcowork/models/` | `{app_data_dir}/models/` |

**Fix**: Replace all `dirs::home_dir().join(".localcowork")` calls with Tauri's `app_data_dir()`. This is a ~6 file change.

---

### C-3: System Prompt Hardcodes macOS Paths ✅ FIXED (session-026)

**File**: `src-tauri/src/commands/chat.rs` (lines 47, 69-77)

The `SYSTEM_PROMPT_RULES` constant contains hardcoded `/Users/alex/Documents/` paths in rules and examples. On Windows, these should be `C:\Users\alex\Documents\`. The model will generate macOS-style paths on Windows machines.

```
Rule 1: "ALWAYS use absolute paths (e.g. /Users/alex/Documents/file.png)"
Example 1: ocr.extract_text_from_image({"path": "/Users/alex/Documents/receipt.png"})
Example 2: filesystem.list_dir({"path": "/Users/alex/Documents"})
```

**Fix**: Make the system prompt dynamic — inject the actual home directory path at runtime. Replace `/Users/alex` with the real user's home dir detected via `dirs::home_dir()`.

---

### C-4: No macOS Code Signing / Entitlements Configuration ✅ FIXED (session-026)

**File**: `src-tauri/tauri.conf.json`

The `bundle` section lacks:
- `macOS.entitlements` — required for notarization and any App Store submission
- `macOS.signingIdentity` — required for code signing
- `macOS.provisioningProfile` — required for App Store

Without entitlements, the app cannot be notarized (even for direct distribution outside the App Store). macOS Gatekeeper will block unsigned apps.

**Fix**: Add a `src-tauri/entitlements.plist` with minimum entitlements, and reference it in `tauri.conf.json`:
```json
"bundle": {
  "macOS": {
    "entitlements": "entitlements.plist",
    "minimumSystemVersion": "12.0"
  }
}
```

---

### C-5: No Windows Code Signing Configuration ✅ FIXED (session-026)

**File**: `src-tauri/tauri.conf.json`

The `bundle` section has no Windows signing config:
- No `windows.certificateThumbprint`
- No `windows.digestAlgorithm`
- No `windows.wix` or `windows.nsis` installer configuration

Without signing, Windows SmartScreen will show a "Windows protected your PC" warning, and Windows Store submission is impossible.

**Fix**: Add Windows bundle configuration for NSIS installer and signing.

---

## High Issues (Will Fail on One Platform)

### H-1: PathBreadcrumb is macOS-Only ✅ FIXED (session-027)

**File**: `src/components/FileBrowser/PathBreadcrumb.tsx` (lines 25-42)

```typescript
const homeDir = path.match(/^\/Users\/[^/]+/)?.[0];  // Only matches macOS paths
const parts = displayPath.split("/").filter((p) => p.length > 0);  // Unix separator only
segments.push({ label, fullPath: `${prevPath}/${label}` });  // Forward-slash only
```

**Impact**: On Windows, the breadcrumb will not detect the home directory, will not split paths correctly (Windows uses `\`), and will generate invalid paths.

**Fix**: Create a shared `pathUtils.ts` that detects the platform and handles path operations correctly:
- Home dir detection: `/Users/<name>` (macOS) or `C:\Users\<name>` (Windows)
- Path splitting: handle both `/` and `\`
- Path joining: use platform-appropriate separator

---

### H-2: FileBrowser Navigate-Up is macOS-Only ✅ FIXED (session-027)

**File**: `src/components/FileBrowser/FileBrowser.tsx` (line 41)

```typescript
const parentIndex = rootPath.lastIndexOf("/");
```

**Impact**: On Windows, paths like `C:\Users\chintan\Documents` won't find the last separator because it searches for `/` not `\`.

**Fix**: Use the shared `pathUtils.ts` to find the parent path.

---

### H-3: Onboarding Default Directory is macOS-Only ✅ FIXED (session-027)

**File**: `src/stores/onboardingStore.ts` (line 160)

```typescript
workingDirectory: "~/Documents",
```

**Impact**: `~` is not expanded on Windows. The default working directory should be resolved at runtime from the Tauri API or detected from the OS.

**Fix**: Use Tauri's `documentDir()` from `@tauri-apps/api/path` to get the platform-correct documents directory.

---

### H-4: GPU Detection Returns `None` on Windows ✅ FIXED (session-027)

**File**: `src-tauri/src/commands/hardware.rs` (lines 71-80)

```rust
fn detect_gpu() -> Option<GpuInfo> {
    if is_apple_silicon() {
        return Some(GpuInfo { vendor: "Apple".to_string(), ... });
    }
    None  // Always None on Windows!
}
```

**Impact**: Windows machines with NVIDIA/AMD GPUs get no GPU info. The onboarding model recommendation step will not detect available VRAM and may recommend suboptimal quantization.

**Fix**: Use the `wgpu` crate or shell out to `nvidia-smi` / `wmic` for Windows GPU detection. At minimum, check for NVIDIA via `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`.

---

### H-5: `expand_tilde` Uses `$HOME` (Missing on Windows) ✅ FIXED (session-026)

**File**: `src-tauri/src/inference/config.rs` (lines 180-187)

```rust
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix('~') {
        if let Ok(home) = std::env::var("HOME") {  // HOME not set on Windows
            return format!("{home}{rest}");
        }
    }
    path.to_string()
}
```

**Impact**: On Windows, `$HOME` is typically not set. The correct env var is `USERPROFILE` (or use `dirs::home_dir()`). Any config paths using `~` will not expand on Windows.

**Fix**: Use `dirs::home_dir()` instead of `std::env::var("HOME")`, or check both `HOME` and `USERPROFILE`.

---

### H-6: MCP Server PATH Separator Hardcoded as `:` ✅ FIXED (session-027)

**Files**: 9 MCP server `index.ts` files + 3 Python `server.py` files

```typescript
// TypeScript servers
process.env.LOCALCOWORK_ALLOWED_PATHS.split(':')  // : is the macOS/Linux separator

# Python servers
allowed_paths = allowed_paths_str.split(":")  # Same issue
```

**Impact**: On Windows, `PATH` and path lists use `;` as the separator. Splitting on `:` will break paths like `C:\Users\...` by splitting at the colon after the drive letter.

**Fix**: Use `path.delimiter` (Node.js) or `os.pathsep` (Python) instead of hardcoded `:`.

Affected files:
- `mcp-servers/filesystem/src/index.ts`
- `mcp-servers/calendar/src/index.ts`
- `mcp-servers/email/src/index.ts`
- `mcp-servers/task/src/index.ts`
- `mcp-servers/data/src/index.ts`
- `mcp-servers/audit/src/index.ts`
- `mcp-servers/security/src/main.py`
- `mcp-servers/ocr/src/server.py`
- `mcp-servers/document/src/server.py`

---

### H-7: Hidden File Filter is Unix-Only ✅ FIXED (session-027)

**File**: `src-tauri/src/commands/filesystem.rs` (line 51)

```rust
if name.starts_with('.') { continue; }
```

**Impact**: On Windows, hidden files use the `FILE_ATTRIBUTE_HIDDEN` attribute, not a dot prefix. This filter will miss Windows hidden files and incorrectly hide files that start with `.` but aren't actually hidden on Windows.

**Fix**: On Windows, check the file attribute via `std::os::windows::fs::MetadataExt::file_attributes()` and test for `FILE_ATTRIBUTE_HIDDEN` (0x2).

---

### H-8: MCP Child Process Windows Console Suppression ✅ FIXED (session-027)

**Files**: 10 MCP server `index.ts` files

```typescript
process.on('SIGINT', () => { server.close(); process.exit(0); });
process.on('SIGTERM', () => { server.close(); process.exit(0); });
```

**Impact**: `SIGINT` and `SIGTERM` are Unix signals. On Windows, Node.js emulates `SIGINT` (via Ctrl+C) but `SIGTERM` is not supported and will be silently ignored. The `kill_process.ts` tool also sends Unix signals which won't work on Windows.

**Fix**: Add Windows-specific shutdown via `process.on('message', ...)` or `process.on('disconnect', ...)`. The `system` server's `kill_process` tool needs a Windows implementation using `taskkill.exe`.

---

### H-9: MCP Server DB Paths Default to `~/.localcowork/` ✅ FIXED (session-027)

**Files**: `mcp-servers/calendar/src/db.ts`, `email/src/db.ts`, `task/src/db.ts`, `audit/src/db.ts`, `filesystem/src/tools/delete_file.ts`, `knowledge/src/db.py`, `document/src/tools/create_pdf.py`

```typescript
path.join(os.homedir(), '.localcowork', 'calendar.db')
```

**Impact**: Same as C-2 — non-standard locations that bypass platform conventions. These should all use the same data directory as the main app.

**Fix**: Pass the resolved `app_data_dir` as an environment variable from the Tauri host to all MCP servers, and use it instead of `os.homedir() + '.localcowork'`.

---

## Medium Issues (Functionality Gaps)

### M-1: CSP is Disabled ✅ FIXED (session-026)

**File**: `src-tauri/tauri.conf.json` (line 24)

```json
"security": { "csp": null }
```

**Impact**: No Content Security Policy means the webview can load arbitrary external resources. Both app stores require a CSP. Even for direct distribution, this is a security concern.

**Fix**: Set a restrictive CSP:
```json
"csp": "default-src 'self'; connect-src 'self' http://localhost:11434; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:"
```

---

### M-2: `npx` Command May Need `.cmd` Extension on Windows ✅ FIXED (session-026)

**File**: `src-tauri/src/mcp_client/discovery.rs` (line 41)

```rust
fn ts_config(name: &str) -> ServerConfig {
    ServerConfig { command: "npx".to_string(), ... }
}
```

**Impact**: On Windows, `npx` is typically a batch script (`npx.cmd`). Using `Command::new("npx")` may fail unless the `.cmd` extension is added or the command is run through `cmd.exe /c npx`.

**Fix**: On Windows, use `npx.cmd` or wrap in `cmd.exe /c`:
```rust
if cfg!(target_os = "windows") { "npx.cmd" } else { "npx" }
```

---

### M-3: `python` Command May Not Exist on macOS 12.3+ ✅ FIXED (session-026)

**File**: `src-tauri/src/mcp_client/discovery.rs` (line 60)

```rust
fn py_config(name: &str, server_dir: &Path) -> ServerConfig {
    ServerConfig { command: "python".to_string(), ... }
}
```

**Impact**: Apple removed the default `python` symlink in macOS 12.3 Monterey. Only `python3` exists on fresh macOS installations. The `python_env.rs` already handles this correctly with `find_system_python()`, but the discovery module hardcodes `"python"`.

**Fix**: Use `python3` as the default command, or use the venv's python binary (which the path resolution in `lib.rs` already does correctly when a venv is present).

---

### M-4: Tauri Capabilities are Minimal ✅ FIXED (session-027)

**File**: `src-tauri/capabilities/default.json`

```json
"permissions": ["core:default", "shell:allow-open", "dialog:default"]
```

**Impact**: The app uses `tauri-plugin-shell` for process spawning but doesn't declare `shell:allow-execute` or `shell:allow-spawn`. It reads/writes the filesystem but doesn't declare filesystem permissions. This may cause runtime permission errors on some Tauri builds.

**Fix**: Declare all required capabilities explicitly.

---

### M-5: `tauri-plugin-dialog` Uses RC Version ✅ FIXED (session-026)

**File**: `src-tauri/Cargo.toml` (line 32)

```toml
tauri-plugin-dialog = "2.0.0-rc.8"
```

**Impact**: Release Candidate versions may have breaking changes or bugs. App stores may flag RC dependencies.

**Fix**: Upgrade to the stable `2.x` release.

---

### M-6: MCP Server Fallback PATH Uses `HOME` Env Var ✅ FIXED (session-027)

**Files**: 6 TypeScript MCP server index files

```typescript
: [process.env.HOME ?? '/tmp'];
```

**Impact**: On Windows, `HOME` may not be set. The fallback of `/tmp` doesn't exist on Windows. Should use `os.homedir()` which works cross-platform, and the temp fallback should use `os.tmpdir()`.

**Fix**: Replace with `[os.homedir()]` and `os.tmpdir()` fallback.

---

### M-7: No Windows Installer Configuration

**File**: `src-tauri/tauri.conf.json`

**Impact**: No NSIS or WiX configuration means `cargo tauri build` will produce a basic `.msi` with default settings. No Start Menu shortcut, no custom install directory, no license agreement display.

**Fix**: Add NSIS config for a polished Windows installer.

---

### M-8: Bundle Metadata Incomplete

**File**: `src-tauri/tauri.conf.json`

**Impact**: Missing `bundle.category`, `bundle.copyright`, `bundle.publisher`, `bundle.longDescription`. These are required for store submissions and recommended for direct distribution.

**Fix**: Add metadata to the bundle section.

---

## Low Issues (Polish)

### L-1: CSS Font Smoothing is macOS-Only

**File**: `src/styles.css`

```css
-webkit-font-smoothing: antialiased;
```

**Impact**: This property only works on WebKit (macOS). No visual bug on Windows, but no equivalent smoothing applied either. Windows uses ClearType by default.

**Fix**: No action needed, but consider adding `-moz-osx-font-smoothing: grayscale` for Firefox on macOS.

---

### L-2: Window Title Bar May Need Platform Tuning

**File**: `src-tauri/tauri.conf.json`

**Impact**: Tauri uses native window decorations by default. On macOS, the traffic lights (close/minimize/maximize) are on the left. On Windows, they're on the right. If any CSS uses `-webkit-app-region: drag` for a custom title bar, it may conflict with native decorations.

**Fix**: Test on both platforms. If using native decorations (current config), this is fine.

---

### L-3: `resolve_project_root()` Heuristic May Fail in Packaged App ✅ FIXED (session-027)

**File**: `src-tauri/src/lib.rs` (lines 103-120)

```rust
pub(crate) fn resolve_project_root() -> std::path::PathBuf {
    let cwd = std::env::current_dir().unwrap_or_default();
    if cwd.join("mcp-servers").is_dir() { cwd }
    else if cwd.join("..").join("mcp-servers").is_dir() { ... }
```

**Impact**: In a packaged app (`.app` bundle on macOS, installed NSIS on Windows), the CWD will not be the project root. The `mcp-servers/` directory won't be at the expected relative location.

**Fix**: Use `tauri::api::path::resource_dir()` to locate bundled resources, or embed the MCP servers directory path at build time.

---

### L-4: No Minimum Window Size Set

**File**: `src-tauri/tauri.conf.json`

**Impact**: Users can resize the window to very small sizes, breaking the layout.

**Fix**: Add `"minWidth": 800, "minHeight": 600` to the window config.

---

### L-5: Missing App Icon Sizes

**File**: `src-tauri/tauri.conf.json`

```json
"icon": ["icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png", "icons/icon.icns", "icons/icon.ico"]
```

**Impact**: Missing some sizes that Windows/macOS expect (16x16, 48x48, 256x256 for Windows; 512x512 and 1024x1024 for macOS Retina). May cause blurry icons in certain contexts.

**Fix**: Generate all standard sizes using `cargo tauri icon <source-image>`.

---

## What Works Well (Cross-Platform Strengths)

These areas are already well-handled:

1. **Python venv provisioning** (`python_env.rs`): Correctly handles `bin/` vs `Scripts/`, `python` vs `python.exe`, `pip` vs `pip.exe`
2. **PATH separator in venv resolution** (`lib.rs`): Uses `;` on Windows, `:` on macOS/Linux
3. **Ollama integration** (`ollama.rs`): HTTP-based, no platform-specific code
4. **OllamaInstallGuide component**: Detects platform and shows correct install instructions
5. **Tauri dialog plugin** for folder selection: Cross-platform by design
6. **Inference client** (`client.rs`): Pure HTTP, no platform dependencies
7. **SQLite via `rusqlite` with `bundled`**: Compiles SQLite from source, works everywhere
8. **Frontend state management** (Zustand stores): No platform dependencies

---

## Recommended Fix Priority

### Phase 1: Pre-Release ✅ COMPLETE (session-026, commit 081cf8d)
1. ~~**C-3**: Make system prompt paths dynamic~~ ✅
2. ~~**C-4**: Add macOS entitlements file~~ ✅
3. ~~**C-5**: Add Windows signing config placeholder~~ ✅
4. ~~**H-5**: Fix `expand_tilde` for Windows~~ ✅
5. ~~**M-1**: Set a proper CSP~~ ✅
6. ~~**M-5**: Upgrade `tauri-plugin-dialog` to stable~~ ✅

### Phase 2: Windows Support ✅ COMPLETE (session-027)
7. ~~**H-1, H-2**: Create `pathUtils.ts`, fix PathBreadcrumb and FileBrowser~~ ✅
8. ~~**H-3**: Fix onboarding default directory~~ ✅
9. ~~**H-6**: Fix PATH separator in all MCP servers~~ ✅
10. ~~**H-7**: Fix hidden file filter for Windows~~ ✅
11. ~~**M-2**: Fix `npx` → `npx.cmd` on Windows~~ ✅ (session-026)
12. ~~**M-3**: Fix `python` → `python3` default~~ ✅ (session-026)
13. ~~**M-6**: Fix HOME fallback in MCP servers~~ ✅

### Phase 3: Production Polish ✅ MOSTLY COMPLETE (session-027)
14. ~~**C-2, H-9**: Migrate all data dirs to `app_data_dir()`~~ ✅
15. ~~**H-4**: Add Windows GPU detection~~ ✅
16. ~~**H-8**: Add Windows CREATE_NO_WINDOW for MCP child processes~~ ✅
17. ~~**M-4**: Expand Tauri capabilities~~ ✅
18. **M-7, M-8**: Add installer config and bundle metadata — M-8 done (session-026), M-7 remaining
19. ~~**L-3**: Fix project root resolution for packaged apps~~ ✅
20. **L-4, L-5**: Window size and icon fixes — L-4 done (session-026), L-5 remaining

### Deferred: Resolved
- **C-1**: MCP child process architecture — resolved: direct download distribution, no App Store sandbox needed

---

## Distribution Strategy Recommendation

| Option | Mac App Store | Windows Store | Direct Download | Effort |
|--------|:---:|:---:|:---:|--------|
| A: Direct download + notarize | No | No | Yes | Low |
| B: Homebrew + winget | No | No | Yes (dev-friendly) | Low |
| C: App Store (requires rearchitecture) | Yes | Yes | Yes | Very High |
| D: Hybrid (direct + stores later) | Later | Later | Yes | Low now |

**Recommendation**: Start with **Option D** — ship via direct download with notarization (macOS) and basic code signing (Windows). Add Homebrew and winget formulae for developer distribution. Revisit store submission as a post-launch milestone when the architecture has stabilized.
