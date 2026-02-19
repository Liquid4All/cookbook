# ADR-001: Tauri 2.0 over Electron

## Status
Accepted

## Context
LocalCowork needs a cross-platform desktop shell (macOS + Windows) that hosts a React frontend and manages native capabilities (filesystem access, child process management, system tray). The two primary options are Electron and Tauri.

The critical constraint is that LocalCowork bundles a 14GB local language model. The application shell must be as lightweight as possible to leave maximum memory for model inference.

## Decision
Use Tauri 2.0 with a Rust backend.

## Rationale

| Factor | Tauri 2.0 | Electron |
|--------|-----------|----------|
| Bundle size | < 30 MB | 150+ MB |
| RAM usage (shell only) | ~20 MB | ~100+ MB |
| Backend language | Rust | Node.js |
| Security model | Capability-based permissions | Unrestricted by default |
| IPC | Tauri commands (typed, fast) | IPC renderer (untyped by default) |
| Ecosystem maturity | Stable 2.0 (2024) | Very mature |

Three decisive factors:

1. **Memory footprint.** On a 32GB machine running a 14GB model, every MB of app overhead matters. Tauri's ~20MB shell vs Electron's ~100MB+ is significant — it's the difference between comfortable and swap-thrashing.

2. **Capability-based security.** Tauri's permission system maps naturally to LocalCowork's sandboxed MCP server model. Each server can be granted only the filesystem paths and system APIs it needs. Electron requires manual implementation of equivalent sandboxing.

3. **Rust backend performance.** The Agent Core (conversation manager, tool router, MCP client, context window manager) benefits from Rust's performance characteristics, especially for JSON-RPC message parsing and routing.

## Trade-offs
- Tauri's ecosystem is smaller than Electron's. Fewer third-party plugins.
- Rust learning curve for frontend developers contributing to the backend.
- Some native APIs (e.g., system tray behaviors) may require more platform-specific code in Tauri.

## Contingency
If Tauri 2.0 proves problematic for a specific platform, an Electron fallback can be built reusing the same React frontend. The Agent Core would need to be rewritten in Node.js/TypeScript.
