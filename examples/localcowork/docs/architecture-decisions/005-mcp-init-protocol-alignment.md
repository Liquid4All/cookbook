# ADR-005: MCP Initialization Protocol — Request-Response Alignment

## Status
Accepted

## Context
LocalCowork's Rust MCP client communicates with MCP server child processes via JSON-RPC 2.0 over stdio. During server startup, the client performs an initialization handshake to discover the server's tool definitions (names, parameter schemas, metadata).

Two server base classes exist — one in Python (`mcp-servers/_shared/py/mcp_base.py`) and one in TypeScript (`mcp-servers/_shared/ts/mcp-base.ts`). They share identical tool registration and execution patterns, but differed in how they handled the initialization handshake.

**The problem:** All 8 TypeScript MCP servers (filesystem, calendar, email, task, data, audit, clipboard, system) failed to start with error `-32601: Unknown method: initialize`. Only the Python servers (document, ocr, knowledge, meeting, security) initialized successfully.

**Root cause:** A protocol mismatch between the Rust client and TS server:

| Behavior | Rust Client (lifecycle.rs) | Python Server | TS Server (before fix) |
|----------|--------------------------|---------------|----------------------|
| Init pattern | Sends `{"method": "initialize", "id": 1}` and waits for response | Handles `initialize` in request dispatcher, responds with tool manifest | No `initialize` handler; returns `-32601` |
| Proactive message | None | None | Sends unsolicited `{"id": "init", ...}` on startup |
| ID type | Expects numeric `u64` | Responds with caller's numeric id | Sends string `"init"` (silently discarded by Rust transport) |

The TS server tried to be "helpful" by proactively pushing its tool manifest on startup, but used a string `id` that the Rust transport's JSON-RPC parser silently discarded (expecting `u64`). The actual `initialize` request from Rust then hit `handleRequest()` which had no case for it, returning method-not-found.

A second compounding issue: the TS base class's `zodToJsonSchema()` method was a stub that returned `{ type: 'object' }` for every tool — meaning even if a TS server started, the LLM would receive zero parameter information (no property names, types, descriptions, or required fields).

## Decision
Align the TypeScript MCP base class with the request-response protocol that the Rust client expects and that the Python base class already implements correctly.

### Changes to `mcp-servers/_shared/ts/mcp-base.ts`

1. **Add `initialize` handler** to `handleRequest()` — responds to `{"method": "initialize"}` with the server info and tool definitions, echoing back the caller's numeric `id`.

2. **Remove `sendInitialize()`** — delete the proactive push that used a string `id` and caused the protocol mismatch.

3. **Replace `zodToJsonSchema()` stub** — use the `zod-to-json-schema` library to convert zod schemas to full JSON Schema with properties, required fields, and descriptions.

4. **Extract `buildToolDefinitions()`** — consolidate the tool manifest construction (previously duplicated between `sendInitialize()` and `handleToolList()`) into a single private method.

## Rationale

### Why request-response and not proactive push?

The MCP specification (and LocalCowork's implementation in `lifecycle.rs`) uses a synchronous request-response handshake:

```
Client → Server:  {"jsonrpc": "2.0", "method": "initialize", "id": 1}
Server → Client:  {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": ..., "tools": [...]}}
```

This is the correct pattern because:
- It's deterministic — the client controls when initialization happens
- It fits JSON-RPC 2.0 semantics (requests have numeric IDs, notifications have no ID)
- It allows the client to enforce a timeout (10s in `lifecycle.rs`)
- Python servers already implement it correctly, proving the pattern works

The proactive push pattern is problematic because:
- The server sends data before the client asks for it, creating a race condition
- Using a string `id` violates the transport's parser expectations
- There's no guarantee the client is ready to receive when the server sends

### Why real JSON Schema matters

The `zodToJsonSchema()` stub returned `{ type: 'object' }` for every tool. This means the LLM received tool definitions like:

```json
{
  "name": "filesystem.move_file",
  "description": "Move or rename a file",
  "parameters": { "type": "object" }
}
```

With no `properties`, `required`, or `description` fields, the model had to guess parameter names. For a 20B local model (smaller than GPT-4), this is catastrophic — it either invents parameters or passes none.

With real schema conversion via `zod-to-json-schema`, the same tool now sends:

```json
{
  "name": "filesystem.move_file",
  "description": "Move or rename a file",
  "parameters": {
    "type": "object",
    "properties": {
      "source": { "type": "string", "description": "Source file path" },
      "destination": { "type": "string", "description": "Destination path" }
    },
    "required": ["source", "destination"]
  }
}
```

## Consequences

### Positive
- All 8 TypeScript MCP servers now initialize successfully
- The filesystem server's 9 tools (list_dir, read_file, write_file, move_file, copy_file, delete_file, search_files, get_metadata, watch_folder) are available to the agent
- The LLM receives full parameter schemas for all TS tools, enabling accurate tool calling
- Python and TypeScript servers now follow identical initialization protocols
- Future TS servers automatically inherit the correct behavior

### Negative
- New dependency: `zod-to-json-schema` added to `mcp-servers/_shared/ts/package.json`
- If a TS server relied on the proactive init push (none do currently), it would need updating

## Files Changed
- `mcp-servers/_shared/ts/mcp-base.ts` — protocol fix + schema conversion
- `mcp-servers/_shared/ts/package.json` — added `zod-to-json-schema` dependency
