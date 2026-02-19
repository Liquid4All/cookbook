---
name: tauri-dev
description: >
  Skill for Tauri 2.0 and Rust backend development in LocalCowork. Use when
  working on the Rust backend, Tauri IPC commands, frontend-backend communication,
  Tauri permissions/capabilities, the application shell, or the Agent Core modules
  (ConversationManager, ToolRouter, MCP Client, Inference Client, ContextWindowManager).
  MANDATORY TRIGGERS: "Tauri", "Rust backend", "IPC command", "tauri.conf.json",
  "Cargo.toml", "capabilities", "agent core", "tool router", "conversation manager",
  "inference client", "MCP client", "context window", or anything related to the
  desktop application shell or the Rust-side orchestration layer.
---

# Tauri 2.0 Development Skill

## Architecture Overview

LocalCowork uses Tauri 2.0 with a three-layer architecture. This skill covers
the Rust backend (middle layer) and its integration with both the React frontend
(top layer) and the MCP servers + inference backend (bottom layer).

```
React Frontend ←──Tauri IPC──→ Rust Backend ←──JSON-RPC/stdio──→ MCP Servers
                                    │
                                    └──OpenAI API──→ Local LLM (Ollama/llama.cpp)
```

## Key References

- `docs/architecture-decisions/001-tauri-over-electron.md` — why Tauri
- `docs/architecture-decisions/003-model-abstraction-layer.md` — the OpenAI API contract
- `docs/patterns/human-in-the-loop.md` — confirmation/undo flow
- `docs/patterns/context-window-management.md` — 32k token budget

## Rust Backend Modules

### agent_core/conversation.rs — ConversationManager

Manages conversation state, history, and persistence.

```rust
pub struct ConversationManager {
    db: SqlitePool,           // Conversation history stored in SQLite
    current_session: Session,
    context_manager: ContextWindowManager,
}

impl ConversationManager {
    /// Create a new conversation session
    pub async fn new_session(&mut self) -> Result<SessionId>;

    /// Add a user message and get the model's response
    pub async fn send_message(&mut self, message: &str) -> Result<MessageStream>;

    /// Get conversation history for context window
    pub fn get_history(&self, max_tokens: usize) -> Vec<Message>;

    /// Persist a message to SQLite
    async fn persist_message(&self, message: &Message) -> Result<()>;
}
```

### agent_core/tool_router.rs — ToolRouter

Routes model tool calls to the appropriate MCP server.

```rust
pub struct ToolRouter {
    mcp_client: MCPClient,
    audit_logger: AuditLogger,
}

impl ToolRouter {
    /// Process a tool call from the model
    pub async fn dispatch(&self, tool_call: ToolCall) -> Result<ToolResult> {
        // 1. Look up tool in the MCP registry
        // 2. Check if confirmation is required
        // 3. If confirmed (or not required): send JSON-RPC call to server
        // 4. Log to audit trail
        // 5. If undo supported: push to undo stack
        // 6. Return result
    }

    /// Check if a tool requires user confirmation
    fn requires_confirmation(&self, tool_name: &str) -> bool;

    /// Push to undo stack for reversible actions
    async fn push_undo(&self, tool_call: &ToolCall, result: &ToolResult) -> Result<()>;
}
```

### agent_core/context_window.rs — ContextWindowManager

Manages the 32k token budget. See `docs/patterns/context-window-management.md`.

```rust
pub struct ContextWindowManager {
    max_tokens: usize,       // 32,768
    tokenizer: Tokenizer,    // tiktoken-rs
    system_prompt: String,
    tool_definitions: String,
}

impl ContextWindowManager {
    /// Build the full prompt for the model
    pub fn build_prompt(
        &self,
        history: &[Message],
        active_context: Option<&str>,
    ) -> Result<Vec<ChatMessage>>;

    /// Count tokens for a string
    pub fn count_tokens(&self, text: &str) -> usize;

    /// Evict old messages when context is tight
    fn evict_oldest(&mut self, history: &mut Vec<Message>);
}
```

### mcp_client/ — MCP Client

Manages MCP server processes and JSON-RPC communication.

```rust
pub struct MCPClient {
    servers: HashMap<String, MCPServerProcess>,
    tool_registry: ToolRegistry,
}

impl MCPClient {
    /// Start all configured MCP servers
    pub async fn start_servers(&mut self, config: &MCPConfig) -> Result<()>;

    /// Get the aggregated tool definitions for the LLM
    pub fn get_tool_definitions(&self) -> Vec<ToolDefinition>;

    /// Send a tool call to the appropriate server
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value>;

    /// Gracefully shutdown all servers
    pub async fn shutdown(&mut self) -> Result<()>;
}

struct MCPServerProcess {
    child: Child,        // tokio::process::Child
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    tools: Vec<ToolDefinition>,
}
```

### inference/ — Inference Client

OpenAI-compatible API client for the local LLM.

```rust
pub struct InferenceClient {
    base_url: String,        // e.g., "http://localhost:11434/v1"
    model: String,           // e.g., "qwen2.5:32b-instruct"
    http_client: reqwest::Client,
}

impl InferenceClient {
    /// Send a chat completion request (streaming)
    pub async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<impl Stream<Item = StreamChunk>>;

    /// Parse tool calls from model response
    fn parse_tool_calls(response: &str) -> Result<Vec<ToolCall>>;
}
```

## Tauri IPC Commands

The frontend communicates with the Rust backend via Tauri commands.

```rust
// src-tauri/src/commands/chat.rs
#[tauri::command]
async fn send_message(
    state: tauri::State<'_, AppState>,
    message: String,
) -> Result<String, String> {
    let mut conv = state.conversation_manager.lock().await;
    let response = conv.send_message(&message).await
        .map_err(|e| e.to_string())?;
    Ok(response)
}

#[tauri::command]
async fn confirm_action(
    state: tauri::State<'_, AppState>,
    action_id: String,
    confirmed: bool,
) -> Result<(), String> {
    let router = state.tool_router.lock().await;
    if confirmed {
        router.execute_confirmed(&action_id).await.map_err(|e| e.to_string())?;
    } else {
        router.reject_action(&action_id).await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
async fn undo_last_action(
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let router = state.tool_router.lock().await;
    router.undo_last().await.map_err(|e| e.to_string())
}
```

Frontend invocation:
```typescript
import { invoke } from '@tauri-apps/api/core';

const response = await invoke<string>('send_message', { message: userInput });
await invoke('confirm_action', { actionId: 'act-001', confirmed: true });
await invoke('undo_last_action');
```

## Tauri Permissions (tauri.conf.json)

Each capability is granted explicitly:

```json
{
  "app": {
    "security": {
      "capabilities": [
        {
          "identifier": "filesystem-access",
          "description": "Access user-granted directories",
          "permissions": [
            "fs:allow-read",
            "fs:allow-write",
            "fs:scope-$DOCUMENTS",
            "fs:scope-$DOWNLOADS"
          ]
        },
        {
          "identifier": "process-management",
          "description": "Manage MCP server child processes",
          "permissions": [
            "shell:allow-spawn",
            "shell:allow-kill"
          ]
        },
        {
          "identifier": "clipboard-access",
          "permissions": ["clipboard-manager:allow-read", "clipboard-manager:allow-write"]
        }
      ]
    }
  }
}
```

## Coding Standards (Rust)

- Edition 2021
- `cargo clippy -- -D warnings` must pass (zero warnings)
- All public functions have doc comments (`///`)
- Error handling: `thiserror` for custom errors, `anyhow` for application errors
- Async: `tokio` runtime (multi-threaded)
- Max 300 lines per file — extract to submodules when approaching
- Use `tracing` crate for structured logging (integrates with shared Logger)
- No `unwrap()` in production code — use `?` operator or explicit error handling

## Dependencies (Cargo.toml)

Key crates:
```toml
[dependencies]
tauri = { version = "2", features = ["shell-open"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sqlx = { version = "0.7", features = ["runtime-tokio", "sqlite"] }
reqwest = { version = "0.12", features = ["json", "stream"] }
tiktoken-rs = "0.5"
thiserror = "1"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
```
