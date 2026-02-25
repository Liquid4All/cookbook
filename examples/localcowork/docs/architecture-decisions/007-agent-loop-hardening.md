# ADR-007: Agent Loop Hardening — Dynamic Budget, Result Capping, Error Recovery

## Status
Accepted

## Context
ADR-006 introduced three layers of reliability (context discipline, accurate estimation, behavioral rules). Production testing with the OCR-rename workflow revealed five additional failure modes that these layers did not address:

### Failure Mode 5: Tool Definition Budget is Static
`TOOL_DEFINITIONS_BUDGET` was hardcoded to 2,000 tokens. This was accurate when tool schemas were stubs (`{ type: 'object' }`). After ADR-005 introduced real JSON Schema via `zod-to-json-schema`, 15 tools with full property/description/required fields consume 5,000-8,000+ tokens. The conversation budget calculation thought it had ~20,000 tokens for history, but actually had ~14,000. Result: eviction triggered far too late, context overflowed silently.

### Failure Mode 6: Malformed Model JSON → Fatal Error
When the model generates syntactically broken JSON in tool call arguments (e.g., `"destination":""/Users/...` — double-quote), Ollama returns HTTP 500. The `is_retriable()` function only matched 502-504, so HTTP 500 was treated as a non-retriable error, which routed directly to the static fallback ("model not available"). The entire agent loop stopped on what should have been a recoverable transient error.

### Failure Mode 7: Unbounded Tool Result Size
One OCR extraction returned 32,613 bytes in a single tool result. This single result consumed more context than all previous rounds combined. With no cap on result size, a single verbose tool call could exhaust the entire remaining budget in one round.

### Failure Mode 8: No Tool Result Truncation
`extract_mcp_result_text()` passed results through verbatim. The built-in `read_file` tool truncated at 8,000 chars, but MCP tool results had no equivalent limit. Combined with Failure Mode 7, large results accumulated in conversation history with no safeguard.

### Failure Mode 9: Budget Blind to Tool Definitions
The `get_budget()` method used a fixed constant for tool definitions regardless of how many tools were actually registered. When tools were stub schemas (pre-ADR-005), the 2,000 estimate was reasonable. Post-ADR-005, the discrepancy between estimated (2,000) and actual (5,000-8,000) tokens meant eviction thresholds were never reached when they should have been.

## Decision
Address each failure mode with a targeted, minimal fix that strengthens the existing reliability layers from ADR-006.

### Fix 1: Dynamic Tool Definition Budget
Replace the static `TOOL_DEFINITIONS_BUDGET` constant with a measured value. `ConversationManager` gains a `tool_definitions_budget` field (defaults to 2,000) and a `set_tool_definitions_budget()` method. In `send_message()`, after building the tool list, serialize the tool definitions, estimate their token count via `estimate_tool_definitions_tokens()`, and set the real value on the manager. The `get_budget()` method now uses this measured value.

### Fix 2: HTTP 500 Retriable
Add `500` to the `is_retriable()` match in `InferenceClient`. Local model servers (Ollama, llama.cpp) return HTTP 500 for model-generated parse errors (malformed JSON in tool call arguments). This is a transient error — the same prompt with the same model may succeed on retry, and if not, the fallback chain provides the next model.

### Fix 3: Tool Result Size Cap
Add `MAX_TOOL_RESULT_CHARS = 6,000` constant. At ~2.8 chars/token (JSON), 6,000 chars ≈ 2,143 tokens — about 10% of the conversation budget. This prevents a single tool result from starving subsequent rounds.

### Fix 4: Tool Result Truncation
`execute_tool()` now runs all results through `truncate_tool_result()` before returning. Results exceeding `MAX_TOOL_RESULT_CHARS` are truncated with a notice: `[... truncated: showing first 6000 of N chars]`. This applies uniformly to built-in and MCP tools.

## Rationale

### Why measure tool tokens instead of increasing the constant?
Increasing `TOOL_DEFINITIONS_BUDGET` to 8,000 would work today but break tomorrow when servers are added or removed. Measuring at runtime is the correct approach because:
- It adapts automatically when MCP servers start or stop
- It handles the difference between 2 servers (13 tools) and 8 servers (40+ tools)
- It eliminates an entire class of estimation errors

### Why make HTTP 500 retriable?
The semantic meaning of HTTP 500 from a model server is different from a traditional web server. Ollama returns 500 when:
- The model generates invalid JSON that Ollama can't parse
- Memory allocation fails mid-generation
- The model hits an internal error during decoding

All of these are transient and benefit from retry or fallback. True permanent errors (wrong model name, unsupported API) return 400 or 404, which remain non-retriable.

### Why cap at 6,000 chars and not higher?
The 32K context window with 12,768 tokens of fixed overhead leaves ~20,000 tokens for conversation. At 2.8 chars/token, 6,000 chars ≈ 2,143 tokens — about 10% of the conversation budget. This means:
- A workflow with 10 tool results uses ~21,430 tokens for results alone, which is already near the budget
- With sequential processing (Rule 6 from ADR-006), older results get evicted before new ones arrive
- The cap is generous enough for most useful content (OCR of a receipt, file listing, structured data) while preventing pathological cases

### Why truncate in `execute_tool()` and not in `add_tool_result_message()`?
Truncating at the execution boundary means:
- The full result is still emitted to the frontend via `tool-result` event (the UI shows everything)
- Only the persisted conversation history is capped
- The truncation notice is visible in the conversation, so the model knows the result was truncated

## Consequences

### Positive
- Context budget is now accurate regardless of how many tools are registered
- Malformed model JSON no longer kills the entire agent loop
- No single tool result can exhaust the conversation budget
- The system self-adapts when MCP servers are added/removed — no manual constant tuning

### Negative
- `MAX_TOOL_RESULT_CHARS = 6,000` may truncate legitimately large results (e.g., a full-page OCR of a dense document). The truncation notice makes this visible, and the full result is shown in the UI.
- HTTP 500 retry adds latency when Ollama is genuinely broken (e.g., OOM). Mitigation: the fallback chain is limited to `remaining_fallbacks()` attempts, typically 1-2 retries before static fallback.
- Dynamic budget measurement adds a `serde_json::to_value()` serialization per `send_message()` call. This is negligible (~1ms) compared to LLM inference.

### Risks
- If `estimate_tool_definitions_tokens()` significantly overestimates, the conversation budget shrinks more than necessary. Mitigation: the JSON estimator (2.8 chars/token) is already calibrated for this content type.
- Tool result truncation could cause the model to make incorrect decisions based on incomplete data. Mitigation: the truncation notice explicitly tells the model data was cut, and Rule 9 (truthfulness) instructs it to acknowledge limitations.

## Files Changed
- `src-tauri/src/agent_core/conversation.rs` — dynamic `tool_definitions_budget` field + setter
- `src-tauri/src/commands/chat.rs` — `MAX_TOOL_RESULT_CHARS`, `truncate_tool_result()`, measured budget in `send_message()`
- `src-tauri/src/inference/client.rs` — HTTP 500 added to `is_retriable()`
