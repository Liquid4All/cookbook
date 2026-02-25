# ADR-006: Agent Loop Reliability — Context Discipline and Behavioral Strategy

## Status
Accepted

## Context
LocalCowork's agent loop (`src-tauri/src/commands/chat.rs`) implements the core agentic cycle: receive user message, call LLM with tools, execute tool calls, feed results back, repeat until the model produces a text response. This loop powers every use case (UC-1 through UC-10).

Testing revealed three failure modes that compound to produce unreliable behavior across any multi-step workflow:

### Failure Mode 1: Unbounded Context Growth
The agent loop called `evict_if_needed()` once before the loop started, then accumulated messages across 7+ rounds (14+ messages) with no eviction. On a 32K token context window, this leads to silent quality degradation — the model produces progressively worse outputs, then goes empty (0 text, 0 tool calls).

### Failure Mode 2: Optimistic Token Estimation
Token estimation used 4.0 chars/token for all content. JSON/structured content (tool arguments, tool results, schema definitions) actually tokenizes at ~2.8-3.2 chars/token due to punctuation density. The ~25% underestimation meant the eviction threshold was never triggered when it should have been.

### Failure Mode 3: Confabulation on Forced Summary
When the model produced consecutive empty responses (context overflow), the agent loop injected a summary prompt: "provide a concise summary of what was found." This unconstrained prompt invited the model to fabricate results it never received — reporting file names, OCR extractions, and rename operations that never happened.

### Failure Mode 4: No Workflow Strategy
The system prompt listed tools and basic rules but provided no guidance for multi-step workflows. A 20B local model (smaller than GPT-4/Claude) without explicit behavioral instructions would: re-list directories already listed, batch-read all files instead of processing sequentially, lose track of progress between rounds, and invent results.

## Decision
Implement three layers of reliability that each independently improve robustness and together make the agent loop reliable across all use cases.

### Layer 1: Context Window Discipline

**Mid-loop eviction** — after each round of tool executions, call `evict_if_needed()` before rebuilding messages for the next LLM call. This prevents unbounded context growth during multi-step workflows.

**Token budget gate** — at the top of each loop iteration, check remaining token budget. If below `MIN_ROUND_TOKEN_BUDGET` (1500 tokens), exit the loop early rather than risk context overflow. 1500 tokens accommodates a model response (~500 tokens) plus a tool result (~1000 tokens).

**Anti-confabulation forced summary** — when the model goes silent and a summary must be forced, the prompt now explicitly constrains the model:
- "ONLY report results you actually received from tool calls above"
- "If a file was not processed, say 'not processed'"
- "If no tool results are visible, say 'I was unable to complete the task'"

### Layer 2: Accurate Token Estimation

**Prose ratio** — `CHARS_PER_TOKEN` changed from 4.0 to 3.2. Conservative (overestimates), which is safer — underestimating causes silent context overflow.

**JSON-specific ratio** — new `JSON_CHARS_PER_TOKEN = 2.8` and `estimate_json_tokens()` function. Used for tool call arguments (always JSON), tool results (role == Tool), and tool definitions. This ~15% tighter estimation means eviction triggers earlier for JSON-heavy conversations (which is every multi-tool workflow).

**Role-aware estimation** — `estimate_message_tokens()` now dispatches to the appropriate estimator based on message role: prose estimator for User/Assistant/System content, JSON estimator for Tool content.

### Layer 3: Behavioral Strategy

Four rules added to the system prompt, each addressing a specific cross-use-case failure mode:

| Rule | Purpose | Failure it prevents |
|------|---------|-------------------|
| 6. Sequential Processing | Process one file completely before the next | Context overflow from batch-loading (UC-1, UC-3, UC-9) |
| 7. No Redundant Calls | Never call a tool with the same arguments twice | Wasted rounds (observed in every test run) |
| 8. Progress Tracking | State what you did and what's next after each file | Lost progress between rounds (UC-2, UC-4) |
| 9. Truthfulness | Only report results actually received from tools | Confabulation (defense-in-depth with anti-confabulation prompt) |

These rules are tool-agnostic and workflow-agnostic — they apply to any combination of tools and any use case. When new MCP servers are added (knowledge, meeting, security), these rules automatically govern the agent's behavior with them.

## Rationale

### Why mid-loop eviction instead of a fixed round limit?

A fixed round limit (e.g., MAX_TOOL_ROUNDS = 10) is a blunt instrument. Some tasks legitimately need 15+ rounds (batch processing 10 files). Others overflow context in 5 rounds (if each tool result is large). Mid-loop eviction adapts to the actual content size — short results allow more rounds, large results trigger eviction sooner.

### Why a separate JSON token ratio?

Consider a tool call argument: `{"source": "/Users/chintan/Desktop/screenshot.png", "destination": "/Users/chintan/Desktop/meeting-notes.png"}`. At 4.0 chars/token, this estimates to 28 tokens. At 2.8 chars/token, it estimates to 40 tokens. Over 14 messages with tool calls and results, the cumulative error is ~3000 tokens — enough to overflow the conversation budget without triggering eviction.

### Why behavioral rules in the system prompt and not in code?

Code-level enforcement (e.g., deduplicating tool calls, forcing sequential execution) would be fragile and model-specific. The rules work as soft constraints via the system prompt because:
- They compose with any tool set without code changes
- They allow the model flexibility (e.g., the model can batch 2-3 small operations if appropriate)
- They're easy to tune as model capabilities improve
- They follow the same pattern used by Claude, ChatGPT, and other production agents

The anti-confabulation prompt is the one place where code-level enforcement matters — it's the last line of defense, executed when the model has already failed to produce normal output.

## Consequences

### Positive
- Multi-step workflows (UC-1 batch receipts, UC-3 download triage, UC-9 data analysis) now process files sequentially without context overflow
- The model no longer fabricates results when it runs out of context
- Token budget is tracked accurately for JSON-heavy conversations
- The agent loop self-regulates — it exits gracefully when running low on context instead of degrading silently

### Negative
- Slightly more conservative token estimation means eviction may happen earlier than strictly necessary, potentially summarizing turns that could have been kept
- `MIN_ROUND_TOKEN_BUDGET = 1500` is a heuristic — some tool results may exceed 1000 tokens, but the budget gate prevents the model from even attempting the call
- The behavioral rules add ~400 characters to the system prompt (~125 tokens), reducing the conversation budget slightly

### Risks
- If a model is instruction-tuned to ignore system prompt rules (unlikely for local models we control), the behavioral layer degrades. Mitigation: the code-level layers (eviction, budget gate, anti-confabulation) still protect against the worst outcomes.
- The 3.2 chars/token ratio may be too aggressive for models with larger vocabularies. Mitigation: this is tunable via the `CHARS_PER_TOKEN` constant, and can be replaced with `tiktoken-rs` for exact counts when the model is finalized.

## Files Changed
- `src-tauri/src/agent_core/tokens.rs` — dual-ratio token estimation
- `src-tauri/src/commands/chat.rs` — budget gate, mid-loop eviction, anti-confabulation prompt, behavioral rules
