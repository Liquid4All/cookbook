# GPT-OSS-20B — Model Behavior Reference

**Last updated**: 2026-02-14 (session-025)
**Status**: Active development model
**Config**: `_models/config.yaml` entry `gpt-oss-20b`

---

## Model Identity

| Property | Value |
|----------|-------|
| Display name | GPT-OSS-20B (Dev) |
| Ollama tag | `gpt-oss:20b` |
| Runtime | Ollama at `localhost:11434/v1` |
| Context window | 32,768 tokens |
| VRAM | ~14 GB |
| Tool call format | `native_json` (OpenAI-compatible) |
| Max output tokens | 4,096 |
| Sampling (tool turns) | temperature=0.1, top_p=0.2 |
| Sampling (conversation) | temperature=0.7, top_p=0.9 |
| Fallback chain | gpt-oss-20b -> qwen3-30b-moe -> static_response |
| `force_json_response` | false (not yet enabled) |

---

## Behavioral Failure Taxonomy

Twelve failure modes observed across sessions 014-024 (~11 sessions of live testing).
Organized into three categories by root cause type.

### Category A: Structural Failures (Format Errors)

These are serialization errors — the model's *intent* is correct but the *output format*
is broken. Fixable by grammar-constrained decoding (GBNF) or JSON repair.

#### FM-1: Malformed JSON Arguments

The model generates syntactically invalid JSON in tool call arguments.

- **Example**: `{"destination":""/Users/chintan/Desktop/file.png"}` — double-quote
  after colon makes JSON unparseable
- **Evidence**: Session-017 agent.log — Ollama returned HTTP 500 with
  `error parsing tool call: raw='{"create_dirs":true,"destination":""/Users/...`
- **Frequency**: ~5% of tool calls
- **Impact**: Entire tool turn fails. Triggers fallback chain walk
  (gpt-oss-20b -> qwen3-30b-moe -> static_response). User sees
  "the local AI model is not currently available."
- **Workaround**: `repair_malformed_tool_call_json()` in
  `src-tauri/src/inference/tool_call_parser.rs:175-210`. Four repair heuristics:
  1. Fix double-quotes after colon (`":"` -> `:"`)
  2. Remove trailing commas before `}` or `]`
  3. Append missing closing braces
  4. Strip non-printable control characters
- **Root fix**: Enable GBNF grammar constraints in llama.cpp (Layer 1 of ADR-008)

#### FM-2: Empty Responses

Model returns 0 text AND 0 tool calls in a single round.

- **Evidence**: Sessions 014, 023 agent.log — `round_text_len=0 tool_calls_count=0`
- **Frequency**: ~10% of rounds in long conversations (>8 messages)
- **Impact**: Agent loop stalls. Without mitigation, identical retry causes same stall.
- **Workaround**: Nudge prompt injection in `chat.rs:830-855`.
  After `MAX_EMPTY_RETRIES=2` consecutive empties, injects contextual message:
  "You returned an empty response after processing N tool call(s). If there are more
  files to process, call the next tool now." If still empty, forces summary.
- **Root cause**: Likely timeout or context overflow. 180s stream timeout (up from 30s)
  reduced frequency.

---

### Category B: Semantic Failures (Decision Errors)

The model makes wrong *choices* — calls the wrong tool, uses wrong parameters, or
stops when it shouldn't. These are model quality issues, not format issues.

#### FM-3: Conversational Deflection (Primary Bug)

After receiving a tool result, the model asks the user what to do instead of
proceeding with the next tool call as instructed.

- **Example**: User asks "rename screenshots by extracting content." Model calls
  `list_dir`, gets file listing, then responds: *"I've listed the files on your
  Desktop. What would you like me to do with them?"* — 80 chars, 0 tool calls.
- **Evidence**: Session `971fb71b` (2026-02-14). Round 0: `list_dir` succeeds.
  Round 1: 80 chars text, 0 tools. Agent loop exits.
  Second attempt in same session: same behavior after 3 rounds.
- **Frequency**: ~80% of multi-step tasks after `list_dir`
- **Impact**: Task never starts. User must rephrase and retry.
- **Workaround**: **NONE**. The `is_incomplete_response()` function (14 signal
  patterns) does not catch this because "What would you like me to do?" contains
  no incomplete signals ("remaining", "next file", etc.). The text looks like a
  *complete* conversational response.
- **Root cause**: With 57 tools (~8,670 tokens of definitions) in context, the
  model cannot reason about which tool to call next. It defaults to the safe
  conversational response of asking the user. This is directly caused by FM-11
  (tool cognitive overload).
- **Interaction**: FM-11 (overload) + FM-12 (context pressure) -> FM-3 (deflection)

#### FM-4: Example Path Leakage

The model copies placeholder paths from the system prompt's few-shot examples
instead of using the user's actual path.

- **Example**: System prompt contains `/Users/alex/Documents/receipt.png`.
  Model calls `filesystem.list_dir({"path": "/Users/name/Desktop"})` using
  the old example path (before the fix) instead of the user's `/Users/chintan/Desktop`.
- **Evidence**: Session `971fb71b`, second attempt, round 0. Tool returned error:
  `Path "/Users/name/Desktop" is outside the sandboxed directories.`
- **Frequency**: ~15% of first tool calls after a fresh session
- **Impact**: Tool call fails with sandbox violation. Model retries with correct
  path on next round (self-correcting).
- **Workaround**: **Fixed in session-025** — system prompt examples now use
  `/Users/alex/Documents` (clearly fictional, different directory) to reduce
  leakage risk.

#### FM-5: Parameter Name Hallucination

The model uses a parameter name that doesn't exist in the tool schema.

- **Example**: Called `ocr.extract_text_from_image({"image_path": "..."})` when the
  actual parameter is `path` (per `docs/mcp-tool-registry.yaml` line 120).
- **Evidence**: Session `43a12b14`. MCP error: `missing required field: 'path'`.
  Model self-corrected on next round.
- **Frequency**: ~20% on first OCR call per session
- **Impact**: First call fails, wastes a round. Model usually self-corrects.
- **Root cause**: The system prompt was teaching `"image_path"` in its few-shot
  examples (lines 69, 74 of `chat.rs`). The model learned the wrong name from
  *our own prompt*.
- **Workaround**: **Fixed in session-025** — system prompt examples now use `"path"`.

#### FM-6: Case-Insensitive Filter

The model uses lowercase glob patterns on case-sensitive filesystems.

- **Example**: User asks for files with "Screenshot" in name. Model uses
  `filter: "*screenshots*"` (lowercase) -> 0 results on macOS (case-sensitive HFS+).
- **Evidence**: Session `43a12b14`, first attempt.
- **Frequency**: ~30% when user mentions filenames with mixed case
- **Impact**: Returns empty results. User must rephrase with exact case.
- **Workaround**: **NONE**. Could be mitigated in the filesystem MCP server
  (case-insensitive glob option) but not yet implemented.

#### FM-7: Premature Task Abandonment ("Model Fatigue")

The model stops processing after 3-4 files and emits a partial summary, even though
7+ files remain.

- **Example**: User asks to rename 7 screenshots. Model processes 3, then says
  "I've renamed 3 files so far. Here's what was done..."
- **Evidence**: Sessions 014-015. `is_incomplete_response()` catches phrases like
  "remaining", "more files", "will process" in the partial summary.
- **Frequency**: ~60% of tasks with >5 files
- **Impact**: Task incomplete. Continuation prompt usually works (1-2 extra rounds).
- **Workaround**: `is_incomplete_response()` in `chat.rs:403-455` detects 14
  incomplete signal patterns. Injects ephemeral continuation: "You stopped before
  finishing. Continue processing the remaining files." These prompts are in-memory
  only (not persisted to DB) to prevent history pollution.

#### FM-8: Unprefixed Tool Names

The model drops the server prefix from tool names.

- **Example**: Calls `move_file` instead of `filesystem.move_file`.
- **Evidence**: Sessions 014-015 agent.log.
- **Frequency**: ~25% of tool calls (improved to ~10% after few-shot examples)
- **Impact**: Tool dispatch fails with "unknown tool" error.
- **Workaround**: `resolve_tool_name()` in `chat.rs:274-327`. Three-step matching:
  1. Exact match in registry
  2. If contains dot -> already qualified, return as-is
  3. Search for registered tool ending with `.{name}` -> resolve if unique

#### FM-9: Relative Paths

The model uses `~/Desktop` instead of absolute paths like `/Users/chintan/Desktop`.

- **Evidence**: Observed in early sessions. System prompt rule 1 addresses this.
- **Frequency**: ~10% (reduced by few-shot examples showing absolute paths)
- **Impact**: Tool call fails (MCP servers require absolute paths).
- **Workaround**: System prompt rule only. No code-level fix.

#### FM-10: Redundant Tool Calls

The model calls the same tool with identical arguments twice in one conversation.

- **Example**: Calls `filesystem.list_dir("/Users/chintan/Desktop")` after already
  having the result from a previous round.
- **Evidence**: Session `971fb71b`, second attempt — called `list_dir` twice.
  Also session `43a12b14` — called with different filter each time (not truly
  redundant but wasteful).
- **Frequency**: ~20% in multi-round conversations
- **Impact**: Wastes a tool round (~5-15 seconds per redundant call).
- **Workaround**: System prompt rule 7: "NO REDUNDANT CALLS". No code-level fix.

---

### Category C: Systemic Failures (Capacity Limits)

These are inherent limitations of running a 20B model with 57 tools in a 32K
context window. No prompt engineering can fully solve them.

#### FM-11: Tool Cognitive Overload

With 57 tools registered (~8,670 tokens of JSON schema definitions), the model
struggles to select the correct next tool.

- **Evidence**: Token measurement from `chat.rs` log: `tool_count=59 tool_tokens=8670`.
  After receiving a list_dir result (2,212 bytes), the model must reason about which
  of 57 tools to call next. It defaults to asking the user (FM-3).
- **Frequency**: Correlates directly with FM-3 occurrence (~80% of multi-step tasks)
- **Impact**: Root cause of conversational deflection.
- **Workaround**: **NONE** currently.
- **Proposed fix**: RAG-based tool selection to reduce visible tools from 57 to 5-10.
  See "Multi-Model Architecture Analysis" section below.

#### FM-12: Context Window Pressure

The 32K context window is heavily loaded before any conversation starts.

- **Token budget breakdown**:
  - System prompt: ~900 tokens
  - Tool definitions: ~8,670 tokens
  - Output reservation: ~2,000 tokens
  - Safety buffer: ~768 tokens
  - **Available for conversation: ~20,430 tokens**
- **Evidence**: Log shows `prompt_tokens=898` for system prompt,
  `tool_tokens=8670` for definitions. That's ~9,568 tokens of overhead.
- **Impact**: Less "thinking space" for the model. In sessions with 10+ rounds,
  the sliding window eviction must aggressively compress older tool results.
- **Workaround**: Mid-loop eviction in `chat.rs:996-1013`. Sliding window with
  3-tier compression (recent verbatim, middle compressed, evicted summarized).
  Tool result truncation at 6,000 chars. Session summary capped at 500 tokens.

---

## System Prompt Bugs (Self-Inflicted)

Three bugs in the few-shot examples (`chat.rs` lines 66-78) where the system prompt
actively taught the model wrong behavior. **All fixed in session-025.**

| Bug | Location | Wrong | Correct | Impact |
|-----|----------|-------|---------|--------|
| Parameter name | Lines 69, 74 | `"image_path"` | `"path"` | FM-5: model hallucinated `image_path` |
| Placeholder path | Lines 67-77 | `/Users/name/Desktop` | `/Users/alex/Documents` | FM-4: model copied example path literally |
| Tool name | Line 73 | `filesystem.list_directory` | `filesystem.list_dir` | Model confused between example and real tool |

---

## Workaround Registry

All code-level mitigations deployed as of session-025:

| Workaround | File | Function / Location | Fixes |
|---|---|---|---|
| Tool name resolution | `chat.rs` | `resolve_tool_name()` (lines 274-327) | FM-8 |
| JSON repair (4-step) | `tool_call_parser.rs` | `repair_malformed_tool_call_json()` (lines 175-210) | FM-1 |
| Incomplete detection | `chat.rs` | `is_incomplete_response()` (lines 403-455) | FM-7 |
| Ephemeral continuation | `chat.rs` | Lines 871-899 (in-memory only) | FM-7 |
| Nudge prompts | `chat.rs` | Lines 830-855 | FM-2 |
| Tool result truncation | `chat.rs` | `truncate_tool_result()` (lines 372-392) | FM-12 |
| Dynamic sampling | `chat.rs` | `TOOL_TURN_SAMPLING` / `CONVERSATIONAL_SAMPLING` (lines 128-138) | FM-1, FM-8 |
| Forced summary | `chat.rs` | Lines 1032-1085 | FM-2, FM-7 |
| HTTP 500 repair | `client.rs` | `try_repair_from_error()` (lines 265-290) | FM-1 |
| Mid-loop eviction | `chat.rs` | Lines 996-1013 | FM-12 |
| Token budget gate | `chat.rs` | `MIN_ROUND_TOKEN_BUDGET=1500` (lines 717-731) | FM-12 |
| Sliding window | `conversation.rs` | `build_windowed_chat_messages()` | FM-12 |

### Open Gaps (No Workaround)

| Failure Mode | Why No Fix Exists |
|---|---|
| FM-3: Conversational deflection | Response looks "complete" — no signal patterns to detect |
| FM-6: Case-insensitive filter | Would require MCP server change, not agent-level |
| FM-9: Relative paths | System prompt rule only |
| FM-10: Redundant tool calls | System prompt rule only |
| FM-11: Tool cognitive overload | Structural — 57 tools exceed model's selection capacity |

---

## Failure Interaction Map

Failures compound in predictable chains:

```
FM-11 (57 tools overload) + FM-12 (9.5K token overhead)
   |
   v
Model cannot reason about next tool
   |
   +---> FM-3 (conversational deflection): "What would you like me to do?"
   |       -> Agent loop exits. Task never starts.
   |
   +---> FM-2 (empty response): 0 text, 0 tools
   |       -> Nudge prompt -> usually recovers
   |
   +---> FM-7 (premature abandonment): partial summary after 3 files
           -> Ephemeral continuation -> usually recovers

FM-4 (path leakage) + FM-5 (param hallucination)  [FIXED in session-025]
   |
   v
Tool call fails (sandbox error or missing param)
   |
   v
Model retries with different approach
   |
   +---> Correct retry (self-correcting) ~70%
   +---> FM-10 (redundant call) or FM-3 (deflection) ~30%
```

The critical unbroken chain is: **FM-11 -> FM-3 -> exit**. This is the "80% failure
on multi-step tasks" that prompted this analysis.

---

## Multi-Model Architecture Analysis

### Problem Statement

A single 20B model handles conversation, reasoning, AND tool selection from 57 tools.
The tool selection task alone consumes ~8,670 tokens of context and requires reasoning
over 57 JSON schemas. This exceeds the model's practical selection capacity, causing
conversational deflection (FM-3) in ~80% of multi-step tasks.

**Core question**: Should we split responsibilities across multiple models?

### Option A: Single Model (Status Quo)

One 20B model handles everything.

| Dimension | Assessment |
|-----------|-----------|
| Memory | ~16 GB (14 GB model + 1.8 GB vision) |
| Complexity | Low — single inference path |
| Tool accuracy | Low (~20-40% multi-step completion) |
| Latency | ~15s per round (one model call) |
| Improvement path | ADR-008 Layers 1-4, then fine-tuning |

**Verdict**: Current approach. Not working for multi-step tool tasks.

### Option B: RAG-Based Tool Selection

Use an embedding model to semantically match user queries to relevant tools,
then pass only the top-K tools to the main model.

```
User query
   |
   v
Embedding model (sentence-transformers, ~100 MB)
   |
   v
Cosine similarity against tool definition embeddings
   |
   v
Top 5-10 tools (instead of 57)
   |
   v
20B model with reduced tool set (~800-1,500 tokens instead of 8,670)
```

| Dimension | Assessment |
|-----------|-----------|
| Memory | +~100 MB for embedding model = ~16.1 GB total |
| Complexity | Medium — new embedding pipeline, tool index |
| Tool accuracy | Improved (Anthropic RAG-MCP: 13% -> 43% accuracy on large toolsets) |
| Latency | +~50ms per query for embedding + similarity |
| Token savings | ~7,000 tokens freed (8,670 -> ~1,500) |
| Implementation | Intercept `build_all_tool_definitions()` in `chat.rs`. The knowledge server (WS-4A) already has embedding infrastructure with SQLite-vec. |

**Verdict**: High value-to-cost ratio. Minimal memory, addresses FM-11 directly.

### Option C: Router Model + Conversational Model

Dedicate a small model (1-3B) to tool selection/execution, and the 20B to conversation.

```
User query
   |
   v
Router model (1-3B, e.g. LFM2-1.2B-Tool or fine-tuned Llama-3.2-1B)
   |
   +---> Tool calls needed? ---> Router selects & executes tools
   |                                |
   |                                v
   |                             Tool results
   |                                |
   +---> Conversational? --------> 20B model synthesizes response
```

| Dimension | Assessment |
|-----------|-----------|
| Memory | +2-4 GB for router = ~18-20 GB total |
| Complexity | High — two inference paths, routing logic, model orchestration |
| Tool accuracy | High (TinyAgent/Berkeley: fine-tuned 1B at 78.89%, 7B at 83.09%) |
| Latency | +~2-5s per query for router inference |
| Ollama support | `OLLAMA_MAX_LOADED_MODELS` for concurrent serving |
| Implementation | New routing layer in agent core, dual model config, result aggregation |

**Verdict**: Best accuracy, but significantly more complex. Requires fine-tuning
the router on LocalCowork's 57-tool schema.

### Option D: RAG Filter + Single Model (Recommended)

Combine Options A and B: use RAG to narrow tools, keep a single model.

```
User query
   |
   v
Embedding model (100 MB) -> cosine similarity -> top 5-10 tools
   |
   v
20B model sees only 5-10 relevant tools
   |
   v
Tool execution (same as today)
```

| Dimension | Assessment |
|-----------|-----------|
| Memory | +~100 MB = ~16.1 GB total |
| Complexity | Medium (similar to Option B) |
| Tool accuracy | Significantly improved — 20B is adequate with 5-10 tools |
| Latency | +~50ms (embedding is fast) |
| Token savings | ~7,000 tokens freed for conversation history |
| Migration path | If accuracy still <85% -> escalate to Option C |

**Key advantages over other options**:
- No new model server (no `OLLAMA_MAX_LOADED_MODELS` config)
- Reuses knowledge server's embedding infrastructure (`mcp-servers/knowledge/`)
- Consistent with ADR-008: free/low-cost improvements before fine-tuning
- Directly addresses both FM-11 (tool overload) and FM-12 (context pressure)

### Memory Footprint Comparison

All measurements assume M4 Max with 36 GB unified memory (user's machine):

| Option | Model VRAM | Vision VRAM | Additional | Total | Headroom |
|--------|-----------|-------------|------------|-------|----------|
| A: Single model | 14 GB | 1.8 GB | — | ~16 GB | 20 GB |
| B: RAG only | 14 GB | 1.8 GB | 100 MB (embeddings) | ~16.1 GB | 20 GB |
| C: Router + main | 14 GB | 1.8 GB | 2-4 GB (router) | ~18-20 GB | 16-18 GB |
| **D: RAG + single** | **14 GB** | **1.8 GB** | **100 MB** | **~16.1 GB** | **20 GB** |

All options fit comfortably on the M4 Max. Option C consumes the most but still
leaves 16+ GB headroom.

### Implementation Complexity

| Option | New Files | Config Changes | New Dependencies | Risk |
|--------|-----------|---------------|-----------------|------|
| A | 0 | 0 | 0 | None (status quo) |
| B | 2-3 (embedder + tool index) | 1 (model config) | sentence-transformers | Low |
| C | 5-8 (router, orchestrator, config) | 3+ | router model, Ollama multi-model | High |
| **D** | **2-3** | **1** | **sentence-transformers** | **Low** |

### Recommendation

1. **Immediate** (session-025): Fix system prompt bugs (FM-4, FM-5). Document all failure modes.
2. **Next** (WS-9D): Evaluate Qwen3-30B-A3B MoE (3B active params). If it handles 57 tools better, it may obviate the need for RAG. Test with the same rename-screenshots workflow.
3. **If Qwen3 insufficient**: Implement Option D (RAG tool selection). ~2-3 sessions of work.
4. **If Option D insufficient**: Escalate to Option C (router model). ~4-6 sessions.
5. **Last resort**: Fine-tune the router model on LocalCowork's 57-tool schema (ADR-008 Layer 5).

---

## References

- [Project Learnings and Recommendations](./project-learnings-and-recommendations.md) — cross-model synthesis of the 50+ tool problem
- [LFM2-24B-A2B Benchmark Results](./lfm2-24b-a2b-benchmark.md) — the model that broke through at 80% single-step
- [ADR-006: Agent Loop Reliability](../architecture-decisions/006-agent-loop-reliability.md)
- [ADR-007: Agent Loop Hardening](../architecture-decisions/007-agent-loop-hardening.md)
- [ADR-008: Tool-Calling Optimization Strategy](../architecture-decisions/008-tool-calling-optimization-strategy.md)
- [Anthropic RAG-MCP Study](https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/) — 13% to 43% accuracy improvement
- [TinyAgent (Berkeley AI)](https://bair.berkeley.edu/blog/2024/05/29/tiny-agent/) — fine-tuned 1B at 78.89%
- [NexusRaven](https://openreview.net/pdf?id=5lcPe6DqfI) — zero-shot function calling
- [Ollama Multi-Model](https://www.glukhov.org/post/2025/05/how-ollama-handles-parallel-requests/) — `OLLAMA_MAX_LOADED_MODELS`
- [Gorilla/ToolBench](https://github.com/ShishirPatil/gorilla) — tool-calling benchmarks
