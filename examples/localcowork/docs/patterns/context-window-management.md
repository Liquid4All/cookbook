# Context Window Management

> Strategy for managing the 32k token context window of the local LLM.

## Token Budget

The LFM2.5-24B (and Qwen2.5-32B dev proxy) has a 32,768 token context window.

| Component | Budget | Eviction | Notes |
|-----------|--------|----------|-------|
| System prompt | ~500 tokens | Never evicted | Static persona + rules |
| Tool definitions | ~2,000 tokens | Lazy-load servers | See "Lazy Loading" below |
| Conversation history | ~20,000 tokens | Rolling window, oldest first | Preserves last N turns |
| Active context (file/doc) | ~9,500 tokens | Replaced per request | Current file, search results, etc. |
| Safety buffer | ~768 tokens | Reserved | Prevents truncation edge cases |

**Total: 32,768 tokens**

## System Prompt (Fixed, ~500 tokens)

The system prompt is loaded once at conversation start and never evicted:

```
You are LocalCowork, a private desktop assistant. You help users manage files,
documents, emails, calendars, and tasks — entirely on their local machine.

Rules:
1. Never write code. Use only the tools provided.
2. For mutable actions, show preview and wait for confirmation.
3. For destructive actions, show explicit warning.
4. Explain what you'll do before calling tools.
5. For multi-step tasks, plan first, then execute step by step.
```

## Tool Definitions (~2,000 tokens)

### Full Load (Default)

All 13 servers' tools are included in the system prompt. This uses ~2,000 tokens because each tool definition is kept concise (one-line description, minimal param descriptions).

### Lazy Loading (Optional, for tight context)

If context is tight (e.g., large document in active context), tool definitions can be lazy-loaded:

1. Always include **Tier 1** tools (filesystem, data, task) — ~600 tokens
2. Include **Tier 2** tools when relevant keywords detected — ~800 tokens
3. Include **Tier 3** tools only when explicitly requested — ~600 tokens

| Tier | Servers | Trigger |
|------|---------|---------|
| Tier 1 (always) | filesystem, data, task, audit, clipboard | Every conversation |
| Tier 2 (on demand) | document, ocr, knowledge, security | File/document keywords detected |
| Tier 3 (explicit) | meeting, calendar, email, system | User explicitly mentions these |

## Conversation History (~20,000 tokens)

### Rolling Window Strategy

The conversation history uses a sliding window with intelligent eviction:

1. **Most recent 5 turns** — always kept in full (both user messages and assistant responses).
2. **Turns 6-20** — kept with tool call results summarized (replace full results with one-line summaries).
3. **Turns 21+** — evicted entirely.

### Summarization on Eviction

When a turn is evicted, its key information is captured in a running summary:

```json
{
  "session_summary": "User organized Downloads folder (47 files → 5 categories). Processed 3 receipts into CSV. Currently working on contract comparison.",
  "files_touched": ["~/Downloads/", "~/Documents/Finance/receipts_feb.csv"],
  "decisions_made": ["Receipts categorized by vendor, not date", "Ignored .dmg files"]
}
```

This summary is prepended to the conversation history, using ~200 tokens but preserving critical context.

### Tool Call Result Handling

Tool call results can be large (e.g., `filesystem.list_dir` returning 100 files). Strategy:

| Result Size | Handling |
|-------------|----------|
| < 500 tokens | Include in full |
| 500–2,000 tokens | Include in full for the current turn, summarize in older turns |
| > 2,000 tokens | Truncate with "... and N more items. Use search_files for specific items." |

## Active Context (~9,500 tokens)

The active context slot holds the most relevant content for the current request:

### Content Types

- **File content** — when the user asks about a specific file, its content fills this slot
- **Search results** — semantic search results from the knowledge server
- **OCR output** — extracted text from images/PDFs
- **Document diff** — the diff output when comparing documents

### Replacement Strategy

Active context is **fully replaced** on each new user request that requires different content. There's no accumulation — it's always the most relevant content for the current turn.

### RAG Integration

For documents that exceed the active context budget (e.g., a 60-page PPM for UC-8):

1. Index the document via `knowledge.index_folder`
2. For each question/analysis step, retrieve the top-K most relevant chunks
3. Place retrieved chunks in the active context slot
4. The model answers grounded in these chunks

This means the model never sees the full document at once — it works with relevant excerpts, guided by the RAG pipeline.

## Context Window Monitor

The Agent Core tracks token usage in real-time:

```rust
struct ContextWindowManager {
    max_tokens: usize,          // 32,768
    system_prompt_tokens: usize, // ~500
    tool_def_tokens: usize,      // ~2,000
    history_tokens: usize,       // rolling count
    active_context_tokens: usize, // current slot

    fn remaining(&self) -> usize {
        self.max_tokens - self.system_prompt_tokens - self.tool_def_tokens
            - self.history_tokens - self.active_context_tokens - SAFETY_BUFFER
    }

    fn should_evict(&self) -> bool {
        self.remaining() < 1000 // evict if less than 1K tokens free
    }
}
```

When `should_evict()` returns true:
1. Summarize the oldest conversation turn
2. Replace tool call results with summaries
3. If still tight, switch to lazy tool loading
4. If still tight, truncate active context

## Token Counting

Use tiktoken (or the model's native tokenizer) for accurate counting:
- Rust: `tiktoken-rs` crate
- TypeScript: `tiktoken` npm package
- Python: `tiktoken` pip package

Important: token counts vary by model. When swapping from Qwen to LFM2.5, the tokenizer must be updated.
