# Prompt Engineering for Small On-Device LLMs

> Patterns and lessons learned from prompt-tuning LFM2-24B-A2B (24B total, 2.3B active per token)
> for tool-calling accuracy in LocalCowork.

**Status:** Validated in production (March 2026)
**Model:** LFM2-24B-A2B-GGUF, Q4_K_M quantization, llama.cpp runtime
**Implementation:** `src-tauri/src/commands/chat.rs` (`build_system_prompt`, `send_message`)

**Key model trait:** LFM2-24B-A2B uses lightweight instruct post-training with
**no reasoning traces** (no chain-of-thought). This means the model won't
"think through" ambiguous instructions — it either knows or it guesses from
training priors. Every pattern below compensates for this by making the correct
answer explicit rather than requiring inference.

---

## Core Insight

Small LLMs (7B-30B) behave fundamentally differently from large frontier models
(70B+, GPT-4, Claude) when processing system prompts. Techniques that work at scale
often fail completely at this size. The patterns below were discovered through
iterative testing with LFM2-24B-A2B running locally.

---

## Pattern 1: User Message Injection (Most Important)

**Problem:** The model ignored dates and file paths provided in the system prompt,
hallucinating values from training data instead (e.g., `2023-10-05` instead of
`2026-03-02`, `/path/to/contracts` instead of actual paths).

**Root cause:** Small LLMs have strong priors from training data. System prompt
information competes with these priors and often loses. User messages receive
significantly more attention weight than system prompts during generation.

**Solution:** Inject critical context directly into the user message as a bracketed
prefix. This places the information where the model cannot ignore it.

```
[Today is 2026-03-02. Tomorrow is 2026-03-03. This week is 2026-03-02 to 2026-03-08.]
[Working folder: /Users/chintan/tests/sample_files. Files: /Users/.../file1.txt, /Users/.../file2.pdf. Use ONLY these paths.]
What's on my calendar today?
```

**Why it works:** The model treats user message content as the primary input to
respond to. By embedding context as a prefix, it becomes part of the "query" rather
than background instruction. The bracketed format signals metadata without confusing
the model into treating it as conversational text.

**Implementation:** Detect temporal keywords (`today`, `tomorrow`, `calendar`,
`schedule`, `meeting`) and working folder presence, then prepend to the last user
message before sending to the LLM. This is a runtime overlay — not persisted to
the conversation database.

**Before (failed):**
```
System: Current date: 2026-03-02.
User: What's on my calendar today?
Model calls: calendar.list_events({"start_date": "2023-10-05", ...})  // WRONG
```

**After (works):**
```
System: <datetime>today = 2026-03-02</datetime>
User: [Today is 2026-03-02.] What's on my calendar today?
Model calls: calendar.list_events({"start_date": "2026-03-02", ...})  // CORRECT
```

---

## Pattern 2: Three-Layer Reinforcement

Critical information should appear in three places, not one:

| Layer | Position | Purpose |
|-------|----------|---------|
| System prompt block | Top of system prompt (primacy) | Sets context for the conversation |
| Few-shot examples | Middle of system prompt | Shows the expected output format |
| User message prefix | In the actual query (strongest) | Forces the model to see it |

Each layer alone is insufficient for a 24B model. All three together achieve
reliable behavior. This is not redundancy — each layer serves a different function
in the model's attention mechanism.

---

## Pattern 3: XML Section Tags

**Problem:** Flat-text system prompts with numbered rules cause the model to lose
track of section boundaries. Information in the "middle" of the prompt gets
deprioritized (the "lost in the middle" problem).

**Solution:** Use XML tags to create explicit section boundaries:

```
You are LocalCowork, a private on-device AI assistant. You call tools to help the user.

<datetime>
today = 2026-03-02 (Monday)
tomorrow = 2026-03-03
this_week = 2026-03-02 to 2026-03-08
current_time = 14:30
NEVER ask the user for a date.
</datetime>

<capabilities>
Available capabilities (24 tools across 7 servers): ...
</capabilities>

<rules>
1. Use fully-qualified tool names.
2. Use absolute paths.
...
</rules>

<examples>
Example 1 — calendar query:
...
</examples>
```

**Why it works:** Small models trained on code and markup data have learned to parse
XML-like structure. The tags act as "section headers" that the model can index,
preventing information from bleeding across boundaries. Research confirms this is
especially effective for locally-hosted models under 30B parameters.

---

## Pattern 4: Pre-Computed Values

**Problem:** Providing `Current date: 2026-03-02` and expecting the model to infer
that "today" = that date, or "tomorrow" = 2026-03-03, requires reasoning. Small
models fail at this inference step.

**Solution:** Pre-compute every relative value the model might need:

```
<datetime>
today = 2026-03-02 (Monday)
tomorrow = 2026-03-03
this_week = 2026-03-02 to 2026-03-08
current_time = 14:30
</datetime>
```

**Rule:** If the model would need to do arithmetic or date manipulation to derive
a value, compute it at prompt-build time and provide it as a literal. Never rely
on the model to reason about time, dates, or relative paths.

---

## Pattern 5: Rule Compression

**Problem:** 12 numbered behavioral rules caused the model to lose track of
instructions. Rules beyond position 5-7 were frequently ignored.

**Solution:** Consolidate to 6 or fewer rules. Merge overlapping instructions.
Remove rules that are already enforced by the system (e.g., confirmation dialogs).

**Before (12 rules, model ignored rules 8-12):**
```
1. Use absolute paths...
2. READ operations: call immediately...
3. WRITE operations: call directly...
4. Be concise...
5. SEQUENTIAL PROCESSING: ...
6. NO REDUNDANT CALLS: ...
7. PROGRESS TRACKING: ...
8. TRUTHFULNESS: ...
9. COMPLETE ALL FILES: ...
10. KNOW WHEN TO STOP: ...
11. REPORT, DON'T ACT: ...
12. DATE RESOLUTION: ...
```

**After (6 rules, all followed):**
```
1. Use fully-qualified tool names.
2. Use absolute paths. If WORKING FOLDER set, use ONLY those paths.
3. READ tools: call immediately. WRITE tools: call directly.
4. After scan results, present findings and STOP.
5. Never call same tool with same args twice.
6. Be concise. Respond after 1-3 tool calls.
```

**Guideline:** If a rule requires more than one sentence to explain, it's too
complex for a small model. Split it into a rule + example instead.

---

## Pattern 6: Primacy and Recency Positioning

**Problem:** Information placed in the middle of a long system prompt is
deprioritized by the model's attention mechanism. This is well-documented as the
"lost in the middle" effect and is more severe in small models.

**Solution:** Place the most important information at the very beginning (primacy)
and very end (recency) of the system prompt. Use a "sandwich pattern" for critical
context:

```
[Identity — 1 line]

[CRITICAL CONTEXT HERE — working folder, dates]       <-- PRIMACY

[Capabilities, rules, examples — middle section]

[REPEATED CRITICAL CONTEXT — reminder block]           <-- RECENCY
```

**Implementation in LocalCowork:** The working folder and file listing appear
as a `<working_folder>` block immediately after the identity line, and again as
a `<reminder>` block at the very end of the system prompt.

---

## Pattern 7: Few-Shot Example Ordering

**Problem:** The model consistently followed the patterns shown in Example 1 but
often ignored patterns from later examples.

**Solution:** Order few-shot examples by importance, not by complexity. The most
critical pattern (calendar date resolution) moved from Example 4 to Example 1.

**Additional finding:** Examples with multiple sub-cases (today, tomorrow, this
week) in a single example block work better than separate examples. The model
learns "this is a pattern with variations" rather than treating each as independent.

---

## Anti-Patterns (What Does NOT Work)

| Approach | Why It Failed |
|----------|---------------|
| Date in system prompt only | Model hallucinated training-data dates (2023) |
| Verbose multi-sentence rules | Model stopped following rules after position 5-7 |
| `Current date: YYYY-MM-DD` without pre-computed relatives | Model couldn't infer "today" = that date |
| Repeating instructions verbatim (copy-paste) | Wasted tokens; diminishing returns after 2 repetitions |
| Long identity paragraph | Competed with actual instructions for attention budget |
| Generic path examples (e.g., `{home}/Documents/contract_v1.pdf`) | Model latched onto example paths instead of real working folder paths |
| File listing in user message prefix | Model already had the answer, so it skipped the tool call and returned text. User never saw the tool trace UI. |

### The "Answer Leak" Trap

When injecting context into the user message, be careful not to include the
**answer** to the user's question. If the user asks "list my files" and the
user message prefix already contains the file listing, the model will respond
with text (regurgitating the listing) instead of calling `filesystem.list_dir`.

**Rule:** The user message prefix should contain just enough context to anchor
the model's tool arguments (e.g., the folder path), but NOT the data the user
is asking for. Full data listings belong in the system prompt, where they guide
tool argument selection without short-circuiting the tool call.

---

## Token Budget Impact

The optimized prompt is shorter than the original despite having more structure:

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| Identity | ~25 tokens | ~15 tokens | -40% |
| Date context | ~20 tokens | ~50 tokens (pre-computed) | +150% |
| Rules | ~350 tokens | ~120 tokens | -66% |
| Examples | ~250 tokens | ~180 tokens | -28% |
| User message prefix | 0 tokens | ~40-80 tokens (dynamic) | new |
| **Total system prompt** | **~645 tokens** | **~365 tokens** | **-43%** |
| **Total with user prefix** | **~645 tokens** | **~445 tokens** | **-31%** |

The net savings leave more room for conversation history and tool results.

---

## Validation

These patterns were validated through iterative testing with the production model:

| Scenario | Before | After |
|----------|--------|-------|
| "What's on my calendar today?" | Hallucinated 2023-10-05 | Correct 2026-03-02 |
| "Scan my working folder for secrets" | Used `/path/to/contracts` | Used actual working folder path |
| "What do I have this week?" | Asked user for the date | Called `list_events` with correct week range |
| Tool name format | Occasional `list_dir` without prefix | Consistent `filesystem.list_dir` |
| Post-scan behavior | Auto-chained to mutable tools | Stopped after presenting findings |

---

## Applicability

These patterns apply to any small LLM (7B-30B) used for tool-calling:

- **Confirmed:** LFM2-24B-A2B (Liquid AI, hybrid MoE)
- **Likely applicable:** Qwen2.5-32B, Llama-3.x-8B/70B, Mistral-7B/22B, Phi-3
- **Less needed:** Models >70B or frontier APIs (GPT-4, Claude) which have
  stronger instruction-following from system prompts alone

The general principle: the smaller the model, the more you must shift critical
context from system prompt to user message, and from implicit reasoning to
explicit pre-computed values.

---

## References

- `src-tauri/src/commands/chat.rs` — Implementation (system prompt + user message injection)
- `docs/model-analysis/lfm2-24b-a2b-benchmark.md` — Model benchmark results
- `docs/patterns/context-window-management.md` — Token budget allocation
- `_models/config.yaml` — Model and tool configuration
