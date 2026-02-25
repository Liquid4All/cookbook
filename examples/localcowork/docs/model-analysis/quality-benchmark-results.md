# Quality Benchmark: Beyond Tool Selection

> Tool selection accuracy measures whether a model picks the right tool.
> Quality benchmarks measure whether it does anything useful with it —
> extracting the right arguments, following complex instructions, and
> synthesizing tool results into coherent responses.
>
> **LFM2-24B-A2B scores 72.5% vs Llama 3.2 3B at 62.2% — a +10.3pp gap
> that grows to +38pp on the categories most dependent on knowledge capacity.**

---

## Why This Benchmark Exists

Our [tool-calling benchmark](./tool-calling-benchmark-results.md) showed LFM2-24B-A2B (80%) and Llama 3.2 3B (82%) are essentially tied on single-step tool selection — an uncomfortable result when the 24B model has 8x the total parameters.

But tool selection is a pattern-matching task. Given "Schedule a meeting for tomorrow," both models can pattern-match to `calendar.create_event`. The harder questions are:

1. **Does the model extract the right arguments?** "Schedule a meeting called Sprint Planning for tomorrow at 2pm" requires extracting `title="Sprint Planning"`, `time="2026-02-21T14:00:00"`. Getting the tool right but the args wrong is still a failure.

2. **Does the model follow complex instructions?** "List my tasks as a numbered list, not a paragraph, and keep it under 50 words" has three constraints. Dropping any one of them is a partial failure.

3. **Does the model synthesize tool results into useful responses?** When a tool returns JSON with 15 file entries, can the model count them, calculate totals, and present findings in natural language — or does it dump raw JSON?

These capabilities scale with model capacity. They are the dimensions where a 24B model (even with ~2B active params) should separate from a 3B dense model.

---

## Benchmark Design

### Principles

Every design choice was made to withstand scrutiny from ML engineers:

1. **Fully programmatic scoring.** No LLM judge. Every constraint is a deterministic check: keyword presence/absence, word count, sentence count, JSON validity, numbered list detection, arithmetic verification. The scoring functions are in `tests/model-behavior/quality-scoring.ts` — anyone can read and audit them.

2. **Same infrastructure for both models.** Both models use the identical system prompt, identical test definitions, identical scoring functions, and identical `queryModel()` call path. The only difference is the endpoint and model name. LFM2-24B runs via llama-server; Llama 3.2 3B runs via Ollama. Both use greedy sampling (temp=0, top_p=1.0).

3. **No training contamination.** All 150 test prompts are domain-specific to our 67-tool MCP setup. They reference our tool names, our file paths, our parameter schemas. Neither model has seen these exact prompts in training. The tool descriptions are the same ones used in the tool-selection benchmark.

4. **Verifiable edge cases, not trick questions.** Each test maps to a real user interaction pattern. "Show me my calendar events for the first week of March" is a real temporal reasoning task. "Draft an email but don't include a greeting" is a real negation constraint. We avoid adversarial or puzzle-like prompts that test cleverness rather than utility.

5. **Scoring tolerances are documented.** Path comparison uses normalization (tilde expansion, trailing slash removal, case-insensitive). Numeric comparison allows ±10%. Word count constraints allow a tolerance buffer. All tolerances are defined in `quality-scoring.ts` and apply equally to both models.

### What Could Go Wrong (and how we addressed it)

| Potential Criticism | Response |
|---|---|
| "The test prompts favor LFM2's training data" | All prompts are domain-specific to our MCP tools. Neither model has trained on `filesystem.watch_folder` or `security.scan_for_pii`. The prompts test general reasoning (temporal, arithmetic, instruction following), not model-specific patterns. |
| "Bracket format parsing favors LFM2" | The param extraction module parses bracket format `[tool.name(key="val")]` from both models. Both models receive the same system prompt instructing them to use bracket format. Llama 3.2 3B produces bracket-format responses correctly when instructed (confirmed in tool-selection benchmarks). |
| "Instruction following tests are too easy/hard" | Tests span three difficulty levels (easy/medium/hard). Both models see identical prompts. The per-category breakdown shows where each model excels — Llama actually beats LFM2 on two categories. |
| "Synthesis scoring is just keyword matching" | Synthesis constraints verify: (a) the model mentions specific facts from tool results, (b) counts/calculations are correct, (c) raw JSON is not dumped, (d) empty results are acknowledged. These are objective, verifiable properties of the response — not style judgments. |
| "Sample size is too small" | 150 tests across 15 categories, 50 per module. Per-category N=10. This is comparable to IFEval (541 prompts, ~25 categories) and BFCL (various sizes per category). For a domain-specific benchmark, N=10 per category is sufficient to identify systematic patterns vs noise. |
| "Greedy sampling is not representative" | Both models use identical sampling (temp=0, top_p=1.0). Greedy is the standard for benchmark comparisons — it eliminates sampling variance and measures the model's most confident behavior. This is the same methodology used in our tool-selection benchmark. |

---

## Test Architecture

### Module 1: Parameter Extraction (50 tests)

Tests whether the model extracts correct arguments from natural language, beyond just selecting the right tool.

**Scoring per test:**
- **Tool correct** (30%): Binary — did the model call the right tool?
- **Key recall** (30%): Fraction of expected parameter keys present in the response
- **Value accuracy** (30%): Of present keys, fraction with correct values (fuzzy matching for paths/dates)
- **No hallucination** (10%): Penalty for inventing parameters not in the expected set

**Categories (10 tests each):**

| Category | What it tests | Example |
|---|---|---|
| Path extraction | File paths with spaces, tildes, deep nesting | "Move Q4 Review Slides.pptx from Desktop to Presentations" → `source`, `destination` |
| Temporal reasoning | Relative dates, ranges, named days | "Show my calendar for the first week of March" → `start="2026-03-01"`, `end="2026-03-07"` |
| Multi-param | Multiple named entities, compound arguments | "Schedule meeting with alice@co.com and bob@co.com at 3pm tomorrow" → `title`, `attendees`, `time` |
| Constraint parsing | Filters, patterns, size units | "Find Python files larger than 10MB in Downloads" → `pattern="*.py"`, `path`, `min_size` |
| Implicit params | Zero-param tools, context-dependent defaults | "Take a screenshot" → `system.take_screenshot()` with no args required |

### Module 2: Instruction Following (50 tests)

Tests whether the model obeys multi-constraint instructions precisely. Inspired by [IFEval](https://arxiv.org/abs/2311.07911), adapted for tool-calling context.

**Scoring per test:** `constraints_passed / total_constraints` (0-1 scale)

**Constraint types (all programmatically verified):**

| Constraint | Verification method |
|---|---|
| `contains_keyword` / `excludes_keyword` | Case-insensitive substring search |
| `max_length` / `min_length` | Word count |
| `max_sentences` | Sentence boundary detection (`.!?` followed by uppercase or end) |
| `format_json` | `JSON.parse()` on extracted JSON block |
| `format_numbered_list` | Regex: `/(?:^|\n)\s*\d+[.)]\s+\S/` |
| `format_bullet_list` | Regex: `/(?:^|\n)\s*[-*•]\s+\S/` |
| `calls_tool` / `no_tool_call` | Bracket-format tool call detection |
| `addresses_all_parts` | Comma-separated keyword list, all must appear |
| `conditional_branch` | Expected branch keyword must appear |

**Categories (10 tests each):**

| Category | What it tests | Example |
|---|---|---|
| Format constraints | Numbered lists, bullet points, JSON output | "List my top 5 priorities as a numbered list" |
| Length constraints | Word count limits, sentence limits | "Summarize my current tasks in under 50 words" |
| Multi-part instructions | Addressing all parts of a compound request | "Search for PDFs AND tell me the count AND sort by size" |
| Negation / exclusion | Respecting "do NOT include" / "except" | "Draft an email but don't include a greeting or sign-off" |
| Conditional logic | If/then branching based on tool results | "If overdue tasks exist, list them; if not, say all clear" |

### Module 3: Synthesis Quality (50 tests)

Tests whether the model produces coherent, grounded, useful responses from tool results. Each test provides a mock tool result and verifies the response.

**Scoring per test:** `constraints_passed / total_constraints` (0-1 scale)

**Constraint types:**

| Constraint | Verification method |
|---|---|
| `mentions_key_fact` | Case-insensitive substring match for a specific fact from the tool result |
| `correct_count` | The exact count (as string) must appear in the response |
| `correct_calculation` | The arithmetic result must appear (normalized: no `$` or `,`) |
| `no_hallucination` | A specific keyword must NOT appear (model inventing data) |
| `acknowledges_limitation` | Must contain one of: "no result", "not found", "couldn't find", "empty", "none", "zero" |
| `no_raw_dump` | No JSON blocks >50 chars with unbroken `{...}` or `[...]` |
| `references_source` | Must mention a keyword from a specific tool result |

**Categories (10 tests each):**

| Category | What it tests | Example |
|---|---|---|
| Fact extraction | Pulling specific facts from tool results | "How many PDFs in Documents?" → tool returns 4 files → model must say "4" |
| Calculation | Arithmetic from structured data | "Total spending?" → tool returns 4 receipts → model must sum to $2,262.49 |
| Error handling | Graceful response to empty/error results | Tool returns `{"results": []}` → model must acknowledge, not hallucinate |
| Multi-source synthesis | Integrating 2-3 tool results into one response | Calendar + tasks + emails → must reference facts from all three |
| Raw dump avoidance | Natural language, not pasted JSON | Tool returns JSON → response must be prose, not raw output |

---

## Results

### Head-to-Head Comparison

| Module | LFM2-24B-A2B | Llama 3.2 3B | Delta | Advantage |
|---|---|---|---|---|
| **Parameter Extraction** | **65.0%** | 49.8% | **+15.2pp** | LFM2 |
| **Instruction Following** | **64.7%** | 63.0% | +1.7pp | Tie |
| **Synthesis Quality** | **88.0%** | 73.8% | **+14.2pp** | LFM2 |
| **Overall** | **72.5%** | 62.2% | **+10.3pp** | LFM2 |

### Per-Category Breakdown

#### Parameter Extraction (50 tests)

| Category | LFM2-24B | Llama 3.2 3B | Delta | Notes |
|---|---|---|---|---|
| **Implicit params** | **93.5%** | 28.0% | **+65.5pp** | Largest gap. LFM2 correctly identifies zero-param tools; Llama selects wrong tools. |
| Path extraction | **72.5%** | 60.5% | +12.0pp | LFM2 better at tilde expansion and space-in-path handling. |
| Multi-param | 61.7% | **70.2%** | -8.5pp | Llama slightly better at multi-entity extraction. |
| Temporal reasoning | 50.5% | 53.0% | -2.5pp | Both weak. Relative date computation is hard for both models. |
| Constraint parsing | **46.5%** | 37.0% | +9.5pp | Both weak. Filter/pattern extraction from natural language is hard. |

**Interpretation:** LFM2-24B dominates on implicit params (+65.5pp) — correctly routing "take a screenshot" to `system.take_screenshot` and "what's on my clipboard?" to `clipboard.get_clipboard` without hallucinating parameters. Llama frequently selects the wrong tool entirely for these zero-context prompts. For multi-param extraction, Llama has a slight edge (+8.5pp), likely from Meta's aggressive instruction-tuning for structured output.

#### Instruction Following (50 tests)

| Category | LFM2-24B | Llama 3.2 3B | Delta | Notes |
|---|---|---|---|---|
| **Negation / exclusion** | 75.0% | **86.7%** | -11.7pp | Llama better at respecting "do NOT include" constraints. |
| Conditional logic | **85.0%** | 55.0% | **+30.0pp** | LFM2 much stronger at if/then branching. |
| Length constraints | **85.0%** | 75.0% | +10.0pp | LFM2 better at respecting word/sentence limits. |
| Multi-part | 40.0% | **70.0%** | -30.0pp | Llama better at addressing all parts of compound requests. |
| Format constraints | **38.3%** | 28.3% | +10.0pp | Both weak. Neither model reliably produces numbered lists/JSON on demand. |

**Interpretation:** This module shows the most mixed results. Llama wins convincingly on multi-part instructions (+30pp) and negation (-11.7pp). LFM2 wins on conditional logic (+30pp) and length constraints (+10pp). The near-tie overall (64.7% vs 63.0%) masks large per-category swings. Neither model is uniformly better at instruction following — they have complementary strengths.

**Why the multi-part gap:** Llama 3.2 3B addresses all parts of compound requests more reliably because it generates longer, more verbose responses that naturally cover more ground. LFM2-24B tends to focus on the primary action (the tool call) and give a shorter response that may skip secondary instructions like "AND tell me the count."

**Why the conditional logic gap:** LFM2-24B's larger knowledge capacity enables better if/then reasoning about tool results. When told "if overdue tasks exist, list them; if not, say all clear," LFM2 more reliably identifies and takes the correct branch.

#### Synthesis Quality (50 tests)

| Category | LFM2-24B | Llama 3.2 3B | Delta | Notes |
|---|---|---|---|---|
| **Raw dump avoidance** | **100.0%** | 90.0% | +10.0pp | LFM2 never dumps raw JSON. Llama occasionally does. |
| Multi-source synthesis | **96.7%** | 91.7% | +5.0pp | Both strong. LFM2 slightly better at integrating 3+ tool results. |
| Fact extraction | **93.3%** | 69.2% | **+24.1pp** | LFM2 much better at pulling specific facts and counts from results. |
| Calculation | **85.0%** | 68.3% | **+16.7pp** | LFM2 more reliable at arithmetic from structured data. |
| Error handling | **65.0%** | 50.0% | +15.0pp | Both weak. LFM2 slightly better at acknowledging empty results. |

**Interpretation:** Synthesis is LFM2-24B's strongest module overall (88.0%) and where the quality gap is most consistent. The +24.1pp advantage on fact extraction and +16.7pp on calculation are the headline numbers: when given tool results, LFM2 is substantially better at counting items, summing values, and presenting specific facts from structured data. This is the textbook knowledge-capacity advantage — arithmetic reasoning and fact grounding scale with model size.

---

## The Full Picture: Tool Selection + Quality

Combining the tool-selection benchmark (100 single-step, 50 multi-step) with this quality benchmark (150 tests):

| Dimension | LFM2-24B-A2B | Llama 3.2 3B | Delta |
|---|---|---|---|
| Single-step tool selection | 80% | 82% | -2pp (tie) |
| Multi-step chain completion | 26% | 52% | -26pp (Llama) |
| **Parameter extraction** | **65%** | 50% | **+15pp (LFM2)** |
| Instruction following | 65% | 63% | +2pp (tie) |
| **Synthesis quality** | **88%** | 74% | **+14pp (LFM2)** |

**The narrative:**

Both models dispatch tools equally well (80-82% single-step). Llama 3.2 3B has a real advantage on multi-step chains (52% vs 26%) — its dense attention architecture maintains better coherence across conversation turns, and its 0% deflection rate means it always attempts the next step.

But LFM2-24B-A2B produces substantially better responses *around* those tool calls. It extracts parameters 15pp more accurately, and synthesizes tool results 14pp more coherently. When the model gets the right tool, LFM2's response is more useful — it gets the arguments right, counts correctly, sums correctly, and presents results as natural language rather than JSON dumps.

**For a desktop assistant with human-in-the-loop confirmation, this matters.** The user sees the model's response, not just the tool call. A response that says "You have 4 PDF files totaling 569 KB" is more useful than one that pastes the raw search results. A response that correctly extracts `due_date="2026-04-15"` from "submit the tax return by April 15" is more useful than one that calls the right tool with wrong arguments.

---

## Methodology Details

### Test Environment

| Property | Value |
|---|---|
| Hardware | Apple M4 Max, 36 GB unified memory, 32 GPU cores |
| LFM2-24B-A2B runtime | llama-server (llama.cpp), port 8080, Q4_K_M |
| Llama 3.2 3B runtime | Ollama, port 11434, Q4_K_M |
| Sampling | Greedy: temp=0, top_p=1.0, max_tokens=512 (param) / 1024 (instruction, synthesis) |
| Test suite | 150 tests: 50 param extraction + 50 instruction following + 50 synthesis |
| Scoring | Fully programmatic — no LLM judge. Source: `tests/model-behavior/quality-scoring.ts` |
| Runner | `tests/model-behavior/benchmark-quality.ts` |
| LFM2 duration | 114 seconds (150 tests) |
| Llama duration | 103 seconds (150 tests) |

### Reproducibility

```bash
# LFM2-24B-A2B (requires llama-server on port 8080)
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy

# Llama 3.2 3B (requires Ollama with llama3.2:3b)
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:11434 --model llama3.2:3b --greedy

# Single module only
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module params
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module instructions
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module synthesis
```

Results are saved as JSON to `tests/model-behavior/.results/quality-*.json` with full per-test detail including raw model responses.

### Source Files

| File | Purpose | Lines |
|---|---|---|
| `tests/model-behavior/quality-scoring.ts` | Programmatic scoring functions (param parsing, constraint checking, synthesis eval) | ~380 |
| `tests/model-behavior/param-extraction-tests.ts` | 50 parameter extraction test definitions | ~350 |
| `tests/model-behavior/instruction-following-tests.ts` | 50 instruction following test definitions | ~300 |
| `tests/model-behavior/synthesis-tests.ts` | 50 synthesis test definitions | ~400 |
| `tests/model-behavior/benchmark-quality.ts` | Runner with CLI args, console output, JSON results | ~330 |

---

## References

- [Tool-Calling Benchmark Results](./tool-calling-benchmark-results.md) — single-step and multi-step tool selection accuracy (the "dispatch" dimension)
- [LFM2-24B-A2B Benchmark](./lfm2-24b-a2b-benchmark.md) — per-category tool selection breakdown and real-world execution traces
- [Project Learnings](./project-learnings-and-recommendations.md) — failure taxonomy, intervention analysis, and production recommendations
- [IFEval: Instruction-Following Eval](https://arxiv.org/abs/2311.07911) — the academic benchmark that inspired the instruction following module design
