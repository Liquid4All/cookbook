# Small Model Tool-Calling: A Benchmark Study

> We tested 8 models (3B to 27B parameters) across two tiers against 67 MCP tools.
> The combination of hybrid architecture and MoE sparsity delivers the best latency-to-accuracy
> trade-off — but a 3B dense model (Llama 3.2) proved surprisingly competitive with much larger models.

---

## Why This Exists

LocalCowork is a desktop AI agent that runs entirely on-device. It uses a locally-hosted LLM to orchestrate 67 tools across 13 MCP servers (filesystem, document, OCR, knowledge, security, calendar, email, task, data, meeting, audit, clipboard, system). The model never writes code. It selects and calls pre-built tools.

Every model we tested hit the same structural barrier: **tool cognitive overload**. With 67 tool definitions consuming 8,670 tokens (26.5% of a 32K context window), models lose the ability to reason about which tool to call. 75% of those tool definitions are irrelevant to any given task.

This directory documents what we learned across 5 models, 150+ benchmark scenarios, and a dual-model orchestrator prototype. The findings are transferable to anyone building local AI agents with tool calling.

---

## The Models We Tested

### Tier 1 — Desktop-class (13-27 GB VRAM)

| Model | Architecture | Active Params | VRAM | Single-Step (greedy) | Multi-Step (greedy) | Latency | Key Finding |
|-------|-------------|--------------|------|---------------------|-------------------|---------|-------------|
| Mistral-Small-24B | Dense transformer | 24B | 14 GB | **85%** | **66%** | 1,425ms | Best overall; stable across sampling configs |
| **LFM2-24B-A2B** | **Hybrid MoE conv+attn** | **~2B** | **~14.5 GB** | **80%** | **26%** | **390ms** | Production model; 0pp delta greedy vs near-greedy |
| Gemma 3 27B | Dense transformer | 27B | 19 GB | 91% | 48% | 21,464ms | Highest accuracy but impractical latency; 0pp delta |
| Qwen3-30B-A3B-Instruct-2507 | MoE transformer | ~3B | 19 GB | 71% | 42% | 610ms | Massive improvement over original A3B (44%→71%) |
| GPT-OSS-20B | MoE transformer | ~3.6B | 14 GB | 51% | 0% | 2,221ms | Namespace confusion; 0% multi-step (pure deflection) |

### Tier 2 — Small model class (2-4 GB VRAM)

| Model | Architecture | Active Params | VRAM | Single-Step (greedy) | Multi-Step (greedy) | Latency | Key Finding |
|-------|-------------|--------------|------|---------------------|-------------------|---------|-------------|
| **Llama 3.2 3B** | Dense transformer | 3B | ~2.0 GB | **82%** | **52%** | **305ms** | Best small model; beats several 20B+ models |
| Phi-4-mini (3.8B) | Dense transformer | 3.8B | ~2.5 GB | 60% | 14% | 549ms | Good tool call rate (94%), high wrong-tool rate (34%) |
| Qwen3-4B | Dense transformer | 4B | ~2.5 GB | 20% | 0% | 5,837ms | Same `<think>` problem as larger Qwen3; 79% no-tool-call rate |

**Dropped from active benchmarks:**
- Qwen3 32B — only had partial run (40/100 tests), extreme latency
- Qwen3-30B-A3B (original) — replaced by Instruct-2507 variant
- Qwen2.5-32B — dev proxy, too large for target hardware

**Benchmark conditions:** 100 single-step prompts and 50 multi-step chains against all 67 tools (unfiltered), using the same test suite across all models. Tests run on Apple M4 Max (36 GB unified memory) via llama-server (LFM2) or Ollama (all others). Greedy sampling (temp=0) for all results unless noted. See [Tool-Calling Benchmark Results](./tool-calling-benchmark-results.md) for detailed results. For instructions on running these benchmarks and viewing results, see [Benchmark Infrastructure](#benchmark-infrastructure) below.

### Quality benchmark (LFM2-24B-A2B vs Llama 3.2 3B)

Tool selection only measures dispatch accuracy. The [quality benchmark](./quality-benchmark-results.md) measures what happens after dispatch: parameter extraction, instruction following, and synthesis of tool results.

| Dimension | LFM2-24B-A2B | Llama 3.2 3B | Delta |
|---|---|---|---|
| Single-step tool selection | 80% | 82% | -2pp (tie) |
| Multi-step chains | 26% | 52% | -26pp (Llama) |
| **Parameter extraction** | **65%** | 50% | **+15pp (LFM2)** |
| Instruction following | 65% | 63% | +2pp (tie) |
| **Synthesis quality** | **88%** | 74% | **+14pp (LFM2)** |

Both models dispatch tools equally well. LFM2-24B extracts parameters 15pp more accurately and synthesizes tool results 14pp more coherently. See [Quality Benchmark: Beyond Tool Selection](./quality-benchmark-results.md) for the full analysis.

### What the numbers mean

- **Single-step accuracy**: Given a user prompt, does the model select the correct tool from all 67 available? Tested with 100 prompts spanning 12 categories.
- **Multi-step chains**: Can the model complete a 3-6 step workflow (e.g., list files, OCR each, rename)? 50 chains of varying difficulty (simple, medium, complex).
- **Active params**: For MoE models, the number of parameters activated per token. LFM2-24B-A2B has 24B total but only routes ~2B per token.
- **Quality scores**: 150 programmatically-scored tests across parameter extraction (50), instruction following (50), and synthesis (50). No LLM judge — all constraints verified with deterministic checks.

---

## The Core Problem: Tool Cognitive Overload

When you give a local model 67 tool definitions, four things happen simultaneously:

### 1. Token pressure

67 tools = 8,670 tokens = 26.5% of a 32K context window. After reserving 4K for output, the model has ~20K tokens for conversation. A third of that is already consumed by tool schemas the model must parse and reason about on every turn.

### 2. Attention dilution

System prompt workflow examples (showing correct tool chains) sit at ~token position 600. Tool definitions span tokens 900-9,570. By the time the model makes its tool-selection decision, the examples are ~9,000 tokens away. This is the classic "lost in the middle" problem in transformer attention.

### 3. Same-namespace bias

After calling `filesystem.list_dir`, models preferentially select other `filesystem.*` tools rather than crossing to `ocr.*`. Tool names with shared prefixes have higher embedding similarity, creating an anchor effect. The model never reaches for `ocr.extract_text_from_image` because `filesystem.search_files` is semantically adjacent and shares the active namespace.

### 4. Choice overload (the K=15 sweet spot)

Systematic benchmarking of pre-filter sizes revealed a paradox of choice:

| Tools shown to model | Accuracy | Filter Hit Rate |
|---------------------|----------|-----------------|
| 5 | 54% | 75% |
| 10 | 60% | 84% |
| **15** | **78%** | **87%** |
| 20 | 63% | 90% |
| 67 (all) | 36% | n/a |

K=15 is the Pareto-optimal point. Below it, the filter misses the correct tool too often. Above it, semantic near-neighbors confuse the model. This one intervention (RAG-based pre-filtering to K=15) delivered a **117% relative accuracy improvement** at zero training cost.

---

## Five Key Discoveries

### 1. The hybrid design + MoE sparsity combination delivers the best speed-accuracy trade-off

LFM2-24B-A2B activates ~2B parameters per token through its hybrid Mixture-of-Experts architecture (convolution blocks + grouped query attention). GPT-OSS-20B is also MoE (32 experts, top-4 routing) activating ~3.6B parameters per token.

Result: the ~2B-active hybrid MoE scores **80%** while the ~3.6B-active standard MoE scores **51%** (updated from earlier ~36% estimate). LFM2 is also 6x faster (390ms vs 2,303ms). The speed advantage comes from the combination of the hybrid conv+attention design and MoE sparsity. The accuracy difference likely reflects differences in training data and methodology rather than architecture alone — we would not attribute the quality improvement specifically to the hybrid block design.

### 2. Every model fails at the same point

The cross-server tool transition (e.g., `filesystem.*` to `ocr.*`) is the universal failure barrier. It's model-independent:

- **GPT-OSS-20B** deflects to the user ("Which screenshots should I process?")
- **Qwen3-30B-A3B** fixates on the current namespace (calls `filesystem.search_files` four times with identical arguments)
- **LFM2-1.2B-Tool** selects the wrong tool 56% of the time in multi-step context

LFM2-24B-A2B was the first model to demonstrate any cross-server success: 7 of 15 simple chains passed, all requiring multi-server navigation.

### 3. Filtering is the highest-ROI intervention

For LFM2-1.2B-Tool, a RAG embedding pre-filter narrowing 67 tools to 15 candidates raised single-step accuracy from **36% to 78%** (117% relative improvement). The cost: one embedding lookup per query (~10ms). No training, no fine-tuning, no architecture changes.

Counterintuitively, the same filter *degraded* LFM2-24B-A2B from 80% to 72%. The larger model handles the full tool surface better than the filtered subset. This suggests filtering is most valuable for smaller models that can't manage cognitive load natively.

**Takeaway for the community:** If you're building a local agent with 20+ tools, implement tool filtering before anything else. It's the single highest-ROI intervention.

### 4. Decomposition beats scale

A dual-model orchestrator that decomposes multi-step tasks into isolated single-turn decisions delivers **100% completion on 1-2 step workflows** (vs chaotic looping in single-model mode). The architecture:

1. **Plan** (LFM2-24B-A2B): Decompose the task into self-contained steps (bracket-format plans)
2. **Execute** (LFM2.5-1.2B-Router-FT-v2 per step): Select one tool per step from K=15 filtered candidates
3. **Synthesize** (LFM2-24B-A2B): Stream a user-facing summary

Each step is a fresh single-turn decision with no conversation history to corrupt. A/B testing confirmed the dual-model approach eliminates 5 behavioral pathologies (deflection, looping, confabulation, category lock-in, hallucination) and is 2.5x faster than the single-model agent loop.

**Current limitation:** The planner under-decomposes complex 4+ step workflows (collapses to 1 step). See [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md) for the full A/B test analysis.

### 5. Infrastructure amplifies model capability

Same model (LFM2-24B-A2B), same task (rename screenshots by OCR content):

| Session | Infrastructure | Files renamed | Tool calls | Duration |
|---------|---------------|--------------|-----------|----------|
| Pre-fix | Basic agent loop | **0** (confabulated completion) | 8 | 28s |
| Post-fix | +semantic aliases, +correction context, +confabulation detection | **1** (partial success) | 14 | 46s |

The model went from fabricating results to actually completing a cross-server workflow. The infrastructure changes that made the difference:

- **Semantic aliases**: `rename_file` resolves to `move_file` (semantic match at score 1.0, vs Levenshtein which incorrectly suggests `read_file`)
- **Correction context**: Error messages explain what happened and how to recover
- **Confabulation detection**: Claims of "successfully renamed all 9 files" are verified against tool call history
- **Deflection trust gate**: After 5+ successful tool calls, the model's text response is trusted as a summary (not flagged as deflection)

---

## The Failure Taxonomy

Across 4 models and 150+ scenarios, we identified 12 failure modes in three categories:

| Category | ID | Failure | Frequency | Mitigation |
|----------|----|---------|-----------|-----------|
| **Structural** | FM-1 | Malformed JSON | ~5% | GBNF grammar constraints |
| | FM-2 | Empty responses | ~10% | Nudge prompt injection |
| | FM-8 | Unprefixed tool names | 10-25% | Fuzzy name resolver |
| **Semantic** | FM-3 | Conversational deflection | 80% (GPT-OSS) | 21-pattern deflection detector |
| | FM-4/5 | Parameter hallucination | ~30% | Validation layer |
| | FM-7 | Premature abandonment | ~60% (>5 files) | Incomplete response detection |
| | FM-10 | Redundant tool calls | ~20% | MAX_ROUNDS cap |
| **Systemic** | FM-11 | Tool cognitive overload | Universal | K=15 pre-filter |
| | FM-12 | Context window pressure | Cumulative | Mid-loop eviction |

The critical interaction chain: FM-11 (overload) triggers FM-3 (deflection), which causes the agent loop to exit prematurely. This chain accounts for the majority of multi-step failures.

See [gpt-oss-20b.md](./gpt-oss-20b.md) for the complete taxonomy with evidence and workarounds.

---

## Per-Category Accuracy (LFM2-24B-A2B, 67 tools unfiltered)

| Category | Accuracy | Tests |
|----------|----------|-------|
| Calendar | **100%** | 7/7 |
| Audit | **100%** | 3/3 |
| Security/Privacy | **90%** | 9/10 |
| Task Management | **88%** | 7/8 |
| Document Processing | **83%** | 10/12 |
| File Operations | **80%** | 12/15 |
| System/Clipboard | **80%** | 4/5 |
| OCR/Vision | 75% | 6/8 |
| Email | 75% | 6/8 |
| Knowledge/Search | 71% | 5/7 |
| Meeting/Audio | 63% | 5/8 |
| Data Operations | **60%** | 6/10 |

**Strongest domains:** Structured, unambiguous tools (calendar, audit, security). **Weakest:** Tools with semantic overlap (data operations, meeting/audio) where multiple tools could plausibly apply.

---

## Reading Guide

### Start here

- **[Project Learnings and Recommendations](./project-learnings-and-recommendations.md)** — 5 models, 12 failure modes, and the 3 interventions that work (filtering, fine-tuning, orchestration) — with measured impact for each.

### Competitive comparison

- **[Tool-Calling Benchmark Results](./tool-calling-benchmark-results.md)** — LFM2-24B-A2B (2B active) vs 7 models across 2 tiers. Includes greedy sampling re-run (temp=0) and Tier 2 small model comparison (3-4B dense). Same 67-tool benchmark, same Apple M4 Max hardware.
- **[Quality Benchmark: Beyond Tool Selection](./quality-benchmark-results.md)** — 150 tests measuring parameter extraction accuracy, instruction following precision, and synthesis quality. LFM2-24B-A2B (72.5%) vs Llama 3.2 3B (62.2%). Fully programmatic scoring, no LLM judge. Shows where the 24B model's knowledge capacity separates from a 3B dense model.

### Per-model deep dives

- **[GPT-OSS-20B](./gpt-oss-20b.md)** — 12 failure modes with evidence, workaround registry, and the multi-model architecture options that emerged from this analysis. Read this to understand the failure taxonomy.
- **[Qwen3-30B-A3B](./qwen3-30b-a3b-tool-calling.md)** — MoE tool fixation loop analysis. Read this to understand why MoE != efficient tool calling, and the 4 root cause hypotheses.
- **[LFM2-24B-A2B Benchmark](./lfm2-24b-a2b-benchmark.md)** — Formal benchmark results (80% single-step, 26% multi-step), per-category breakdown, real-world execution traces, and infrastructure impact analysis.

### Orchestrator & fine-tuning

- **[Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md)** — Architecture (Flow A/B), the 11 fixes that made the orchestrator production-viable, A/B test methodology and results, head-to-head single-model vs dual-model comparison, and operational serving guides.
- **[Fine-Tuning Results](./fine-tuning-results.md)** — Two iterations of LoRA fine-tuning on LFM2.5-1.2B-Instruct for tool routing. V1 vs V2 comparison, training data generation, live test results, failure analysis.

### Roadmap

- **[Accuracy Improvement Roadmap](./accuracy-improvement-roadmap.md)** — Strategic roadmap from 80% to 90%+ single-step and 26% to 60-75% multi-step. Covers hierarchical routing, dedicated embeddings, essential tool set, multi-step interventions (M1-M5), and GRPO reinforcement learning — with a phased implementation sequence.

### Architecture decisions driven by these findings

- **[ADR-008: Tool-Calling Optimization Strategy](../architecture-decisions/008-tool-calling-optimization-strategy.md)** — 4-layer optimization stack (grammar, few-shot, sampling, model evaluation)
- **[ADR-009: Dual-Model Orchestrator](../architecture-decisions/009-dual-model-orchestrator.md)** — Plan-execute-synthesize pipeline design (operational for 1-2 step workflows)
- **[ADR-010: RAG Pre-Filter Benchmark](../architecture-decisions/010-rag-prefilter-benchmark-analysis.md)** — K-value optimization and per-category analysis

---

## The Big Takeaway: Dispatchers, Not Autonomous Agents

> **TL;DR — 24B-class local models are good dispatchers, not good autonomous agents. Design the UX around single-turn interactions with human confirmation, not autonomous multi-step workflows. That turns 80% model accuracy into near-100% effective accuracy — and that's a genuinely useful product.**

### What works well: single-turn, single-tool tasks

80% accuracy at 67 tools means 4 out of 5 times, the user says something and the model picks the right tool. For a desktop assistant, that's usable — especially in the high-signal categories:

| Category | Accuracy | Verdict |
|----------|----------|---------|
| Calendar/scheduling | 100% | Just works |
| Audit/session history | 100% | Just works |
| Security scanning | 90% | Just works |
| Task management | 88% | Just works |
| File operations | 80% | Works most of the time |

These are the bread-and-butter interactions — a user asks one thing, the model picks one tool, the tool executes, done.

### What doesn't work: autonomous multi-step chains

26% chain completion means 3 out of 4 multi-step workflows fail somewhere. "Download my receipts, extract text from each PDF, and reconcile against the bank CSV" — that's a 3-step chain, and it'll fail ~74% of the time. The orchestrator doesn't help — it adds pipeline stages that each have their own failure rate.

### The practical product strategy

Design the UX around single-turn interactions, not autonomous workflows:

1. **One tool at a time, human in the loop.** The model picks the tool, shows a preview, the user confirms. If the model picks wrong (20% of the time), the user redirects. This turns 80% accuracy into near-100% effective accuracy because the human catches errors.

2. **Guided multi-step, not autonomous multi-step.** Instead of "do these 5 things in sequence automatically," the UX is conversational: do step 1, show the result, let the user say "now do step 2 with that." Each step is a single-turn with human verification — avoiding the compounding failure problem entirely.

3. **The 20% error isn't a blocker — it's a UX problem.** The model never does anything without confirmation (that's the human-in-the-loop pattern). Wrong tool selection means one extra click to redirect, not a destructive mistake.

4. **Use the model's strengths for what matters.** Calendar, tasks, file management, security scanning — these are high-frequency daily actions where 80-100% accuracy with confirmation is genuinely faster than doing it manually. That's the core value proposition.

### What 24B-class models can and can't do in agentic setups

**Can do:**
- Understand user intent and map it to the right tool with high accuracy
- Handle a surprisingly large tool catalog (67 tools, only 5pp degradation from 31)
- Respond fast enough for interactive use (395ms)

**Can't do:**
- Maintain coherence across a multi-step plan (error compounds too fast)
- Self-correct when a step fails (no reliable re-planning)
- Replace a human's judgment about what to do next

### The honest framing

LocalCowork isn't a local Claude that autonomously handles complex workflows. It's a fast, private, tool-aware assistant that makes individual actions quicker — and that's actually a solid product.

---

## Getting Maximum Leverage from 24B-Class Models

If you're deploying a 24B-class local model for tool calling, here's what to do — ranked by measured impact from our benchmarks. Each intervention is independent; they compound when stacked.

### Tier 1: Do these first (validated, zero/low cost)

| # | Intervention | Impact | Cost | Evidence |
|---|-------------|--------|------|----------|
| 1 | **RAG pre-filter to K=15 tools** | 36% → 78% accuracy (+117%) | ~10ms per query | [ADR-010](../architecture-decisions/010-rag-prefilter-benchmark-analysis.md) |
| 2 | **GBNF grammar-constrained decoding** | Eliminates ~100% of malformed JSON | Zero runtime cost | [ADR-008](../architecture-decisions/008-tool-calling-optimization-strategy.md) Layer 1 |
| 3 | **Contrastive tool descriptions** | Reduces sibling confusion from 31% to ~18% | One-time authoring effort | Each tool description says what it does AND what the similar tool does instead |
| 4 | **Dynamic inference parameters** | ~10% accuracy improvement on tool-calling turns | Zero cost | Temperature 0.1 / top_p 0.2 for tool selection; 0.7 / 0.9 for conversation |
| 5 | **Human-in-the-loop confirmation** | Turns 80% model accuracy into ~100% effective accuracy | UX design | The 20% error becomes one extra click, not a wrong action |

**Note on filtering + LFM2-24B-A2B specifically:** The K=15 pre-filter actually *degraded* the 24B model from 80% to 72%. This model handles the full 67-tool surface better unfiltered. Pre-filtering is most valuable for sub-7B models. For the 24B model, skip #1 and focus on #2-5.

### Tier 2: Infrastructure hardening (moderate effort, compounding returns)

| # | Intervention | Impact | What it fixes |
|---|-------------|--------|---------------|
| 6 | **Semantic tool name resolution** | Catches 10-25% of unprefixed/aliased tool calls | Model says `rename_file` → resolver maps to `filesystem.move_file` |
| 7 | **Confabulation detection** | Catches fabricated success claims | Model says "renamed all 9 files" but tool history shows 1 call |
| 8 | **Deflection detection (21 patterns)** | Catches 80% of conversational deflection | Model asks "Which files?" instead of calling `list_dir` |
| 9 | **Mid-loop context eviction** | Prevents silent context overflow after round 5+ | Evict after each tool round based on actual token usage, not round count |
| 10 | **JSON-specific token estimation** | Prevents 25% underestimation on structured content | 2.8 chars/token for JSON vs 4.0 for prose — the gap compounds across turns |

These individually seem small but they interact: confabulation detection (#7) catches the failures that deflection detection (#8) misses, and both depend on accurate token estimation (#10) to avoid context overflow that triggers the failures in the first place.

### Tier 3: Model-level improvements (higher effort, highest ceiling)

| # | Intervention | Expected impact | Cost |
|---|-------------|----------------|------|
| 11 | **Fine-tune on your tool schemas (QLoRA)** | 80% → 85-90% single-step | $20-100, 1-2 days on consumer GPU |
| 12 | **Hierarchical category→tool routing** | Eliminates cross-server confusion (16% of errors) | 2-3s extra latency per query |
| 13 | **Dedicated embedding model for pre-filter** | 87% → 95%+ filter hit rate | ~30ms latency, 137M param model |
| 14 | **Plan-execute-synthesize orchestrator** | 26% → 50-60% chain completion (projected) | ~7-8 GB total VRAM, added complexity |

Fine-tuning (#11) is the highest-confidence path: the training data already exists (your benchmark scenarios), the remaining errors (sibling confusion, cross-server confusion) are exactly what domain-specific training fixes, and the accuracy improvement compounds across chains (0.85^4 = 52% vs 0.80^4 = 41%).

### What doesn't help

- **Larger K values (K > 20)** — accuracy drops due to choice overload, even though filter coverage improves
- **More tools** — adding tools makes the problem worse; every 10 additional tools costs ~2-3pp accuracy
- **Scaling the same architecture** — GPT-OSS-20B (MoE, ~3.6B active) scored 51% vs LFM2-24B-A2B (hybrid MoE, ~2B active) at 80%. More active parameters with a standard MoE architecture doesn't close the gap
- **Longer system prompts with more examples** — pushes tool definitions further from the decision point, worsening the "lost in the middle" effect
- **Autonomous multi-step without decomposition** — error compounds geometrically (0.80^n), so a 5-step chain has 33% success even at 80% per-step

For the deep dive on each strategy with implementation details, see [Project Learnings and Recommendations](./project-learnings-and-recommendations.md#8-strategies-worth-exploring-ranked-by-evidence).

---

## The Bottom Line

**LFM2-24B-A2B at 80% single-step accuracy is the production model for LocalCowork.** It activates only ~2B parameters per token, fits in 13 GB VRAM, and responds in under 400ms. It's the first model in our testing to successfully navigate cross-server tool transitions.

The gap to cloud-hosted models (Claude, GPT-4 — likely 90%+ chain completion with 200K+ context and 100B+ parameters) is real: ~30-40 percentage points on multi-step chains. Three paths to close it:

1. **Filter aggressively** — Don't show the model tools it doesn't need. K=15 pre-filtering gave us 117% relative improvement on the 1.2B model.
2. **Orchestrate deliberately** — Decompose multi-step tasks into isolated single-turn decisions. Each step is a fresh context with no accumulated errors. See [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md) for how this works in practice.
3. **Invest in infrastructure** — Semantic aliases, correction context, confabulation detection, and deflection gating turned 0 successful file renames into 1. That's the difference between "doesn't work" and "works sometimes."

For the broader community building local AI agents: the era of "throw all tools at a big model" is over for on-device deployment. Architecture, filtering, and infrastructure tooling matter more than parameter count. A well-instrumented 2B-active model outperforms a naive 20B dense model by a wide margin.

---

## Benchmark Infrastructure

### Source Files

| File | What it does | Tests |
|------|-------------|-------|
| `tests/model-behavior/benchmark-lfm.ts` | Single-step tool selection | 100 prompts across 12 categories |
| `tests/model-behavior/benchmark-multi-step.ts` | Multi-step chain completion | 50 chains (simple, medium, complex) |
| `tests/model-behavior/benchmark-orchestrator.ts` | Dual-model orchestrator | 50 chains via plan-execute-synthesize |
| `tests/model-behavior/benchmark-quality.ts` | Quality: param extraction, instruction following, synthesis | 150 tests (50 per module) |
| `tests/model-behavior/quality-scoring.ts` | Programmatic scoring functions (no LLM judge) | — |
| `tests/model-behavior/param-extraction-tests.ts` | 50 param extraction test definitions | — |
| `tests/model-behavior/instruction-following-tests.ts` | 50 instruction following test definitions | — |
| `tests/model-behavior/synthesis-tests.ts` | 50 synthesis test definitions | — |
| `scripts/benchmark-lfm2-24b.sh` | All-in-one runner (single-model + orchestrator + report) | — |
| `_models/config.yaml` | Model config (active model, fallback chain, sampling params) | — |

### Running Benchmarks

**Prerequisites:** Start a model server before running any benchmark.

```bash
# LFM2-24B-A2B via llama-server (default port 8080)
llama-server -m _models/LFM2-24B-A2B-Preview-Q4_K_M.gguf --port 8080

# Any Ollama model (default port 11434)
ollama run llama3.2
```

**Tool selection — single-step (100 prompts):**

```bash
# LFM2-24B via llama-server
npx tsx tests/model-behavior/benchmark-lfm.ts --endpoint http://localhost:8080 --greedy

# Llama 3.2 3B via Ollama
npx tsx tests/model-behavior/benchmark-lfm.ts --endpoint http://localhost:11434 --model llama3.2 --greedy

# With RAG pre-filter (narrows 67 tools to top-K candidates)
npx tsx tests/model-behavior/benchmark-lfm.ts --endpoint http://localhost:8080 --greedy --filter --topk 15
```

**Tool selection — multi-step chains (50 chains):**

```bash
npx tsx tests/model-behavior/benchmark-multi-step.ts --endpoint http://localhost:8080 --greedy
npx tsx tests/model-behavior/benchmark-multi-step.ts --endpoint http://localhost:11434 --model llama3.2 --greedy
```

**Quality benchmark (150 tests — param extraction + instruction following + synthesis):**

```bash
# All 3 modules
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy

# Individual modules
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module params
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module instructions
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:8080 --greedy --module synthesis

# Against Ollama
npx tsx tests/model-behavior/benchmark-quality.ts --endpoint http://localhost:11434 --model llama3.2 --greedy
```

**Orchestrator (dual-model plan-execute-synthesize):**

```bash
npx tsx tests/model-behavior/benchmark-orchestrator.ts --endpoint http://localhost:8080 --greedy
```

**All-in-one runner script:**

```bash
./scripts/benchmark-lfm2-24b.sh
```

**CLI flags reference:**

| Flag | Default | Description |
|------|---------|-------------|
| `--endpoint` | `http://localhost:8080` | Model server URL |
| `--model` | _(none)_ | Model name for Ollama (required for Ollama, omit for llama-server) |
| `--greedy` | off | Greedy sampling: temp=0, top_p=1.0 (recommended for reproducibility) |
| `--filter` | off | Enable RAG embedding pre-filter |
| `--topk` | 15 | Number of tools to show model when `--filter` is on |
| `--timeout` | 30000 | Per-request timeout in ms |
| `--module` | _(all)_ | Quality benchmark only: `params`, `instructions`, or `synthesis` |

### Viewing Results

Results are saved as JSON to `tests/model-behavior/.results/`. Each run creates a timestamped file.

**File naming convention:**

| Pattern | Benchmark type |
|---------|---------------|
| `lfm-unfiltered-k0-<timestamp>.json` | Single-step, all 67 tools |
| `lfm-filtered-k15-<timestamp>.json` | Single-step, RAG pre-filtered to K tools |
| `lfm-multistep-all-<timestamp>.json` | Multi-step chain completion |
| `orchestrator-all-<timestamp>.json` | Dual-model orchestrator |
| `quality-param-extraction-instruction-following-synthesis-<timestamp>.json` | Quality benchmark (all 3 modules) |

**JSON structure — tool selection results:**

```json
{
  "runId": "lfm-1771786409609",
  "timestamp": "2026-02-22T...",
  "endpoint": "http://localhost:8080",
  "sampling": { "mode": "greedy", "temperature": 0, "topP": 1 },
  "summary": { "total": 100, "passed": 80, "failed": 20, "accuracy": 0.80 },
  "results": [
    { "testId": "ts-file-001", "status": "passed", "expectedTools": ["filesystem.list_dir"], "actualTools": ["filesystem.list_dir"], "durationMs": 312 }
  ]
}
```

**JSON structure — quality benchmark results:**

```json
{
  "runId": "quality-1771791851178",
  "sampling": { "mode": "greedy", "temperature": 0, "topP": 1 },
  "modules": {
    "param-extraction": { "totalTests": 50, "avgScore": 0.649, "categories": { "path-extraction": { "total": 10, "avgScore": 0.725 } } },
    "instruction-following": { "totalTests": 50, "avgScore": 0.647, "categories": { ... } },
    "synthesis": { "totalTests": 50, "avgScore": 0.880, "categories": { ... } }
  }
}
```

**Quick commands to extract scores:**

```bash
# Overall accuracy from a tool selection run
cat tests/model-behavior/.results/lfm-unfiltered-k0-*.json | jq '.summary.accuracy'

# Per-module scores from a quality run
cat tests/model-behavior/.results/quality-*.json | jq '.modules | to_entries[] | {module: .key, score: .value.avgScore}'

# Per-category breakdown
cat tests/model-behavior/.results/quality-*.json | jq '.modules["param-extraction"].categories'

# List all failed tests from a run
cat tests/model-behavior/.results/lfm-unfiltered-k0-*.json | jq '[.results[] | select(.status == "failed")] | length'

# Compare two runs side by side
diff <(cat .results/run1.json | jq '.summary') <(cat .results/run2.json | jq '.summary')
```
