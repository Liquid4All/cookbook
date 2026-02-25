# Local AI Agent with 50+ Tools: What We Learned

**Project:** LocalCowork — on-device desktop AI agent
**Last Updated:** 2026-02-18
**Scope:** 5 models tested, 83 tools across 15 MCP servers, a fine-tuned router, and a dual-model orchestrator validated via A/B testing

---

## TL;DR

Can a local model handle 50+ MCP tools? Not naively. But with the right architecture, yes.

We tested five models against 83 tools. Every model failed at cross-server tool transitions when given the full tool surface. Three interventions fixed this — each with measured impact:

| Intervention | Single-step accuracy | Multi-step completion | Cost |
|---|---|---|---|
| Baseline (all tools, no help) | 36% | 0-8% | — |
| RAG pre-filter (K=15) | 78% (on 1.2B router) | 8% | ~10ms per query |
| Fine-tuned 1.2B router (LoRA V2) | 84% (100% at K=25) | — | $5 training, 6 min on H100 |
| Dual-model orchestrator | 100% tool selection | 100% (1-2 step), partial (4+) | 2 models, ~14.5 GB VRAM |

**Start with tool filtering.** It's the single highest-ROI intervention for any local agent with 20+ tools.

---

## Models Tested

| Model | Active Params | VRAM | Single-step (67 tools) | Multi-step | Key Failure Mode |
|---|---|---|---|---|---|
| GPT-OSS-20B | ~3.6B (MoE) | ~14 GB | ~36% | ~0% | Conversational deflection (80% of tasks) |
| Qwen2.5-32B | 32B (dense) | ~20 GB | ~36% | ~0% | Same cross-server failures |
| Qwen3-30B-A3B | ~3B (MoE) | ~5 GB | ~36% | ~0% | Tool fixation loops (repeats wrong tool 4x) |
| LFM2-1.2B-Tool | 1.2B | ~2.3 GB | 78% (K=15) | 8% | Wrong tool under context pressure |
| **LFM2-24B-A2B** | **~2B (MoE)** | **~13 GB** | **80%** | **26%** | Wrong tool (54%), but first to cross servers |

**Cross-cutting finding:** The cross-server tool transition failure is model-independent. All models fail when a task requires jumping from one server namespace (e.g., `filesystem.*`) to another (e.g., `ocr.*`). This is a structural problem — tool count and context layout — not a model deficiency.

---

## The 50+ Tool Problem: Why Models Fail

Four root causes, each empirically validated:

1. **Token pressure.** 83 tool definitions consume ~14,000 tokens — 43% of a 32K context. The model's reasoning capacity is halved before the conversation starts.

2. **Attention dilution.** Few-shot workflow examples in the system prompt (~token 600) are buried under ~14,000 tokens of tool schemas by the time the model makes its selection decision. Classic "lost in the middle."

3. **Same-namespace bias.** After using `filesystem.list_dir`, models preferentially select other `filesystem.*` tools over `ocr.extract_text_from_image`. Shared prefixes create anchor bias.

4. **Choice overload (K=15 sweet spot).** Accuracy peaks at 15 candidate tools. Above K=20, it drops — even though filter coverage improves. Five extra semantically-similar tools are enough to confuse a 1.2B model.

---

## What We Built and What It Achieved

### 1. RAG Tool Pre-Filter (highest ROI)

Embedding-based filter narrows 83 tools to K=15 per query using cosine similarity. Combined with contrastive tool descriptions that differentiate sibling tools.

| Metric | No filter (83 tools) | K=15 filtered | K=15 + prompt engineering |
|---|---|---|---|
| Accuracy (1.2B) | 36% | 68% | **78%** |
| No tool call rate | 53% | 13% | 6% |

**Counterintuitive:** The same filter *degraded* LFM2-24B-A2B from 80% to 72%. The larger model handles the full tool surface better. **Lesson:** Always benchmark filtering per model.

### 2. Fine-Tuned Router (LoRA, 2 iterations)

LoRA fine-tuned LFM2.5-1.2B-Instruct on project-specific tool schemas. V2 (4,314 examples, r=64) is production.

| Metric | Base model | V1 (841 examples) | V2 (4,314 examples) |
|---|---|---|---|
| Live accuracy (K=15) | 78% | 83% | 84% |
| K=25 accuracy | collapsed | not tested | **100%** |
| New servers (3) | 0% | 0% | 86% |
| Training time (H100) | — | 2 min | 6 min |

**Key insight:** V2's remaining failures (8/49) are dominated by cross-server confusion (62.5%), not format errors. See [Fine-Tuning Results](./fine-tuning-results.md).

### 3. Dual-Model Orchestrator (A/B tested)

Plan (LFM2-24B-A2B) → Execute (fine-tuned 1.2B router per step) → Synthesize (LFM2-24B-A2B). Each step is a clean single-turn decision — no conversation history to corrupt.

| Metric | Single-Model (24B solo) | Dual-Model Orchestrator |
|---|---|---|
| 1-step task | Correct, then looped 8x | Clean single-pass |
| 2-step cross-server | Correct, then 7 wasted calls | Clean two-step |
| 4+ step complex | 3/6 tools (stuck) | 1/4 tools (planner limit) |
| Behavioral pathologies | 5 types | **None** |
| Wall time (3 test msgs) | 54s | **21s** |
| Wasted model calls | 16/27 | **0/10** |

Dual-model eliminates deflection, looping, confabulation, category lock-in, and hallucination on 1-2 step workflows. See [Orchestrator Performance](./dual-model-orchestrator-performance.md).

### 4. Agent Loop Infrastructure

Same model, same task (rename screenshots by OCR content):

| Session | Infrastructure | Files Renamed | Error Recovery |
|---|---|---|---|
| Pre-fix | Basic agent loop | **0** (confabulated completion) | 0 of 4 errors |
| Post-fix | Aliases + correction context + confabulation detection | **1** | 3 of 5 errors |

Infrastructure doesn't prevent errors (models always hallucinate tool names). It **enables recovery from errors**. The pattern: model hallucinates → error with context → model self-corrects → success.

---

## Failure Taxonomy (12 Modes)

Organized by what fixes them:

### Fixed by grammar/infrastructure (cheap)

| FM | Failure | Fix |
|---|---|---|
| FM-1 | Malformed JSON/bracket syntax | GBNF grammar constraints |
| FM-2 | Empty responses | Nudge prompt injection |
| FM-8 | Unprefixed tool names | `resolve_tool_name()` fuzzy matcher |

### Fixed by prompt engineering (free)

| FM | Failure | Fix |
|---|---|---|
| FM-4 | Example path leakage | Use fictional paths in examples |
| FM-5 | Parameter name hallucination | Correct few-shot examples |
| FM-9 | Relative paths (`~/`) | System prompt rule |
| FM-10 | Redundant tool calls | System prompt rule |

### Requires architecture (expensive but high-impact)

| FM | Failure | Fix |
|---|---|---|
| FM-3 | Conversational deflection | Deflection detection (21 patterns) + dual-model orchestrator |
| FM-7 | Premature task abandonment | `is_incomplete_response()` + continuation prompts |
| FM-11 | Tool cognitive overload | RAG pre-filter to K=15 |
| FM-12 | Context window pressure | Mid-loop eviction, result truncation |

**Critical chain:** FM-11 (overload) → FM-3 (deflection) → agent loop exits. Breaking FM-11 with a pre-filter is the single most impactful intervention.

---

## Transferable Patterns

These apply to any system running a local LLM agent with tools.

**JSON token estimation.** Structured content tokenizes at ~2.8 chars/token, not ~4.0 for prose. After 14 tool-calling turns, the cumulative error reaches ~3,000 tokens — enough to silently overflow a 32K context. Use role-aware estimation.

**Mid-loop eviction.** A fixed `MAX_ROUNDS = 10` is too blunt. Evict after each tool round based on actual token usage. Short results → more rounds. Large results → earlier eviction.

**Anti-confabulation.** When injecting "summarize what was found," constrain it: "ONLY report results you received from tool calls above." Otherwise the model fabricates results.

**Tool result truncation.** Cap results at 6,000 chars. A single 50KB file listing can starve the entire context.

**Deflection detection.** "I see the files. What would you like me to do?" looks like a complete response. Build a dedicated detector: 21 patterns + short-question heuristic (< 300 chars + `?` + tools were executed).

**The 4-layer optimization stack** (apply in order, each is free or near-free):
1. Grammar constraints (GBNF) — eliminate structural errors
2. Few-shot examples — correct tool names, paths, multi-step sequences
3. Dynamic inference params — temperature 0.1 for tool turns, 0.7 for conversation
4. Model evaluation — benchmark before switching

Only fine-tune after all four layers are deployed.

---

## What's Next (Ranked by Evidence)

### Proven — implement first

- **Tool filtering** for any agent with 20+ tools. K=15 gave 117% relative improvement for zero training cost.
- **Dual-model orchestrator** for multi-step workflows. Eliminates 5 pathology classes, 2.5x faster.
- **LoRA fine-tuning** on project-specific schemas. 6 minutes on an H100, pushes accuracy from 78% to 84%.

### High confidence — not yet validated

- **Hierarchical routing** (category → tool). Two-stage: pick 2-3 of 15 servers, then select tool within those. Eliminates cross-server confusion entirely.
- **GRPO reinforcement learning** on the router. Reward: +1 correct tool, -1 wrong. Expected to push past 90%.

### Worth monitoring

- MCP hierarchical tool management (lazy loading, category-based discovery)
- On-device fine-tuning (Apple MLX, Qualcomm AI Engine)
- Larger context windows (64-128K becoming standard in 2026 local models)

---

## The Gap to Cloud Quality

| Dimension | Cloud Agent (estimated) | LocalCowork (best) |
|---|---|---|
| Context window | 200K+ tokens | 32K tokens |
| Parameters | 100B+ (dense) | ~2B active (MoE) + 1.2B router |
| Chain completion | 90%+ | 100% (1-2 step), partial (4+) |
| Latency per step | ~1-2s | ~3-5s (orchestrator) |

**What closes the gap:** Better planners (decompose 4+ step tasks), fine-tuned routers, larger context windows.
**What doesn't:** Prompt engineering alone (proven ceiling), larger K values (accuracy drops above K=20), just adding parameters (GPT-OSS-20B and Qwen3-30B had the same cross-server failure as smaller models).

---

## References

- [LFM2-24B-A2B Benchmark](./lfm2-24b-a2b-benchmark.md) — 80% single-step, 26% multi-step, per-category breakdown
- [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md) — A/B test results, 11 fixes, architecture
- [Fine-Tuning Results](./fine-tuning-results.md) — V1/V2 LoRA training on LFM2.5-1.2B-Instruct
- [GPT-OSS-20B](./gpt-oss-20b.md) — 12 failure modes taxonomy
- [Qwen3-30B-A3B Analysis](./qwen3-30b-a3b-tool-calling.md) — MoE tool fixation analysis
- [ADR-008: Tool-Calling Optimization](../architecture-decisions/008-tool-calling-optimization-strategy.md) — 4-layer stack
- [ADR-009: Dual-Model Orchestrator](../architecture-decisions/009-dual-model-orchestrator.md) — plan/execute/synthesize pipeline
- [ADR-010: RAG Pre-Filter](../architecture-decisions/010-rag-prefilter-benchmark-analysis.md) — K-value analysis
