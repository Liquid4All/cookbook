# LFM2-24B-A2B-Preview — Benchmark & Execution Results

**Status:** Production model (planner + synthesizer in dual-model orchestrator)
**Date:** 2026-02-18 (updated with real-world execution traces, orchestrator A/B results)
**Benchmark script:** `scripts/benchmark-lfm2-24b.sh`
**Config entry:** `_models/config.yaml` → `lfm2-24b-a2b`

> **Related:** For the dual-model orchestrator A/B test results (single-model vs dual-model),
> see [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md).

---

## Model Identity

| Property         | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| Model            | LFM2-24B-A2B-Preview                                               |
| Architecture     | Hybrid MoE (convolution + grouped query attention)                 |
| Total params     | ~24B                                                               |
| Active per token | ~2B                                                                |
| Context window   | 32,768 tokens                                                      |
| Quantization     | Q4_K_M                                                             |
| VRAM             | ~13 GB (GGUF file size)                                            |
| Runtime          | llama-server (llama.cpp)                                           |
| Port             | 8080                                                               |
| Decode speed     | ~121 tokens/sec (Apple Silicon, Metal)                             |
| Tool call format | LFM bracket syntax (`<\|tool_call_start\|>...<\|tool_call_end\|>`) |
| Source           | https://huggingface.co/LiquidAI/LFM2-24B-A2B-Preview (gated)       |

### Why this model

LFM2-24B-A2B shares the same hybrid architecture as LFM2-8B-A1B (double-gated short-range convolution blocks + grouped query attention blocks) but scaled to 24B total / ~2B active per token. The key hypothesis: Liquid AI's convolution-heavy hybrid architecture may handle tool schemas differently than pure transformer MoE models (Qwen3-30B-A3B), potentially breaking through the cross-server transition failure that all previously tested models exhibited.

Additionally, LFM2 models are optimized for on-device deployment with native tool-calling support, making this a strong candidate for the production target originally designated for LFM2.5-24B.

---

## Path A: Single-Model Agent Loop (main branch)

The main branch runs a single model in the agent loop. All 67 tools are sent as definitions in every request. This tests the model's raw ability to handle the full tool surface without filtering or decomposition.

### Test 1: 100 Single-Step Tool Selection (all 67 tools, no pre-filter)

| Metric       | LFM2-24B-A2B (Run 2) | LFM2-24B-A2B (Run 1) | GPT-OSS-20B (baseline) | Qwen3-30B-A3B (baseline) | LFM2-1.2B-Tool (baseline) |
| ------------ | --------------------- | --------------------- | ---------------------- | ------------------------ | ------------------------- |
| Accuracy     | **80%** ✅             | 78%                   | ~36%\*                 | ~36%\*                   | 36%                       |
| Wrong tool   | 14%                   | 14%                   | 17%                    | —                        | 17%                       |
| No tool call | 6%                    | 8%                    | 53%                    | —                        | 53%                       |
| Tool call rate | 94%                 | 92%                   | ~47%                   | —                        | ~47%                      |
| Restraint    | 0.86                  | 0.86                  | —                      | —                        | —                         |
| Avg latency  | 395ms                 | 436ms                 | ~5-21s                 | ~22-34s                  | ~0.4s                     |

\* Estimated from observed multi-step behavior. Formal single-step benchmarks not run with all 67 tools for GPT-OSS-20B and Qwen3-30B-A3B.

Run 2 was performed after agent loop hardening (semantic aliases, correction context, confabulation detection). The model improved from 78% → **80%** with faster average latency (395ms vs 436ms) and a higher tool call rate (94% vs 92%).

**Key finding:** LFM2-24B-A2B achieves **80% accuracy with all 67 tools unfiltered** — more than double the 36% baseline. This exceeds the accuracy that LFM2-1.2B-Tool achieved with K=15 pre-filtering (78%). The model has a 94% tool call rate (vs ~47% for baselines), meaning it almost always attempts a tool call rather than deflecting.

### Test 2: 100 Single-Step Tool Selection (K=15 pre-filter, self-embeddings)

| Metric            | LFM2-24B-A2B (self-embed) | LFM2-1.2B-Tool (self-embed baseline) |
| ----------------- | ------------------------- | ------------------------------------ |
| Accuracy          | **72%**                   | 78%                                  |
| Filter hit rate   | 94%                       | 94%                                  |
| Wrong tool        | 20%                       | 16%                                  |
| No tool call      | 8%                        | 6%                                   |
| Avg tools sent    | 15                        | 15                                   |
| Avg latency       | 992ms                     | —                                    |

**Surprising finding:** K=15 pre-filtering actually **degraded** accuracy from 78% (unfiltered) to 72% (filtered). Two factors:

1. **Embedding quality mismatch:** LFM2-24B-A2B's embeddings (dim=2048) produce different similarity rankings than LFM2-1.2B-Tool's. The filter hit rate is 94% (same as 1.2B), but the wrong tools in the filtered set may confuse the larger model differently.
2. **Latency doubling:** Each query requires an embedding call + completion call, pushing per-query latency from 436ms to 992ms.
3. **The model is already strong at 67 tools:** The unfiltered accuracy is high enough that filtering introduces more harm (wrong tools in the reduced set) than benefit (narrower decision space).

**Implication:** For LFM2-24B-A2B specifically, skip the pre-filter and use all 67 tools directly. The model handles the full tool surface better than any model tested so far.

### Test 3: 50 Multi-Step Chains (all 67 tools, no pre-filter)

| Metric               | LFM2-24B-A2B (Run 2) | LFM2-24B-A2B (Run 1) | GPT-OSS-20B | Qwen3-30B-A3B | LFM2-1.2B-Tool |
| -------------------- | --------------------- | --------------------- | ----------- | ------------- | -------------- |
| Chain completion     | **26%**               | 24%                   | ~0%         | ~0%           | 8%             |
| Step completion      | **31%**               | 31%                   | —           | —             | 16%            |
| Avg steps/chain      | 1.4                   | 1.4                   | —           | —             | —              |
| FM-3 deflection rate | 12%                   | 10%                   | ~80%        | ~0%           | 4%             |
| Wrong tool rate      | 54%                   | 56%                   | —           | —             | —              |
| No tool call rate    | 8%                    | 10%                   | —           | —             | —              |
| Tool fixation loops  | No                    | No                    | No          | Yes           | No             |
| Duration             | 58s                   | 65s                   | —           | —             | —              |

**By difficulty (Run 2):**

| Difficulty | Chain completion | Chains passed |
| ---------- | ---------------- | ------------- |
| Simple     | 47%              | 7/15          |
| Medium     | 25%              | 5/20          |
| Complex    | 7%               | 1/15          |

**Key findings:**

1. **3x+ improvement over LFM2-1.2B-Tool:** 26% chain completion vs 8% — the largest improvement any model has shown in multi-step chains. Run 2 gained one more medium chain (5/20 vs 4/20).
2. **Cross-server transitions partially working:** The model completed chains that span multiple MCP servers (7 simple chains passed, all requiring cross-server tool selection). This is the first model to demonstrate any cross-server success in the unfiltered setting.
3. **Wrong tool is the dominant failure mode (54%):** Not deflection (12%) or no-tool (8%). The model tries to call tools but picks the wrong one, especially in medium/complex chains where context accumulates.
4. **Simple chains approaching viability (47%):** Nearly half of 3-step same-server chains succeed. This suggests the model could handle production workloads for simple tasks.
5. **Complex chains remain hard (7%):** Only 1 of 15 complex chains (6+ steps, cross-server) completed. The model degrades as conversation history grows.

### Per-Category Accuracy Breakdown (unfiltered, 67 tools)

| Category            | LFM2-24B-A2B Run 2 | LFM2-24B-A2B Run 1 | LFM2-1.2B-Tool (K=15) |
| ------------------- | ------------------- | ------------------- | --------------------- |
| calendar            | **100%** (7/7)      | **100%**            | 86%                   |
| audit               | **100%** (3/3)      | **100%**            | 100%                  |
| security-privacy    | **90%** (9/10)      | 80%                 | 80%                   |
| task-management     | **88%** (7/8)       | 100%                | 100%                  |
| document-processing | **83%** (10/12)     | 83%                 | 67%                   |
| file-operations     | **80%** (12/15)     | 80%                 | 60%                   |
| system-clipboard    | **80%** (4/5)       | 80%                 | 80%                   |
| ocr-vision          | **75%** (6/8)       | 75%                 | **88%**               |
| email               | **75%** (6/8)       | 75%                 | 63%                   |
| meeting-audio       | **71%** (5/7)       | 71%                 | **86%**               |
| knowledge-search    | **71%** (5/7)       | 71%                 | 71%                   |
| data-operations     | **60%** (6/10) ⬆    | 40%                 | **90%**               |

**Notable patterns:**
- **Data-operations improved from 40% → 60%:** Still the weakest category, but 4 of the no-tool-call deflections from Run 1 now produce tool calls in Run 2. The remaining failures are still "no tool call" deflections on data/SQL queries — likely a training artifact.
- **Security improved from 80% → 90%:** The agent loop hardening may help the model distinguish between similar security tools.
- **Task-management dropped from 100% → 88%:** One test (`list_tasks` vs `daily_briefing`) flipped. Normal variance within error margin for 8 samples.
- **Calendar and audit remain at 100%:** Clean tool boundaries + clear prompt language → perfect accuracy.
- **OCR and meeting-audio slightly worse than K=15 baseline:** The 1.2B model with filtered tools actually does better on these categories, suggesting the larger context (67 tools) adds noise for these specific categories.

---

## Real-World Execution: Benchmark vs Practice

Formal benchmarks test single-step tool selection in isolation. Real-world execution involves multi-round context, error compounding, and filename hallucination. We ran the same task (rename screenshots by OCR content) before and after agent loop infrastructure fixes.

### Pre-fix vs Post-fix (same model, same task)

| Dimension | Pre-fix (session 60024ab0) | Post-fix (session 558fadf3) |
|---|---|---|
| Rounds | 4 | 8 |
| Tool calls | 8 | 14 |
| Files OCR'd | 1 (wrong file) | 2 |
| **Files renamed** | **0** | **1** |
| Cross-server transitions | 1 | 3 |
| Error recovery | 0 of 4 errors | 3 of 5 errors |
| Confabulated exit | Yes (claimed 9 renames) | No |
| Duration | ~28s | ~46s |

### What the infrastructure fixes changed

| Fix | Impact |
|---|---|
| **Semantic aliases** (`rename_file` → `move_file`) | Stopped `rename_file` resolving to `read_file` (Levenshtein distance failure) |
| **Correction context** in error messages | Enabled 3 self-correction loops (model reads error, fixes on next round) |
| **Confabulation detection** | Prevented premature exit when model claimed "all files successfully processed" with zero mutations |

### Key insight: infrastructure enables error recovery

The agent loop's value is not preventing errors — models will always hallucinate tool names. It's enabling **recovery from errors**:

```
Round 0: Model hallucinated tool name → Error with context
Round 1: Model self-corrected → Success
Round 3: Model used wrong server prefix → Suggestions provided
Round 4: Model used correct server → Success
```

Each error-correction pair takes ~5-10s. Without the infrastructure, the model gets a confusing error and either gives up or repeats the same mistake.

### Remaining real-world gaps

1. **Partial task completion** — processes 2 of 10+ files and stops. Most impactful remaining problem.
2. **Filename hallucination** — invents filenames not in `list_dir` results.
3. **78% benchmark ≠ 78% real-world** — multi-round context, error compounding, and suboptimal tool ordering make effective accuracy much lower on complex tasks.

---

## Path B: Dual-Model Orchestrator

### Initial attempt: JSON plans — FAILED (0% chain completion)

The first orchestrator attempt asked LFM2-24B-A2B to produce raw JSON structured plans. Result: 94% of plans had JSON parse errors. The model uses bracket format natively and cannot reliably produce JSON outside of its tool-calling syntax.

### Solution: bracket-format plans — WORKING

After 11 fixes (F1-F11), the orchestrator was redesigned:
- **Planner** produces bracket-format plans (`[plan.add_step(...)]`) instead of JSON
- **Router** uses the fine-tuned LFM2.5-1.2B-Router-FT-v2 (not the base 1.2B model)
- **Argument override** system compensates for router's hallucinated arguments

A/B testing (2026-02-18) confirmed the dual-model approach is strictly better for 1-2 step tasks: zero behavioral pathologies, 2.5x faster, zero wasted calls. For 4+ step tasks, the planner under-decomposes.

See [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md) for the full A/B test results and fix details.

---

## Comparison Summary

### Single-Model Architecture (Path A)

| Model            | Active params | Single-step (67 tools) | Multi-step chains | VRAM   |
| ---------------- | ------------- | ---------------------- | ----------------- | ------ |
| GPT-OSS-20B      | ~3.6B (MoE)   | ~36%                   | ~0% (deflection)  | 14 GB  |
| Qwen3-30B-A3B    | ~3B (MoE)     | ~36%                   | ~0% (fixation)    | 5 GB   |
| LFM2-1.2B-Tool   | 1.2B          | 36%                    | 8%                | 2.3 GB |
| **LFM2-24B-A2B** | **~2B (MoE)** | **80%**                | **26%**           | **13 GB** |

### Improvement over baselines

| Metric            | LFM2-24B-A2B (Run 2) | Best previous | Improvement |
| ----------------- | --------------------- | ------------- | ----------- |
| Single-step (67)  | 80%                   | 36% (all)     | **+44pp (2.2x)** |
| Multi-step chains | 26%                   | 8% (1.2B)     | **+18pp (3.25x)** |
| Tool call rate    | 94%                   | ~47%          | **+47pp** |
| Deflection rate   | 12%                   | 80% (GPT-OSS) | **-68pp** |
| Avg latency       | 395ms                 | 400ms (1.2B)  | Comparable |

### Dual-Model Architecture (Path B — after 11 fixes)

| Planner | Router | 1-2 step tasks | 4+ step tasks | Total VRAM |
|---|---|---|---|---|
| **LFM2-24B-A2B** | **LFM2.5-1.2B-Router-FT-v2** | **100% (clean execution)** | Partial (planner limit) | **~14.5 GB** |

**Verdict:** The orchestrator is the recommended mode for 1-2 step workflows (zero pathologies, 2.5x faster). Single-model (Path A) remains the fallback for complex multi-step chains where the planner under-decomposes.

---

## Conclusions

### Hypothesis validation

1. **H1: VALIDATED (speed), INCONCLUSIVE (quality).** LFM2-24B-A2B with ~2B active params outperforms both GPT-OSS-20B (~3.6B MoE) and Qwen3-30B-A3B (~3B MoE) on single-step tool selection — 80% vs 51%/44% — while being 6x faster. The speed advantage comes from the combination of hybrid conv+attention design and MoE sparsity. The accuracy difference likely reflects training data and methodology differences rather than architecture alone.

2. **H2: PARTIALLY VALIDATED.** LFM2-24B-A2B shows the first cross-server transition success in multi-step chains (47% simple chain completion, 26% overall). Previous models all scored ~0% on chains requiring cross-server transitions. However, complex chains (6+ steps) still largely fail (7%), indicating the improvement is real but incomplete.

3. **H3: INITIALLY REFUTED, THEN RESOLVED.** LFM2-24B-A2B as orchestrator planner initially achieved 0% chain completion due to inability to generate JSON plans. After switching to bracket-format plans and 11 infrastructure fixes (F1-F11), the orchestrator achieves 100% on 1-2 step workflows with zero behavioral pathologies. See [Orchestrator Performance](./dual-model-orchestrator-performance.md).

4. **H4: REFUTED.** LFM2-24B-A2B at K=15 pre-filter (72%) is actually worse than unfiltered (78%). The model's own embeddings produce a different similarity space than LFM2-1.2B-Tool, and the reduced tool set introduces more confusion than it removes. For this model, skip the pre-filter.

### Key takeaways

1. **The hybrid design + MoE sparsity delivers the best latency-to-accuracy trade-off.** LFM2-24B-A2B (~2B active) beats GPT-OSS-20B (~3.6B active MoE) by 29 percentage points and is 6x faster. The speed comes from the combination of the hybrid conv+attention design and MoE sparsity. The accuracy gap likely reflects training differences rather than architecture alone.

2. **The 67-tool problem is solvable without decomposition.** 80% single-step accuracy with all 67 tools means the "tool overload" failure mode (FM-11) is not an inherent limitation — it's a model capability issue. The right model handles it without needing tool pre-filtering.

3. **Multi-step chains remain the hard problem.** Even at 80% single-step, chain completion is only 26%. The compounding effect of errors (0.80^4 = 41% theoretical for 4-step) combined with growing conversation context and wrong-tool propagation means multi-step requires either better models or architectural mitigation (retry, hierarchical routing, or agent loop hardening like the semantic alias + confabulation detection implemented in this session).

4. **The orchestrator architecture is format-dependent.** The initial orchestrator required JSON plans, which LFM2-24B-A2B couldn't produce. Switching to bracket-format plans (matching the model's native syntax) resolved this entirely. Lesson: design the orchestrator around the model's strengths, not against them.

5. **Data-operations is a training gap, but improving.** The 60% accuracy on data tools (up from 40% in Run 1, vs 100% on task/calendar) suggests the agent loop hardening helped somewhat, but this category still needs fine-tuning on synthetic data tool-calling examples.

6. **Pre-filtering is model-specific.** The K=15 strategy that doubled LFM2-1.2B-Tool's accuracy (36% → 78%) actually hurt LFM2-24B-A2B (78% → 72%). Each model's embedding space and decision-making interacts differently with filtered tool sets. Always benchmark filtering per model.

7. **Agent loop hardening provides incremental gains.** Run 2 (post-fix) improved single-step from 78% → 80% and multi-step from 24% → 26%. The semantic alias layer, correction context, and confabulation detection are each small improvements that compound positively.

### Recommended next steps

1. ~~Orchestrator~~ → **Done.** Bracket-format plans resolved the JSON failure. Dual-model is now the default for 1-2 step tasks.
2. **Improve planner decomposition** for 4+ step workflows (currently collapses to 1 step).
3. **Fine-tune on data-operations examples** to close the 60% → 80%+ gap on the weakest category.
4. **Explore hierarchical routing** (category → tool) as an alternative to flat K=15 pre-filter.

---

## References

- [LFM2-8B-A1B Model Card](https://huggingface.co/LiquidAI/LFM2-8B-A1B) — architecture reference (same family, scaled down)
- [LFM2 Technical Report](https://arxiv.org/abs/2511.23404) — hybrid architecture details
- [Project Learnings](./project-learnings-and-recommendations.md) — full analysis of the 50+ tool problem
- [GPT-OSS-20B Model Behavior Reference](./gpt-oss-20b.md) — failure taxonomy baseline (12 failure modes)
- [Fine-Tuning Results](./fine-tuning-results.md) — V1/V2 router training, accuracy, and failure analysis
- [Qwen3-30B-A3B Analysis](./qwen3-30b-a3b-tool-calling.md) — baseline comparison data
- [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md) — A/B test results, 11 fixes, architecture
- [ADR-009: Dual-Model Orchestrator](../architecture-decisions/009-dual-model-orchestrator.md) — orchestrator architecture
- [ADR-010: RAG Pre-Filter](../architecture-decisions/010-rag-prefilter-benchmark-analysis.md) — K=15 validation data
