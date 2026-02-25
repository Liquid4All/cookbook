# Roadmap: Tool-Calling Accuracy from 80% to 90%+

**Last Updated:** 2026-02-19
**Status:** Strategic roadmap — not yet implemented
**Current Baseline:** 80% single-step, 26% multi-step (LFM2-24B-A2B)

---

## Where the 20% is lost today

| Failure bucket | % of errors | Root cause |
|---|---|---|
| **Wrong tool (cross-server)** | 62.5% of router failures (5/8) | Model picks `filesystem.read_file` when it needs `document.extract_text` |
| **Wrong tool (sibling)** | 25% of router failures (2/8) | `knowledge.search_documents` vs `knowledge.ask_about_files` |
| **No tool call (deflection)** | 6-12% | Model asks "what would you like?" instead of acting |
| **Planner under-decomposition** | 100% of 4+ step failures | Collapses "scan + create task + email" into 1 step |
| **Data-operations gap** | 60% accuracy (weakest category) | Training data underrepresents SQL/CSV patterns |

### The critical insight: multi-step is the biggest gap

Single-step at 80% with human confirmation is effectively ~100%. Multi-step at 26% is unusable. Every intervention is ranked by its impact on multi-step chaining.

All multi-step interventions (M1-M6) operate within the **dual-model orchestrator** (ADR-009). The orchestrator's plan-execute-synthesize architecture is the right foundation — the planner is the bottleneck, not the architecture.

---

## Interventions (ranked by priority)

### M2. Few-Shot Decomposition Examples (do first — free)

**The problem:** The planner gets rules ("if multiple actions, create one step per action") but rules don't work at 1-2B scale. The model needs to *see* correct decomposition, not just be told about it.

**The fix:** Add 5 few-shot examples of correct multi-step decomposition to `PLANNER_SYSTEM_PROMPT`. Cover: 2-step, 3-step, 4-step, cross-server, and dependent-chain patterns.

**Why first:** Zero cost, zero risk, immediate signal. If this alone pushes decomposition from 60% to 80%, it tells us the problem is prompt quality, not model capacity.

| Metric | Before | After (expected) |
|---|---|---|
| Decomposition rate (4+ steps) | ~0% | 60-80% |
| Multi-step overall | 26% | ~35% |

> **⚠️ Tested (2026-02-20):** We tried benchmark-level few-shot examples (3 examples + 9 rules + contrastive tool descriptions) in the benchmark system prompt. Result: single-step regressed 80% → 77%, multi-step unchanged at 24%. The few-shot approach didn't work at the benchmark/tool-selection layer. This reinforces that M2 must operate at the **orchestrator planner level** (decomposing multi-step plans), not at the tool-selection level (picking the right tool). The planner is the bottleneck, not tool selection per step. Code stashed: `git stash@{0}`. See `tool-calling-benchmark-results.md` → "Experiment: Prompt Optimization" section.

**Cost:** System prompt change. **Risk:** None. **Time:** Hours.

---

### M1. Template-Based Decomposition (parallel with M2)

Hard-code decomposition for the 10 validated use cases. Pattern match user input -> inject pre-built plan, bypassing the planner entirely.

```
UC-1 (Receipt Reconciliation): list_dir -> extract_text -> query_sqlite -> create_task
UC-4 (Download Triage):        list_dir -> extract_text -> scan_pii -> move_file -> create_task
UC-7 (Contract Copilot):       extract_text -> search_documents -> draft_email
```

**Why parallel with M2:** Templates give 100% decomposition for known patterns regardless of planner quality. They're the safety net while we improve the planner's general decomposition.

| Metric | Before | After |
|---|---|---|
| Templated UC multi-step | 26% | ~100% |
| Non-templated multi-step | 26% | 26% (no change) |

**Cost:** ~1 day. **Risk:** None — templates are a pure fallback. **Time:** 1 day.

---

### M3. Step-Result Forwarding

Each step currently gets a clean context (no prior results). Step 2 can't reference step 1's output. Fix: pass 1-2 line condensed summary of prior step results.

```
Step 1: list_dir("/Downloads") → "Found 14 PDF files, 3 CSV files"
Step 2: extract_text(file="...") → now knows WHICH files to extract from
```

**Why here in priority:** Without this, even perfect decomposition fails on dependent chains. Step 2 needs step 1's output. This is the difference between "plan correctly" and "execute correctly."

**Expected:** Unlocks dependent chains. Cost: one summarization call per step (~200ms).

---

### ⏸ BENCHMARK GATE: Measure M2 + M1 + M3 Combined

After implementing M2, M1, and M3, benchmark multi-step performance on:
- 20 compound prompts (4+ steps) for decomposition rate
- 10 validated use cases (UC-1 through UC-10) for end-to-end completion
- 50 multi-step scenarios from `tests/model-behavior/benchmark-multi-step.ts`

| Metric | Baseline | Target |
|---|---|---|
| Decomposition rate (4+ steps) | ~0% | > 70% |
| Multi-step overall | 26% | > 45% |
| Templated UC completion | 26% | ~100% |

**Decision:** If multi-step > 50% → skip M6, go to Phase 2 (router improvements). If multi-step < 40% and decomposition is the bottleneck → try M6 (thinking model). If decomposition is fine but execution fails → skip to M4 (iterative re-planning).

---

### M6. Thinking Model as Planner (contingent — only if M2+M1+M3 plateau)

**The hypothesis:** If planner under-decomposition persists after few-shot examples, it's a *reasoning* failure, not a prompt quality issue. A thinking model that generates chain-of-thought traces before committing to a plan should decompose more reliably.

**The candidate:** [LFM2.5-1.2B-Thinking](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb)

| Spec | LFM2-24B-A2B (current planner) | LFM2.5-1.2B-Thinking |
|---|---|---|
| Active params | ~2B (from 24B MoE) | 1.2B |
| VRAM | ~13 GB | ~900 MB |
| BFCLv3 (tool use) | N/A (not benchmarked) | 57% |
| Multi-IF (instruction following) | N/A | 69% (vs 61% instruct) |
| MATH-500 (reasoning proxy) | N/A | 88% (vs 63% instruct) |
| Context window | 32K | 32K |
| GGUF available | Yes | Yes (llama.cpp, MLX, vLLM) |
| Doom-loop rate | N/A | 0.36% (RL-fixed from 15.7%) |

**Why it fits the planner role specifically:**

1. **Planning doesn't need 83 tools in context.** The planner sees server capability summaries (~1200 tokens), not tool definitions. A 1.2B model has plenty of capacity for this.
2. **Thinking traces produce step-by-step reasoning.** This is exactly what `parse_bracket_plan()` needs to extract — the model reasons through "the user wants X *and* Y *and* Z" before committing to a bracket-format plan.
3. **VRAM savings are massive.** Freeing ~12 GB means the router and planner can coexist comfortably, or the headroom can be used for the dedicated embedding model.
4. **The format is compatible.** LFM2.5-1.2B-Thinking uses the same Liquid AI architecture family. Bracket-format output should work with minimal prompt tuning. The thinking tokens appear before the plan output and don't interfere with parsing (strip `<think>...</think>` prefix).

**Why it might not work:**

1. **1.2B may lack world knowledge** for complex decomposition. "Reconcile receipts against bank statement" requires understanding what reconciliation *means* across multiple servers. The 24B model has more latent knowledge even if only 2B params are active per token.
2. **Thinking traces consume context.** Budget ~200-500 tokens per planning call. Current planner uses ~6,700 of 32K — headroom is fine, but worth monitoring.
3. **Not fine-tuned on bracket format.** Will need prompt engineering to produce `[plan.add_step(...)]` output reliably. May require a short SFT pass on 50-100 planning examples.

**Integration point:** Config-only change in `_models/config.yaml`:

```yaml
orchestrator:
  planner_model: lfm25-1.2b-thinking   # Was: lfm2-24b-a2b
  router_model: lfm25-1.2b-router-ft   # Unchanged
```

Plus a new model entry with `base_url`, `context_window`, `tool_call_format: bracket`. The `InferenceClient::from_config_with_model()` constructor handles the rest.

**Experiment design:**

1. Use the M2+M1+M3 benchmark results as baseline.
2. Swap planner to LFM2.5-1.2B-Thinking with the same few-shot prompt. Re-run the same benchmark suite.
3. Compare: if thinking model improves decomposition by >10% over 24B planner → adopt it. If equal or worse → the bottleneck is knowledge, not reasoning → skip to M4.

| Metric | After M2+M1+M3 | After M6 (expected) |
|---|---|---|
| Decomposition rate (4+ steps) | 60-80% | 80-95% |
| Multi-step overall | ~40-50% | ~55-65% |
| Planner VRAM | 13 GB | ~900 MB |

**Cost:** Model download (~900 MB GGUF) + prompt tuning (~1 day). **Risk:** Medium — format compatibility untested. **Time:** 2-3 days including benchmarking.

---

### M4. Iterative Re-Planning (backstop — plan 1 step -> execute -> re-plan)

Instead of producing all steps upfront, produce one step at a time. The planner can always do 1 step. After execution, re-plan with the result. Chain length limited by step counter (max 8).

**Why this is the backstop:** If M2 + M1 + M3 (and optionally M6) still can't decompose 4+ step requests reliably, iterative re-planning eliminates the problem entirely. The planner never needs to produce more than 1 step. But it adds ~3-5s latency per step, so we try cheaper options first.

| Metric | Before | After (expected) |
|---|---|---|
| Multi-step overall | 26% | 50-65% |
| Decomposition rate | ~0% for 4+ | 100% (by construction) |
| Latency per step | ~2s | ~5-8s (plan + execute) |

**Cost:** Medium engineering effort. **Risk:** Latency penalty. **Time:** 1-2 weeks.

---

### M5. Mid-Chain Context Eviction

After each step, check context usage. If >80% of 32K, evict oldest step summaries.

**Expected:** Enables 8-10 step chains. Required for long workflows but only matters once 4-step chains work.

---

### 1. Hierarchical Routing: Server-Scoped Tool Selection

Don't make one hard 83-way decision. Make two easy ones.

**Stage 1 (already happens):** The planner picks which server a step belongs to. It already outputs `expected_server` in its bracket-format plans.

**Stage 2 (the change):** The router only sees tools from that server + its semantic neighbors. Not the full 83. A scoped, server-aware K=15.

### How it works

```
User: "Scan my Downloads for SSNs and create a remediation task"

Planner outputs:
  Step 1: server="security"
  Step 2: server="task"

Step 1 routing:
  security.* (6 tools) + neighbors [audit, filesystem] = 15 tools
  Cross-server confusion with task/email/calendar is IMPOSSIBLE

Step 2 routing:
  task.* (5 tools) + neighbors [calendar, email] = 14 tools
  Cross-server confusion with filesystem/security is IMPOSSIBLE
```

### Semantic neighbor graph

```
filesystem  -> [document, knowledge, data]
document    -> [ocr, filesystem, knowledge]
ocr         -> [document, knowledge, screenshot]
security    -> [audit, filesystem]
task        -> [calendar, email]
calendar    -> [task, email, meeting]
email       -> [task, calendar]
meeting     -> [calendar, email, knowledge]
audit       -> [security, task]
```

Already validated in training data generation (`scripts/generate_training_data_v2.py`).

### Fallbacks

1. No server hint -> fall back to RAG K=15 (current behavior)
2. Wrong server hint -> neighbor graph provides safety net
3. Orchestrator failure -> single-model agent loop

**Expected impact:** Router accuracy 83.7% -> ~90-92%.

---

### 2. Dedicated Embedding Model for Pre-Filter

### The problem

Pre-filter uses the router's own embeddings (mean-pooled LLM tokens). These are trained for next-token prediction, not semantic similarity. Filter hit rate: 94% at K=15.

Known misses: "backup" misses `copy_file`, "anomalies" misses `summarize_anomalies`, "lock down" misses `encrypt_file`.

### Recommendation: nomic-embed-text-v1.5

| Model | Params | MTEB Score | GGUF Size | Latency |
|---|---|---|---|---|
| **nomic-embed-text-v1.5** | 137M | 0.696 | 140 MB | ~30ms/query |
| LFM2.5-1.2B (current) | 1.2B | N/A | N/A | ~100ms/query |

Run as separate llama-server on port 8085. Zero code changes in `tool_prefilter.rs` -- just change the endpoint URL.

**Expected impact:** Filter hit rate 94% -> 97-99%.

---

### 3. Essential Tool Set: 5 Tools x 3 Servers

### What normal people do daily on their computers

1. Find a file
2. Read a document
3. Search for something
4. Create a reminder
5. Check their schedule

### The 5 tools

| Tool | Server | Accuracy | Frequency |
|---|---|---|---|
| `filesystem.list_dir` | filesystem | 80% | Very high |
| `filesystem.read_file` | filesystem | 80% | Very high |
| `filesystem.search_files` | filesystem | 80% | High |
| `task.create_task` | task | 88% | High |
| `calendar.list_events` | calendar | 100% | High |

**Expected accuracy at 5 tools: 95-98%.** Decision space is tiny; only filesystem sibling confusion remains.

### Progressive expansion (only after 95%+ per wave)

- **Wave 2 (8 tools):** + `move_file`, `extract_text`, `list_tasks`
- **Wave 3 (12 tools):** + `search_emails`, `draft_email`, `create_event`, `update_task`

### Tools to merge (83 -> ~74)

| Tool(s) | Recommendation |
|---|---|
| `knowledge.ask_about_files` | Merge into `search_documents` |
| `data.summarize_anomalies` | Merge into `query_sqlite` |
| `system.get_cpu/memory/disk/network` (4 tools) | Merge into `get_resource_usage` |
| `screenshot.extract_ui/suggest_actions` | Merge into `capture_and_extract` |

---

### 4. GRPO: Reinforcement Learning on Router

### SFT vs GRPO

**SFT:** "Here's the right answer -- learn to produce it."
**GRPO:** "Here are 4 attempts you made -- the ones that scored higher, do more of those."

GRPO applies **asymmetric penalties** that SFT can't: cross-server mistakes penalized 3x more than sibling confusion. Directly targets the 62.5% failure mode.

### Cost

~$5-10, ~30 min H100. Script exists (`scripts/fine-tune-grpo.py`), unused.

### Risks

Reward hacking, catastrophic forgetting, untested on LFM2 architecture.

**Bottom line:** Low cost, uncertain reward. Do after architectural fixes, not before.

---

## Recommended sequence

```
Phase 1 — Fix the planner (1-2 weeks, low risk):
  -> M2: Few-shot decomposition examples (free, do first, hours)
  -> M1: Template decomposition for UC-1, UC-4, UC-7 (parallel, 1 day)
  -> M3: Step-result forwarding (unlocks dependent chains, days)
  -> BENCHMARK: measure decomposition rate + multi-step completion
  -> M6: Thinking model experiment (ONLY if benchmark shows decomposition gap)
  Expected: ~40-55% multi-step, ~100% templated UCs

Phase 2 — Fix the router (2-4 weeks, medium risk):
  -> Hierarchical routing (server-scoped candidates)
  -> Dedicated embedding model (nomic-embed-text-v1.5)
  -> 5-tool starter mode (validate 95%+ single-step)
  -> Tool schema audit (83 -> ~74 tools)
  -> V3 fine-tune with cross-server contrastive data
  Expected: 93-95% single-step, ~55-65% multi-step

Phase 3 — Ceiling push (4-6 weeks, highest ceiling):
  -> M4: Iterative re-planning (backstop if decomposition still < 70%)
  -> M5: Mid-chain context eviction (enables 8-10 step chains)
  -> GRPO on router ($5-10)
  -> Templates for remaining UCs
  -> Progressive tool expansion (5 -> 8 -> 12 -> full)
  Expected: 95%+ single-step, 60-75% multi-step
```

### Decision gates

- **After M2+M1+M3 benchmark:** If multi-step > 50% → skip M6, go to Phase 2. If decomposition < 70% → try M6 (thinking model). If decomposition is fine but execution fails → skip to Phase 2 + M4.
- **After M6 (if run):** If decomposition improves >10% → adopt thinking model. If not → problem is knowledge not reasoning, go to M4.
- **After Phase 2:** If multi-step > 60% → ship it. If < 50% → Phase 3 is mandatory.

---

## What won't get us there

- Bigger K values (drops above K=20)
- Longer system prompts (worsens deflection)
- More parameters alone (GPT-OSS-20B at 20B scored 36%)
- Prompt engineering alone (proven ceiling ~80%)
- More retries (errors compound: 0.80^n)
- Just improving single-step (0.95^4 = 81%; multi-step needs architecture)
- Thinking model as router (BFCLv3 at 57% is below fine-tuned router's 83.7%)

---

## References

- [LFM2-24B-A2B Benchmark](./lfm2-24b-a2b-benchmark.md)
- [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md)
- [Fine-Tuning Results](./fine-tuning-results.md)
- [Project Learnings](./project-learnings-and-recommendations.md)
- [ADR-009: Dual-Model Orchestrator](../architecture-decisions/009-dual-model-orchestrator.md)
- [ADR-010: RAG Pre-Filter](../architecture-decisions/010-rag-prefilter-benchmark-analysis.md)
- [LFM2.5-1.2B-Thinking Blog Post](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb)
