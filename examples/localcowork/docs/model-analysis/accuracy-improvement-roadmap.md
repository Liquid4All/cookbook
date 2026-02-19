# Roadmap: Tool-Calling Accuracy from 80% to 90%+

**Last Updated:** 2026-02-18
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

---

## 1. Hierarchical Routing: Server-Scoped Tool Selection

### The core idea

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

## 2. Dedicated Embedding Model for Pre-Filter

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

## 3. Essential Tool Set: 5 Tools x 3 Servers

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

## 4. Multi-Step: Why 26% Is So Bad and What Fixes It

### Three stacked failures

**A. Error compounding:** 80%^4 = 41% theoretical, actual is 7%. Wrong tool in step 2 corrupts context for step 3.

**B. Context accumulation:** By step 5, ~21K of 32K tokens consumed. Decision capacity halves.

**C. Planner under-decomposition:** Orchestrator fixes A and B by isolating each step. But the planner collapses 4-step requests into 1 step.

### The orchestrator architecture is right, the planner is the bottleneck

- 1-step: 100% (clean)
- 2-step: 100% (planner decomposes correctly)
- 4+ step: fails (planner won't produce 4 steps)

### Five interventions

**M1. Template-based decomposition (bypass the planner for known UCs)**

Hard-code decomposition for the 10 validated use cases. Pattern match user input -> inject pre-built plan.

Expected: 100% decomposition for known UCs. Cost: ~1 day.

**M2. Few-shot decomposition examples (free)**

The planner gets rules ("if multiple actions, create one step per action") but rules don't work. Add 5 few-shot examples of correct multi-step decomposition.

Expected: 60% -> 80% decomposition. Cost: system prompt change.

**M3. Step-result forwarding**

Each step currently gets a clean context (no prior results). Step 2 can't reference step 1's output. Fix: pass 1-2 line condensed summary of prior results.

Expected: unlocks dependent chains. Cost: one summarization call per step.

**M4. Iterative re-planning (plan 1 step -> execute -> re-plan)**

Instead of producing all steps upfront, produce one step at a time. The planner can always do 1 step. Chain length limited by step counter (max 8).

Expected: eliminates under-decomposition entirely, 26% -> 50-65%. Cost: ~3-5s latency per step.

**M5. Mid-chain context eviction**

After each step, check context usage. If >80% of 32K, evict oldest step summaries.

Expected: enables 8-10 step chains.

### Projected improvement

| Intervention combo | Multi-step |
|---|---|
| Current baseline | 26% |
| M2 alone (free) | ~35% |
| M1 + M3 + M4 combined | **60-75%** |

At 60-70% with human confirmation on each step, multi-step is usable.

---

## 5. GRPO: What It Is

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
Phase 1 (1-2 weeks, low risk):
  -> 5-tool starter mode (validate 95%+)
  -> Few-shot planner examples (M2, free)
  -> Template decomposition for UC-1, UC-4, UC-7 (M1)
  -> Tool schema audit (83 -> ~74 tools)
  Expected: 95%+ single-step, ~35% multi-step, ~100% templated UCs

Phase 2 (2-4 weeks, medium risk):
  -> Hierarchical routing (server-scoped candidates)
  -> Dedicated embedding model (nomic-embed-text-v1.5)
  -> Step-result forwarding (M3)
  -> V3 fine-tune with cross-server contrastive data
  Expected: 93-95% single-step, ~50% multi-step

Phase 3 (4-6 weeks, highest ceiling):
  -> Iterative re-planning (M4)
  -> GRPO on router ($5-10)
  -> Templates for remaining UCs
  -> Progressive tool expansion (5 -> 8 -> 12 -> full)
  Expected: 95%+ single-step, 60-75% multi-step
```

---

## What won't get us there

- Bigger K values (drops above K=20)
- Longer system prompts (worsens deflection)
- More parameters alone (GPT-OSS-20B at 20B scored 36%)
- Prompt engineering alone (proven ceiling ~80%)
- More retries (errors compound: 0.80^n)
- Just improving single-step (0.95^4 = 81%; multi-step needs architecture)

---

## References

- [LFM2-24B-A2B Benchmark](./lfm2-24b-a2b-benchmark.md)
- [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md)
- [Fine-Tuning Results](./fine-tuning-results.md)
- [Project Learnings](./project-learnings-and-recommendations.md)
- [ADR-009: Dual-Model Orchestrator](../architecture-decisions/009-dual-model-orchestrator.md)
- [ADR-010: RAG Pre-Filter](../architecture-decisions/010-rag-prefilter-benchmark-analysis.md)
