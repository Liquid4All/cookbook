# Dual-Model Orchestrator: Architecture & Performance

**Last Updated:** 2026-02-18
**Status:** Operational for 1-2 step workflows; single-model fallback for complex chains
**Related ADR:** [ADR-009: Dual-Model Orchestrator](../architecture-decisions/009-dual-model-orchestrator.md)
**Related:** [Fine-Tuning Results](./fine-tuning-results.md) (router model training details)

---

## 1. Architecture Overview

LocalCowork supports two model execution flows, selectable via `_models/config.yaml`.

### Flow A: Single-Model Mode (24B Solo)

The default configuration (`orchestrator.enabled: false`). One model handles everything.

```
User message
  |
  v
Tauri IPC (send_message)
  |
  v
Orchestrator check --> DISABLED, skip
  |
  v
Two-pass tool selection (enabled):
  |
  |  Pass 1: LFM2-24B-A2B sees ~15 category meta-tools
  |  Model selects 2-3 relevant categories
  |  Categories expanded to real tools (~15-25 tools)
  |
  v
Agent loop (up to 10 rounds):
  |
  |  LFM2-24B-A2B selects tool from expanded set
  |      |
  |      v
  |  MCP server executes tool
  |      |
  |      v
  |  Result fed back to LFM2-24B-A2B
  |  (loop continues if more work needed)
  |
  v
Final text response streamed to frontend
```

**Characteristics:**
- Only **one model** runs: LFM2-24B-A2B (port 8080, ~13 GB VRAM)
- The 1.2B fine-tuned router is NOT used
- Two-pass narrows ~83 tools to ~15-25 (saves ~7,000 tokens per turn)
- The 24B model handles planning, tool selection, execution, and response generation
- Up to 10 tool-call rounds with deflection/incomplete detection and auto-correction

### Flow B: Dual-Model Orchestrator

Enable with `orchestrator.enabled: true` in `_models/config.yaml`. Implements the plan-execute-synthesize pipeline from ADR-009.

```
User message
  |
  v
Tauri IPC (send_message)
  |
  v
Orchestrator check --> ENABLED
  |
  v
Phase 1 -- PLAN (LFM2-24B-A2B):
  |  Receives server capability summaries (no tool definitions)
  |  Decomposes request into self-contained steps
  |  Output: bracket-format plan [plan.add_step(...)]
  |
  v
Phase 2 -- EXECUTE (per step):
  |
  |  For each step:
  |    RAG pre-filter embeds step description
  |    Selects K=15 most relevant tools from 83
  |        |
  |        v
  |    LFM2.5-1.2B-Router-FT-v2 selects one tool from K=15
  |    (text-based system prompt matching training format)
  |        |
  |        v
  |    Argument override from step description + user message
  |        |
  |        v
  |    MCP server executes tool
  |        |
  |        v
  |    Result collected (retries up to 3x if no tool call)
  |
  v
Phase 3 -- SYNTHESIZE (LFM2-24B-A2B):
  |  Receives all step results
  |  Streams user-facing summary to frontend
  |
  v
Fallback: if any phase fails --> single-model agent loop (Flow A)
```

**Characteristics:**
- Two models run simultaneously on separate llama-server instances
- Each step is a clean single-turn call to the 1.2B router (no conversation history accumulation)
- The RAG pre-filter uses embedding similarity to select the 15 most relevant tools per step
- Graceful fallback: if orchestration fails at any phase, the system falls through to Flow A

### Model Roles

| Model | Role | Port | VRAM | Used In |
|-------|------|------|------|---------|
| LFM2-24B-A2B | Main agent + planner + synthesizer | 8080 | ~13 GB | Flow A (everything), Flow B (plan + synthesize) |
| LFM2.5-1.2B-Router-FT-v2 | Tool router (fine-tuned) | 8082 | ~1.5 GB | Flow B only (execute phase) |
| LFM2.5-VL-1.6B | Vision/OCR | 8081 | ~1.8 GB | OCR MCP server (both flows) |

### VRAM Budget

| Mode | Models Loaded | Total VRAM |
|------|--------------|------------|
| Single-model (Flow A) | LFM2-24B-A2B | ~13 GB |
| Dual-model (Flow B) | LFM2-24B-A2B + LFM2.5-1.2B-Router-FT-v2 | ~14.5 GB |
| With OCR | Add LFM2.5-VL-1.6B | +1.8 GB |

### Configuration

```yaml
# _models/config.yaml
active_model: lfm2-24b-a2b           # Primary model for all reasoning
orchestrator:
  enabled: true                       # Flow B enabled
  planner_model: lfm2-24b-a2b         # Plans in Flow B
  router_model: lfm25-1.2b-router-ft  # Routes in Flow B (V2 fine-tuned)
  router_top_k: 15                     # RAG pre-filter K value
two_pass_tool_selection: true          # Two-pass categories in Flow A (fallback)
```

---

## 2. Key Innovations (Fixes F1-F11)

During A/B testing, 11 fixes and improvements were implemented to make the dual-model orchestrator production-viable. These are grouped by the problem they solve.

### Critical: Training/Production Format Alignment (F1)

The router model was fine-tuned on a text-based tool list in the system prompt (numbered list, bracket-syntax calls). But the orchestrator originally sent tools via the OpenAI `tools` JSON parameter, which llama-server reformatted via its chat template. The model had never seen this format during training.

**Fix:** Replace the `tools` JSON parameter with a text-based system prompt matching the training format exactly. The router now receives tools as:
```
1. filesystem.list_dir -- List directory contents including files and subdirectories.
2. filesystem.read_file -- Read the full text content of a file.
...
```

This single fix took router tool selection from **0% to 100%**.

### Critical: Argument Override from Step Description (F6)

The 1.2B router selects the correct tool with high accuracy but fills arguments with memorized training-data paths (e.g., `~/Documents/example.txt` instead of the actual path from the user's message). This is overfitting to example paths in the training data.

**Fix:** The router's role is tool selection only. After selection, arguments are constructed from the step description and user message using heuristic extraction:
- Path parameters: extract from backtick-quoted text, absolute paths, or well-known directory references ("Downloads folder" -> `~/Downloads`)
- Title parameters: extract from quoted strings or description context
- Date parameters: extract relative date references ("Friday", "tomorrow")

Router's hallucinated args are only used as fallback when context extraction produces nothing.

### Smart Merge with Placeholder Detection (F10)

The argument override (F6) was initially too aggressive, overwriting correct router values. Fix F10 introduced `is_placeholder_value()` detection:
- Only override path-like parameters unconditionally (router always hallucinates these)
- For other parameters, only override if the router's value is a detected placeholder (`"example.txt"`, `"ISO 8601"`, `"value"`, etc.)
- Preserve good router values like `title="Review Q4 numbers"`

### All Fixes Summary

| # | Fix | Phase | Problem Solved |
|---|-----|-------|----------------|
| F1 | Tool format alignment | 2c | Router never produced tool calls (0% -> 100% selection) |
| F2 | Retry prompt with bracket example | 2c | Retry still used wrong format |
| F3 | Fallback on all-steps-failed | 2c | User saw hallucinated synthesis from empty results |
| F4 | Planner few-shot examples | 2c | Planner lacked multi-step decomposition examples |
| F5 | Diagnostic logging | 2c | No visibility into router raw responses |
| F6 | Argument override | 2d | Router hallucinated training-data paths as arguments |
| F7 | Planner decomposition rules | 2d | Complex requests collapsed to single step |
| F8 | Bracket parser string-awareness | 2d | `[` and `]` inside quoted args broke parsing |
| F9 | Tool execution result logging | 2d | No visibility into MCP tool results |
| F10 | Smart merge (placeholder detection) | 2e | Argument override was too aggressive, overwrote good values |
| F11 | Post-plan decomposition check | 2e | Planner under-decomposed compound requests |

---

## 3. A/B Test Methodology

### Test Messages

Three messages of increasing complexity, designed to stress different aspects of the system:

| # | Message | Complexity | Servers Involved | Expected Steps |
|---|---------|-----------|-----------------|---------------|
| 1 | "What files are in my Downloads folder?" | Single-tool | filesystem | 1 |
| 2 | "Read the file `tests/fixtures/uc4/downloads/quarterly_report.txt` and create a task to review the Q4 numbers by Friday." | Two-step chain | filesystem, task | 2 |
| 3 | "Scan the files in `tests/fixtures/uc3/sample_files/` for SSNs and API keys, then tell me what you found and create a task to follow up on the sensitive files." | Multi-step cross-server | filesystem, security, task | 4-6 |

### Protocol

1. **Phase 1 (Single-Model):** `orchestrator.enabled: false`, `two_pass_tool_selection: true`. Run all 3 messages as new chat sessions. Save logs.
2. **Phase 2 (Dual-Model):** `orchestrator.enabled: true`, start 1.2B router on port 8082. Run the same 3 messages. Save logs.
3. **Analysis:** Compare tool selection accuracy, argument correctness, round efficiency, behavioral pathologies, and wall time.

Phase 2 was iterative (2b through 2e) as fixes were implemented and retested.

---

## 4. Single-Model Results (Phase 1)

Executed 2026-02-18. LFM2-24B-A2B solo with two-pass tool selection.

### Message 1: "What files are in my Downloads folder?"

| Metric | Result |
|--------|--------|
| **Verdict** | Correct tool on round 0, then stuck in loop |
| Total rounds | 10 (hit max limit) |
| Tool calls | 9 total: 1x `list_directory` (correct), 8x redundant re-calls |
| Behavioral detections | 1x FM-3 deflection (round 1) |
| Wall time | 23 seconds |

The model got the correct answer in round 0 but the agent loop couldn't detect task completion. It called `list_directory` 8 more times with identical results before hitting the 10-round cap.

### Message 2: "Read file...create task to review Q4 numbers"

| Metric | Result |
|--------|--------|
| **Verdict** | Core task succeeded, then couldn't stop |
| Total rounds | 10 (hit max limit) |
| Tool calls | 7 total: `read_file` -> `task.create_task` (correct), then 4 unnecessary calls |
| Cross-server transition | filesystem -> task (via two-pass category expansion) |
| Behavioral detections | 2x confabulation |
| Wall time | 10 seconds |

Cross-server transition worked via two-pass category expansion. But after creating the task (round 2), the model made 7 more unnecessary calls including `task.daily_briefing` (completely unrelated).

### Message 3: "Scan files for SSNs and API keys...create task"

| Metric | Result |
|--------|--------|
| **Verdict** | Partial -- security scans succeeded, task never created |
| Total rounds | 10 (hit max limit) |
| Tool calls | 10: 4 successful security calls, 6 failures |
| Tool failures | 2x sandbox path error, 4x filename-as-tool hallucination |
| Cross-server transition | Never reached task server |
| Wall time | 21 seconds |

The model successfully scanned for PII and secrets (rounds 4-6) but got stuck in the security category due to two-pass lock-in. It never transitioned to the task server. It also hallucinated the filename `has_api_key.env` as a tool name (4 failed calls).

### Phase 1 Scorecard

| Metric | Msg 1 | Msg 2 | Msg 3 |
|--------|-------|-------|-------|
| Core task success | Round 0 | Rounds 0-2 | Partial (scans only) |
| Clean completion | No (8 extra calls) | No (4 extra calls) | No (never finished) |
| Rounds used | 10 (needed 1) | 10 (needed 3) | 10 (needed ~6) |
| Cross-server | N/A | Success | Failure |
| Wall time | 23s | 10s | 21s |

**Systemic issues identified:**
1. **Can't-stop problem** -- model completes task but agent loop can't detect completion
2. **Tool call looping** -- identical tool calls with identical params repeated
3. **Two-pass category lock-in** -- once categories expand, model can't request additional categories mid-session
4. **Filename-as-tool hallucination** -- model confuses file content with tool names

---

## 5. Dual-Model Results (Phases 2b-2e)

### Phase 2b: Baseline (All Fixes Disabled)

The orchestrator ran but **zero tools were executed across all 3 messages**. The router model never produced a single tool call. Every session: planner created steps -> router failed -> planner synthesized a hallucinated response from empty results.

**Root cause:** Training/production format mismatch (see Fix F1).

### Phase 2c: After Fixes F1-F5

| Metric | Msg 1 | Msg 2 | Msg 3 |
|--------|-------|-------|-------|
| Router selects correct tool | Yes | Yes (both steps) | Yes |
| Correct arguments | No (hallucinated path) | Step 1 yes, Step 2 failed parse | No (hallucinated path) |
| Steps executed | 1 | 2 | 1 (planner only made 1 step) |

Fix F1 was transformative: router tool selection went from 0% to 100%. But arguments were filled with memorized training-data paths (`~/Documents/example.txt` for everything).

### Phase 2d: After Fixes F6-F9

| Metric | Msg 1 | Msg 2 | Msg 3 |
|--------|-------|-------|-------|
| Router selects correct tool | Yes | Yes (both steps) | Yes |
| Argument override | Correct (`~/Downloads`) | Path correct, title overridden wrong | Correct (backtick path) |
| Real tool results | 2376 bytes, actual files | Q4 report + task_id=5 | SSN found in has_ssn.txt |
| Steps executed | 1 | 2 | 1 (planner still only 1 step) |

Argument override (F6) fixed paths. But title extraction was too aggressive -- it overwrote the router's correct `title="Review Q4 numbers"` with `title="content"`.

### Phase 2e: After Fixes F10-F11 (Final)

| Metric | Msg 1 | Msg 2 | Msg 3 |
|--------|-------|-------|-------|
| Router selects correct tool | Yes | Yes (both steps) | Yes |
| Argument override | Correct | **Title preserved!** Due date extracted | Correct path |
| Real tool results | 2376 bytes | Q4 report + task_id=6 | SSN found (493 bytes) |
| Steps executed | 1 | 2 | 1 (re-plan triggered but model still produced 1 step) |

Smart merge (F10) correctly preserved `title="Review Q4 numbers"` while still overriding the hallucinated path. The decomposition check (F11) detected the compound request in Message 3 and triggered a re-plan, but the 24B planner still produced only 1 step.

### Progress Across All Phases

| Issue | Phase 2b | Phase 2c | Phase 2d | Phase 2e |
|-------|---------|---------|---------|---------|
| Router tool selection | 0/3 | **3/3** | 3/3 | 3/3 |
| Arguments correct | N/A | 1/5 | 4/5 | **5/5** |
| Real tool results | 0 tools | wrong data | correct data | correct data |
| Bracket parsing (no retry) | N/A | 4/5 | 5/5 | **5/5** |
| Cross-server workflows | N/A | 1/1 | 1/1 | 1/1 |
| Planner decomposition | N/A | fail | fail | fail (model limit) |

---

## 6. Head-to-Head Comparison

### Message 1: "What files are in my Downloads folder?"

| Metric | Single-Model (24B Solo) | Dual-Model (24B + 1.2B) |
|--------|------------------------|-------------------------|
| **Verdict** | Correct tool, then looped 8x | **Clean single-pass** |
| Correct tool | Yes (round 0) | Yes (step 1) |
| Correct arguments | Yes | Yes (after override) |
| Total model calls | 10 (hit max) | 3 (plan + route + synthesize) |
| Wasted calls | 9 | 0 |
| Behavioral issues | 1x deflection, 7x identical re-calls | None |
| Wall time | 23s | ~12s |

### Message 2: "Read file...create task to review Q4 numbers by Friday"

| Metric | Single-Model (24B Solo) | Dual-Model (24B + 1.2B) |
|--------|------------------------|-------------------------|
| **Verdict** | Core task done, then looped | **Clean two-step execution** |
| Step 1: Read file | Yes (round 0) | Yes (step 1) |
| Step 2: Create task | Yes (round 2) | Yes (step 2) |
| Task title | Unknown | "Review Q4 numbers" (preserved) |
| Task due date | Unknown | "Friday" (extracted) |
| Cross-server transition | Yes (two-pass expansion) | Yes (planner decomposition) |
| Total model calls | 10 (hit max) | 4 (plan + 2x route + synthesize) |
| Wasted calls | 7 | 0 |
| Behavioral issues | 2x confabulation | None |
| Wall time | 10s | ~4s |

### Message 3: "Scan files for SSNs and API keys...create task"

| Metric | Single-Model (24B Solo) | Dual-Model (24B + 1.2B) |
|--------|------------------------|-------------------------|
| **Verdict** | Partial (scans done, no task) | Partial (1 scan done, no task) |
| PII scan | Yes (round 4, after 2 failures) | Yes (step 1, first try) |
| Secrets scan | Yes (round 5) | No (only 1 step planned) |
| Task creation | Never reached task server | Never reached task server |
| Tool failures | 6 (sandbox + hallucination) | 0 |
| Path correctness | Failed twice before fixing | Correct on first try |
| Total model calls | 10 (hit max) | 3 (plan + route + synthesize) |
| Wall time | 21s | ~5s |

Both modes fail Message 3, but for different reasons: single-model gets stuck in the security category (two-pass lock-in); dual-model's planner refuses to decompose into multiple steps (model limitation).

### Aggregate Metrics

| Metric | Single-Model | Dual-Model | Winner |
|--------|-------------|-----------|--------|
| Tool selection accuracy | 3/3 (with retries) | 3/3 (first try) | **Dual** |
| Argument accuracy | Unknown | 5/5 correct | **Dual** |
| Tool call failures | 6 across 3 messages | 0 | **Dual** |
| Wasted model calls | 16 of 27 total | 0 of 10 total | **Dual** |
| Cross-server transition | 1/2 | 1/1 | **Dual** |
| Clean task completion | 0/3 (always hit round limit) | 3/3 (natural stop) | **Dual** |
| Behavioral pathologies | 3x deflection, 2x confabulation, looping, hallucination | None | **Dual** |
| Complex workflows (4+ steps) | 3/6 tools executed | 1/4 tools executed | **Single** |
| Total wall time (3 messages) | 54 seconds | ~21 seconds | **Dual** (2.5x faster) |
| VRAM usage | ~13 GB | ~14.5 GB | **Single** (1.5 GB less) |

### Behavioral Pathologies Eliminated by Dual-Model

The single-model agent loop exhibits five pathological behaviors, all eliminated by the orchestrator:

1. **Can't-stop loop** -- model completes the task but keeps calling tools until the 10-round hard cap. The orchestrator provides deterministic plan-execute-stop flow.

2. **FM-3 deflection** -- model asks "Would you like me to..." instead of executing. The orchestrator's router is trained to always call a tool, never ask questions.

3. **Confabulation** -- model claims actions it didn't take. The orchestrator's synthesis phase receives actual tool results, not model-generated claims.

4. **Two-pass category lock-in** -- once categories expand, the model can't access tools from other servers. The orchestrator plans all steps upfront, each routed independently.

5. **Filename-as-tool hallucination** -- model confuses filenames in tool results with tool names. The orchestrator gives each step a fresh context with only tool definitions, no prior results.

### Architecture Insights

| Insight | Evidence |
|---------|----------|
| 1.2B is sufficient for tool selection | 100% accuracy on tool name selection across all tests |
| 1.2B is not sufficient for argument construction | Memorizes training-data paths; argument override needed |
| 24B handles 1-2 step decomposition | Correctly produces 2-step plans for read+task workflows |
| 24B fails at 4+ step decomposition | Collapses complex multi-server workflows to 1 step even with explicit instructions |
| Single-turn isolation is the key advantage | No context accumulation, no looping, no behavioral pathologies |
| Argument override is a novel pattern | Extracting args from context is more reliable than model-generated args for small models |

---

## 7. Recommendation

**Use the dual-model orchestrator as the default mode.** It is:
- Strictly more reliable for 1-2 step requests (clean pass vs looping)
- 2.5x faster (21s vs 54s total across 3 test messages)
- Free of behavioral pathologies (no deflection, confabulation, looping)
- Cleanly terminable (natural stop vs forced round limit)

**Keep the single-model fallback** for:
- Orchestrator errors (planner failure, router server down)
- Complex 4+ step workflows where the planner under-decomposes

**Future improvements:**
- Code-side decomposition for compound requests (bypass planner for known patterns)
- Fine-tune the 24B planner on decomposition training data
- Fine-tune the 1.2B router on argument construction (not just tool selection)
- GRPO reinforcement learning on the router to push past 83.7% tool selection accuracy

---

## 8. Serving & Operations

### Serving Commands

**Single-model mode (Flow A):**
```bash
# Only the 24B model is needed
llama-server \
  --model _models/LFM2-24B-A2B-Preview-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --flash-attn
```

**Dual-model mode (Flow B):**
```bash
# Terminal 1: 24B planner/synthesizer
llama-server \
  --model _models/LFM2-24B-A2B-Preview-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --flash-attn

# Terminal 2: 1.2B router (with embeddings for RAG pre-filter)
llama-server \
  --model _models/LFM2.5-1.2B-Router-FT-v2-Q8_0.gguf \
  --port 8082 \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --embeddings
```

### Readiness Checklists

**For Single-Model Mode (Flow A):**
- [ ] `_models/LFM2-24B-A2B-Preview-Q4_K_M.gguf` exists
- [ ] llama-server running on port 8080
- [ ] `active_model: lfm2-24b-a2b` in config.yaml
- [ ] `orchestrator.enabled: false` in config.yaml
- [ ] `two_pass_tool_selection: true` in config.yaml
- [ ] MCP servers auto-discovered (check Settings panel in app)
- [ ] Send a test message and verify tool trace shows tool calls

**For Dual-Model Mode (Flow B):**
- [ ] All of the above, plus:
- [ ] `_models/LFM2.5-1.2B-Router-FT-v2-Q8_0.gguf` exists (1.2 GB)
- [ ] Second llama-server on port 8082 with `--embeddings` flag
- [ ] `orchestrator.enabled: true` in config.yaml
- [ ] Verify embeddings: `curl http://localhost:8082/v1/embeddings -d '{"input":["test"],"model":"any"}'`
- [ ] Send a multi-step request and check for plan-execute-synthesize in logs
