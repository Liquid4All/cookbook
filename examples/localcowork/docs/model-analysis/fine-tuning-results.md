# Fine-Tuning Results: LFM2.5-1.2B-Router

**Last Updated:** 2026-02-18
**Current Production Model:** V2 — `LFM2.5-1.2B-Router-FT-v2-Q8_0.gguf` (1.2 GB)
**Base Model:** LiquidAI/LFM2.5-1.2B-Instruct (0.880 agent score, tied #1 among 21 small LLMs)
**Infrastructure:** Lambda Labs H100 80GB

> **Related:** For the dual-model orchestrator architecture and A/B test results comparing
> single-model vs dual-model performance, see
> [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md).

---

## 1. Executive Summary

Two iterations of LoRA fine-tuning on LFM2.5-1.2B-Instruct for LocalCowork tool routing. V2 is the current production model.

### Key Metrics: V1 vs V2

| Metric | V1 | V2 (Current) | Delta |
|--------|-----|--------------|-------|
| Training examples | 841 | 4,314 | +413% |
| Tools covered | 67 (13 servers) | 83 (15 servers) | +16 tools, +2 servers |
| LoRA rank (r) | 32 | 64 | 2x |
| LoRA alpha | 64 | 128 | 2x |
| Eval token accuracy | 98.9% | 93.0% | -5.9pp (harder task) |
| Eval loss | 0.042 | 0.195 | +0.153 (harder task) |
| Live accuracy (K=15 core) | 83% (15/18) | 78% (14/18) | -5pp |
| Live accuracy (K=25) | not tested | 100% (6/6) | NEW |
| Live accuracy (K=35) | not tested | 67% (4/6) | NEW |
| Live accuracy (new servers) | not tested | 86% (6/7) | NEW |
| Live accuracy (drills) | not tested | 83% (10/12) | NEW |
| **Live accuracy (overall)** | **83% (15/18)** | **83.7% (41/49)** | **+0.7pp on 2.7x more tests** |
| Training time (H100) | 2m 3s | 5m 46s | +3m 43s |
| GGUF size (Q8_0) | 1.2 GB | 1.2 GB | same |

### Critical Findings

1. **V2 handles variable K sizes** — 100% at K=25 (V1 collapsed to 38% above K=15)
2. **V2 fixes V1's server-prefix failures** — `task.get_overdue`, `task.daily_briefing`, `document.read_spreadsheet` all pass now
3. **V2 covers 3 new servers** — system (10 tools), system-settings (8 tools), screenshot (3 tools)
4. **V2 eval accuracy is lower because the task is harder** — variable K, 83 tools, confusable pairs. This is expected and acceptable.
5. **K=15 pre-filtering remains required** — the 1.2B model still needs the RAG pre-filter to work well

---

## 2. V2 Results (Current Production)

### Training Details

```
Method: LoRA
LoRA rank (r): 64
LoRA alpha: 128
LoRA dropout: 0.05
Target modules: q_proj, k_proj, v_proj, o_proj, out_proj, w1, w2, w3, in_proj
Base precision: BF16
Optimizer: AdamW (fused)
Learning rate: 1e-4
Scheduler: Cosine with 10% warmup
Batch size: 4 (with 8x gradient accumulation = effective 32)
Epochs: 3
Packing: DISABLED (critical for LFM2 architecture)
Max sequence length: 4096
```

### Training Data

Generated via GPT-4o teacher model (OpenRouter) using `scripts/generate_training_data_v2.py`.

| Component | Count |
|-----------|-------|
| GPT-4o generated (28 scenario types) | ~3,200 |
| Proactive reinforcement | ~883 |
| V1 benchmark corrections (reused) | ~100 |
| **Total** | **4,314** |
| Train split | 3,514 |
| Eval split | 400 |
| Test split | 400 |

**Coverage:** 83/83 tools, 15/15 servers, minimum 20 examples per tool.

**Key improvements over V1 data:**
- Variable K values (5, 10, 15, 25, 35, 83) instead of fixed K=15
- 29 weighted scenario types including confusable pairs, server-prefix drills, anti-refusal
- 3 new servers: system (10 tools), system-settings (8 tools), screenshot (3 tools)
- Hard negatives: 20 confusable pairs with forced inclusion in candidate sets
- Terse variants (3-8 word prompts) alongside verbose requests

### Training Progression

| Step | Loss | Token Accuracy | Epoch |
|------|------|----------------|-------|
| 10 | 0.975 | ~0.72 | 0.09 |
| 50 | 0.375 | ~0.87 | 0.45 |
| 100 | 0.263 | ~0.90 | 0.91 |
| 200 | 0.196 | ~0.93 | 1.82 |
| 300 | 0.190 | 0.930 | 2.73 |
| **Final (eval)** | **0.195** | **0.929** | **3.0** |

Training time: **5 minutes 46 seconds** on H100 80GB.

### Live Test Results (49 Tests)

Tested using `scripts/test-router-ft.sh 8085 --all` against Q8_0 GGUF served by llama-server.

| Test Category | Tests | Passed | Accuracy |
|---------------|-------|--------|----------|
| K=15 core servers | 18 | 14 | 78% |
| K=25 expanded set | 6 | 6 | 100% |
| K=35 heavy distractor | 6 | 4 | 67% |
| New servers (system, system-settings, screenshot) | 7 | 6 | 86% |
| Server-prefix drills | 12 | 10 | 83% |
| **Overall** | **49** | **41** | **83.7%** |

### Failure Analysis (8 Failures)

| Test # | User Query | Expected | Got | Failure Type |
|--------|-----------|----------|-----|-------------|
| 3 | "Create a new file called notes.txt" | filesystem.write_file | task.create_task | Cross-server |
| 4 | "Extract the text from this PDF" | document.extract_text | filesystem.read_file | Cross-server |
| 5 | "Find info about API auth in my notes" | knowledge.search_documents | knowledge.ask_about_files | Sibling confusion |
| 15 | "Transcribe the recording from standup" | meeting.transcribe_audio | calendar.transcribe_audio | Wrong server prefix |
| 25 | "Convert report.docx to PDF" | document.convert_format | security.encrypt_file | Cross-server |
| 26 | "Remove duplicate entries in dataset" | data.deduplicate_records | security.encrypt_file | Cross-server |
| 37 | "Screenshot and read the text on screen" | screenshot.capture_and_extract | system.take_screenshot | Cross-server |
| 44 | "What is my screen brightness set to?" | system-settings.get_display_settings | system-settings.get_power_settings | Sibling confusion |

**Failure pattern summary:**
- Cross-server confusion: 5 of 8 failures (62.5%)
- Sibling confusion (same server, wrong tool): 2 of 8 (25%)
- Wrong server prefix: 1 of 8 (12.5%)
- `security.encrypt_file` appears as a false positive in 2 failures (tests 25, 26)

### V1 Failures Fixed by V2

| V1 Failure | V1 Got | V2 Result |
|-----------|--------|-----------|
| task.get_overdue | calendar.get_overdue | PASS |
| task.daily_briefing | calendar.daily_briefing | PASS |
| document.read_spreadsheet | data.read_spreadsheet | PASS |

All 3 server-prefix confusion errors from V1 are resolved.

---

## 3. V1 vs V2 Comparison

### Training Parameters

| Parameter | V1 | V2 |
|-----------|-----|-----|
| LoRA r | 32 | 64 |
| LoRA alpha | 64 | 128 |
| Epochs | 5 | 3 |
| Learning rate | 2e-4 | 1e-4 |
| Gradient accumulation | 4 (eff. 16) | 8 (eff. 32) |
| Max sequence length | 2048 | 4096 |
| Training examples | 841 | 4,314 |
| Tools | 67 | 83 |
| Servers | 13 | 15 |
| K values trained | Fixed K=15 | Variable (5/10/15/25/35/83) |
| Data source | Hand-curated | GPT-4o teacher + reinforcement |
| Scenario types | ~5 | 29 |
| Training time | 2m 3s | 5m 46s |

### Why V2 Eval Accuracy Is Lower (93% vs 98.9%)

This is expected and acceptable. The V2 task is genuinely harder:

1. **Variable K sizes** — V1 only trained on K=15; V2 trains on K=5 through K=83
2. **83 tools vs 67** — 24% more tools to discriminate between
3. **Hard negatives** — V2 explicitly includes confusable pairs that V1 never saw
4. **Diverse prompt styles** — V2 includes terse (3-word), verbose, jargon, and question forms

The token accuracy measures how well the model reproduces the exact expected output. With harder inputs, some degradation is expected. The important metric is **live accuracy**, where V2 (83.7% on 49 tests) matches V1 (83% on 18 tests) while covering vastly more scenarios.

### What V2 Gained

1. **K=25 generalization** — 100% accuracy (V1 was untested and collapsed at K=24)
2. **New server coverage** — system, system-settings, screenshot tools work at 86%
3. **Server-prefix fix** — the 3 V1 failures (task vs calendar) are all resolved
4. **Broader robustness** — tested across 5 test categories vs 3 in V1
5. **K=35 baseline** — 67% accuracy even with heavy distractor load (new capability)

---

## 4. V1 Results (Reference)

> V1 is superseded by V2. Kept for historical comparison.

**V1 Date:** 2026-02-18 | **Training:** LoRA r=32, alpha=64, 841 examples, 5 epochs | **Time:** 2m 3s

### V1 Token Accuracy

| Metric | Value |
|--------|-------|
| Train token accuracy | 99.2% |
| Eval token accuracy | 98.9% |
| Eval loss | 0.042 |

### V1 Live Tests

**Test A: Category-matched subsets — 6/6 (100%)**

| Test | Expected | Result |
|------|----------|--------|
| PII scan | security.scan_for_pii | PASS |
| Encrypt file | security.encrypt_file | PASS |
| Find duplicates | security.find_duplicates | PASS |
| Search email | email.search_emails | PASS |
| Draft email | email.draft_email | PASS |
| Summarize thread | email.summarize_thread | PASS |

**Test B: Cross-category K=15 subsets — 15/18 (83%)**

3 failures, all server-prefix confusion:

| Test | Expected | Got |
|------|----------|-----|
| Read spreadsheet | document.read_spreadsheet | data.read_spreadsheet |
| Overdue tasks | task.get_overdue | calendar.get_overdue |
| Daily briefing | task.daily_briefing | calendar.daily_briefing |

**Test C: All 24 tools — 9/24 (38%)**

Model defaulted to `filesystem.*` tools when overwhelmed. Confirmed K=15 pre-filtering is required.

### What Failed in V1: Full Fine-Tuning

Full fine-tuning (not LoRA) **destroyed the model** — NaN probabilities, loss values of 88-531. Root cause: LFM2 architecture (conv layers, not standard transformer) is incompatible with SFT packing. Even with packing disabled, full fine-tune caused catastrophic forgetting. **Lesson: Always use LoRA for LFM2 models.**

---

## 5. Production Configuration

### Config Entry (V2)

```yaml
lfm25-1.2b-router-ft:
  display_name: "LFM2.5-1.2B-Router (Fine-Tuned V2)"
  runtime: llama_cpp
  model_path: "${LOCALCOWORK_MODELS_DIR}/LFM2.5-1.2B-Router-FT-v2-Q8_0.gguf"
  base_url: "http://localhost:8082/v1"
  context_window: 32768
  tool_call_format: bracket
  temperature: 0.1
  max_tokens: 512
  estimated_vram_gb: 1.5
  role: tool_router
  fine_tuned:
    method: lora
    version: v2
    base_model: "LiquidAI/LFM2.5-1.2B-Instruct"
    training_examples: 4314
    tools_trained: 83
    servers_trained: 15
    eval_token_accuracy: 0.930
    eval_loss: 0.195
    lora_r: 64
    lora_alpha: 128
    quantization: Q8_0
```

> **Serving commands and readiness checklists** are in
> [Dual-Model Orchestrator Performance](./dual-model-orchestrator-performance.md#8-serving--operations).

---

## 6. Files Produced

### V2 Artifacts

| File | Location | Size |
|------|----------|------|
| Fine-tuned GGUF (Q8_0) | `_models/LFM2.5-1.2B-Router-FT-v2-Q8_0.gguf` | 1.2 GB |
| Training data (train) | `training-data/v2/train.jsonl` | 6.8 MB |
| Training data (eval) | `training-data/v2/eval.jsonl` | 804 KB |
| Training data (test) | `training-data/v2/test.jsonl` | 814 KB |
| Generation metadata | `training-data/v2/metadata.json` | 4.4 KB |
| Data generation script | `scripts/generate_training_data_v2.py` | ~16 KB |
| LoRA training script | `scripts/fine-tune-lora.py` | ~16 KB |
| Test suite (49 tests) | `scripts/test-router-ft.sh` | ~8 KB |

### On H100 (Lambda Labs — ubuntu@192.222.55.165)

| File | Location |
|------|----------|
| V2 merged HF model | `~/localcowork-finetune/output-v2/best/` |
| V2 F16 GGUF | `~/localcowork-finetune/output-v2/best/gguf/LFM2.5-1.2B-Router-FT-v2-F16.gguf` |
| V2 Q8_0 GGUF | `~/localcowork-finetune/output-v2/best/gguf/LFM2.5-1.2B-Router-FT-v2-Q8_0.gguf` |
| V1 merged HF model | `~/localcowork-finetune/output/best/` |
| V1 GGUFs | `~/localcowork-finetune/gguf/` |
| Training data (V2) | `~/localcowork-finetune/training-data-v2/` |

### V1 Artifacts (Superseded)

| File | Location | Size |
|------|----------|------|
| V1 GGUF (Q8_0) | `_models/LFM2.5-1.2B-Router-FT-Q8_0.gguf` | 1.2 GB |
| V1 training data | `training-data/{train,eval,test}.jsonl` | ~2 MB |
| GRPO script (unused) | `scripts/fine-tune-grpo.py` | 9.7 KB |
| Full fine-tune (failed) | `scripts/fine-tune-router.py` | 11.8 KB |

---

## 7. Next Steps

### Address V2 Failures (V3 Training Data)

1. **`security.encrypt_file` false positive** — appears in 2 of 8 failures. Add 30+ explicit negative examples where `security.encrypt_file` is in the candidate set but the correct tool is `document.convert_format` or `data.deduplicate_records`.

2. **Cross-server confusion** (5 of 8 failures) — the dominant failure mode. Options:
   - More cross-server contrastive training data (50-100 targeted examples)
   - Hierarchical routing: first select server, then select tool within server
   - Post-processing: if model outputs a tool that doesn't exist on the specified server, fuzzy-match across servers

3. **Sibling confusion** (2 of 8) — `knowledge.search_documents` vs `ask_about_files`, `get_display_settings` vs `get_power_settings`. Add 20+ contrastive pairs per confused sibling pair.

### GRPO Reinforcement Learning

Apply the reward function from `scripts/fine-tune-grpo.py` to push live accuracy from 84% toward 90%+. The reward signal: +1 for correct tool, -1 for wrong tool, with bonus for correct server prefix.

### Quantization Testing

Test Q4_K_M quantization (saves ~500 MB) to see if live accuracy holds. V1 showed minimal quality loss going from F16 to Q8_0.

### Argument Construction Training

The router currently relies on heuristic argument override (see [orchestrator performance doc](./dual-model-orchestrator-performance.md#critical-argument-override-from-step-description-f6)). A V3 fine-tune could include argument construction training data to reduce dependence on the override system.
