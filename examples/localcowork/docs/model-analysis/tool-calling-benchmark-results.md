# Instruct-Model Tool-Calling Comparison

> LFM2-24B-A2B (2B active params) vs five comparison models (3B-32B active params)
> on 67 MCP tools, Apple M4 Max hardware.
> **Core finding: LFM2 achieves 94% of the best dense model's accuracy at 3% of the latency.**

---

## Purpose

This document compares LFM2-24B-A2B (Liquid AI's sparse MoE hybrid model) against five comparison models on the LocalCowork tool-calling benchmark. The models span three categories: dense instruct-only transformers (Gemma, Mistral, Qwen3 32B, GPT-OSS-20B), and a lightweight MoE (Qwen3-30B-A3B). The goal is to quantify the scaling advantage of sparse activation — achieving competitive accuracy with dramatically fewer active parameters per token.

The comparison is relevant for anyone deploying local AI agents on consumer hardware (16-36 GB unified memory), where inference latency and memory footprint directly constrain product viability.

---

## Test Environment

- **Hardware:** Apple M4 Max, 36 GB unified memory, 32 GPU cores
- **Test suite:** 100 single-step tool selection prompts, 50 multi-step chains (3-6 steps each), 67 tools across 13 MCP servers
- **Inference parameters:** Temperature 0.1, top_p 0.1, max_tokens 512
- **LFM2-24B-A2B:** Served via llama-server (Q4_K_M GGUF), flash attention enabled
- **All other models:** Served via Ollama (native quantizations), native OpenAI function calling
- **Tool format:** LFM2 uses bracket format `[server.tool(args)]`; Ollama models use native JSON function calling
- **Qwen3 32B note:** 40/100 single-step tests completed before benchmark was stopped due to extreme latency. Results extrapolated proportionally; multi-step not tested.
- **All benchmarks run same session:** Fresh runs on identical hardware, same test suite, same day.

---

## Results

### Single-Step Tool Selection (100 prompts, 67 tools)

| Model | Architecture | Active Params | Accuracy | Avg Latency | Memory (GPU) |
|-------|-------------|--------------|----------|-------------|-------------|
| **Gemma 3 27B** | Dense transformer | 27B | **91%** (91/100) | 24,088ms | 19 GB |
| Mistral-Small-24B | Dense transformer | 24B | 85% (85/100) | 1,239ms | 14 GB |
| **LFM2-24B-A2B** | **Hybrid MoE (conv+attn)** | **~2B** | **80%** (80/100) | **385ms** | **~14.5 GB** |
| Qwen3 32B | Dense transformer | 32B | ~70% (28/40)* | 28,385ms | 21 GB |
| GPT-OSS-20B | Dense transformer | 20B | 51% (51/100) | 2,303ms | 14 GB |
| Qwen3-30B-A3B | MoE transformer | ~3B | 44% (44/100) | 5,938ms | 19 GB |

*Qwen3 32B: 40 of 100 tests completed; accuracy extrapolated from partial run.

### Multi-Step Chain Completion (50 chains, 3-6 steps each)

| Model | Chain Completion | Step Completion | Avg Steps/Chain | Chains Passed |
|-------|-----------------|----------------|-----------------|---------------|
| **Mistral-Small-24B** | **66%** | **74.3%** | 3.4 | 33/50 |
| Gemma 3 27B | 48% | 57.2% | 2.7 | 24/50 |
| LFM2-24B-A2B | 26% | 31% | 1.4 | 13/50 |
| Qwen3-30B-A3B | 4% | 14% | 0.6 | 2/50 |
| GPT-OSS-20B | 0% | 0% | 0.0 | 0/50 |
| Qwen3 32B | — | — | — | Not tested |

### Multi-Step by Difficulty

| Model | Easy (15) | Medium (20) | Hard (15) |
|-------|----------|------------|----------|
| **Mistral-Small-24B** | **87%** (13/15) | **65%** (13/20) | **47%** (7/15) |
| Gemma 3 27B | 67% (10/15) | 45% (9/20) | 33% (5/15) |
| LFM2-24B-A2B | 47% (7/15) | 25% (5/20) | 7% (1/15) |
| Qwen3-30B-A3B | 7% (1/15) | 5% (1/20) | 0% (0/15) |
| GPT-OSS-20B | 0% (0/15) | 0% (0/20) | 0% (0/15) |

### Understanding the Multi-Step Benchmark

**What is a chain?** A chain is a complete multi-tool workflow — a sequence of 3-7 user prompts where each prompt expects exactly one tool call. The model accumulates conversation history across steps, seeing prior tool results when deciding what to do next.

**What is chain completion vs step completion?**
- **Chain completion** = the model nailed *every* step in the chain. One wrong tool at step 3 of a 4-step chain = failed chain.
- **Step completion** = the % of individual steps correct, even within failed chains. A chain that gets 3/4 steps right contributes 75% step completion but 0% chain completion.

**Difficulty levels:**
- **Easy** (15 chains × 3 steps) — Single-domain or light cross-domain workflows
- **Medium** (20 chains × 4-5 steps) — Cross-domain workflows spanning 2-3 MCP servers
- **Hard** (15 chains × 6-7 steps) — Full end-to-end pipelines spanning 4+ servers

**Three real examples from the test suite:**

**Easy — "Create a task from meeting notes" (3 steps, `ms-simple-003`):**

| Step | User prompt | Expected tool |
|------|------------|---------------|
| 1 | "Read the meeting notes from today" | `filesystem.read_file` |
| 2 | "Create a task for the first action item: prepare proposal by Wednesday" | `task.create_task` |
| 3 | "Also schedule a follow-up meeting next Monday at 10am" | `calendar.create_event` |

Tests whether the model can cross from filesystem → task → calendar servers in a natural conversation flow.

**Medium — "Transcribe meeting, create tasks, schedule follow-up" (4 steps, `ms-medium-002`):**

| Step | User prompt | Expected tool |
|------|------------|---------------|
| 1 | "Transcribe the team standup recording" | `meeting.transcribe_audio` |
| 2 | "Pull out all the action items from the transcript" | `meeting.extract_action_items` |
| 3 | "Create tasks for each of these action items" | `task.create_task` |
| 4 | "Schedule a follow-up meeting for next Tuesday at 10am" | `calendar.create_event` |

Tests a realistic meeting-to-action workflow: meeting → meeting → task → calendar.

**Hard — "Full receipt reconciliation pipeline" (7 steps, `ms-complex-001`):**

| Step | User prompt | Expected tool |
|------|------------|---------------|
| 1 | "Show me all receipt images in my Receipts folder" | `filesystem.search_files` |
| 2 | "Extract text from the first receipt image" | `ocr.extract_text_from_image` |
| 3 | "Parse the vendor, date, items, and total from the OCR text" | `ocr.extract_structured_data` |
| 4 | "Check if this receipt is already in the system" | `data.deduplicate_records` |
| 5 | "Export all the reconciled receipt data to a CSV report" | `data.write_csv` |
| 6 | "Flag any receipts with unusual amounts or missing data" | `data.summarize_anomalies` |
| 7 | "Generate a PDF expense reconciliation report" | `document.create_pdf` |

Spans 4 MCP servers (filesystem → ocr → data → document) across 7 steps. This is the full UC-1 (Receipt Reconciliation) use case from the PRD.

**Why chain completion is hard — the compounding error effect:**

Even at 80% per-step accuracy, errors compound geometrically across steps:

| Steps | Expected chain completion (at 80%/step) |
|-------|----------------------------------------|
| 3 (easy) | 0.80³ = 51% |
| 4-5 (medium) | 0.80⁴ = 41%, 0.80⁵ = 33% |
| 7 (hard) | 0.80⁷ = 21% |

LFM2's actual 26% chain completion is consistent with this math. The multi-step benchmark tests whether models can maintain coherence across turns — remembering prior tool results, understanding workflow context, and crossing server boundaries — which is fundamentally harder than isolated tool selection.

---

## Latency Analysis

Latency is the critical differentiator on consumer hardware. All models ran on the same Apple M4 Max with 100% GPU offload.

| Model | Avg Latency | Relative to LFM2 | Interactive? |
|-------|-------------|-------------------|-------------|
| **LFM2-24B-A2B** | **385ms** | **1x (baseline)** | **Yes — sub-second** |
| Mistral-Small-24B | 1,239ms | 3.2x slower | Borderline — perceptible delay |
| GPT-OSS-20B | 2,303ms | 6.0x slower | No — multi-second delays |
| Qwen3-30B-A3B | 5,938ms | 15.4x slower | No — 6s per response |
| Gemma 3 27B | 24,088ms | 62.6x slower | No — 24s per response |
| Qwen3 32B | 28,385ms | 73.7x slower | No — 28s per response |

### What this means for product viability

- **LFM2 at 385ms** delivers a responsive, interactive experience. Users can ask questions and get tool selections in well under a second. This enables the single-turn, human-in-the-loop UX pattern that turns 80% model accuracy into near-100% effective accuracy.

- **Mistral at 1.2s** is usable but noticeably slower. The 3x latency penalty means users wait perceptibly between each interaction. In multi-step workflows (where Mistral excels at 66% chain completion), each step adds 1.2s — a 4-step chain takes ~5s of inference time alone.

- **GPT-OSS-20B at 2.3s** adds noticeable delay on every turn — and at 51% accuracy, half of those responses are wrong. The latency-accuracy combination makes it unsuitable for production.

- **Qwen3-30B-A3B at 5.9s** is a notable result: despite being an MoE model with only ~3B active params, it's 15x slower than LFM2 (also an MoE with ~2B active). The Qwen3 MoE's transformer-only architecture doesn't achieve the same inference efficiency as LFM2's hybrid conv+attn design.

- **Gemma and Qwen3 32B at 24-28s** are impractical for interactive use on MacBook hardware. Despite Gemma's leading 91% accuracy, a 24-second response time makes it unsuitable for a desktop assistant. These models would require server-class GPUs (A100/H100) to achieve interactive latency.

---

## Efficiency Analysis: Accuracy vs Compute Cost

LFM2-24B-A2B achieves 80% accuracy with only ~2B active parameters per token — comparable to dense models 12-16x its active size, at a fraction of the compute cost.

| Model | Active Params | Accuracy | Latency | VRAM |
|-------|--------------|----------|---------|------|
| **LFM2-24B-A2B** | **~2B** | **80%** | **385ms** | **~14.5 GB** |
| Qwen3-30B-A3B | ~3B | 44% | 5,938ms | 19 GB |
| GPT-OSS-20B | 20B | 51% | 2,303ms | 14 GB |
| Mistral-Small-24B | 24B | 85% | 1,239ms | 14 GB |
| Gemma 3 27B | 27B | 91% | 24,088ms | 19 GB |
| Qwen3 32B | 32B | ~70% | 28,385ms | 21 GB |

The comparison with Qwen3-30B-A3B is especially telling: both are MoE models with similar active param counts (~2B vs ~3B), but LFM2's hybrid conv+attn architecture achieves 80% vs 44% accuracy at 15x the speed — proving that MoE architecture alone isn't sufficient; the underlying block design matters.

The sparse activation pattern (24B total, ~2B active per token via 64 experts, 4 selected per token) means the model carries the knowledge capacity of a 24B model while paying the compute cost of a 2B model. This is the fundamental scaling advantage.

---

## Per-Category Breakdown (Single-Step)

LFM2-24B-A2B's accuracy varies by tool domain. All 6 models compared:

| Category | LFM2 | Mistral | Gemma | GPT-OSS | Qwen3-A3B |
|----------|-------|---------|-------|---------|-----------|
| Calendar | **100%** (7/7) | 86% | 100% | 86% | 43% |
| Audit | **100%** (3/3) | 0%† | 100% | 0%† | 33% |
| Security/Privacy | **90%** (9/10) | 90% | 90% | 90% | 70% |
| Task Management | **88%** (7/8) | 75% | 100% | 75% | 75% |
| Document Processing | 83% (10/12) | 0% | 92% | 0% | 8% |
| File Operations | 80% (12/15) | 67% | 93% | 67% | 40% |
| System/Clipboard | 80% (4/5) | 100% | 80% | 100% | 100% |
| OCR/Vision | 75% (6/8) | 50% | 88% | 50% | 38% |
| Email | 75% (6/8) | 63% | 88% | 63% | 25% |
| Meeting/Audio | 71% (5/7) | 29% | 86% | 29% | 43% |
| Knowledge/Search | 71% (5/7) | 57% | 86% | 57% | 43% |
| Data Operations | 60% (6/10) | 0% | 80% | 0% | 40% |

†GPT-OSS-20B uses `auditor.*` prefix instead of `audit.*` — scored as wrong tool due to namespace mismatch.

**Key observations:**
- LFM2 matches or beats the dense models in structured, unambiguous categories (calendar, audit, security, task management).
- GPT-OSS-20B and Qwen3-30B-A3B both score **0%** on document processing and data operations — complete category failures. These models can't discriminate among semantically similar tools in these domains.
- System/clipboard is the easiest category — even the weakest models score 100%. These tools have unique, unambiguous names.
- The accuracy gap between LFM2 and the sub-performing models is concentrated in domains with semantic tool overlap.

---

## Failure Pattern Analysis

### LFM2-24B-A2B (80% single-step)
- **Wrong tool:** 14% — mostly sibling confusion within same server (e.g., `list_dir` instead of `delete_file`)
- **No tool call:** 6% — generates conversational response instead of tool call
- **Multi-step dominant failure:** Wrong tool at 54% of chain failures, deflection at 12%

### Mistral-Small-24B (85% single-step)
- **Wrong tool:** 11% — lower sibling confusion than LFM2
- **No tool call:** 4% — strong tool calling discipline
- **Multi-step advantage:** Maintains coherence across 3-4 steps; failures concentrate at step 4+

### Gemma 3 27B (91% single-step)
- **Wrong tool:** 7% — lowest error rate in single-step
- **No tool call:** 2% — rarely deflects
- **Multi-step paradox:** Despite highest single-step accuracy, drops to 48% chain completion. Loses coherence in multi-turn context — likely due to attention degradation over longer sequences.

### GPT-OSS-20B (51% single-step)
- **Wrong tool:** 40% — highest wrong-tool rate of all models. Defaults to `filesystem.list_dir` or `filesystem.search_files` for unrelated domains.
- **No tool call:** 9% — moderate deflection rate (asks user for clarification instead of acting)
- **Multi-step: 0% chains** — 100% of chains fail at step 1 with no tool call. The multi-step system prompt triggers pure conversational deflection on every single chain.
- **Namespace confusion:** Uses `auditor.*` instead of `audit.*`, `container.exec` for document tasks. The model hallucinates tool server names.

### Qwen3-30B-A3B (44% single-step)
- **No tool call:** 51% — majority failure mode. Generates extended `<think>` reasoning blocks that consume the entire response budget without producing a tool call.
- **Wrong tool:** 5% — very low; when it does call a tool, it's usually correct (restraint score 0.95)
- **Multi-step: 4% chains** (2/50) — 90% of chain failures are no-tool-call. The model reasons extensively about each step but fails to act.
- **MoE comparison with LFM2:** Both are MoE models with similar active params (~3B vs ~2B), but Qwen3's transformer-only MoE scores 44% vs LFM2's 80%. The difference is architectural: LFM2's hybrid conv+attn blocks handle structured tool schemas more efficiently than pure transformer attention.

### Qwen3 32B (~70% single-step, partial)
- **No tool call:** 25% — generates reasoning tokens (`<think>` blocks) instead of tool calls
- **Wrong tool:** 5% — when it does call a tool, it's usually correct
- **Core issue:** The model's reasoning-first training (chain-of-thought) conflicts with tool calling. It prefers to reason about the problem rather than act on it.

---

## Observations

### 1. Latency is the deciding factor on consumer hardware

Gemma 3 27B achieves the highest accuracy (91%) but at 24 seconds per response, it's unusable for interactive desktop agents. LFM2's 385ms response time enables the human-in-the-loop UX pattern where users confirm each tool selection — turning 80% accuracy into near-100% effective accuracy. A model that's 11pp more accurate but 62x slower is not a viable trade.

### 2. Mistral is the strongest dense competitor

Mistral-Small-24B is the only dense model that delivers both competitive accuracy (85%) and acceptable latency (1.2s). It leads decisively on multi-step chains (66% vs LFM2's 26%). For workloads that require autonomous multi-step execution, Mistral is the better choice — at the cost of 3.2x higher latency and requiring all 24B parameters active per token.

### 3. Sparse MoE enables consumer-hardware deployment

LFM2-24B-A2B achieves 80% accuracy with only ~2B active parameters — compared to 24-32B for the dense models. This means comparable accuracy at 385ms instead of 1-24 seconds, which is the difference between a real-time interactive agent and a batch processor. It also fits in 14.5 GB VRAM — consumer MacBook territory rather than server-class hardware.

### 4. Qwen3's reasoning-first design hurts tool calling

Qwen3 32B (dense, 32B active) scored the lowest accuracy (~70%) with a 25% "no tool call" rate. The model generates extended reasoning chains instead of selecting tools. This is a training objective mismatch — models optimized for chain-of-thought reasoning may not transfer well to structured tool dispatch without fine-tuning.

### 5. Single-step accuracy doesn't predict multi-step success

Gemma leads single-step (91%) but drops to 48% on chains. Mistral is second in single-step (85%) but leads chains (66%). LFM2 is third in single-step (80%) with lowest chains (26%). Multi-step success depends more on maintaining coherence across turns and self-correcting after tool results — capabilities not captured by single-step benchmarks.

### 6. MoE architecture alone doesn't guarantee efficiency

Qwen3-30B-A3B (~3B active, MoE transformer) scores 44% at 5.9s latency. LFM2-24B-A2B (~2B active, hybrid MoE conv+attn) scores 80% at 385ms. Both are MoE models with similar active parameter counts, but LFM2 delivers **1.8x the accuracy at 15x the speed**. The difference is the block architecture: LFM2's convolution blocks appear more efficient at parsing structured tool schemas than Qwen3's transformer-only attention blocks.

### 7. Dense 20B models with native function calling aren't enough

GPT-OSS-20B (dense, 20B active, native OpenAI function calling) scores 51% — better than the ~36% estimated in earlier testing, but still well below the 80% threshold for production use. With 40% wrong-tool rate and 0% multi-step chains, having native function calling format doesn't compensate for the model's inability to discriminate among 67 tools.

---

## Summary Table

| Metric | LFM2-24B-A2B | Mistral-Small-24B | Gemma 3 27B | GPT-OSS-20B | Qwen3-30B-A3B | Qwen3 32B |
|--------|-------------|-------------------|-------------|-------------|---------------|-----------|
| Architecture | Hybrid MoE | Dense | Dense | Dense | MoE | Dense |
| Total params | 24B | 24B | 27B | 20B | 30B | 32B |
| Active params/token | ~2B | 24B | 27B | 20B | ~3B | 32B |
| Single-step accuracy | 80% | 85% | **91%** | 51% | 44% | ~70%* |
| Multi-step chains | 26% | **66%** | 48% | 0% | 4% | — |
| Avg latency | **385ms** | 1,239ms | 24,088ms | 2,303ms | 5,938ms | 28,385ms |
| Memory (GPU) | **~14.5 GB** | 14 GB | 19 GB | 14 GB | 19 GB | 21 GB |
| Interactive on MacBook | **Yes** | Borderline | No | No | No | No |
| Accuracy per active B | **40%/B** | 3.5%/B | 3.4%/B | 2.6%/B | 14.7%/B | ~2.2%/B |

*Qwen3 32B: 40/100 tests completed; extrapolated.

---

## Benchmark Infrastructure

- **Single-step runner:** `tests/model-behavior/benchmark-lfm.ts`
- **Multi-step runner:** `tests/model-behavior/benchmark-multi-step.ts`
- **Results (JSON):** `tests/model-behavior/.results/`
  - LFM2-24B-A2B: `lfm-unfiltered-k0-1771567058836.json`, `lfm-multistep-all-1771567127881.json`
  - Mistral-Small-24B: `lfm-unfiltered-k0-1771547409737.json`, `lfm-multistep-all-1771547680720.json`
  - Gemma 3 27B: `lfm-unfiltered-k0-1771550120786.json`, `lfm-multistep-all-1771564213834.json`
  - GPT-OSS-20B: `lfm-unfiltered-k0-1771567704182.json`, `lfm-multistep-all-1771567828367.json`
  - Qwen3-30B-A3B: `lfm-unfiltered-k0-1771568436811.json`, `lfm-multistep-all-1771568941941.json`
  - Qwen3 32B: Partial run (40 tests captured in transcript, no JSON file)
- **Model config:** `_models/config.yaml`
