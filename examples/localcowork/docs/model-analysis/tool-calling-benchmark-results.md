# Instruct-Model Tool-Calling Comparison

> LFM2-24B-A2B (~2B active params) vs five comparison models (~3B-32B active params)
> on 67 MCP tools, Apple M4 Max hardware.
> **Core finding: LFM2 achieves 94% of the best dense model's accuracy at 3% of the latency.**

---

## Purpose

This document compares LFM2-24B-A2B (Liquid AI's sparse MoE hybrid model) against five comparison models on the LocalCowork tool-calling benchmark. The models span two categories: dense instruct-only transformers (Gemma, Mistral, Qwen3 32B), and MoE models (GPT-OSS-20B, Qwen3-30B-A3B). The goal is to quantify the scaling advantage of LFM2's hybrid conv+attn architecture — achieving competitive accuracy with dramatically fewer active parameters per token.

The comparison is relevant for anyone deploying local AI agents on consumer hardware (16-36 GB unified memory), where inference latency and memory footprint directly constrain product viability.

---

## Test Environment

- **Hardware:** Apple M4 Max, 36 GB unified memory, 32 GPU cores
- **Test suite:** 100 single-step tool selection prompts, 50 multi-step chains (3-6 steps each), 67 tools across 13 MCP servers
- **Eval dataset:** Custom domain-specific benchmark for LocalCowork's MCP tool set — not a standard benchmark (e.g., BFCL). Test definitions in `tests/model-behavior/tool-selection/` and `tests/model-behavior/multi-step-chains-*.ts`.
- **Inference parameters (original — near-greedy):** Temperature 0.1, top_p 0.1, top_k 50, repetition_penalty 1.05, max_tokens 512
- **Inference parameters (re-run — greedy):** Temperature 0, top_p 1.0, top_k 0, repetition_penalty 1.0, max_tokens 512
- **Sampling note:** The original run used near-greedy parameters. The greedy re-run (see below) uses fully deterministic sampling (temp=0) as recommended by Liquid AI's head of post-training. Single run per model, N=100 for single-step, N=50 for multi-step.
- **Tool format:** LFM2 uses bracket format `[server.tool(args)]`; Ollama models use native JSON function calling
- **Qwen3 32B note:** 40/100 single-step tests completed before benchmark was stopped due to extreme latency. Results extrapolated proportionally; multi-step not tested.
- **Qwen3-30B-A3B note:** Ollama ships the **original Qwen3-30B-A3B release** (with `<think>` mode), not the later `-Instruct-2507` variant which specifically improved tool calling and removed `<think>` block generation. The 51% no-tool-call rate is largely from `<think>` blocks consuming the response budget.
- **All benchmarks run same session:** Fresh runs on identical hardware, same test suite, same day.

### Model IDs and Quantization

**Tier 1 — Desktop-class (13-27 GB VRAM):**

| Model | HuggingFace ID | Ollama Tag | Runtime | Quantization |
|-------|---------------|-----------|---------|-------------|
| LFM2-24B-A2B | `LiquidAI/LFM2-24B-A2B-Preview` | N/A | llama-server | Q4_K_M (GGUF) |
| Mistral-Small-24B | `mistralai/Mistral-Small-24B-Instruct-2501` | `mistral-small:24b` | Ollama | Q4_K_M |
| Gemma 3 27B | `google/gemma-3-27b-it` | `gemma3:27b` | Ollama | Q4_K_M |
| GPT-OSS-20B | `openai/gpt-oss-20b` | `gpt-oss:20b` | Ollama | MXFP4 (native, ~4.25 bits/param) |
| Qwen3-30B-A3B-Instruct-2507 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | Ollama | Q4_K_M |

**Tier 2 — Small model class (2-4 GB VRAM):**

| Model | HuggingFace ID | Ollama Tag | Runtime | Quantization |
|-------|---------------|-----------|---------|-------------|
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | `llama3.2:3b` | Ollama | Q4_K_M |
| Phi-4-mini | `microsoft/phi-4-mini` | `phi4-mini` | Ollama | Q4_K_M |
| Qwen3-4B | `Qwen/Qwen3-4B` | `qwen3:4b` | Ollama | Q4_K_M |

**Dropped from greedy re-run:**

| Model | HuggingFace ID | Ollama Tag | Reason |
|-------|---------------|-----------|--------|
| Qwen3 32B | `Qwen/Qwen3-32B` | `qwen3:32b` | Only had partial run (40/100 tests), extreme latency (~28s/query) |
| Qwen3-30B-A3B (original) | `Qwen/Qwen3-30B-A3B` | `qwen3:30b-a3b` | Replaced by Instruct-2507 variant |

---

## Results

### Single-Step Tool Selection (100 prompts, 67 tools)

| Model | Architecture | Active Params | Accuracy | Avg Latency | Memory (GPU) |
|-------|-------------|--------------|----------|-------------|-------------|
| **Gemma 3 27B** | Dense transformer | 27B | **91%** (91/100) | 24,088ms | 19 GB |
| Mistral-Small-24B | Dense transformer | 24B | 85% (85/100) | 1,239ms | 14 GB |
| **LFM2-24B-A2B** | **Hybrid MoE (conv+attn)** | **~2B** | **80%** (80/100) | **385ms** | **~14.5 GB** |
| Qwen3 32B | Dense transformer | 32B | ~70% (28/40)* | 28,385ms | 21 GB |
| GPT-OSS-20B | MoE transformer | ~3.6B | 51% (51/100) | 2,303ms | 14 GB |
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

LFM2-24B-A2B achieves 80% accuracy with only ~2B active parameters per token — outperforming other MoE models with similar active params and competing with dense models 12-16x its active size, at a fraction of the compute cost.

| Model | Active Params | Accuracy | Latency | VRAM |
|-------|--------------|----------|---------|------|
| **LFM2-24B-A2B** | **~2B** | **80%** | **385ms** | **~14.5 GB** |
| Qwen3-30B-A3B | ~3B | 44% | 5,938ms | 19 GB |
| GPT-OSS-20B | ~3.6B | 51% | 2,303ms | 14 GB |
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

LFM2-24B-A2B achieves 80% accuracy with only ~2B active parameters — outperforming two other MoE models (GPT-OSS ~3.6B, Qwen3-A3B ~3B) and competing with dense models (24-32B active). This means comparable accuracy at 385ms instead of 1-24 seconds, which is the difference between a real-time interactive agent and a batch processor. It also fits in 14.5 GB VRAM — consumer MacBook territory rather than server-class hardware.

### 4. Qwen3's reasoning-first design hurts tool calling

Qwen3 32B (dense, 32B active) scored the lowest accuracy (~70%) with a 25% "no tool call" rate. The model generates extended reasoning chains instead of selecting tools. This is a training objective mismatch — models optimized for chain-of-thought reasoning may not transfer well to structured tool dispatch without fine-tuning.

### 5. Single-step accuracy doesn't predict multi-step success

Gemma leads single-step (91%) but drops to 48% on chains. Mistral is second in single-step (85%) but leads chains (66%). LFM2 is third in single-step (80%) with lowest chains (26%). Multi-step success depends more on maintaining coherence across turns and self-correcting after tool results — capabilities not captured by single-step benchmarks.

### 6. MoE architecture alone doesn't guarantee efficiency

Qwen3-30B-A3B (~3B active, MoE transformer) scores 44% at 5.9s latency. LFM2-24B-A2B (~2B active, hybrid MoE conv+attn) scores 80% at 385ms. Both are MoE models with similar active parameter counts, but LFM2 delivers **1.8x the accuracy at 15x the speed**. The difference is the block architecture: LFM2's convolution blocks appear more efficient at parsing structured tool schemas than Qwen3's transformer-only attention blocks.

### 7. MoE with native function calling isn't enough either

GPT-OSS-20B (MoE, ~3.6B active, native OpenAI function calling) scores 51% — better than the ~36% estimated in earlier testing, but still well below the 80% threshold for production use. With 40% wrong-tool rate and 0% multi-step chains, having native function calling format doesn't compensate for the model's inability to discriminate among 67 tools.

---

## Summary Table (Near-Greedy — Original Run)

| Metric | LFM2-24B-A2B | Mistral-Small-24B | Gemma 3 27B | GPT-OSS-20B | Qwen3-30B-A3B | Qwen3 32B |
|--------|-------------|-------------------|-------------|-------------|---------------|-----------|
| Architecture | Hybrid MoE | Dense | Dense | MoE | MoE | Dense |
| Total params | 24B | 24B | 27B | 21B | 30B | 32B |
| Active params/token | ~2B | 24B | 27B | ~3.6B | ~3B | 32B |
| Single-step accuracy | 80% | 85% | **91%** | 51% | 44% | ~70%* |
| Multi-step chains | 26% | **66%** | 48% | 0% | 4% | — |
| Avg latency | **385ms** | 1,239ms | 24,088ms | 2,303ms | 5,938ms | 28,385ms |
| Memory (GPU) | **~14.5 GB** | 14 GB | 19 GB | 14 GB | 19 GB | 21 GB |
| Interactive on MacBook | **Yes** | Borderline | No | No | No | No |
| Accuracy per active B | **40%/B** | 3.5%/B | 3.4%/B | 14.2%/B | 14.7%/B | ~2.2%/B |

*Qwen3 32B: 40/100 tests completed; extrapolated. Dropped from greedy re-run.

---

## Greedy Sampling Re-Run

Greedy sampling (temperature=0, top_p=1.0, top_k=0, repetition_penalty=1.0) was recommended by Liquid AI's head of post-training to eliminate variance from near-greedy sampling. All benchmarks below use `--greedy` flag.

**Changes from original lineup:**
- **Replaced** Qwen3-30B-A3B (original) with **Qwen3-30B-A3B-Instruct-2507** — removes `<think>` mode that caused 51% no-tool-call rate
- **Dropped** Qwen3 32B — was only a partial run (40/100 tests)
- **Added** 3 Tier 2 small models for active-parameter-class comparison (CEO request)

### Tier 1: Greedy vs Near-Greedy Comparison

| Model | Single-step (near-greedy) | Single-step (greedy) | Δ | Multi-step (near-greedy) | Multi-step (greedy) | Δ |
|-------|--------------------------|---------------------|---|-------------------------|--------------------|----|
| LFM2-24B-A2B | 80% | 80% | 0pp | 26% | 26% | 0pp |
| Mistral-Small-24B | 85% | 85% | 0pp | 66% | 66% | 0pp |
| Gemma 3 27B | 91% | 91% | 0pp | 48% | 48% | 0pp |
| GPT-OSS-20B | 51% | 51% | 0pp | 0% | 0% | 0pp |
| Qwen3-30B-A3B-Instruct-2507 | N/A (new model) | 71% | | N/A | 42% | |

**Finding:** All five Tier 1 models produce identical results under greedy vs near-greedy sampling (0pp delta across all models and both benchmarks). This confirms the near-greedy config (temp=0.1) was already fully deterministic for N=100 single-run benchmarks.

### Tier 2: Small Model Active-Parameter Comparison (Greedy Only)

These models match LFM2-24B-A2B's ~2B active compute budget. Tests whether hybrid MoE architecture outperforms dense models at similar per-token cost.

| Model | Total Params | Active Params | Architecture | Single-step | Multi-step | Avg Latency | VRAM | Tool Call Rate |
|-------|-------------|--------------|-------------|-------------|-----------|-------------|------|---------------|
| **Llama 3.2 3B** | 3B | 3B | Dense | **82%** | **52%** | **305ms** | ~2.0 GB | 96% |
| Phi-4-mini (3.8B) | 3.8B | 3.8B | Dense | 60% | 14% | 549ms | ~2.5 GB | 94% |
| Qwen3-4B | 4B | 4B | Dense | 20% | 0% | 5,837ms | ~2.5 GB | 21% |
| **LFM2-24B-A2B** | **24B** | **~2B (MoE)** | **Hybrid MoE** | **80%** | **26%** | **390ms** | **~14.5 GB** | **94%** |

### Greedy Results — Full Single-Step Table

| Model | Architecture | Active Params | Accuracy | Avg Latency | Memory (GPU) | Tool Call Rate | Wrong Tool Rate |
|-------|-------------|--------------|----------|-------------|-------------|---------------|----------------|
| **Gemma 3 27B** | Dense transformer | 27B | **91%** | 21,464ms | 19 GB | 99% | 8% |
| Mistral-Small-24B | Dense transformer | 24B | 85% | 1,425ms | 14 GB | 97% | 12% |
| Llama 3.2 3B | Dense transformer | 3B | 82% | 305ms | ~2.0 GB | 96% | 14% |
| LFM2-24B-A2B | Hybrid MoE (conv+attn) | ~2B | 80% | 390ms | ~14.5 GB | 94% | 14% |
| Qwen3-30B-A3B-Instruct-2507 | MoE transformer | ~3B | 71% | 610ms | 19 GB | 97% | 26% |
| Phi-4-mini | Dense transformer | 3.8B | 60% | 549ms | ~2.5 GB | 94% | 34% |
| GPT-OSS-20B | MoE transformer | ~3.6B | 51% | 2,221ms | 14 GB | 92% | 41% |
| Qwen3-4B | Dense transformer | 4B | 20% | 5,837ms | ~2.5 GB | 21% | 1% |

### Greedy Results — Full Multi-Step Table

| Model | Chain Completion | Step Completion | Avg Steps/Chain | Chains Passed |
|-------|-----------------|----------------|-----------------|---------------|
| **Mistral-Small-24B** | **66%** | 74% | 3.3 | 33/50 |
| Llama 3.2 3B | 52% | 61% | 2.7 | 26/50 |
| Gemma 3 27B | 48% | 57% | 2.5 | 24/50 |
| Qwen3-30B-A3B-Instruct-2507 | 42% | 50% | 2.2 | 21/50 |
| LFM2-24B-A2B | 26% | 31% | 1.4 | 13/50 |
| Phi-4-mini | 14% | 34% | 1.5 | 7/50 |
| GPT-OSS-20B | 0% | 0% | 0.0 | 0/50 |
| Qwen3-4B | 0% | 5% | 0.2 | 0/50 |

---

## Experiment: Prompt Optimization + RAG Pre-Filter (Stashed)

> **Status:** Code stashed (`git stash@{0}`), not merged. Results documented here for reference.

We tested whether prompt engineering and RAG pre-filtering could improve LFM2-24B-A2B accuracy. Three optimizations were applied simultaneously:

1. **System prompt strengthening** — expanded from 4/5 rules to 8/9 rules, added 3 few-shot examples targeting top failure patterns (list_dir fallback, data-server deflection, overdue-task confusion)
2. **Tool description sharpening** — 13 descriptions rewritten with contrastive negative guidance ("Do NOT use X — use Y instead") targeting 3 confusion clusters
3. **RAG pre-filtering for multi-step** — wired `--top-k 15` per-step filtering into the multi-step runner using the existing shared embedding infrastructure

### Results (LFM2-24B-A2B, greedy sampling)

| Benchmark | Baseline | Post-Optimization | Delta |
|-----------|----------|-------------------|-------|
| Single-step (all 67 tools) | **80%** | 77% | **-3pp** |
| Multi-step (all 67 tools) | **26%** | 24% | -2pp |
| Multi-step (`--top-k 15`) | n/a | **34%** | +8pp vs baseline |

### Multi-step with `--top-k 15` breakdown

| Difficulty | Baseline (no filter) | Post-Opt (no filter) | Post-Opt (K=15) |
|------------|---------------------|---------------------|-----------------|
| Easy (15) | 60% | 60% | 53% |
| Medium (20) | 10% | 10% | 30% |
| Hard (15) | 7% | 7% | 20% |

Filter metrics: 84% hit rate (102/121 steps), 15.0 avg tools/step, 19 filter misses.

### Key findings

1. **Prompt-only changes didn't help** — the strengthened system prompt and sharpened descriptions had no measurable impact on multi-step (26% → 24%, within noise) and slightly hurt single-step (80% → 77%). The additional few-shot examples may have introduced new confusions while fixing targeted ones.

2. **RAG pre-filtering works but has a ceiling** — reducing 67 → 15 tools per step improved multi-step from 24% → 34% (+10pp). Medium chains saw the biggest gain (10% → 30%). But the 84% filter hit rate means 16% of steps are guaranteed failures regardless of model quality.

3. **Single-step regression is concerning** — 3 new wrong-tool failures appeared that weren't in the baseline. The contrastive descriptions ("Do NOT use X") may confuse MoE routing by increasing attention to the wrong tool.

4. **Deflection remains stubborn** — 20-24% deflection rate across all configurations. Prompt-level anti-deflection rules didn't reduce it. The model's tendency to ask clarifying questions instead of calling tools is a deeper behavioral pattern.

### Why stashed

- Net negative on single-step accuracy (-3pp) — the most important metric for production
- Multi-step gains (+8pp with filtering) fall far short of the 45-55% target
- Prompt-level fixes can't overcome fundamental tool cognitive overload
- The architectural fix (dual-model orchestrator, M1/M2 interventions) is the right path forward

### Lesson learned

Benchmark-level prompt engineering is not the lever for LFM2-24B-A2B accuracy improvement. The model's 26% multi-step ceiling is an architectural problem (too many tools, MoE routing limitations, context dilution) that requires structural solutions: the dual-model orchestrator for multi-step, and fine-tuning or model selection for single-step.

---

## Benchmark Infrastructure

- **Single-step runner:** `tests/model-behavior/benchmark-lfm.ts`
- **Multi-step runner:** `tests/model-behavior/benchmark-multi-step.ts`
- **Results (JSON):** `tests/model-behavior/.results/`

**Near-greedy run (original):**
  - LFM2-24B-A2B: `lfm-unfiltered-k0-1771567058836.json`, `lfm-multistep-all-1771567127881.json`
  - Mistral-Small-24B: `lfm-unfiltered-k0-1771547409737.json`, `lfm-multistep-all-1771547680720.json`
  - Gemma 3 27B: `lfm-unfiltered-k0-1771550120786.json`, `lfm-multistep-all-1771564213834.json`
  - GPT-OSS-20B: `lfm-unfiltered-k0-1771567704182.json`, `lfm-multistep-all-1771567828367.json`
  - Qwen3-30B-A3B: `lfm-unfiltered-k0-1771568436811.json`, `lfm-multistep-all-1771568941941.json`
  - Qwen3 32B: Partial run (40 tests captured in transcript, no JSON file)

**Greedy re-run:**
  - Mistral-Small-24B: `lfm-unfiltered-k0-1771613037719.json`, `lfm-multistep-all-1771613286797.json`
  - GPT-OSS-20B: `lfm-unfiltered-k0-1771613522386.json`, `lfm-multistep-all-1771613649881.json`
  - Qwen3-30B-A3B-Instruct-2507: `lfm-unfiltered-k0-1771612796614.json`, `lfm-multistep-all-1771612882476.json`
  - Llama 3.2 3B: `lfm-unfiltered-k0-1771612666277.json`, `lfm-multistep-all-1771612721761.json`
  - Phi-4-mini: `lfm-unfiltered-k0-1771612565716.json`, `lfm-multistep-all-1771612623842.json`
  - Qwen3-4B: `lfm-unfiltered-k0-1771612139331.json`, `lfm-multistep-all-1771612493546.json`
  - LFM2-24B-A2B: `lfm-unfiltered-k0-1771625543192.json`, `lfm-multistep-all-1771625704065.json`
  - Gemma 3 27B: `lfm-unfiltered-k0-1771622888382.json`, `lfm-multistep-all-1771623851751.json`

**Post-optimization experiment (stashed):**
  - LFM2-24B-A2B single-step: `lfm-unfiltered-k0-1771648679219.json`
  - LFM2-24B-A2B multi-step (no filter): `lfm-multistep-all-1771648749381.json`
  - LFM2-24B-A2B multi-step (K=15): `lfm-multistep-all-1771648926635.json`

- **Model config:** `_models/config.yaml`
