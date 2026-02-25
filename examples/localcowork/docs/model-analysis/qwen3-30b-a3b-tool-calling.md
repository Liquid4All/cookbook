# Qwen3-30B-A3B: Why MoE ≠ Efficient Tool Calling

**Model:** Qwen3-30B-A3B (MoE, 30B total, ~3B active per token)
**Date:** 2026-02-16
**Verdict:** Rejected. Tool fixation loops make it unusable for multi-step tasks.

---

## What Happened

**Task:** List screenshots on Desktop → OCR each → rename with semantic name.

| Round | Tool Called | Result | Time |
|---|---|---|---|
| 0 | `filesystem.list_dir` | ✅ Found 4 files | 22s |
| 1 | `filesystem.search_files` | Same 4 files (wrong tool) | 31s |
| 2 | `filesystem.search_files` | Identical (stuck in loop) | 32s |
| 3 | `filesystem.search_files` | Identical | 32s |
| 4 | `filesystem.search_files` | Identical | 34s |

**Failure mode: Tool fixation loop.** The model locks onto `filesystem.search_files` and repeats it with identical arguments, never crossing to `ocr.extract_text_from_image`. Each round takes ~30s of reasoning (thinking tokens) yet arrives at the same wrong answer.

For comparison, GPT-OSS-20B on the same task: listed files correctly (21s), then deflected to the user (5.4s). Different failure mode, same bottleneck — neither model crosses from `filesystem.*` to `ocr.*`.

---

## Why It Fails

**1. Tool overload at ~3B active params.** 59 tool definitions (8,670 tokens) consume 30% of usable input context. The correct next tool (`ocr.extract_text_from_image`) competes with 58 alternatives. A ~3B active model can't maintain discriminative attention across this many options.

**2. Same-namespace bias.** After using `filesystem.list_dir`, the model preferentially selects other `filesystem.*` tools. Tool names with shared prefixes have higher embedding similarity, creating anchor bias. The model never reaches `ocr.*` because `filesystem.search_files` always scores higher.

**3. Lost-in-the-middle.** The system prompt's workflow example (`list_dir → extract_text → move_file`) sits at ~token 600. Tool definitions span tokens 900–9,570. By the time the model decides, the example is ~9,000 tokens behind — deeply affected by attention decay.

**4. Confident but wrong.** Unlike GPT-OSS-20B (which bails quickly at 5.4s), the MoE model spends 30+ seconds reasoning per round and converges on the wrong tool every time. It lacks the self-awareness to recognize uncertainty. Inference time increases monotonically (22s → 34s) as context grows — the model reasons harder but not better.

---

## Key Takeaway

**MoE architecture with ~3B active parameters is insufficient for 50+ tool selection.** The parameter efficiency that makes MoE attractive for general inference becomes a liability for tool calling, where the model must discriminate between dozens of semantically similar options simultaneously.

This model was considered as the planner in the dual-model orchestrator (ADR-009) based on its low VRAM (~5 GB). We ultimately chose LFM2-24B-A2B instead — its hybrid convolution + attention architecture handles tool schemas dramatically better (80% single-step vs ~36% for Qwen3-30B-A3B), and it serves as both planner and synthesizer in the production orchestrator.

---

## References

- [LFM2-24B-A2B Benchmark](./lfm2-24b-a2b-benchmark.md) — the model that replaced this one
- [Project Learnings](./project-learnings-and-recommendations.md) — cross-model synthesis
- [ADR-009: Dual-Model Orchestrator](../architecture-decisions/009-dual-model-orchestrator.md) — production architecture
