# ADR-009: Dual-Model Orchestrator (Planner + Router)

**Status**: Accepted — enabled for 1-2 step workflows, single-model fallback for complex chains
**Date**: 2026-02-15 (original), 2026-02-18 (updated with A/B test results)
**Decision makers**: Session-031 benchmark results (original), Phase 2 A/B test (update)
**Relates to**: [ADR-010](./010-rag-prefilter-benchmark-analysis.md), [ADR-008](./008-tool-calling-optimization-strategy.md)

> **Current state:** The orchestrator is fully implemented and operational. LFM2-24B-A2B serves as both planner (bracket-format plans) and synthesizer. LFM2.5-1.2B-Router-FT-v2 handles tool selection with 100% accuracy on tested workflows. A/B testing (Phase 2, 2026-02-18) confirmed the dual-model approach eliminates 5 behavioral pathologies present in the single-model agent loop and is 2.5x faster. Complex 4+ step workflows remain limited by planner decomposition. See [Dual-Model Orchestrator Performance](../model-analysis/dual-model-orchestrator-performance.md) for full A/B test results.

---

## Context

Session-031 benchmarked LFM2-1.2B-Tool on 50 multi-step tool chains. Results:

| Metric | Value |
|--------|-------|
| Single-step accuracy (K=15) | **78%** |
| Multi-step chain completion | **8%** (4/50) |
| Wrong tool (multi-step) | 56% of failures |
| No tool call (multi-step) | 32% of failures |
| Deflection (FM-3) | 4% of failures |

The 78% to 8% drop occurs because conversation history (tool calls + results) degrades the
1.2B model's tool selection accuracy. The model cannot simultaneously hold context, parse
prior results, and select the next tool. This is a parameter-count limitation, not fixable
by prompt engineering.

However, when given a **clean single-step prompt** (no conversation history, filtered to
15 tools), the model selects the correct tool 78% of the time. This is the key insight.

## Decision

Implement a **dual-model orchestrator** with three phases:

1. **Plan** (LFM2-24B-A2B): Decompose user request into a sequence of self-contained steps.
   No tool definitions sent — only server capability summaries. Output: bracket-format plan
   (`[plan.add_step(...)]`). The same model serves as planner and synthesizer.

2. **Execute** (LFM2.5-1.2B-Router-FT-v2, per step): For each step, build a clean single-step
   prompt with RAG pre-filtered tools (K=15). No conversation history. Parse bracket tool call,
   override arguments from step description context, and execute via MCP.

3. **Synthesize** (LFM2-24B-A2B): Generate user-facing response from accumulated step results.
   Streamed to the frontend.

The orchestrator is **opt-in** via `_models/config.yaml` and **wraps** the existing single-model
agent loop. If orchestration fails at any point, control falls through to the single-model
loop unchanged.

## Alternatives Considered

| Alternative | Why Not |
|-------------|---------|
| Fine-tune LFM2 on multi-step chains | Requires training data we don't have; model size is the fundamental limit |
| Use planner model for everything | Larger model is overloaded by 67 tool definitions (FM-11); ADR-010 confirmed this |
| Bigger model (LFM2.5-24B) | Not yet released; architecture should work regardless of model size |
| Single model + better prompts | Session-031 proved prompt engineering alone can't bridge the 78% to 8% gap |

## Actual Performance (A/B Test, 2026-02-18)

Tested with 3 messages of increasing complexity. Full results in [Orchestrator Performance](../model-analysis/dual-model-orchestrator-performance.md).

| Metric | Single-Model (24B Solo) | Dual-Model (24B + 1.2B) |
|--------|------------------------|-------------------------|
| 1-step task | Correct, then looped 8x extra | Clean single-pass |
| 2-step cross-server | Correct, then 7 wasted calls | Clean two-step execution |
| 4+ step complex | Partial (3/6 tools, stuck in category) | Partial (1/4 tools, planner limit) |
| Behavioral pathologies | 5 types (deflection, looping, confabulation, lock-in, hallucination) | None |
| Total wall time (3 msgs) | 54s | 21s |
| Wasted model calls | 16/27 | 0/10 |

**Key finding:** The dual-model approach is strictly better for 1-2 step requests. For 4+ step workflows, the planner under-decomposes (produces 1 step instead of 4-6), limiting effectiveness. The single-model approach attempts more tools via its chaotic looping but with higher failure rates.

## Resource Requirements

| Model | VRAM | Endpoint |
|-------|------|----------|
| LFM2-24B-A2B (planner + synthesizer) | ~13 GB | localhost:8080 (llama-server) |
| LFM2.5-1.2B-Router-FT-v2 (router) | ~1.5 GB | localhost:8082 (llama-server + embeddings) |
| **Total** | **~14.5 GB** | |

Fits on 16 GB systems (Apple M-series with unified memory, RTX 4060+).

## Model Compatibility

The orchestrator is **LFM-family only**. Both the planner and router depend on bracket-format tool calling (`[server.tool(args)]`), which is hardcoded in three places:

| Component | Hardcoded Format | Why |
|-----------|-----------------|-----|
| `PLANNER_SYSTEM_PROMPT` | Bracket: `[plan.add_step(...)]` | LFM2-24B-A2B had 94% JSON parse failure; bracket is its native format |
| `build_router_system_prompt()` | Bracket: `[server.tool(param="value")]` | Fine-tuned router was trained on this exact format |
| Tool delivery | Text list in system prompt (not OpenAI `tools` param) | 1.2B router accuracy drops to 0% with chat-template-reformatted tools |

**Non-LFM models (GPT-OSS, Qwen, etc.):** The orchestrator will fail at the plan phase — the model won't produce parseable bracket-format plans — and fall back to the single-model agent loop. This fallback is graceful (no crash, no data loss), but wastes ~2-3s on the failed planner call.

**For non-LFM models, set `orchestrator.enabled: false`** to skip the orchestrator entirely.

**The single-model agent loop (`chat.rs`) is fully format-portable.** It sends tools via the standard OpenAI `tools` JSON parameter and parses responses using the model's configured `tool_call_format` (native_json, pythonic, or bracket). GPT-OSS and Qwen work correctly in single-model mode.

| Mode | LFM2 (bracket) | GPT-OSS / Qwen (native_json) |
|------|----------------|------------------------------|
| Single-model agent loop | ✅ Works | ✅ Works |
| Orchestrator (plan-execute-synthesize) | ✅ Works | ❌ Falls back to single-model |

Making the orchestrator format-agnostic is tracked as a future improvement (format-aware planner prompts + JSON plan parser + OpenAI tools parameter for router). Not prioritized because the orchestrator's value is tightly coupled to the fine-tuned LFM router anyway.

## Rollback

Set `orchestrator.enabled: false` in `_models/config.yaml`. The single-model agent loop
(chat.rs) continues to work exactly as before — the orchestrator is a separate code path
that returns early if enabled, falls through if disabled or failed.

## Consequences

- Two models must run simultaneously (separate endpoints, separate processes)
- Latency: ~3 model calls per request (plan + N steps + synthesis) vs current single-round
- ToolPreFilter (WS-11D) is a hard dependency (implemented in Rust)
- Frontend receives new events (`plan-created`, `step-executing`, `step-completed`)
- Step descriptions must be self-contained — planner prompt engineering is critical
- Router requires training-format-aligned system prompt (text list, not OpenAI tools JSON)
- Argument override system compensates for router's tendency to hallucinate training-data paths

## Implementation

- Config: `_models/config.yaml` -> `orchestrator:` section
- Types: `src-tauri/src/inference/config.rs` -> `OrchestratorConfig`
- Client: `src-tauri/src/inference/client.rs` -> `from_config_with_model()`
- PreFilter: `src-tauri/src/agent_core/tool_prefilter.rs`
- Orchestrator: `src-tauri/src/agent_core/orchestrator.rs` (11 fixes: F1-F11)
- Integration: `src-tauri/src/commands/chat.rs` -> check before agent loop

## References

- [Dual-Model Orchestrator Performance](../model-analysis/dual-model-orchestrator-performance.md) — A/B test results, architecture diagrams, fix details
- [Fine-Tuning Results](../model-analysis/fine-tuning-results.md) — V1/V2 router training, accuracy, and failure analysis
- [ADR-010: RAG Pre-Filter](./010-rag-prefilter-benchmark-analysis.md) — K=15 validation
- [ADR-008: Tool-Calling Optimization](./008-tool-calling-optimization-strategy.md) — 4-layer strategy
