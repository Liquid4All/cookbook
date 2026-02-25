# ADR-008: Tool-Calling Optimization Strategy — Grammar Constraints Before Fine-Tuning

**Date**: 2026-02-13
**Status**: Accepted
**Context**: The 20B local model (GPT-OSS-20B, production target LFM2.5-24B) exhibits 6 failure modes during MCP tool calls. Should we fine-tune the model, or are cheaper alternatives sufficient?

## Decision

**Don't fine-tune yet.** Implement 4 layers of free/low-cost improvements first. Fine-tune only if tool selection accuracy remains below 85% after all 4 layers are deployed.

## Observed Failure Modes

From `agent.log` and workaround code in `chat.rs` (session-015 analysis):

| Failure Mode | Evidence | Current Workaround | Structural vs Semantic |
|---|---|---|---|
| Unprefixed tool names (`move_file` instead of `filesystem.move_file`) | agent.log line 381 | `resolve_tool_name()` | Semantic |
| Malformed JSON arguments (unescaped quotes) | agent.log line 157, HTTP 500 | Retry + fallback chain | **Structural** |
| Empty responses (0 text, 0 tools) | agent.log lines 43, 262-268 | `MAX_EMPTY_RETRIES=2` + forced summary | **Structural** |
| Premature task abandonment (stops at 3/7 files) | `is_incomplete_response()` | Ephemeral continuation injection | Semantic |
| Redundant tool calls (same args twice) | System prompt rule 7 | None (prompt-only) | Semantic |
| Relative paths (`~/` instead of absolute) | System prompt rule 1 | None (prompt-only) | Semantic |

**Key insight**: Only 2 of 6 failures are structural (fixable by grammar constraints). The other 4 are semantic (model decision quality).

## The 4-Layer Strategy

### Layer 1: llama.cpp Grammar-Constrained Tool Calling

llama.cpp has built-in tool-calling support with GBNF grammar enforcement (via `--jinja`):
- **Guarantees 100% valid JSON** at the token level — eliminates HTTP 500 malformed JSON errors
- Auto-generates grammar from OpenAI-format tool definitions
- Has native format handlers for Qwen2.5 (current model family)
- Grammar sampler runs before temperature/top-k/top-p, masking invalid tokens

**Implementation**: Switch Ollama to native tool calling mode. `InferenceClient` already sends tools in the correct OpenAI format. Ollama passes these to llama.cpp which enforces grammar constraints internally.

**Files**: `src-tauri/src/inference/client.rs`, `_models/config.yaml`
**Cost**: Zero. **Effort**: A few hours. **Eliminates**: Malformed JSON errors entirely.

**Rationale**

The agent log from session-017 testing contains a concrete example of why this layer is critical:

```
WARN localcowork::commands::chat: all models unavailable, using static fallback
  error=HTTP 500: {"error":{"message":"error parsing tool call:
  raw='{\"create_dirs\":true,\"destination\":\"\"/Users/chintan/Desktop/Screenshot 2026-02-11 at 2.48.48 PM.png\",
  \"source\":\"/Users/chintan/Desktop/Screenshot 2026-02-11 at 2.48.48 PM.png\"}',
  err=invalid character '/' after object key:value pair"}}
```

The model generated `"destination":""/Users/...` — a double-quote that makes the JSON invalid. It knew the right tool (`filesystem.move_file`), the right arguments (`source`, `destination`, `create_dirs`), and the right file path. But one malformed character caused the entire turn to fail with HTTP 500, which triggered the fallback chain (`gpt-oss-20b` → `qwen3-30b-moe` → `static_response`), and the user saw "the local AI model is not currently available."

This failure class is **entirely structural** — the model's intent was correct, only the serialization was wrong. Grammar constraints operate at the token sampling level inside llama.cpp: before each token is selected, the GBNF grammar masks out any token that would produce invalid JSON. The model literally cannot generate `""` where a single `"` is expected, because the grammar won't allow that token.

Key properties of grammar-constrained generation:
- **No inference speed penalty** — grammar masking is applied during the sampling step, not as post-processing
- **No accuracy penalty** — the model still picks from its preferred tokens; the grammar only removes structurally invalid options
- **Compounding benefit with Layers 2-3** — few-shot examples (Layer 2) help the model pick the right tool; low temperature (Layer 3) reduces argument hallucination; grammar constraints (this layer) ensure the format is always valid. These are three orthogonal improvements.
- **Eliminates the HTTP 500 → fallback cascade** — today, malformed JSON triggers `is_retriable()` which walks the entire fallback chain before reaching `static_response`. With grammar constraints, this failure mode disappears entirely.

Additionally, a second failure mode observed in the same session involved the primary model timing out after 18 seconds on round 11 (22 messages in context), falling back to `qwen3:30b-a3b` which returned HTTP 404 (model not pulled in Ollama). The 404 was not in `is_retriable()`, so the chain stopped instead of continuing to `static_response`. This was fixed in session-017 by adding HTTP 404 to retriable errors, but grammar constraints would have prevented the original timeout by producing a valid tool call on the first attempt.

### Layer 2: Few-Shot Examples in System Prompt

Add 2-3 exemplar tool calls to `SYSTEM_PROMPT` showing:
1. Correct fully-qualified name usage: `filesystem.move_file` (not `move_file`)
2. Correct absolute path format: `/Users/chintan/Desktop/file.png` (not `~/Desktop/file.png`)
3. Correct multi-step sequencing: OCR extract -> propose rename -> filesystem.move_file -> next file

**Token cost**: ~200-300 tokens from the ~500 system prompt budget. Few-shot prompting improves tool-call accuracy by 15-40% in benchmarks (Berkeley Function Calling Leaderboard research).

**Files**: `src-tauri/src/commands/chat.rs` (SYSTEM_PROMPT constant)
**Cost**: Zero. **Effort**: 1 hour. **Eliminates**: Most unprefixed name and relative path errors.

### Layer 3: Dynamic Inference Parameters for Tool Turns

`InferenceClient` currently uses a single temperature for all turns. Tool-calling turns benefit from lower temperature:

| Turn Type | Temperature | Top-P |
|---|---|---|
| Tool-calling turn (tools parameter present) | 0.1 | 0.2 |
| Conversational turn (no tools) | 0.7 | 0.9 |

**Implementation**: In `InferenceClient::chat_completion_stream()`, check if `tools.is_some()` and adjust parameters accordingly.

**Files**: `src-tauri/src/inference/client.rs`
**Cost**: Zero. **Effort**: 30 minutes.

### Layer 4: Evaluate Qwen3-30B-A3B (MoE)

Before fine-tuning the current model, benchmark a potentially better base model:
- Qwen3-30B-A3B: Only 3B active parameters (MoE) for much faster inference
- Improved tool calling in thinking and non-thinking modes
- Scores 69.6 on Tau2-Bench (agent benchmark) vs Qwen2.5's ~63
- Apache 2.0 license

Run the existing 100-prompt tool-selection test suite against it. If it scores >=80% on the 15-tool set, it's a better choice than fine-tuning the current model.

**Cost**: Free to benchmark. **Effort**: Download + test run (~2 hours).

## When Fine-Tuning Becomes Worth It

Fine-tune only if Layers 1-4 are deployed and:
- Tool selection accuracy remains below 85% on the 15-tool set
- Persistent multi-step chain failures (UC-1 through UC-10)
- Argument hallucination that retries cannot fix

Recommended approach if needed:
- Base dataset: NousResearch Hermes Function Calling V1
- Augment with 500-1,000 examples from integration tests (UC-1 through UC-10)
- QLoRA with Unsloth on a single RTX 4090 (~$20-50, 1-2 days)
- Include 10% general conversation data to prevent catastrophic forgetting
- Must retrain when tool schemas change (13 MCP servers still evolving)

## Alternative: Toolshim Model (Hybrid Architecture)

If the primary model handles conversation well but struggles with tool selection, run a small specialized model (e.g., LFM2-1.2B-Tool) as a tool-calling intermediary:

```
User prompt -> Large model (reasoning) -> Small model (tool formatting) -> MCP servers
```

This decouples tool-calling quality from the primary model but adds architectural complexity. Consider only if Layers 1-4 fail and fine-tuning is impractical.

**Update (2026-02-15):** This approach was validated and evolved into [ADR-009: Dual-Model Orchestrator](./009-dual-model-orchestrator.md). LFM2-1.2B-Tool achieves 78% single-step accuracy with a RAG pre-filter (K=15, see [ADR-010](./010-rag-prefilter-benchmark-analysis.md)), but only 8% multi-step chain completion. ADR-009 addresses this with a plan-execute-synthesize pipeline: GPT-OSS-20B plans, LFM2-1.2B-Tool executes each step independently, GPT-OSS-20B synthesizes.

## Cost Comparison

| Approach | Fixes | Cost | Maintenance | Priority |
|---|---|---|---|---|
| Grammar-constrained tool calling | Malformed JSON (100%) | Free | Zero | **Layer 1** |
| Few-shot examples in prompt | Unprefixed names, bad paths (~30%) | Free | Low | **Layer 2** |
| Dynamic inference params | Wrong tool selection (~10%) | Free | Zero | **Layer 3** |
| Evaluate Qwen3-30B-A3B | Potentially all semantic issues | Free | Low | **Layer 4** |
| QLoRA fine-tuning | Remaining semantic issues (~20-40%) | $20-100 + days | High | **Defer** |
| Toolshim model | Tool formatting | Moderate | Medium | **Defer** |

## References

- Berkeley Function Calling Leaderboard (BFCL) V4: https://gorilla.cs.berkeley.edu/leaderboard.html
- llama.cpp function calling docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
- llama.cpp GBNF grammar: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- NousResearch Hermes Function Calling V1: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- Unsloth (fast QLoRA): https://github.com/unslothai/unsloth
- ToolBench (ICLR'24): https://github.com/OpenBMB/ToolBench
- Qwen3 announcement: https://qwenlm.github.io/blog/qwen3/
- Goose Toolshim approach: https://block.github.io/goose/blog/2025/04/11/finetuning-toolshim/
- LFM2 (Liquid Foundation Models): https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models
