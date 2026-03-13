# Benchmark Summary

| Date | Model | Score | % |
|------|-------|-------|---|
| 2026-03-13 | gpt-4o-mini | 19/19 | 100% |
| 2026-03-13 | LFM2.5-1.2B-Instruct-Q4_0.gguf | 11/19 | 58% |
| 2026-03-13 | gpt-4o-mini (pre-prompt fix) | 17/19 | 89% |
| 2026-03-12 | gpt-4o-mini | 15/15 | 100% |
| 2026-03-12 | LFM2.5-1.2B-Instruct-Q4_0.gguf | 11/15 | 73% |
| 2026-03-12 | LFM2.5-1.2B-Instruct-Q8_0.gguf | 10/15 | 67% |
| 2026-03-12 | LFM2.5-1.2B-Instruct-Q4_K_M.gguf | 10/15 | 67% |
| 2026-03-12 | LFM2.5-1.2B-Thinking-Q4_0.gguf | 3/15 | 20% |

## Per-task results (19-task benchmark, 2026-03-13)

| # | Task | Diff | gpt-4o-mini | Instruct-Q4_0 |
|---|------|------|-------------|---------------|
| 1 | Turn on kitchen lights | easy | PASS | PASS |
| 2 | Lock the front door | easy | PASS | PASS |
| 3 | Heat house to 72 degrees | easy | PASS | PASS |
| 4 | Get status of all devices | easy | PASS | PASS |
| 5 | Activate movie night scene | medium | PASS | PASS |
| 6 | Unlock the garage door | medium | PASS | FAIL |
| 7 | Check bedroom light status | medium | PASS | PASS |
| 8 | Cool house to 74 degrees | medium | PASS | PASS |
| 9 | Away scene via indirect phrasing | hard | PASS | PASS |
| 10 | Lock back door + off office lights | hard | PASS | PASS |
| 11 | Turn on all lights (multi-tool) | hard | PASS | FAIL |
| 12 | Turn off bedroom light (pronoun reference) | hard | PASS | PASS |
| 13 | Correct bulk action (hallway off) | hard | PASS | FAIL |
| 14 | Relative thermostat increase (+2 degrees) | hard | PASS | PASS |
| 15 | Unlock first door (3-turn back-reference) | hard | PASS | FAIL |
| 16 | Reject: unsupported device (dim lights) | easy | FAIL | FAIL |
| 17 | Reject: off-topic request | easy | PASS | FAIL |
| 18 | Reject: incomplete request | easy | PASS | FAIL |
| 19 | Reject: ambiguous request | easy | FAIL | FAIL |

## Notes

### 2026-03-13 (19-task benchmark)

Tasks 16-19 test `intent_unclear` rejection behavior, added as part of the synthetic
data generation improvements.

- **gpt-4o-mini**: 19/19 (100%) after two targeted prompt fixes. The original system
  prompt had no guidance on when to call `intent_unclear`, causing the model to reply
  in plain text for tasks 16 and 19. Two changes fixed all four rejection tasks: (1) an
  explicit instruction in the system prompt ("Call intent_unclear, never plain text, when...")
  with per-reason examples, and (2) sharper enum descriptions in the tool schema
  distinguishing `ambiguous` (multiple valid home control actions) from `incomplete`
  (no target device at all). The `_verify_task_4` verifier was also relaxed to accept
  individual-type calls in addition to `device_type="all"`, since both are correct
  responses to "get all device status".

- **LFM2.5-1.2B-Instruct-Q4_0**: 11/19 (58%). Identical to its prior 11/15 on the
  original tasks. Fails all four rejection tasks (16-19), confirming zero generalisation
  to `intent_unclear` without fine-tuning. This is the primary gap the new training
  examples are designed to close.

### 2026-03-12 (15-task benchmark, pre-rejection tasks)

- **LFM2.5-1.2B-Instruct-Q4_0**: best local variant tested. Consistent across most easy/medium tasks; fails on multi-room lights (11), bulk action (13), and multi-turn back-reference (15).
- **LFM2.5-1.2B-Instruct-Q8_0**: tied with Q4_K_M at 67%. Gains task 15 over Q4_0 but regresses on tasks 3, 4, 7.
- **LFM2.5-1.2B-Instruct-Q4_K_M**: tied with Q8_0 at 67% but smaller file (731 MB vs 1.2 GB). Gains tasks 3, 4, 6 over Q4_0 but loses task 9.
- **LFM2.5-1.2B-Thinking-Q4_0**: not suited for tool calling in this setup. Outputs reasoning text rather than structured tool calls, leading to near-zero scores.
- **LFM2-8B-A1B** (prior run): outputs tool calls in native `<|tool_call_start|>...<|tool_call_end|>` format instead of OpenAI-compatible structured tool calls. Incompatible with the current agent setup without a custom parser.
