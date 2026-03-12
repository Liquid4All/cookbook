# Benchmark Summary

| Date | Model | Score | % |
|------|-------|-------|---|
| 2026-03-12 | gpt-4o-mini | 15/15 | 100% |
| 2026-03-12 | LFM2.5-1.2B-Instruct-Q4_0.gguf | 11/15 | 73% |
| 2026-03-12 | LFM2.5-1.2B-Instruct-Q8_0.gguf | 10/15 | 67% |
| 2026-03-12 | LFM2.5-1.2B-Instruct-Q4_K_M.gguf | 10/15 | 67% |
| 2026-03-12 | LFM2.5-1.2B-Thinking-Q4_0.gguf | 3/15 | 20% |

## Per-task results

| # | Task | Diff | gpt-4o-mini | Instruct-Q4_0 | Instruct-Q8_0 | Instruct-Q4_K_M | Thinking-Q4_0 |
|---|------|------|-------------|---------------|---------------|-----------------|---------------|
| 1 | Turn on kitchen lights | easy | PASS | PASS | PASS | PASS | FAIL |
| 2 | Lock the front door | easy | PASS | PASS | PASS | PASS | PASS |
| 3 | Heat house to 72 degrees | easy | PASS | PASS | FAIL | PASS | FAIL |
| 4 | Get status of all devices | easy | PASS | PASS | FAIL | PASS | FAIL |
| 5 | Activate movie night scene | medium | PASS | PASS | PASS | PASS | FAIL |
| 6 | Unlock the garage door | medium | PASS | FAIL | PASS | PASS | FAIL |
| 7 | Check bedroom light status | medium | PASS | PASS | FAIL | FAIL | FAIL |
| 8 | Cool house to 74 degrees | medium | PASS | PASS | PASS | PASS | FAIL |
| 9 | Away scene via indirect phrasing | hard | PASS | PASS | PASS | FAIL | FAIL |
| 10 | Lock back door + off office lights | hard | PASS | PASS | PASS | PASS | PASS |
| 11 | Turn on all lights (multi-tool) | hard | PASS | FAIL | FAIL | FAIL | FAIL |
| 12 | Turn off bedroom light (pronoun reference) | hard | PASS | PASS | PASS | PASS | PASS |
| 13 | Correct bulk action (hallway off) | hard | PASS | FAIL | FAIL | FAIL | FAIL |
| 14 | Relative thermostat increase (+2 degrees) | hard | PASS | PASS | PASS | PASS | FAIL |
| 15 | Unlock first door (3-turn back-reference) | hard | PASS | FAIL | PASS | FAIL | FAIL |

## Notes

- **LFM2.5-1.2B-Instruct-Q4_0**: best local variant tested. Consistent across most easy/medium tasks; fails on multi-room lights (11), bulk action (13), and multi-turn back-reference (15).
- **LFM2.5-1.2B-Instruct-Q8_0**: tied with Q4_K_M at 67%. Gains task 15 over Q4_0 but regresses on tasks 3, 4, 7.
- **LFM2.5-1.2B-Instruct-Q4_K_M**: tied with Q8_0 at 67% but smaller file (731 MB vs 1.2 GB). Gains tasks 3, 4, 6 over Q4_0 but loses task 9.
- **LFM2.5-1.2B-Thinking-Q4_0**: not suited for tool calling in this setup. Outputs reasoning text rather than structured tool calls, leading to near-zero scores.
- **LFM2-8B-A1B** (prior run): outputs tool calls in native `<|tool_call_start|>...<|tool_call_end|>` format instead of OpenAI-compatible structured tool calls. Incompatible with the current agent setup without a custom parser.
