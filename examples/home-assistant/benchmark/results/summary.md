# Benchmark Summary

| Date | Model | Score | % |
|------|-------|-------|---|
| 2026-03-12 | gpt-4o-mini | 15/15 | 100% |
| 2026-03-12 | LFM2.5-1.2B-Instruct-Q8_0.gguf | 10/15 | 67% |
| 2026-03-12 | LFM2.5-1.2B-Thinking-Q8_0.gguf | 3/15 | 20% |
| 2026-03-12 | LFM2-8B-A1B-Q4_0.gguf | 0/15 | 0% |

## Per-task results

| # | Task | Diff | gpt-4o-mini | LFM2.5-1.2B-Instruct | LFM2.5-1.2B-Thinking | LFM2-8B-A1B |
|---|------|------|-------------|----------------------|----------------------|-------------|
| 1 | Turn on kitchen lights | easy | PASS | PASS | FAIL | FAIL |
| 2 | Lock the front door | easy | PASS | PASS | PASS | FAIL |
| 3 | Heat house to 72 degrees | easy | PASS | FAIL | FAIL | FAIL |
| 4 | Get status of all devices | easy | PASS | FAIL | FAIL | FAIL |
| 5 | Activate movie night scene | medium | PASS | PASS | FAIL | FAIL |
| 6 | Unlock the garage door | medium | PASS | PASS | PASS | FAIL |
| 7 | Check bedroom light status | medium | PASS | FAIL | FAIL | FAIL |
| 8 | Cool house to 74 degrees | medium | PASS | PASS | FAIL | FAIL |
| 9 | Away scene via indirect phrasing | hard | PASS | PASS | FAIL | FAIL |
| 10 | Lock back door + off office lights | hard | PASS | PASS | FAIL | FAIL |
| 11 | Turn on all lights (multi-tool) | hard | PASS | FAIL | FAIL | FAIL |
| 12 | Turn off bedroom light (pronoun reference) | hard | PASS | PASS | PASS | FAIL |
| 13 | Correct bulk action (hallway off) | hard | PASS | PASS | FAIL | FAIL |
| 14 | Relative thermostat increase (+2 degrees) | hard | PASS | PASS | FAIL | FAIL |
| 15 | Unlock first door (3-turn back-reference) | hard | PASS | FAIL | FAIL | FAIL |

## Notes

- **LFM2-8B-A1B**: outputs tool calls in native `<|tool_call_start|>...<|tool_call_end|>` format instead of OpenAI-compatible structured tool calls. Incompatible with the current agent setup without a custom parser.
- **LFM2.5-1.2B-Thinking**: not suited for tool calling in this setup. Significantly underperforms the base instruct variant.
- **LFM2.5-1.2B-Instruct**: best local model tested. Fails on thermostat tasks (3, 8 partial), status queries (4, 7), multi-room lights (11), and multi-turn back-reference (15).
