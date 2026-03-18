# Benchmark Summary

| Date | Model | Score | % |
|------|-------|-------|---|
| 2026-03-17 | gpt-4o-mini | 19/19 | 100% |
| 2026-03-17 | LFM2.5-1.2B-Instruct-Q4_0.gguf | 13/19 | 68% |
| 2026-03-17 | LFM2.5-1.2B-Instruct-Q8_0.gguf | 10/19 | 53% |
| 2026-03-17 | LFM2-350M-Q8_0.gguf | 7/19 | 37% |

## Per-task results (2026-03-17)

| # | Task | Diff | gpt-4o-mini | Q4_0 1.2B | Q8_0 1.2B | Q8_0 350M |
|---|------|------|-------------|-----------|-----------|-----------|
| 1 | Turn on kitchen lights | easy | PASS | PASS | PASS | PASS |
| 2 | Lock the front door | easy | PASS | PASS | PASS | PASS |
| 3 | Heat house to 72 degrees | easy | PASS | PASS | PASS | FAIL |
| 4 | Get status of all devices | easy | PASS | PASS | FAIL | FAIL |
| 5 | Activate movie night scene | medium | PASS | PASS | PASS | FAIL |
| 6 | Unlock the garage door | medium | PASS | PASS | PASS | PASS |
| 7 | Check bedroom light status | medium | PASS | PASS | FAIL | FAIL |
| 8 | Cool house to 74 degrees | medium | PASS | PASS | PASS | FAIL |
| 9 | Away scene via indirect phrasing | hard | PASS | PASS | PASS | FAIL |
| 10 | Lock back door + off office lights | hard | PASS | PASS | PASS | PASS |
| 11 | Turn on all lights (multi-tool) | hard | PASS | FAIL | FAIL | FAIL |
| 12 | Turn off bedroom light (pronoun reference) | hard | PASS | PASS | PASS | PASS |
| 13 | Correct bulk action (hallway off) | hard | PASS | FAIL | FAIL | FAIL |
| 14 | Relative thermostat increase (+2 degrees) | hard | PASS | PASS | PASS | PASS |
| 15 | Unlock first door (3-turn back-reference) | hard | PASS | FAIL | FAIL | PASS |
| 16 | Reject: unsupported device (dim lights) | easy | PASS | FAIL | FAIL | FAIL |
| 17 | Reject: off-topic request | easy | PASS | PASS | FAIL | FAIL |
| 18 | Reject: incomplete request | easy | PASS | FAIL | FAIL | FAIL |
| 19 | Reject: ambiguous request | easy | PASS | FAIL | FAIL | FAIL |
