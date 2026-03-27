# Benchmark Results

All runs: 100 tasks, 1 run per task, local llama-server (llama.cpp) or OpenAI API.

## Summary

| Model | Source | Raw Tool Call Parsing | Score | Accuracy |
|-------|--------|-----------------------|-------|----------|
| gpt-4o-mini | OpenAI | No | 93/100 | 93% |
| LFM2.5-1.2B-Instruct-Q4_0.gguf | HuggingFace | Yes | 72/100 | 72% |
| LFM2.5-1.2B-Instruct-Q4_0.gguf | HuggingFace | No | 71/100 | 71% |
| LFM2-350M-Q8_0.gguf | HuggingFace | No | 29/100 | 29% |
| LFM2-350M-Q8_0.gguf | HuggingFace | Yes | 29/100 | 29% |
| LFM2.5-350M-Q8_0.gguf | HuggingFace | No | 19/100 | 19% |
| LFM2.5-350M-Q8_0.gguf | HuggingFace | Yes | 18/100 | 18% |

## By Capability

| Capability | gpt-4o-mini | LFM2.5-1.2B (no raw) | LFM2.5-1.2B (raw) | LFM2-350M (no raw) | LFM2-350M (raw) | LFM2.5-350M (no raw) | LFM2.5-350M (raw) |
|------------|-------------|----------------------|--------------------|---------------------|-----------------|----------------------|-------------------|
| doors | 93.8% | 81.2% | 81.2% | 75.0% | 75.0% | 56.2% | 56.2% |
| lights | 91.7% | 75.0% | 75.0% | 41.7% | 41.7% | 29.2% | 29.2% |
| multi_tool | 91.7% | 66.7% | 75.0% | 8.3% | 8.3% | 16.7% | 8.3% |
| rejection | 100.0% | 8.3% | 8.3% | 0.0% | 0.0% | 8.3% | 8.3% |
| scene | 100.0% | 80.0% | 80.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| status | 100.0% | 80.0% | 80.0% | 20.0% | 20.0% | 0.0% | 0.0% |
| thermostat | 81.2% | 93.8% | 93.8% | 25.0% | 25.0% | 0.0% | 0.0% |

## By Inference Depth

| Depth | gpt-4o-mini | LFM2.5-1.2B (no raw) | LFM2.5-1.2B (raw) | LFM2-350M (no raw) | LFM2-350M (raw) | LFM2.5-350M (no raw) | LFM2.5-350M (raw) |
|-------|-------------|----------------------|--------------------|---------------------|-----------------|----------------------|-------------------|
| literal | 100.0% | 86.9% | 88.5% | 39.3% | 39.3% | 21.3% | 21.3% |
| semantic | 74.1% | 63.0% | 63.0% | 18.5% | 18.5% | 18.5% | 14.8% |
| boundary | 100.0% | 8.3% | 8.3% | 0.0% | 0.0% | 8.3% | 8.3% |

## By Phrasing

| Phrasing | gpt-4o-mini | LFM2.5-1.2B (no raw) | LFM2.5-1.2B (raw) | LFM2-350M (no raw) | LFM2-350M (raw) | LFM2.5-350M (no raw) | LFM2.5-350M (raw) |
|----------|-------------|----------------------|--------------------|---------------------|-----------------|----------------------|-------------------|
| imperative | 94.1% | 79.4% | 82.4% | 38.2% | 38.2% | 17.6% | 17.6% |
| colloquial | 87.5% | 75.0% | 75.0% | 29.2% | 29.2% | 20.8% | 16.7% |
| implicit | 91.7% | 54.2% | 54.2% | 12.5% | 12.5% | 16.7% | 16.7% |
| question | 100.0% | 72.2% | 72.2% | 33.3% | 33.3% | 22.2% | 22.2% |

## Result Files

| File | Model | Raw |
|------|-------|-----|
| 2026-03-26_13-47-21_gpt-4o-mini.md | gpt-4o-mini | No |
| 2026-03-26_14-08-37_LFM2-350M-Q8_0.gguf.md | LFM2-350M-Q8_0.gguf | No |
| 2026-03-26_14-10-07_LFM2-350M-Q8_0.gguf.md | LFM2-350M-Q8_0.gguf | Yes |
| 2026-03-26_14-20-42_LFM2.5-350M-Q8_0.gguf.md | LFM2.5-350M-Q8_0.gguf | No |
| 2026-03-26_14-21-08_LFM2.5-350M-Q8_0.gguf.md | LFM2.5-350M-Q8_0.gguf | Yes |
| 2026-03-26_14-22-41_LFM2.5-1.2B-Instruct-Q4_0.gguf.md | LFM2.5-1.2B-Instruct-Q4_0.gguf | No |
| 2026-03-26_14-24-10_LFM2.5-1.2B-Instruct-Q4_0.gguf.md | LFM2.5-1.2B-Instruct-Q4_0.gguf | Yes |
