# Benchmark Results

All runs: 100 tasks, 1 run per task, local llama-server (llama.cpp) or OpenAI API.

## Summary

| Model | Build | Raw Tool Call Parsing | Score | Accuracy |
|-------|-------|-----------------------|-------|----------|
| gpt-4o-mini | N/A | No | 92/100 | 92% |
| LFM2-350M-q8_0.gguf (fine-tuned) | b8533 (26 Mar 2026) | Yes | 48/100 | 48% |
| LFM2-350M-q8_0.gguf (fine-tuned) | b8533 (26 Mar 2026) | No | 48/100 | 48% |
| LFM2-350M-q8_0.gguf (fine-tuned) | 7930 (4 Feb 2026) | Yes | 48/100 | 48% |
| LFM2-350M-q8_0.gguf (fine-tuned) | 7930 (4 Feb 2026) | No | 16/100 | 16% |

## By Capability

| Capability | gpt-4o-mini | ft b8533 (raw) | ft b8533 (no raw) | ft 7930 (raw) | ft 7930 (no raw) |
|------------|-------------|----------------|-------------------|---------------|------------------|
| doors      | 93.8%       | 56.2%          | 56.2%             | 56.2%         | 56.2%            |
| lights     | 87.5%       | 87.5%          | 87.5%             | 79.2%         | 25.0%            |
| multi_tool | 91.7%       | 33.3%          | 33.3%             | 50.0%         | 8.3%             |
| rejection  | 100.0%      | 0.0%           | 0.0%              | 0.0%          | 0.0%             |
| scene      | 100.0%      | 80.0%          | 80.0%             | 90.0%         | 0.0%             |
| status     | 100.0%      | 30.0%          | 30.0%             | 30.0%         | 0.0%             |
| thermostat | 81.2%       | 18.8%          | 18.8%             | 12.5%         | 0.0%             |

## By Inference Depth

| Depth    | gpt-4o-mini | ft b8533 (raw) | ft b8533 (no raw) | ft 7930 (raw) | ft 7930 (no raw) |
|----------|-------------|----------------|-------------------|---------------|------------------|
| literal  | 98.4%       | 57.4%          | 57.4%             | 57.4%         | 19.7%            |
| semantic | 74.1%       | 48.1%          | 48.1%             | 48.1%         | 14.8%            |
| boundary | 100.0%      | 0.0%           | 0.0%              | 0.0%          | 0.0%             |

## By Phrasing

| Phrasing   | gpt-4o-mini | ft b8533 (raw) | ft b8533 (no raw) | ft 7930 (raw) | ft 7930 (no raw) |
|------------|-------------|----------------|-------------------|---------------|------------------|
| imperative | 94.1%       | 55.9%          | 55.9%             | 61.8%         | 17.6%            |
| colloquial | 87.5%       | 45.8%          | 45.8%             | 45.8%         | 16.7%            |
| implicit   | 87.5%       | 45.8%          | 45.8%             | 37.5%         | 12.5%            |
| question   | 100.0%      | 38.9%          | 38.9%             | 38.9%         | 16.7%            |

## Result Files

| File | Model | Build | Raw |
|------|-------|-------|-----|
| 2026-03-27_11-30-07_gpt-4o-mini.md | gpt-4o-mini | N/A | No |
| 2026-03-27_11-26-31_LFM2-350M-q8_0.gguf.md | LFM2-350M-q8_0.gguf (fine-tuned) | b8533 (26 Mar 2026) | Yes |
| 2026-03-27_11-26-29_LFM2-350M-q8_0.gguf.md | LFM2-350M-q8_0.gguf (fine-tuned) | b8533 (26 Mar 2026) | No |
| 2026-03-27_11-26-15_LFM2-350M-q8_0.gguf.md | LFM2-350M-q8_0.gguf (fine-tuned) | 7930 (4 Feb 2026) | Yes |
| 2026-03-27_11-25-50_LFM2-350M-q8_0.gguf.md | LFM2-350M-q8_0.gguf (fine-tuned) | 7930 (4 Feb 2026) | No |
