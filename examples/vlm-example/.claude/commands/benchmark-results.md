---
name: benchmark-results
description: Show completed WandB benchmark runs with accuracy from the last N minutes, for the defect-detection-benchmark project.
argument-hint: [minutes=10] [source=<name>]
allowed-tools: Bash, mcp__wandb__query_wandb_tool
---

Show benchmark results from the last N minutes.

## Arguments

Parse $ARGUMENTS as optional key=value pairs:
- `minutes` (default: 10): how far back to look
- `source` (optional): if provided, filter to only that dataset source (e.g. GoodsAD, VisA); if omitted, include all sources

Examples:
- `/benchmark-results` → last 10 minutes, all sources
- `/benchmark-results minutes=30` → last 30 minutes, all sources
- `/benchmark-results minutes=60 source=VisA` → last 60 minutes, VisA only

## Steps

1. Get the current UTC time with Bash: `date -u +%Y-%m-%dT%H:%M:%SZ`

2. Compute the cutoff timestamp by subtracting `minutes` from now (use Bash with `date -u -v-<N>M +%Y-%m-%dT%H:%M:%SZ` on macOS).

3. Query WandB using `mcp__wandb__query_wandb_tool` with:
   - entity: `paulescu`
   - project: `defect-detection-benchmark`
   - filter: `{"createdAt": {"$gt": "<cutoff>"}}`
   - fields: `displayName`, `state`, `createdAt`, `config`, `summaryMetrics`
   - order: `-createdAt`
   - limit: 50

4. From the results, keep only runs where:
   - `state` is `finished`
   - `summaryMetrics.accuracy` is a number (not null)
   - if `source` argument was provided: `config.source.value` matches it; otherwise include all sources

5. For each run, read the majority class baseline from `summaryMetrics.majority_class_accuracy`.

6. Print a markdown table with columns:
   `Run name | Time (UTC) | Config file | Source | Model | Majority Class Baseline | Accuracy`

   - **Model**: strip the `LiquidAI/` prefix for brevity
   - **Time (UTC)**: show only `HH:MM`
   - **Config file**: value of `config.config_file.value`, or `?` if missing
   - **Source**: value of `config.source.value`
   - **Majority Class Baseline**: from `summaryMetrics.majority_class_accuracy`, formatted as a percentage with one decimal (e.g. `58.3%`), or `?` if missing
   - **Accuracy**: formatted as a percentage with one decimal (e.g. `53.1%`)
   - Sort by `createdAt` descending (newest first)

7. After the table, print a one-line summary: total runs shown, time window, and sources included.

If no runs match, say so clearly.
