#!/usr/bin/env bash
# Benchmark Fine-Tuned Router Model
#
# Runs the full benchmark suite comparing base vs fine-tuned models.
# Requires llama-server instances running for both models.
#
# Usage:
#   # Start base model on port 8082, fine-tuned on port 8083
#   llama-server --model LFM2-1.2B-Tool-F16.gguf --port 8082 --embeddings
#   llama-server --model LFM2-1.2B-Router-FT-Q8_0.gguf --port 8083 --embeddings
#
#   # Run comparison benchmarks
#   ./scripts/benchmark-finetuned.sh
#
# Output: comparison report in tests/model-behavior/.results/

set -euo pipefail

BASE_ENDPOINT="${BASE_ENDPOINT:-http://localhost:8082}"
FT_ENDPOINT="${FT_ENDPOINT:-http://localhost:8083}"
PLANNER_ENDPOINT="${PLANNER_ENDPOINT:-http://localhost:8080}"
TOP_K="${TOP_K:-15}"
RESULTS_DIR="tests/model-behavior/.results"

echo "=== Fine-Tuned Router Benchmark Suite ==="
echo "Base model:     $BASE_ENDPOINT"
echo "Fine-tuned:     $FT_ENDPOINT"
echo "Planner (24B):  $PLANNER_ENDPOINT"
echo "Top-K:          $TOP_K"
echo

# Check endpoints are reachable
for endpoint in "$BASE_ENDPOINT" "$FT_ENDPOINT"; do
    if ! curl -sf "$endpoint/health" > /dev/null 2>&1; then
        # Try the v1 health endpoint
        if ! curl -sf "${endpoint%/v1}/health" > /dev/null 2>&1; then
            echo "ERROR: Cannot reach $endpoint — is the model running?"
            exit 1
        fi
    fi
done

echo "All endpoints reachable."
echo

# ─── Phase 1: Single-Step (base vs fine-tuned) ──────────────────────────────

echo "=== Phase 1: Single-Step Tool Selection (K=$TOP_K) ==="
echo

echo "--- Base model ---"
npx tsx tests/model-behavior/benchmark-lfm.ts \
    --endpoint "$BASE_ENDPOINT" \
    --top-k "$TOP_K" \
    --timeout 30000

echo
echo "--- Fine-tuned model ---"
npx tsx tests/model-behavior/benchmark-lfm.ts \
    --endpoint "$FT_ENDPOINT" \
    --top-k "$TOP_K" \
    --timeout 30000

# ─── Phase 2: Multi-Step Chains ─────────────────────────────────────────────

echo
echo "=== Phase 2: Multi-Step Chain Completion ==="
echo

echo "--- Base model ---"
npx tsx tests/model-behavior/benchmark-multi-step.ts \
    --endpoint "$BASE_ENDPOINT" \
    --top-k "$TOP_K" \
    --difficulty all \
    --timeout 60000

echo
echo "--- Fine-tuned model ---"
npx tsx tests/model-behavior/benchmark-multi-step.ts \
    --endpoint "$FT_ENDPOINT" \
    --top-k "$TOP_K" \
    --difficulty all \
    --timeout 60000

# ─── Phase 3: Orchestrator (if planner available) ───────────────────────────

if curl -sf "${PLANNER_ENDPOINT%/v1}/health" > /dev/null 2>&1; then
    echo
    echo "=== Phase 3: Orchestrator (24B planner + fine-tuned router) ==="
    echo

    npx tsx tests/model-behavior/benchmark-orchestrator.ts \
        --planner-endpoint "$PLANNER_ENDPOINT" \
        --router-endpoint "$FT_ENDPOINT" \
        --top-k "$TOP_K" \
        --difficulty all \
        --timeout 120000
else
    echo
    echo "Skipping Phase 3 (planner not running at $PLANNER_ENDPOINT)"
fi

# ─── Summary ────────────────────────────────────────────────────────────────

echo
echo "=== Benchmark Complete ==="
echo "Results saved to $RESULTS_DIR/"
echo
echo "Latest results:"
ls -lt "$RESULTS_DIR"/*.json 2>/dev/null | head -6
echo
echo "Compare with:"
echo "  jq '.accuracyPercent' $RESULTS_DIR/lfm-filtered-k15-*.json"
