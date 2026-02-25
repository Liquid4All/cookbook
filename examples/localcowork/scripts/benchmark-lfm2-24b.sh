#!/usr/bin/env bash
# benchmark-lfm2-24b.sh — Benchmark LFM2-24B-A2B on both architectures
#
# Runs the full benchmark suite against LFM2-24B-A2B-Preview:
#   Phase 1: Single-model agent loop (main branch architecture)
#   Phase 2: Dual-model orchestrator (feat/dual-model-orchestrator architecture)
#   Phase 3: Comparison report generation
#
# Prerequisites:
#   - GGUF model file downloaded
#   - llama.cpp installed (llama-server binary in PATH)
#   - Node.js 18+ (for benchmark runners)
#   - npm install completed in project root
#
# Usage:
#   ./scripts/benchmark-lfm2-24b.sh --path ~/Projects/_models/LFM2-24B-A2B-Preview-Q4_K_M.gguf
#   ./scripts/benchmark-lfm2-24b.sh --path <gguf> --phase single    # Phase 1 only
#   ./scripts/benchmark-lfm2-24b.sh --path <gguf> --phase orchestrator  # Phase 2 only
#   ./scripts/benchmark-lfm2-24b.sh --phase report                  # Phase 3 only (uses existing results)

set -euo pipefail

# --- Configuration ---
PLANNER_PORT=8080
ROUTER_PORT=8082
CTX_SIZE=32768
RESULTS_DIR="tests/model-behavior/.results"
ROUTER_MODEL_PATH="${LOCALCOWORK_MODELS_DIR:-~/Projects/_models}/LFM2-1.2B-Tool-F16.gguf"
TIMESTAMP=$(date +%s)

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# --- Parse arguments ---
MODEL_PATH=""
PHASE="all"
GPU_LAYERS=99

while [[ $# -gt 0 ]]; do
    case $1 in
        --path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --gpu-layers)
            GPU_LAYERS="$2"
            shift 2
            ;;
        --router-path)
            ROUTER_MODEL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --path <gguf-path> [--phase single|orchestrator|report|all] [--gpu-layers N]"
            echo ""
            echo "Options:"
            echo "  --path          Path to LFM2-24B-A2B GGUF file (required for single/orchestrator phases)"
            echo "  --phase         Which phase to run: single, orchestrator, report, or all (default: all)"
            echo "  --gpu-layers    Number of GPU layers to offload (default: 99 = all)"
            echo "  --router-path   Path to LFM2-1.2B-Tool GGUF for orchestrator phase"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# --- Helpers ---
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

wait_for_server() {
    local url="$1"
    local timeout="$2"
    local elapsed=0
    while ! curl -sf "$url" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$timeout" ]; then
            log_err "Server at $url did not start within ${timeout}s"
            return 1
        fi
    done
    log_ok "Server at $url is ready (${elapsed}s)"
}

stop_server() {
    local port="$1"
    local pid
    pid=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        log_info "Stopped server on port $port (PID: $pid)"
    fi
}

start_llama_server() {
    local model="$1"
    local port="$2"
    local extra_args="${3:-}"

    log_info "Starting llama-server on port $port..."
    log_info "  Model: $model"
    log_info "  Context: $CTX_SIZE tokens"

    # shellcheck disable=SC2086
    llama-server \
        --model "$model" \
        --port "$port" \
        --ctx-size "$CTX_SIZE" \
        --n-gpu-layers "$GPU_LAYERS" \
        --flash-attn on \
        $extra_args \
        > "/tmp/llama-server-${port}.log" 2>&1 &

    wait_for_server "http://localhost:${port}/health" 120
}

mkdir -p "$RESULTS_DIR"

# ============================================================
# PHASE 1: Single-Model Agent Loop (main branch architecture)
# ============================================================
run_phase_single() {
    if [ -z "$MODEL_PATH" ]; then
        log_err "Model path required. Use --path <gguf-path>"
        exit 1
    fi

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PHASE 1: Single-Model Agent Loop (all 67 tools)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # Stop any existing servers on our ports
    stop_server "$PLANNER_PORT"

    # Start LFM2-24B-A2B as the single model
    start_llama_server "$MODEL_PATH" "$PLANNER_PORT"

    local endpoint="http://localhost:${PLANNER_PORT}"

    # 1a. Single-step tool selection (100 tests, NO pre-filter — all 67 tools)
    log_info "Running 100 single-step tool-selection tests (no pre-filter)..."
    npx tsx tests/model-behavior/benchmark-lfm.ts \
        --endpoint "$endpoint" \
        --timeout 60000 \
        2>&1 | tee "${RESULTS_DIR}/lfm2-24b-single-step-${TIMESTAMP}.log"

    log_ok "Single-step results saved to ${RESULTS_DIR}/"

    # 1b. Single-step with K=15 pre-filter (for comparison)
    log_info "Running 100 single-step tool-selection tests (K=15 pre-filter)..."
    npx tsx tests/model-behavior/benchmark-lfm.ts \
        --endpoint "$endpoint" \
        --timeout 60000 \
        --top-k 15 \
        2>&1 | tee "${RESULTS_DIR}/lfm2-24b-single-step-k15-${TIMESTAMP}.log"

    log_ok "Single-step K=15 results saved to ${RESULTS_DIR}/"

    # 1c. Multi-step chains (50 tests, all 67 tools)
    log_info "Running 50 multi-step chain tests (no pre-filter)..."
    npx tsx tests/model-behavior/benchmark-multi-step.ts \
        --endpoint "$endpoint" \
        --timeout 120000 \
        2>&1 | tee "${RESULTS_DIR}/lfm2-24b-multi-step-${TIMESTAMP}.log"

    log_ok "Multi-step results saved to ${RESULTS_DIR}/"

    # Stop the server
    stop_server "$PLANNER_PORT"

    echo ""
    log_ok "Phase 1 complete."
}

# ============================================================
# PHASE 2: Dual-Model Orchestrator
# ============================================================
run_phase_orchestrator() {
    if [ -z "$MODEL_PATH" ]; then
        log_err "Model path required. Use --path <gguf-path>"
        exit 1
    fi

    if [ ! -f "$ROUTER_MODEL_PATH" ]; then
        log_err "Router model not found at: $ROUTER_MODEL_PATH"
        log_err "Set --router-path or LOCALCOWORK_MODELS_DIR env var"
        exit 1
    fi

    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PHASE 2: Dual-Model Orchestrator${NC}"
    echo -e "${BLUE}  Planner: LFM2-24B-A2B (port ${PLANNER_PORT})${NC}"
    echo -e "${BLUE}  Router:  LFM2-1.2B-Tool (port ${ROUTER_PORT})${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # Stop any existing servers
    stop_server "$PLANNER_PORT"
    stop_server "$ROUTER_PORT"

    # Start planner (LFM2-24B-A2B)
    start_llama_server "$MODEL_PATH" "$PLANNER_PORT"

    # Start router (LFM2-1.2B-Tool) with embeddings support
    start_llama_server "$ROUTER_MODEL_PATH" "$ROUTER_PORT" "--embeddings"

    # Run orchestrator benchmark (50 chains)
    log_info "Running 50 orchestrator chain tests..."
    npx tsx tests/model-behavior/benchmark-orchestrator.ts \
        --planner-endpoint "http://localhost:${PLANNER_PORT}" \
        --router-endpoint "http://localhost:${ROUTER_PORT}" \
        --top-k 15 \
        --timeout 180000 \
        2>&1 | tee "${RESULTS_DIR}/lfm2-24b-orchestrator-${TIMESTAMP}.log"

    log_ok "Orchestrator results saved to ${RESULTS_DIR}/"

    # Stop both servers
    stop_server "$PLANNER_PORT"
    stop_server "$ROUTER_PORT"

    echo ""
    log_ok "Phase 2 complete."
}

# ============================================================
# PHASE 3: Comparison Report
# ============================================================
run_phase_report() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PHASE 3: Comparison Report${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    log_info "Generating comparison report..."

    # Find the latest results for each benchmark type
    local latest_single latest_k15 latest_multi latest_orch

    latest_single=$(ls -t "${RESULTS_DIR}"/lfm2-24b-single-step-[0-9]*.log 2>/dev/null | head -1 || true)
    latest_k15=$(ls -t "${RESULTS_DIR}"/lfm2-24b-single-step-k15-[0-9]*.log 2>/dev/null | head -1 || true)
    latest_multi=$(ls -t "${RESULTS_DIR}"/lfm2-24b-multi-step-[0-9]*.log 2>/dev/null | head -1 || true)
    latest_orch=$(ls -t "${RESULTS_DIR}"/lfm2-24b-orchestrator-[0-9]*.log 2>/dev/null | head -1 || true)

    echo ""
    echo "=== LFM2-24B-A2B Benchmark Summary ==="
    echo ""
    echo "Results files:"
    [ -n "$latest_single" ] && echo "  Single-step (all tools): $latest_single" || echo "  Single-step (all tools): NOT RUN"
    [ -n "$latest_k15" ]    && echo "  Single-step (K=15):      $latest_k15"    || echo "  Single-step (K=15):      NOT RUN"
    [ -n "$latest_multi" ]  && echo "  Multi-step chains:       $latest_multi"  || echo "  Multi-step chains:       NOT RUN"
    [ -n "$latest_orch" ]   && echo "  Orchestrator:            $latest_orch"   || echo "  Orchestrator:            NOT RUN"
    echo ""
    echo "=== Baselines for Comparison ==="
    echo ""
    echo "| Model                 | Single (67 tools) | Single (K=15) | Multi-step | Orchestrator |"
    echo "|-----------------------|-------------------|---------------|------------|--------------|"
    echo "| GPT-OSS-20B           | ~36%              | n/a           | ~0% (FM-3) | n/a          |"
    echo "| Qwen3-30B-A3B (MoE)  | ~36%              | n/a           | ~0% (fix.) | n/a          |"
    echo "| LFM2-1.2B-Tool        | 36%               | 78%           | 8%         | proj. 50-60% |"
    echo "| LFM2-24B-A2B          | TBD               | TBD           | TBD        | TBD          |"
    echo ""
    echo "Fill in TBD values from result JSON files, then update:"
    echo "  docs/model-analysis/lfm2-24b-a2b-benchmark.md"
    echo ""

    log_ok "Phase 3 complete. Update the benchmark doc with results."
}

# ============================================================
# Main
# ============================================================
case "$PHASE" in
    single)
        run_phase_single
        ;;
    orchestrator)
        run_phase_orchestrator
        ;;
    report)
        run_phase_report
        ;;
    all)
        run_phase_single
        run_phase_orchestrator
        run_phase_report
        ;;
    *)
        log_err "Unknown phase: $PHASE. Use: single, orchestrator, report, or all"
        exit 1
        ;;
esac

echo ""
log_ok "All requested phases complete. Results in ${RESULTS_DIR}/"
