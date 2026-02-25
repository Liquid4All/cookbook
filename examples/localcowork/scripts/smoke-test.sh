#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LocalCowork — Smoke Test Runner
# ─────────────────────────────────────────────────────────────────────────────
#
# PURPOSE: Discover and run all smoke tests across the project.
# Smoke tests are FAST (<30s total) regression checks that verify:
#   1. Contract compliance — tool signatures match the registry YAML
#   2. Server health — each implemented server responds to `initialize`
#   3. Per-tool smoke — basic round-trip per tool (*.smoke.test.ts / *_smoke_test.py)
#
# USAGE:
#   ./scripts/smoke-test.sh              # Run all smoke tests
#   ./scripts/smoke-test.sh --contract   # Contract tests only
#   ./scripts/smoke-test.sh --health     # Server health tests only
#   ./scripts/smoke-test.sh --tools      # Per-tool smoke tests only
#   ./scripts/smoke-test.sh --server filesystem  # Smoke tests for one server
#   ./scripts/smoke-test.sh --save       # Save results to tests/smoke/results/
#
# EXIT CODES:
#   0 — All tests passed
#   1 — One or more tests failed
#   2 — No tests found (nothing to run)
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SMOKE_DIR="$PROJECT_ROOT/tests/smoke"
RESULTS_DIR="$SMOKE_DIR/results"
REGISTRY="$PROJECT_ROOT/docs/mcp-tool-registry.yaml"
MCP_DIR="$PROJECT_ROOT/mcp-servers"
TIMESTAMP=$(date +"%Y-%m-%d-%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

# CLI flags
RUN_CONTRACT=true
RUN_HEALTH=true
RUN_TOOLS=true
FILTER_SERVER=""
SAVE_RESULTS=false

# ── Parse CLI Arguments ──────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --contract)
            RUN_HEALTH=false; RUN_TOOLS=false; shift ;;
        --health)
            RUN_CONTRACT=false; RUN_TOOLS=false; shift ;;
        --tools)
            RUN_CONTRACT=false; RUN_HEALTH=false; shift ;;
        --server)
            FILTER_SERVER="$2"; shift 2 ;;
        --save)
            SAVE_RESULTS=true; shift ;;
        --help|-h)
            head -26 "$0" | tail -18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helper Functions ─────────────────────────────────────────────────────

print_header() {
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}═══════════════════════════════════════════════════${NC}"
    echo ""
}

print_result() {
    local status="$1"
    local name="$2"
    local detail="${3:-}"
    ((TOTAL++))
    if [ "$status" = "PASS" ]; then
        ((PASSED++))
        echo -e "  ${GREEN}✅ PASS${NC}  $name"
    elif [ "$status" = "FAIL" ]; then
        ((FAILED++))
        echo -e "  ${RED}❌ FAIL${NC}  $name"
        if [ -n "$detail" ]; then
            echo -e "           ${RED}$detail${NC}"
        fi
    elif [ "$status" = "SKIP" ]; then
        ((SKIPPED++))
        echo -e "  ${YELLOW}⏭️  SKIP${NC}  $name  ${YELLOW}($detail)${NC}"
    fi
}

# List all implemented servers (those with at least an src/ directory and entry point)
find_implemented_servers() {
    local servers=()
    for server_dir in "$MCP_DIR"/*/; do
        local server_name
        server_name=$(basename "$server_dir")

        # Skip _shared directory
        [ "$server_name" = "_shared" ] && continue

        # Filter by server name if specified
        if [ -n "$FILTER_SERVER" ] && [ "$server_name" != "$FILTER_SERVER" ]; then
            continue
        fi

        # Check for an entry point (TS or Python)
        if [ -f "$server_dir/src/index.ts" ] || [ -f "$server_dir/src/__init__.py" ] || [ -f "$server_dir/src/main.py" ]; then
            servers+=("$server_name")
        fi
    done
    echo "${servers[@]}"
}

# Detect server language
detect_language() {
    local server_name="$1"
    local server_dir="$MCP_DIR/$server_name"
    if [ -f "$server_dir/package.json" ] || [ -f "$server_dir/tsconfig.json" ]; then
        echo "ts"
    elif [ -f "$server_dir/pyproject.toml" ] || [ -f "$server_dir/requirements.txt" ] || [ -f "$server_dir/setup.py" ]; then
        echo "py"
    else
        echo "unknown"
    fi
}

# Count tool files in a server
count_tool_files() {
    local server_name="$1"
    local server_dir="$MCP_DIR/$server_name/src/tools"
    if [ -d "$server_dir" ]; then
        find "$server_dir" -type f \( -name '*.ts' -o -name '*.py' \) ! -name '__init__*' ! -name 'index.*' | wc -l | tr -d ' '
    else
        echo "0"
    fi
}

# ── Phase 1: Contract Tests ─────────────────────────────────────────────

run_contract_tests() {
    print_header "Phase 1: Contract Compliance"
    echo -e "  ${CYAN}Validating tool signatures against docs/mcp-tool-registry.yaml${NC}"
    echo ""

    local servers
    servers=$(find_implemented_servers)

    if [ -z "$servers" ]; then
        print_result "SKIP" "Contract tests" "No implemented servers found"
        return
    fi

    # Check if registry exists
    if [ ! -f "$REGISTRY" ]; then
        print_result "FAIL" "Registry check" "docs/mcp-tool-registry.yaml not found"
        return
    fi

    # Run the contract validator if it exists
    if [ -f "$SMOKE_DIR/contract-validator.ts" ]; then
        local output
        if output=$(cd "$PROJECT_ROOT" && npx tsx "$SMOKE_DIR/contract-validator.ts" $servers 2>&1); then
            # Parse output — each line is "PASS|FAIL|SKIP server.tool detail"
            while IFS= read -r line; do
                if [[ "$line" =~ ^(PASS|FAIL|SKIP)[[:space:]]+(.*) ]]; then
                    local status="${BASH_REMATCH[1]}"
                    local rest="${BASH_REMATCH[2]}"
                    print_result "$status" "$rest"
                fi
            done <<< "$output"
        else
            print_result "FAIL" "Contract validator" "Script exited with error"
            echo "$output" | head -5 | while IFS= read -r line; do echo "           $line"; done
        fi
    else
        # Fallback: basic structural checks
        for server_name in $servers; do
            local tool_count
            tool_count=$(count_tool_files "$server_name")
            if [ "$tool_count" -gt 0 ]; then
                print_result "PASS" "$server_name — $tool_count tool file(s) present"
            else
                print_result "FAIL" "$server_name — no tool files in src/tools/"
            fi
        done
    fi
}

# ── Phase 2: Server Health ───────────────────────────────────────────────

run_health_tests() {
    print_header "Phase 2: Server Health"
    echo -e "  ${CYAN}Checking each server can start and respond to initialize${NC}"
    echo ""

    local servers
    servers=$(find_implemented_servers)

    if [ -z "$servers" ]; then
        print_result "SKIP" "Health tests" "No implemented servers found"
        return
    fi

    # Run the health checker if it exists
    if [ -f "$SMOKE_DIR/server-health.ts" ]; then
        local output
        if output=$(cd "$PROJECT_ROOT" && npx tsx "$SMOKE_DIR/server-health.ts" $servers 2>&1); then
            while IFS= read -r line; do
                if [[ "$line" =~ ^(PASS|FAIL|SKIP)[[:space:]]+(.*) ]]; then
                    local status="${BASH_REMATCH[1]}"
                    local rest="${BASH_REMATCH[2]}"
                    print_result "$status" "$rest"
                fi
            done <<< "$output"
        else
            print_result "FAIL" "Health checker" "Script exited with error"
        fi
    else
        # Fallback: check entry points exist
        for server_name in $servers; do
            local lang
            lang=$(detect_language "$server_name")
            if [ "$lang" = "ts" ] && [ -f "$MCP_DIR/$server_name/src/index.ts" ]; then
                print_result "PASS" "$server_name — entry point exists (index.ts)"
            elif [ "$lang" = "py" ] && { [ -f "$MCP_DIR/$server_name/src/__init__.py" ] || [ -f "$MCP_DIR/$server_name/src/main.py" ]; }; then
                print_result "PASS" "$server_name — entry point exists (__init__.py)"
            else
                print_result "FAIL" "$server_name — no entry point found"
            fi
        done
    fi
}

# ── Phase 3: Per-Tool Smoke Tests ───────────────────────────────────────

run_tool_smoke_tests() {
    print_header "Phase 3: Per-Tool Smoke Tests"
    echo -e "  ${CYAN}Running *.smoke.test.ts and *_smoke_test.py files${NC}"
    echo ""

    local ts_smoke_tests=()
    local py_smoke_tests=()
    local search_dir="$MCP_DIR"

    # If filtering by server, narrow the search
    if [ -n "$FILTER_SERVER" ]; then
        search_dir="$MCP_DIR/$FILTER_SERVER"
    fi

    # Discover TypeScript smoke tests
    while IFS= read -r -d '' file; do
        ts_smoke_tests+=("$file")
    done < <(find "$search_dir" -name '*.smoke.test.ts' -print0 2>/dev/null)

    # Discover Python smoke tests
    while IFS= read -r -d '' file; do
        py_smoke_tests+=("$file")
    done < <(find "$search_dir" -name '*_smoke_test.py' -print0 2>/dev/null)

    # Also check tests/smoke/ for project-level smoke tests
    while IFS= read -r -d '' file; do
        ts_smoke_tests+=("$file")
    done < <(find "$SMOKE_DIR" -name '*.smoke.test.ts' -print0 2>/dev/null)

    while IFS= read -r -d '' file; do
        py_smoke_tests+=("$file")
    done < <(find "$SMOKE_DIR" -name '*_smoke_test.py' -print0 2>/dev/null)

    local found_any=false

    # Run TypeScript smoke tests
    if [ ${#ts_smoke_tests[@]} -gt 0 ]; then
        found_any=true
        echo -e "  ${CYAN}TypeScript smoke tests: ${#ts_smoke_tests[@]} file(s)${NC}"
        for test_file in "${ts_smoke_tests[@]}"; do
            local relative
            relative=$(realpath --relative-to="$PROJECT_ROOT" "$test_file")
            if (cd "$PROJECT_ROOT" && npx vitest run "$relative" --reporter=dot 2>&1 | tail -1 | grep -q "passed"); then
                print_result "PASS" "$relative"
            else
                print_result "FAIL" "$relative" "vitest failed — run directly for details"
            fi
        done
    fi

    # Run Python smoke tests
    if [ ${#py_smoke_tests[@]} -gt 0 ]; then
        found_any=true
        echo -e "  ${CYAN}Python smoke tests: ${#py_smoke_tests[@]} file(s)${NC}"
        for test_file in "${py_smoke_tests[@]}"; do
            local relative
            relative=$(realpath --relative-to="$PROJECT_ROOT" "$test_file")
            if (cd "$PROJECT_ROOT" && python -m pytest "$relative" -x -q 2>&1 | tail -1 | grep -q "passed"); then
                print_result "PASS" "$relative"
            else
                print_result "FAIL" "$relative" "pytest failed — run directly for details"
            fi
        done
    fi

    if [ "$found_any" = false ]; then
        print_result "SKIP" "Per-tool smoke tests" "No *.smoke.test.ts or *_smoke_test.py files found"
    fi
}

# ── Results Summary ──────────────────────────────────────────────────────

print_summary() {
    print_header "Smoke Test Summary"

    echo -e "  Total:   ${BOLD}$TOTAL${NC}"
    echo -e "  Passed:  ${GREEN}$PASSED${NC}"
    echo -e "  Failed:  ${RED}$FAILED${NC}"
    echo -e "  Skipped: ${YELLOW}$SKIPPED${NC}"
    echo ""

    if [ "$FAILED" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
        echo -e "  ${GREEN}${BOLD}All smoke tests passed!${NC}"
    elif [ "$TOTAL" -eq 0 ]; then
        echo -e "  ${YELLOW}${BOLD}No tests were run.${NC}"
    else
        echo -e "  ${RED}${BOLD}$FAILED test(s) failed. Fix before pushing.${NC}"
    fi
    echo ""
}

save_results() {
    mkdir -p "$RESULTS_DIR"
    local result_file="$RESULTS_DIR/$TIMESTAMP.json"

    cat > "$result_file" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "git_sha": "$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || echo 'unknown')",
  "total": $TOTAL,
  "passed": $PASSED,
  "failed": $FAILED,
  "skipped": $SKIPPED,
  "success": $([ "$FAILED" -eq 0 ] && echo "true" || echo "false")
}
EOF
    echo -e "  Results saved to: ${CYAN}$result_file${NC}"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────────

print_header "LocalCowork Smoke Test Suite"
echo -e "  ${CYAN}Registry:${NC} $REGISTRY"
echo -e "  ${CYAN}Servers:${NC}  $MCP_DIR"
if [ -n "$FILTER_SERVER" ]; then
    echo -e "  ${CYAN}Filter:${NC}   $FILTER_SERVER only"
fi

[ "$RUN_CONTRACT" = true ] && run_contract_tests
[ "$RUN_HEALTH" = true ] && run_health_tests
[ "$RUN_TOOLS" = true ] && run_tool_smoke_tests

print_summary

[ "$SAVE_RESULTS" = true ] && save_results

# Exit with appropriate code
if [ "$FAILED" -gt 0 ]; then
    exit 1
elif [ "$TOTAL" -eq 0 ]; then
    exit 2
else
    exit 0
fi
