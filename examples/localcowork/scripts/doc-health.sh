#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LocalCowork — Doc Health Checker
# ─────────────────────────────────────────────────────────────────────────────
#
# Automated checks for documentation staleness, broken cross-references,
# and drift between code and docs.
#
# Usage:
#   ./scripts/doc-health.sh              # Run all checks
#   ./scripts/doc-health.sh --refs       # Cross-reference integrity only
#   ./scripts/doc-health.sh --staleness  # Staleness detection only
#   ./scripts/doc-health.sh --drift      # Code-doc drift only
#   ./scripts/doc-health.sh --fix        # Show suggested fix commands
#
# Exit codes:
#   0 = all checks pass
#   1 = issues found
#   2 = script error (missing tools, bad args)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

ISSUES=0
WARNINGS=0
SHOW_FIX=false

# Doc files to scan
DOC_FILES=(
    "README.md"
    "CLAUDE.md"
    "PROGRESS.yaml"
    "docs/PRD.md"
    "docs/contributing.md"
    "docs/mcp-tool-registry.yaml"
    "docs/agent-loop-sequence-diagram.md"
)

# Add pattern docs
for f in docs/patterns/*.md; do
    [ -f "$f" ] && DOC_FILES+=("$f")
done

# Add ADRs
for f in docs/architecture-decisions/*.md; do
    [ -f "$f" ] && DOC_FILES+=("$f")
done

# Add skill files
for f in .claude/skills/*/SKILL.md; do
    [ -f "$f" ] && DOC_FILES+=("$f")
done

# Add command files
for f in .claude/commands/*.md; do
    [ -f "$f" ] && DOC_FILES+=("$f")
done

# ── Parse args ──────────────────────────────────────────────────────────────

RUN_REFS=true
RUN_STALENESS=true
RUN_DRIFT=true

if [ $# -gt 0 ]; then
    RUN_REFS=false
    RUN_STALENESS=false
    RUN_DRIFT=false
    for arg in "$@"; do
        case "$arg" in
            --refs)       RUN_REFS=true ;;
            --staleness)  RUN_STALENESS=true ;;
            --drift)      RUN_DRIFT=true ;;
            --fix)        SHOW_FIX=true; RUN_REFS=true; RUN_STALENESS=true; RUN_DRIFT=true ;;
            --help|-h)
                echo "Usage: $0 [--refs] [--staleness] [--drift] [--fix]"
                exit 0
                ;;
            *)
                echo "Unknown argument: $arg"
                exit 2
                ;;
        esac
    done
fi

# ── Helpers ─────────────────────────────────────────────────────────────────

issue() {
    echo -e "  ${RED}✗${NC} $1"
    ISSUES=$((ISSUES + 1))
}

warn() {
    echo -e "  ${YELLOW}△${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

fix() {
    if [ "$SHOW_FIX" = true ]; then
        echo -e "    ${DIM}fix: $1${NC}"
    fi
}

# ── Check 1: Cross-Reference Integrity ──────────────────────────────────────

check_cross_references() {
    echo ""
    echo -e "${CYAN}Cross-Reference Integrity${NC}"
    echo -e "${CYAN}─────────────────────────${NC}"

    local ref_issues=0

    # Extract file paths referenced in backticks across all docs
    for doc in "${DOC_FILES[@]}"; do
        [ -f "$doc" ] || continue

        # Find backtick-quoted paths that look like project files
        # Matches: `docs/something.md`, `mcp-servers/filesystem/src/tools/list_dir.ts`, etc.
        while IFS= read -r ref; do
            # Skip patterns with wildcards, placeholders, or template vars
            [[ "$ref" == *"*"* ]] && continue
            [[ "$ref" == *"<"* ]] && continue
            [[ "$ref" == *"{"* ]] && continue
            [[ "$ref" == *"NNN"* ]] && continue
            [[ "$ref" == *"..."* ]] && continue
            # Skip command-like references
            [[ "$ref" == *"npm "* ]] && continue
            [[ "$ref" == *"cargo "* ]] && continue
            [[ "$ref" == *"git "* ]] && continue
            # Skip references that are clearly code, not paths
            [[ "$ref" == *"("* ]] && continue
            [[ "$ref" == *"="* ]] && continue
            [[ "$ref" == *":"* && "$ref" != *"/"* ]] && continue

            # Check if the referenced path exists (relative to project root)
            if [ ! -e "$ref" ] && [ ! -d "$ref" ]; then
                # Only flag if it looks like a real path (has a slash or a file extension)
                if [[ "$ref" == *"/"* ]] || [[ "$ref" == *".md" ]] || [[ "$ref" == *".yaml" ]] || [[ "$ref" == *".ts" ]] || [[ "$ref" == *".py" ]] || [[ "$ref" == *".rs" ]] || [[ "$ref" == *".sh" ]]; then
                    issue "Broken reference in ${doc}: \`${ref}\` does not exist"
                    fix "grep -n '${ref}' ${doc}"
                    ref_issues=$((ref_issues + 1))
                fi
            fi
        done < <(grep -oP '`([^`]+)`' "$doc" 2>/dev/null | sed 's/`//g' | sort -u)
    done

    if [ "$ref_issues" -eq 0 ]; then
        pass "All cross-references resolve"
    fi
}

# ── Check 2: Staleness Detection ────────────────────────────────────────────

check_staleness() {
    echo ""
    echo -e "${CYAN}Staleness Detection${NC}"
    echo -e "${CYAN}───────────────────${NC}"

    # Check if git is available
    if ! git rev-parse --is-inside-work-tree &>/dev/null; then
        warn "Not a git repo — skipping staleness detection"
        return
    fi

    local stale_count=0

    # For each doc, find the code directories it describes and check if
    # code changed more recently than the doc
    declare -A DOC_CODE_MAP
    DOC_CODE_MAP=(
        ["docs/patterns/mcp-server-pattern.md"]="mcp-servers/_shared"
        ["docs/patterns/human-in-the-loop.md"]="src-tauri/src/agent_core"
        ["docs/patterns/context-window-management.md"]="src-tauri/src/agent_core"
        [".claude/skills/mcp-server-dev/SKILL.md"]="mcp-servers"
        [".claude/skills/tauri-dev/SKILL.md"]="src-tauri/src"
        [".claude/skills/tool-chain-test/SKILL.md"]="tests/integration"
        ["docs/contributing.md"]="scripts"
    )

    for doc in "${!DOC_CODE_MAP[@]}"; do
        [ -f "$doc" ] || continue
        local code_dir="${DOC_CODE_MAP[$doc]}"
        [ -d "$code_dir" ] || continue

        # Get last commit date for the doc and the code directory
        local doc_date code_date
        doc_date=$(git log -1 --format="%at" -- "$doc" 2>/dev/null || echo "0")
        code_date=$(git log -1 --format="%at" -- "$code_dir" 2>/dev/null || echo "0")

        if [ "$code_date" -gt "$doc_date" ] && [ "$doc_date" != "0" ] && [ "$code_date" != "0" ]; then
            local doc_human code_human
            doc_human=$(git log -1 --format="%ar" -- "$doc" 2>/dev/null || echo "unknown")
            code_human=$(git log -1 --format="%ar" -- "$code_dir" 2>/dev/null || echo "unknown")
            warn "${doc} may be stale — last updated ${doc_human}, but ${code_dir}/ changed ${code_human}"
            fix "Review ${doc} and update to match current ${code_dir}/ state"
            stale_count=$((stale_count + 1))
        fi
    done

    # Check registry vs tool implementations
    if [ -f "docs/mcp-tool-registry.yaml" ]; then
        local registry_date
        registry_date=$(git log -1 --format="%at" -- "docs/mcp-tool-registry.yaml" 2>/dev/null || echo "0")

        for server_dir in mcp-servers/*/src/tools; do
            [ -d "$server_dir" ] || continue
            local tools_date
            tools_date=$(git log -1 --format="%at" -- "$server_dir" 2>/dev/null || echo "0")

            if [ "$tools_date" -gt "$registry_date" ] && [ "$registry_date" != "0" ] && [ "$tools_date" != "0" ]; then
                local server_name
                server_name=$(echo "$server_dir" | cut -d'/' -f2)
                warn "Tool registry may be stale for ${server_name} — tools changed after registry was last updated"
                fix "Compare mcp-servers/${server_name}/src/tools/ against docs/mcp-tool-registry.yaml"
                stale_count=$((stale_count + 1))
            fi
        done
    fi

    if [ "$stale_count" -eq 0 ]; then
        pass "No staleness detected"
    fi
}

# ── Check 3: Code-Doc Drift ─────────────────────────────────────────────────

check_drift() {
    echo ""
    echo -e "${CYAN}Code-Doc Drift${NC}"
    echo -e "${CYAN}──────────────${NC}"

    local drift_count=0

    # Check: tools in code but not in registry
    if [ -f "docs/mcp-tool-registry.yaml" ]; then
        for server_dir in mcp-servers/*/src/tools; do
            [ -d "$server_dir" ] || continue
            local server_name
            server_name=$(echo "$server_dir" | cut -d'/' -f2)

            for tool_file in "$server_dir"/*.ts "$server_dir"/*.py; do
                [ -f "$tool_file" ] || continue
                local tool_name
                tool_name=$(basename "$tool_file" | sed 's/\.\(ts\|py\)$//')

                # Skip index/init files
                [[ "$tool_name" == "index" ]] && continue
                [[ "$tool_name" == "__init__" ]] && continue
                [[ "$tool_name" == "mod" ]] && continue

                # Check if tool name appears in registry
                if ! grep -q "$tool_name" docs/mcp-tool-registry.yaml 2>/dev/null; then
                    issue "Tool ${server_name}.${tool_name} exists in code but not in docs/mcp-tool-registry.yaml"
                    fix "Add ${server_name}.${tool_name} to docs/mcp-tool-registry.yaml"
                    drift_count=$((drift_count + 1))
                fi
            done
        done
    fi

    # Check: tools in code without test files
    for server_dir in mcp-servers/*/src/tools; do
        [ -d "$server_dir" ] || continue
        local server_name test_dir
        server_name=$(echo "$server_dir" | cut -d'/' -f2)
        test_dir="mcp-servers/${server_name}/tests"

        for tool_file in "$server_dir"/*.ts "$server_dir"/*.py; do
            [ -f "$tool_file" ] || continue
            local tool_name
            tool_name=$(basename "$tool_file" | sed 's/\.\(ts\|py\)$//')

            [[ "$tool_name" == "index" ]] && continue
            [[ "$tool_name" == "__init__" ]] && continue
            [[ "$tool_name" == "mod" ]] && continue

            # Check for any test file containing the tool name
            if [ -d "$test_dir" ]; then
                if ! find "$test_dir" -name "*${tool_name}*" -type f 2>/dev/null | grep -q .; then
                    warn "Tool ${server_name}.${tool_name} has no test file in ${test_dir}/"
                    fix "Create ${test_dir}/${tool_name}.test.ts (or test_${tool_name}.py)"
                    drift_count=$((drift_count + 1))
                fi
            fi
        done
    done

    # Check: CLAUDE.md Key Paths lists servers that don't exist, or misses ones that do
    if [ -f "CLAUDE.md" ]; then
        for server_dir in mcp-servers/*/; do
            [ -d "$server_dir" ] || continue
            local server_name
            server_name=$(basename "$server_dir")
            [[ "$server_name" == "_shared" ]] && continue

            if ! grep -q "$server_name" CLAUDE.md 2>/dev/null; then
                warn "Server ${server_name} exists but is not mentioned in CLAUDE.md Key Paths"
                fix "Add mcp-servers/${server_name}/ to CLAUDE.md Key Paths section"
                drift_count=$((drift_count + 1))
            fi
        done
    fi

    # Check: ADR count matches what CLAUDE.md Breaking Changes implies
    local adr_count=0
    if [ -d "docs/architecture-decisions" ]; then
        adr_count=$(find docs/architecture-decisions -name "*.md" -type f 2>/dev/null | wc -l | tr -d ' ')
    fi
    pass "Found ${adr_count} ADR(s) in docs/architecture-decisions/"

    if [ "$drift_count" -eq 0 ]; then
        pass "No code-doc drift detected"
    fi
}

# ── Main ────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Doc Health Check"
echo "═══════════════════════════════════════════════════"

[ "$RUN_REFS" = true ] && check_cross_references
[ "$RUN_STALENESS" = true ] && check_staleness
[ "$RUN_DRIFT" = true ] && check_drift

echo ""
echo "─────────────────────────────────────────────────"

if [ "$ISSUES" -gt 0 ]; then
    echo -e "  ${RED}${ISSUES} issue(s)${NC}, ${YELLOW}${WARNINGS} warning(s)${NC}"
    if [ "$SHOW_FIX" = false ]; then
        echo -e "  ${DIM}Run with --fix for suggested fix commands${NC}"
    fi
    echo ""
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "  ${GREEN}0 issues${NC}, ${YELLOW}${WARNINGS} warning(s)${NC}"
    echo ""
    exit 0
else
    echo -e "  ${GREEN}All checks passed${NC}"
    echo ""
    exit 0
fi
