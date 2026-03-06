#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LocalCowork — MCP Server Validation
# Validates all MCP servers against the PRD tool registry.
# Run: ./scripts/validate-mcp-servers.sh [server-name]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REGISTRY="docs/mcp-tool-registry.yaml"
MCP_DIR="mcp-servers"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "═══════════════════════════════════════════════════"
echo "  LocalCowork — MCP Server Validation"
echo "═══════════════════════════════════════════════════"
echo ""

if [ ! -f "$REGISTRY" ]; then
	echo -e "${RED}❌ Registry not found: $REGISTRY${NC}"
	echo "   Run from project root."
	exit 1
fi

# TypeScript servers
TS_SERVERS="filesystem calendar email task data audit clipboard system"
# Python servers
PY_SERVERS="document ocr knowledge meeting security"

ALL_SERVERS="$TS_SERVERS $PY_SERVERS"

# If a specific server was requested
if [ $# -ge 1 ]; then
	ALL_SERVERS="$1"
fi

total_tools=0
total_tests=0
servers_complete=0
servers_total=0

for server in $ALL_SERVERS; do
	servers_total=$((servers_total + 1))
	server_dir="$MCP_DIR/$server"
	echo -e "${BLUE}── $server ──${NC}"

	# Check if server directory exists
	if [ ! -d "$server_dir" ]; then
		echo -e "  ${RED}❌ Directory not found: $server_dir${NC}"
		continue
	fi

	# Check for entry point
	if [ -f "$server_dir/src/index.ts" ]; then
		echo -e "  ${GREEN}✅ Entry point: src/index.ts${NC}"
		lang="ts"
	elif [ -f "$server_dir/src/__init__.py" ]; then
		echo -e "  ${GREEN}✅ Entry point: src/__init__.py${NC}"
		lang="py"
	else
		echo -e "  ${YELLOW}⚠️  No entry point found (src/index.ts or src/__init__.py)${NC}"
		lang="unknown"
	fi

	# Count tool files
	if [ "$lang" = "ts" ]; then
		tool_files=$(find "$server_dir/src/tools" -name "*.ts" 2>/dev/null | wc -l | tr -d ' ')
		test_files=$(find "$server_dir/tests" -name "*.test.ts" 2>/dev/null | wc -l | tr -d ' ')
	elif [ "$lang" = "py" ]; then
		tool_files=$(find "$server_dir/src/tools" -name "*.py" -not -name "__init__.py" 2>/dev/null | wc -l | tr -d ' ')
		test_files=$(find "$server_dir/tests" -name "test_*.py" 2>/dev/null | wc -l | tr -d ' ')
	else
		tool_files=0
		test_files=0
	fi

	echo "  Tools implemented: $tool_files"
	echo "  Test files: $test_files"

	total_tools=$((total_tools + tool_files))
	total_tests=$((total_tests + test_files))

	# Run type check
	if [ "$lang" = "ts" ] && [ -f "$server_dir/tsconfig.json" ]; then
		if npx tsc --noEmit --project "$server_dir/tsconfig.json" 2>/dev/null; then
			echo -e "  ${GREEN}✅ Type check passes${NC}"
		else
			echo -e "  ${RED}❌ Type check failed${NC}"
		fi
	elif [ "$lang" = "py" ]; then
		if mypy --strict "$server_dir/src/" 2>/dev/null; then
			echo -e "  ${GREEN}✅ mypy passes${NC}"
		else
			echo -e "  ${YELLOW}⚠️  mypy check skipped or failed${NC}"
		fi
	fi

	# Run tests
	if [ "$lang" = "ts" ] && [ "$test_files" -gt 0 ]; then
		if cd "$server_dir" && npx vitest run --reporter=dot 2>/dev/null; then
			echo -e "  ${GREEN}✅ Tests pass${NC}"
			servers_complete=$((servers_complete + 1))
		else
			echo -e "  ${RED}❌ Tests failed${NC}"
		fi
		cd - >/dev/null
	elif [ "$lang" = "py" ] && [ "$test_files" -gt 0 ]; then
		if cd "$server_dir" && pytest tests/ --quiet 2>/dev/null; then
			echo -e "  ${GREEN}✅ Tests pass${NC}"
			servers_complete=$((servers_complete + 1))
		else
			echo -e "  ${RED}❌ Tests failed${NC}"
		fi
		cd - >/dev/null
	else
		echo -e "  ${YELLOW}⚠️  No tests to run${NC}"
	fi

	# Check for TODO markers
	todo_count=$(grep -r "TODO" "$server_dir/src/" 2>/dev/null | wc -l | tr -d ' ')
	if [ "$todo_count" -gt 0 ]; then
		echo -e "  ${YELLOW}🔨 $todo_count TODO markers remaining${NC}"
	fi

	echo ""
done

# ── Summary ────────────────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════"
echo "  Summary"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Servers:    $servers_complete / $servers_total complete"
echo "  Tool files: $total_tools total"
echo "  Test files: $total_tests total"
echo ""

if [ "$servers_complete" -eq "$servers_total" ]; then
	echo -e "${GREEN}✅ All servers passing!${NC}"
else
	remaining=$((servers_total - servers_complete))
	echo -e "${YELLOW}⚠️  $remaining servers need work${NC}"
fi
