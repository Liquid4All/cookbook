#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# LFM2-24B Demo Benchmark — Focused Tool Set
#
# Runs single-step, multi-step, and quality benchmarks against a reduced
# set of 9 servers / ~31 tools designed for the three demo use cases:
#
#   Demo 1: Security Steward — scan for leaked secrets, encrypt, audit trail
#   Demo 2: Contract Copilot — diff NDAs, generate PDF, draft email
#   Demo 3: Screenshot-to-Action — screenshot, OCR, clipboard
#
# Prerequisites:
#   - LFM2-24B-A2B running on llama-server (default: http://localhost:8080)
#   - Node.js 18+ with tsx available
#
# Usage:
#   ./scripts/benchmark-demo.sh
#   ./scripts/benchmark-demo.sh --endpoint http://localhost:8082
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_SERVERS="security,audit,document,ocr,data,email,system,clipboard,filesystem"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  LFM2-24B Demo Benchmark — Focused Tool Set"
echo "  Servers: ${DEMO_SERVERS}"
echo "  $(echo "${DEMO_SERVERS}" | tr ',' '\n' | wc -l | tr -d ' ') servers"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "▶ Phase 1: Single-Step Tool Selection (focused)"
echo "──────────────────────────────────────────────────────────────────────"
npx tsx tests/model-behavior/benchmark-lfm.ts \
  --servers "${DEMO_SERVERS}" --greedy "$@"

echo ""
echo "▶ Phase 2: Multi-Step Chain Completion (focused)"
echo "──────────────────────────────────────────────────────────────────────"
npx tsx tests/model-behavior/benchmark-multi-step.ts \
  --servers "${DEMO_SERVERS}" --greedy "$@"

echo ""
echo "▶ Phase 3: Quality Benchmark (full suite)"
echo "──────────────────────────────────────────────────────────────────────"
npx tsx tests/model-behavior/benchmark-quality.ts \
  --greedy "$@"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Done. Results in tests/model-behavior/.results/"
echo "═══════════════════════════════════════════════════════════════════════"
