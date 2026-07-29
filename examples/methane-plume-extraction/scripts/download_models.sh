#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_ID="${BASE_MODEL_ID:-LiquidAI/LFM2.5-VL-1.6B}"
FINE_MODEL_ID="${FINE_MODEL_ID:-felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo}"
BASE_DIR="${BASE_DIR:-models/base}"
FINE_DIR="${FINE_DIR:-models/fine}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install with:"
  echo "  curl -LsSf https://hf.co/cli/install.sh | bash -s"
  exit 1
fi

if [[ -z "${FINE_MODEL_ID}" ]]; then
  echo "Set FINE_MODEL_ID to the private fine-tuned model repo id."
  echo "Example:"
  echo "  export FINE_MODEL_ID=felipeliquid/LFM2.5-1.6B-VL-Extract-Plume-Demo"
  exit 2
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "You are not logged into Hugging Face. Run:"
  echo "  hf auth login"
  exit 3
fi

mkdir -p "${BASE_DIR}" "${FINE_DIR}"

echo "Downloading base model: ${BASE_MODEL_ID} -> ${BASE_DIR}"
hf download "${BASE_MODEL_ID}" --local-dir "${BASE_DIR}"

echo "Downloading fine-tuned model: ${FINE_MODEL_ID} -> ${FINE_DIR}"
hf download "${FINE_MODEL_ID}" --local-dir "${FINE_DIR}"

echo "Done."
