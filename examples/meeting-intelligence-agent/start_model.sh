#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/LFM2-24B-A2B.gguf}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH"
  echo "Download with:"
  echo "  huggingface-cli download LiquidAI/LFM2-24B-A2B-GGUF --include 'LFM2-24B-A2B-Q4_K_M.gguf' --local-dir models/"
  exit 1
fi

llama-server \
  --model "$MODEL_PATH" \
  --ctx-size 32768 \
  --n-gpu-layers 0 \
  --host 0.0.0.0 \
  --port 8080
