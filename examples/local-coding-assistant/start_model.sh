#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Get your token at https://huggingface.co/settings/tokens and run:"
  echo "  HF_TOKEN=hf_... ./start_model.sh"
  exit 1
fi

llama-server \
  -hf liquidai/LFM2-24B-A2B-Q4_0-GGUF \
  -hftoken "$HF_TOKEN" \
  --ctx-size 8192 \
  --port 8080
