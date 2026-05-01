#!/usr/bin/env bash
#
# Copy local Telco Triage GGUF model artifacts into the Xcode target.
#
# The cookbook keeps large GGUF files out of git. Put the files in either:
#
#   ./models/telco/
#
# or set TELCO_MODELS_DIR to another directory before running this script.
#
# Usage:
#
#   cd examples/telco-triage-ios
#   TELCO_MODELS_DIR=/path/to/telco-models ./bootstrap-models.sh
#   xcodegen generate
#
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC="${TELCO_MODELS_DIR:-$SCRIPT_DIR/models/telco}"
DST="$SCRIPT_DIR/VerizonSupportPOC/Resources/Models"

REQUIRED=(
  # CRITICAL: use the LFM2.5-350M Base GGUF. The LoRA adapters below were
  # trained against Base weights, not DPO/instruct weights.
  "lfm25-350m-base-Q4_K_M.gguf"

  # ADR-015 shared classifier adapter. The nine classifier head triplets are
  # small and committed under VerizonSupportPOC/Resources/.
  "telco-shared-clf-v1.gguf"

  # Generative adapters used for grounded answer and tool-argument paths.
  "telco-tool-selector-v3.gguf"
  "chat-mode-router-v2.gguf"
  "kb-extractor-v1.gguf"
)

OPTIONAL=(
  # Transitional classifier adapters. The app prefers telco-shared-clf-v1,
  # but these can be bundled for experiments with the older paired-head path.
  "chat-mode-clf-v1.gguf"
  "kb-extract-clf-v1.gguf"
  "tool-selector-clf-v1.gguf"
)

if [[ ! -d "$SRC" ]]; then
  echo "error: model directory not found: $SRC" >&2
  echo "" >&2
  echo "Create $SCRIPT_DIR/models/telco or set TELCO_MODELS_DIR." >&2
  echo "Required files:" >&2
  for name in "${REQUIRED[@]}"; do
    echo "  - $name" >&2
  done
  exit 1
fi

mkdir -p "$DST"

# Prune stale GGUFs from prior runs so a local demo build does not silently
# bundle retired adapters.
shopt -s nullglob
for existing in "$DST"/*.gguf; do
  base="$(basename "$existing")"
  keep=false
  for name in "${REQUIRED[@]}" "${OPTIONAL[@]}"; do
    if [[ "$base" == "$name" ]]; then
      keep=true
      break
    fi
  done
  if [[ "$keep" == false ]]; then
    rm -f "$existing"
    echo "pruned stale $base"
  fi
done
shopt -u nullglob

for name in "${REQUIRED[@]}"; do
  if [[ ! -f "$SRC/$name" ]]; then
    echo "error: missing required model artifact: $SRC/$name" >&2
    exit 1
  fi
  cp "$SRC/$name" "$DST/$name"
  size_mb="$(du -m "$DST/$name" | cut -f1)"
  echo "copied $name (${size_mb} MB)"
done

for name in "${OPTIONAL[@]}"; do
  if [[ -f "$SRC/$name" ]]; then
    cp "$SRC/$name" "$DST/$name"
    size_mb="$(du -m "$DST/$name" | cut -f1)"
    echo "copied optional $name (${size_mb} MB)"
  fi
done

echo ""
echo "done - $(find "$DST" -maxdepth 1 -name '*.gguf' | wc -l | tr -d ' ') GGUF(s) in $DST"
