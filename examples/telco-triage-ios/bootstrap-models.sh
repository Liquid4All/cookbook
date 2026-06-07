#!/usr/bin/env bash
#
# Copy the current Liquid Telco Triage runtime pack into the iOS bundle.
#
# Canonical source:
#   hf download LiquidAI/TelcoTriage --local-dir models/telco
#
# Usage:
#   examples/telco-triage-ios/bootstrap-models.sh
#   TELCO_MODELS_DIR=/path/to/TelcoTriage examples/telco-triage-ios/bootstrap-models.sh
#
# The GGUFs are gitignored, but Xcode needs them present before a local build.
# Classifier heads and JSON resources are tracked in the app, and this script
# refreshes them from the HF pack when newer copies are available.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC="${TELCO_MODELS_DIR:-$SCRIPT_DIR/models/telco}"
DST_MODELS="$SCRIPT_DIR/TelcoTriage/Resources/Models"
DST_RESOURCES="$SCRIPT_DIR/TelcoTriage/Resources"

REQUIRED_MODELS=(
  "lfm25-350m-base-Q4_K_M.gguf"
  "telco-shared-clf-v1.gguf"
  "telco-turn-relation-v4.gguf"
  "telco-tool-selector-v3.gguf"
  "telco-dialogue-repair-v4.gguf"
)

REQUIRED_JSON=(
  "rag-units-v1.json"
  "page-link-table-v1.json"
  "telco_shared_clf_schema.json"
)

TELCO_HEADS=(
  "telco-support-intent"
  "telco-issue-complexity"
  "telco-routing-lane"
  "telco-cloud-requirements"
  "telco-required-tool"
  "telco-customer-escalation-risk"
  "telco-pii-risk"
  "telco-transcript-quality"
  "telco-slot-completeness"
)

TURN_RELATION_HEADS=(
  "telco-turn-relation"
)

OPTIONAL_TELCO_HEADS=(
  "telco-topic-scope"
)

if [[ ! -d "$SRC" ]]; then
  echo "error: source Telco Triage pack not found at $SRC" >&2
  echo "hint: run:" >&2
  echo "        hf download LiquidAI/TelcoTriage --local-dir $SRC" >&2
  echo "      or set TELCO_MODELS_DIR=/path/to/downloaded/TelcoTriage" >&2
  exit 1
fi

mkdir -p "$DST_MODELS" "$DST_RESOURCES"

copy_required() {
  local from="$1"
  local to="$2"
  local label="$3"
  if [[ ! -f "$from" ]]; then
    echo "error: missing $label at $from" >&2
    exit 1
  fi
  cp "$from" "$to"
  echo "copied $label"
}

copy_classifier_triplet() {
  local head="$1"
  copy_required \
    "$SRC/${head}_classifier_weights.bin" \
    "$DST_RESOURCES/${head}_classifier_weights.bin" \
    "${head}_classifier_weights.bin"
  copy_required \
    "$SRC/${head}_classifier_bias.bin" \
    "$DST_RESOURCES/${head}_classifier_bias.bin" \
    "${head}_classifier_bias.bin"
  copy_required \
    "$SRC/${head}_classifier_meta.json" \
    "$DST_RESOURCES/${head}_classifier_meta.json" \
    "${head}_classifier_meta.json"
}

copy_optional_classifier_triplet() {
  local head="$1"
  local weights="$SRC/${head}_classifier_weights.bin"
  local bias="$SRC/${head}_classifier_bias.bin"
  local meta="$SRC/${head}_classifier_meta.json"
  if [[ -f "$weights" && -f "$bias" && -f "$meta" ]]; then
    cp "$weights" "$DST_RESOURCES/${head}_classifier_weights.bin"
    cp "$bias" "$DST_RESOURCES/${head}_classifier_bias.bin"
    cp "$meta" "$DST_RESOURCES/${head}_classifier_meta.json"
    echo "copied optional ${head}_classifier_{weights,bias,meta}"
  else
    echo "skipped optional ${head}_classifier_{weights,bias,meta}"
  fi
}

echo "Using Telco Triage pack: $SRC"
echo ""

for name in "${REQUIRED_MODELS[@]}"; do
  copy_required "$SRC/$name" "$DST_MODELS/$name" "$name"
done

for name in "${REQUIRED_JSON[@]}"; do
  copy_required "$SRC/$name" "$DST_RESOURCES/$name" "$name"
done

for head in "${TELCO_HEADS[@]}"; do
  copy_classifier_triplet "$head"
done

for head in "${TURN_RELATION_HEADS[@]}"; do
  copy_classifier_triplet "$head"
done

for head in "${OPTIONAL_TELCO_HEADS[@]}"; do
  copy_optional_classifier_triplet "$head"
done

echo ""
echo "done"
echo "models:    $DST_MODELS"
echo "resources: $DST_RESOURCES"
echo ""
echo "Next:"
echo "  cd $SCRIPT_DIR && xcodegen generate"
