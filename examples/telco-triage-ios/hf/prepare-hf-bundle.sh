#!/usr/bin/env bash
#
# Stage the Telco Triage customer delivery bundle for Hugging Face.
#
# Inputs:
#   TELCO_MODELS_DIR  Directory containing required GGUFs.
#                     Defaults to ./models/telco.
#   HF_BUNDLE_DIR     Output directory.
#                     Defaults to ./.hf-bundle/telco-triage-ios.
#
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXAMPLE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
MODELS_DIR="${TELCO_MODELS_DIR:-$EXAMPLE_DIR/models/telco}"
BUNDLE_DIR="${HF_BUNDLE_DIR:-$EXAMPLE_DIR/.hf-bundle/telco-triage-ios}"

REQUIRED_MODELS=(
  "lfm25-350m-base-Q4_K_M.gguf"
  "telco-shared-clf-v1.gguf"
  "telco-turn-relation-v4.gguf"
  "telco-dialogue-repair-v4.gguf"
  "telco-tool-selector-v3.gguf"
)

REQUIRED_RESOURCES=(
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

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "error: model directory not found: $MODELS_DIR" >&2
  exit 1
fi

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

for name in "${REQUIRED_MODELS[@]}"; do
  if [[ ! -f "$MODELS_DIR/$name" ]]; then
    echo "error: missing required GGUF: $MODELS_DIR/$name" >&2
    exit 1
  fi
  cp "$MODELS_DIR/$name" "$BUNDLE_DIR/$name"
done

for name in "${REQUIRED_RESOURCES[@]}"; do
  src="$EXAMPLE_DIR/TelcoTriage/Resources/$name"
  if [[ ! -f "$src" ]]; then
    echo "error: missing required resource: $src" >&2
    exit 1
  fi
  cp "$src" "$BUNDLE_DIR/$name"
done

for head in "${TELCO_HEADS[@]}"; do
  for suffix in weights.bin bias.bin meta.json; do
    name="${head}_classifier_${suffix}"
    src="$EXAMPLE_DIR/TelcoTriage/Resources/$name"
    if [[ ! -f "$src" ]]; then
      echo "error: missing classifier head artifact: $src" >&2
      exit 1
    fi
    cp "$src" "$BUNDLE_DIR/$name"
  done
done

for head in "${TURN_RELATION_HEADS[@]}"; do
  for suffix in weights.bin bias.bin meta.json; do
    name="${head}_classifier_${suffix}"
    src="$EXAMPLE_DIR/TelcoTriage/Resources/$name"
    if [[ ! -f "$src" ]]; then
      echo "error: missing turn-relation classifier artifact: $src" >&2
      exit 1
    fi
    cp "$src" "$BUNDLE_DIR/$name"
  done
done

for head in "${OPTIONAL_TELCO_HEADS[@]}"; do
  have_all=1
  for suffix in weights.bin bias.bin meta.json; do
    if [[ ! -f "$EXAMPLE_DIR/TelcoTriage/Resources/${head}_classifier_${suffix}" ]]; then
      have_all=0
    fi
  done
  if [[ "$have_all" == "1" ]]; then
    for suffix in weights.bin bias.bin meta.json; do
      name="${head}_classifier_${suffix}"
      cp "$EXAMPLE_DIR/TelcoTriage/Resources/$name" "$BUNDLE_DIR/$name"
    done
  fi
done

cp "$SCRIPT_DIR/README.md" "$BUNDLE_DIR/README.md"
cp "$SCRIPT_DIR/model_manifest.json" "$BUNDLE_DIR/model_manifest.json"

(
  cd "$BUNDLE_DIR"
  find . -type f ! -name 'checksums.sha256' -print0 \
    | sort -z \
    | xargs -0 shasum -a 256 \
    > checksums.sha256
)

echo "Prepared Hugging Face bundle:"
echo "  $BUNDLE_DIR"
echo ""
find "$BUNDLE_DIR" -maxdepth 1 -type f -print | sort | sed 's#^#  #'
