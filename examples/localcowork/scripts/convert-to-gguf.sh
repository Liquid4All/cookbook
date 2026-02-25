#!/usr/bin/env bash
# Convert fine-tuned HuggingFace model to GGUF for llama-server deployment.
#
# Usage (on H100):
#   ./convert-to-gguf.sh /home/ubuntu/localcowork-finetune/output/best
#
# Outputs:
#   LFM2.5-1.2B-Router-FT-F16.gguf   (2.3 GB — training reference)
#   LFM2.5-1.2B-Router-FT-Q8_0.gguf  (1.2 GB — production default)
#   LFM2.5-1.2B-Router-FT-Q4_K_M.gguf (0.7 GB — lightweight option)
#
# Prerequisites:
#   pip install gguf sentencepiece
#   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
#   cd ~/llama.cpp && make -j$(nproc)

set -euo pipefail

MODEL_DIR="${1:?Usage: $0 <hf-model-dir>}"
LLAMA_CPP="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$MODEL_DIR")/gguf}"

mkdir -p "$OUTPUT_DIR"

echo "=== GGUF Conversion Pipeline ==="
echo "Model dir: $MODEL_DIR"
echo "llama.cpp: $LLAMA_CPP"
echo "Output:    $OUTPUT_DIR"
echo

# Step 1: Convert HF to GGUF (FP16)
F16_PATH="$OUTPUT_DIR/LFM2.5-1.2B-Router-FT-F16.gguf"
echo "[1/4] Converting HF → GGUF (F16)..."
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outfile "$F16_PATH" \
    --outtype f16

echo "  → $F16_PATH ($(du -h "$F16_PATH" | cut -f1))"

# Step 2: Quantize to Q8_0 (production default — minimal quality loss)
Q8_PATH="$OUTPUT_DIR/LFM2.5-1.2B-Router-FT-Q8_0.gguf"
echo "[2/4] Quantizing → Q8_0..."
"$LLAMA_CPP/llama-quantize" "$F16_PATH" "$Q8_PATH" Q8_0

echo "  → $Q8_PATH ($(du -h "$Q8_PATH" | cut -f1))"

# Step 3: Quantize to Q4_K_M (lightweight option)
Q4_PATH="$OUTPUT_DIR/LFM2.5-1.2B-Router-FT-Q4_K_M.gguf"
echo "[3/4] Quantizing → Q4_K_M..."
"$LLAMA_CPP/llama-quantize" "$F16_PATH" "$Q4_PATH" Q4_K_M

echo "  → $Q4_PATH ($(du -h "$Q4_PATH" | cut -f1))"

# Step 4: Validate with llama-server (quick sanity check)
echo "[4/4] Validating Q8_0 with llama-server..."
"$LLAMA_CPP/llama-server" \
    --model "$Q8_PATH" \
    --port 8099 \
    --ctx-size 2048 \
    --n-gpu-layers 99 &
SERVER_PID=$!

# Wait for server startup
sleep 5

# Quick health check
if curl -s http://localhost:8099/health | grep -q "ok"; then
    echo "  ✓ Server started successfully with Q8_0 model"

    # Quick inference test
    RESPONSE=$(curl -s http://localhost:8099/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [
                {"role": "system", "content": "You are a tool router. Call one tool."},
                {"role": "user", "content": "What files are in my Downloads?"}
            ],
            "temperature": 0.1,
            "max_tokens": 128
        }')
    echo "  Quick test response: $(echo "$RESPONSE" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:200])' 2>/dev/null || echo 'parse error')"
else
    echo "  ✗ Server failed to start — check model compatibility"
fi

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo
echo "=== Conversion Complete ==="
echo "Files:"
ls -lh "$OUTPUT_DIR"/LFM2.5-1.2B-Router-FT-*.gguf
echo
echo "Next steps:"
echo "  1. Transfer Q8_0 to dev machine:"
echo "     rsync -avz $Q8_PATH chintan@<dev-mac>:~/Projects/_models/"
echo "  2. Update _models/config.yaml: model_path → new GGUF path"
echo "  3. Run benchmarks: npx tsx benchmark-lfm.ts --k=15"
