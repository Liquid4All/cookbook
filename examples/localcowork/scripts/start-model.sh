#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LocalCowork — Start Model Server
#
# Starts llama-server for the active model defined in _models/config.yaml.
# Usage:
#   ./scripts/start-model.sh              # Start LFM2-24B-A2B (default)
#   ./scripts/start-model.sh --vision     # Also start vision model on port 8081
#   ./scripts/start-model.sh --check      # Just check if models are downloaded
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

MODELS_DIR="${LOCALCOWORK_MODELS_DIR:-$HOME/Projects/_models}"

# Main model (LFM2-24B-A2B)
MAIN_MODEL="LFM2-24B-A2B-Q4_K_M.gguf"
MAIN_PORT=8080
MAIN_CTX=32768

# Vision model (LFM2.5-VL-1.6B)
VISION_MODEL="LFM2.5-VL-1.6B-Q8_0.gguf"
VISION_MMPROJ="mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"
VISION_PORT=8081

# ── Parse arguments ──────────────────────────────────────────────────────────

START_VISION=false
CHECK_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --vision)  START_VISION=true ;;
        --check)   CHECK_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--vision] [--check]"
            echo ""
            echo "  --vision    Also start the vision model server (port $VISION_PORT)"
            echo "  --check     Check if model files exist (don't start servers)"
            echo ""
            echo "Environment:"
            echo "  LOCALCOWORK_MODELS_DIR    Model directory (default: ~/Projects/_models)"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

# ── Check llama-server ───────────────────────────────────────────────────────

if ! command -v llama-server &> /dev/null; then
    echo "❌ llama-server not found."
    echo ""
    echo "Install via Homebrew (macOS):"
    echo "  brew install llama.cpp"
    echo ""
    echo "Or build from source:"
    echo "  git clone https://github.com/ggml-org/llama.cpp"
    echo "  cd llama.cpp && cmake -B build && cmake --build build --config Release"
    echo "  # Binary at: build/bin/llama-server"
    exit 1
fi

echo "✅ llama-server found: $(command -v llama-server)"

# ── Check model files ────────────────────────────────────────────────────────

echo ""
echo "Models directory: $MODELS_DIR"
echo ""

MAIN_PATH="$MODELS_DIR/$MAIN_MODEL"
VISION_PATH="$MODELS_DIR/$VISION_MODEL"
MMPROJ_PATH="$MODELS_DIR/$VISION_MMPROJ"

if [ -f "$MAIN_PATH" ]; then
    MAIN_SIZE=$(du -h "$MAIN_PATH" | cut -f1)
    echo "✅ Main model:   $MAIN_MODEL ($MAIN_SIZE)"
else
    echo "❌ Main model not found: $MAIN_PATH"
    echo ""
    echo "   Download LFM2-24B-A2B from HuggingFace (gated — request access first):"
    echo "   https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF"
    echo ""
    echo "   pip install huggingface-hub"
    echo "   python3 -c \""
    echo "     from huggingface_hub import hf_hub_download"
    echo "     hf_hub_download('LiquidAI/LFM2-24B-A2B-GGUF',"
    echo "                     'LFM2-24B-A2B-Q4_K_M.gguf',"
    echo "                     local_dir='$MODELS_DIR')"
    echo "   \""
    if [ "$CHECK_ONLY" = true ]; then
        echo ""
    else
        exit 1
    fi
fi

if [ -f "$VISION_PATH" ] && [ -f "$MMPROJ_PATH" ]; then
    echo "✅ Vision model:  $VISION_MODEL + mmproj"
else
    echo "⚠️  Vision model not found (optional — OCR falls back to Tesseract)"
    if [ "$START_VISION" = true ]; then
        echo ""
        echo "   Download from: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF"
        echo ""
        echo "   pip install huggingface-hub"
        echo "   python3 -c \""
        echo "     from huggingface_hub import hf_hub_download"
        echo "     for f in ['$VISION_MODEL', '$VISION_MMPROJ']:"
        echo "         hf_hub_download('LiquidAI/LFM2.5-VL-1.6B-GGUF', f,"
        echo "                         local_dir='$MODELS_DIR')"
        echo "   \""
    fi
fi

if [ "$CHECK_ONLY" = true ]; then
    exit 0
fi

# ── Start main model server ─────────────────────────────────────────────────

if [ ! -f "$MAIN_PATH" ]; then
    echo "Cannot start server — main model file missing."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Starting LFM2-24B-A2B on port $MAIN_PORT"
echo "═══════════════════════════════════════════════════"
echo "  Model:   $MAIN_MODEL"
echo "  Context: $MAIN_CTX tokens"
echo "  API:     http://localhost:$MAIN_PORT/v1"
echo ""

# Start main model in background
llama-server \
    --model "$MAIN_PATH" \
    --port "$MAIN_PORT" \
    --ctx-size "$MAIN_CTX" \
    --n-gpu-layers 99 \
    --flash-attn on &

MAIN_PID=$!
echo "  PID: $MAIN_PID"

# Wait for health check
echo -n "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$MAIN_PORT/health" > /dev/null 2>&1; then
        echo " ready!"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo " timeout (60s). Check logs above for errors."
        exit 1
    fi
    sleep 1
    echo -n "."
done

# ── Start vision model server (optional) ─────────────────────────────────────

if [ "$START_VISION" = true ] && [ -f "$VISION_PATH" ] && [ -f "$MMPROJ_PATH" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════"
    echo "  Starting LFM2.5-VL-1.6B on port $VISION_PORT"
    echo "═══════════════════════════════════════════════════"

    llama-server \
        --model "$VISION_PATH" \
        --mmproj "$MMPROJ_PATH" \
        --port "$VISION_PORT" \
        --ctx-size 32768 &

    VISION_PID=$!
    echo "  PID: $VISION_PID"

    echo -n "  Waiting for server..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:$VISION_PORT/health" > /dev/null 2>&1; then
            echo " ready!"
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo " timeout. Vision OCR will fall back to Tesseract."
        fi
        sleep 1
        echo -n "."
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Model servers running"
echo "═══════════════════════════════════════════════════"
echo "  Main:   http://localhost:$MAIN_PORT/v1  (PID $MAIN_PID)"
if [ "$START_VISION" = true ] && [ -n "${VISION_PID:-}" ]; then
    echo "  Vision: http://localhost:$VISION_PORT/v1  (PID $VISION_PID)"
fi
echo ""
echo "  In another terminal:  cargo tauri dev"
echo "  To stop:              kill $MAIN_PID${VISION_PID:+ $VISION_PID}"
echo ""

# Wait for all background processes
wait
