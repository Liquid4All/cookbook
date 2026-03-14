#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LocalCowork — Development Environment Setup
# Run this once after cloning the repo to install all dependencies.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "═══════════════════════════════════════════════════"
echo "  LocalCowork — Dev Environment Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Detect OS ─────────────────────────────────────────────────────────────

OS="$(uname -s)"
echo "Detected OS: $OS"
echo ""

# ── Install system prerequisites ──────────────────────────────────────────

echo "Installing system prerequisites..."

if [ "$OS" = "Darwin" ]; then
    # ── macOS ──
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Install from https://brew.sh"
        exit 1
    fi

    if ! command -v node &> /dev/null; then
        echo "  Installing Node.js..."
        brew install node
    fi

    if ! command -v python3 &> /dev/null; then
        echo "  Installing Python..."
        brew install python3
    fi

    if ! command -v cargo &> /dev/null; then
        echo "  Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    if ! command -v cmake &> /dev/null; then
        echo "  Installing cmake..."
        brew install cmake
    fi

    if ! xcode-select -p &> /dev/null; then
        echo "  Installing Xcode Command Line Tools..."
        xcode-select --install
        echo "  ⚠️  Please complete the Xcode CLT install dialog, then re-run this script."
        exit 1
    fi

elif [ "$OS" = "Linux" ]; then
    # ── Ubuntu / Debian ──
    if ! command -v apt-get &> /dev/null; then
        echo "❌ apt-get not found. This script supports macOS and Debian/Ubuntu."
        exit 1
    fi

    NEEDS_APT=()

    if ! command -v node &> /dev/null; then
        echo "  Installing Node.js 22..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi

    if ! python3 -m venv --help &> /dev/null; then
        PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        NEEDS_APT+=("python${PY_VER}-venv")
    fi

    for pkg in build-essential cmake libssl-dev libcurl4-openssl-dev \
               libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev \
               librsvg2-dev patchelf; do
        if ! dpkg -s "$pkg" &> /dev/null; then
            NEEDS_APT+=("$pkg")
        fi
    done

    if [ ${#NEEDS_APT[@]} -gt 0 ]; then
        echo "  Installing: ${NEEDS_APT[*]}"
        sudo apt-get install -y "${NEEDS_APT[@]}"
    fi

    if ! command -v cargo &> /dev/null; then
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        else
            echo "  Installing Rust..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source "$HOME/.cargo/env"
        fi
    fi

    # Increase inotify watcher limit for Vite dev server
    if [ "$(cat /proc/sys/fs/inotify/max_user_watches 2>/dev/null)" -lt 524288 ] 2>/dev/null; then
        echo "  Increasing inotify watcher limit..."
        echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.d/99-inotify.conf > /dev/null
        sudo sysctl -p /etc/sysctl.d/99-inotify.conf > /dev/null
    fi

else
    echo "❌ Unsupported OS: $OS (supported: macOS, Linux/Ubuntu)"
    exit 1
fi

# ── Ensure common tool paths are on PATH ─────────────────────────────────

[[ ":$PATH:" != *":/usr/local/bin:"* ]] && export PATH="/usr/local/bin:$PATH"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

# ── Verify prerequisites ─────────────────────────────────────────────────

echo ""
echo "Verifying prerequisites..."

check_ok() {
    if command -v "$1" &> /dev/null; then
        echo "✅ $1 found: $(command -v "$1")"
    else
        echo "❌ $1 not found after install attempt. $2"
        exit 1
    fi
}

check_ok "node" "Install Node.js 20+ from https://nodejs.org"
check_ok "npm" "Comes with Node.js"
check_ok "python3" "Install Python 3.11+ from https://python.org"
check_ok "cargo" "Install Rust from https://rustup.rs"
check_ok "cmake" "Install cmake"

echo ""

# ── Install Rust dependencies ──────────────────────────────────────────────

echo "Installing Rust dependencies..."

# Install Tauri CLI if not present
if ! cargo tauri --version &> /dev/null; then
    echo "  Installing tauri-cli (this takes ~2 minutes on first run)..."
    cargo install tauri-cli 2>&1 | tail -1
fi
echo "✅ tauri-cli $(cargo tauri --version 2>/dev/null || echo 'installed')"

cd src-tauri
if [ -f Cargo.toml ]; then
    cargo check 2>/dev/null && echo "✅ Rust deps OK" || echo "⚠️  Cargo check failed — Cargo.toml may need setup"
else
    echo "⚠️  No Cargo.toml yet — skip (will be created during Foundation phase)"
fi
cd ..

# ── Install TypeScript dependencies ────────────────────────────────────────

echo ""
echo "Installing TypeScript dependencies..."

# Root workspace
if [ -f package.json ]; then
    npm install
    echo "✅ Root npm deps installed"
else
    echo "⚠️  No root package.json yet — skip"
fi

# TypeScript MCP servers
for server_dir in mcp-servers/filesystem mcp-servers/calendar mcp-servers/email mcp-servers/task mcp-servers/data mcp-servers/audit mcp-servers/clipboard mcp-servers/system; do
    if [ -f "$server_dir/package.json" ]; then
        echo "  Installing $server_dir..."
        cd "$server_dir"
        npm install
        cd - > /dev/null
    fi
done

echo "✅ TypeScript deps installed"

# ── Install Python dependencies ────────────────────────────────────────────

echo ""
echo "Installing Python dependencies..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created Python virtual environment at .venv/"
fi

source .venv/bin/activate

# Upgrade pip to avoid compatibility issues with hatchling editable installs
pip install --quiet --upgrade pip

# Install shared Python deps
pip install --quiet pydantic pytest mypy ruff black

# Python MCP servers
for server_dir in mcp-servers/document mcp-servers/ocr mcp-servers/knowledge mcp-servers/meeting mcp-servers/security mcp-servers/screenshot-pipeline; do
    if [ -f "$server_dir/pyproject.toml" ]; then
        echo "  Installing $server_dir..."
        pip install --quiet -e "$server_dir"
    fi
done

echo "✅ Python deps installed (root venv)"

# ── Per-server Python venvs (for Tauri auto-discovery) ─────────────────────
# The Tauri app's auto-discovery (discovery.rs) detects .venv/ dirs inside
# each server directory and activates them at spawn time (lib.rs).
# These isolated venvs prevent dependency conflicts between servers.

echo ""
echo "Creating per-server Python venvs..."

for server_dir in mcp-servers/document mcp-servers/ocr mcp-servers/knowledge mcp-servers/meeting mcp-servers/security mcp-servers/screenshot-pipeline; do
    if [ -f "$server_dir/pyproject.toml" ]; then
        server_name=$(basename "$server_dir")
        if [ ! -d "$server_dir/.venv" ]; then
            echo "  Creating venv for $server_name..."
            python3 -m venv "$server_dir/.venv"
        fi
        echo "  Installing $server_name deps..."
        "$server_dir/.venv/bin/pip" install --quiet --upgrade pip
        "$server_dir/.venv/bin/pip" install --quiet -e "$server_dir"
    fi
done

echo "✅ Per-server venvs ready"

# ── Install OCR dependencies ─────────────────────────────────────────────

echo ""
echo "Installing OCR dependencies..."

# Tesseract binary (fallback OCR engine when vision model is unavailable)
if command -v brew &> /dev/null; then
    if ! command -v tesseract &> /dev/null; then
        echo "  Installing Tesseract OCR via Homebrew..."
        brew install tesseract
    fi
    echo "✅ tesseract found: $(command -v tesseract)"
elif command -v apt-get &> /dev/null; then
    if ! command -v tesseract &> /dev/null; then
        echo "  Installing Tesseract OCR via apt..."
        sudo apt-get install -y tesseract-ocr
    fi
    echo "✅ tesseract found: $(command -v tesseract)"
else
    check_optional_command "tesseract" "Install Tesseract OCR for fallback text extraction (brew install tesseract)"
fi

# pytest-asyncio (needed for async Python MCP server tests)
pip install --quiet pytest-asyncio

echo "✅ OCR deps installed"

# ── Setup local model ────────────────────────────────────────────────────

MODELS_DIR="${LOCALCOWORK_MODELS_DIR:-$HOME/Projects/_models}"
mkdir -p "$MODELS_DIR"

echo ""
echo "Checking local model setup..."
echo "  Models directory: $MODELS_DIR"
echo ""

# Primary model: LFM2-24B-A2B (production, 80% tool-calling accuracy)
MAIN_MODEL="LFM2-24B-A2B-Q4_K_M.gguf"
if [ -f "$MODELS_DIR/$MAIN_MODEL" ]; then
    MAIN_SIZE=$(du -h "$MODELS_DIR/$MAIN_MODEL" | cut -f1)
    echo "✅ LFM2-24B-A2B found ($MAIN_SIZE)"
else
    echo "❌ LFM2-24B-A2B not found — this is the primary production model"
    echo ""
    echo "   Download from HuggingFace (gated — request access first):"
    echo "   https://huggingface.co/LiquidAI/LFM2-24B-A2B-GGUF"
    echo ""
    echo "   pip install huggingface-hub"
    echo "   python3 -c \""
    echo "     from huggingface_hub import hf_hub_download"
    echo "     hf_hub_download('LiquidAI/LFM2-24B-A2B-GGUF',"
    echo "                     '$MAIN_MODEL',"
    echo "                     local_dir='$MODELS_DIR')"
    echo "   \""
    echo ""
fi

# Check for llama-server (required to serve LFM2 models)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$SCRIPT_DIR/llama-server" ]; then
    echo "✅ llama-server found: $SCRIPT_DIR/llama-server"
elif command -v llama-server &> /dev/null; then
    echo "✅ llama-server found: $(command -v llama-server)"
else
    echo "⚠️  llama-server not found"
    if [ "$OS" = "Darwin" ]; then
        echo "    Install via: brew install llama.cpp"
    else
        echo "    Build via:   make llama-server  (auto-detects ROCm GPU)"
    fi
fi

# Alternative: Ollama with GPT-OSS-20B (optional, for development/comparison)
echo ""
if command -v ollama &> /dev/null; then
    if ollama list 2>/dev/null | grep -q "gpt-oss"; then
        echo "✅ gpt-oss:20b model found in Ollama (optional alternative)"
    else
        echo "ℹ️  Ollama installed but gpt-oss:20b not found (optional)"
        echo "    To install: ollama pull gpt-oss:20b"
    fi
else
    echo "ℹ️  Ollama not installed (optional — only needed if using GPT-OSS-20B instead of LFM2)"
fi

# LFM Vision model for OCR (optional, served via llama.cpp)
echo ""
if [ -f "$MODELS_DIR/LFM2.5-VL-1.6B-Q8_0.gguf" ] && [ -f "$MODELS_DIR/mmproj-LFM2.5-VL-1.6b-Q8_0.gguf" ]; then
    echo "✅ LFM2.5-VL-1.6B vision model found (optional — for AI-powered OCR)"
else
    echo "ℹ️  LFM Vision model not found (optional — OCR falls back to Tesseract)"
    echo "    To install (~1.8 GB): ./scripts/start-model.sh --check"
fi

# ── Create local config directories ───────────────────────────────────────

echo ""
echo "Creating local config directories..."
mkdir -p ~/.localcowork/{models,templates,trash}
echo "✅ Config dirs created at ~/.localcowork/"

# ── Install git hooks ────────────────────────────────────────────────────

echo ""
echo "Installing git hooks..."
if [ -d ".git" ]; then
    git config core.hooksPath .git-hooks
    echo "✅ Git hooks installed (core.hooksPath → .git-hooks/)"
else
    echo "⚠️  Not a git repo yet — run 'git init' first, then 'git config core.hooksPath .git-hooks'"
fi

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Next steps:"
if [ ! -x "$SCRIPT_DIR/llama-server" ] && ! command -v llama-server &> /dev/null; then
    echo "  1. Build llama-server:  make llama-server  (auto-detects ROCm GPU)"
    echo "  2. Download model:      see README.md"
    echo "  3. Start model server:  ./scripts/start-model.sh"
    echo "  4. Start dev server:    cargo tauri dev  (in another terminal)"
else
    echo "  1. Start model server:  ./scripts/start-model.sh"
    echo "  2. Start dev server:    cargo tauri dev  (in another terminal)"
fi
echo ""
echo "  Note: The model server must be running before 'cargo tauri dev'."
echo ""
