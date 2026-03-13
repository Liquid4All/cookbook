#!/usr/bin/env bash
set -euo pipefail

missing=0

check_command() {
    local cmd="$1"
    local install_hint="$2"
    if command -v "$cmd" &>/dev/null; then
        echo "[OK]  $cmd is installed ($(command -v "$cmd"))"
    else
        echo "[MISSING]  $cmd is not installed"
        echo "          Install suggestion: $install_hint"
        missing=1
    fi
}

echo "Checking prerequisites..."
echo

check_command "make" \
    "sudo apt-get install -y make       # Debian/Ubuntu
          brew install make               # macOS"

check_command "curl" \
    "sudo apt-get install -y curl       # Debian/Ubuntu
          brew install curl               # macOS"

# ── Radeon iGPU / ROCm checks ──

echo
has_radeon=0
if lspci 2>/dev/null | grep -qi 'vga.*amd\|display.*amd\|vga.*radeon\|display.*radeon'; then
    has_radeon=1
    gpu_name=$(lspci 2>/dev/null | grep -iE 'vga.*amd|display.*amd|vga.*radeon|display.*radeon' | head -1 | sed 's/.*: //')
    echo "[OK]  AMD Radeon GPU detected: $gpu_name"

    # Check kernel driver is loaded (amdgpu)
    if grep -q amdgpu /proc/modules 2>/dev/null; then
        echo "[OK]  amdgpu kernel driver is loaded"
    else
        echo "[MISSING]  amdgpu kernel driver is NOT loaded"
        echo "          Install suggestion: sudo apt-get install -y linux-modules-extra-\$(uname -r)"
        echo "          Then reboot and verify with: lsmod | grep amdgpu"
        missing=1
    fi

    # Check ROCm installation
    if [ -d /opt/rocm ]; then
        rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
        echo "[OK]  ROCm is installed at /opt/rocm (version: $rocm_version)"
    else
        echo "[MISSING]  ROCm is not installed (no /opt/rocm found)"
        echo "          Install suggestion: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
        echo "          Also install: sudo apt install -y libstdc++-14-dev"
        missing=1
    fi

    # Check GPU architecture via rocminfo
    if command -v rocminfo &>/dev/null; then
        gfx_arch=$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || true)
        if [ -n "$gfx_arch" ]; then
            echo "[OK]  GPU architecture: $gfx_arch  (pass HIP_ARCH=$gfx_arch to make)"
        else
            echo "[WARN] Could not determine GPU architecture from rocminfo"
        fi
    else
        echo "[INFO] rocminfo not available — install ROCm to detect GPU architecture"
    fi
else
    echo "[INFO] No AMD Radeon GPU detected — will build in CPU-only mode"
fi

echo
if [ "$missing" -eq 0 ]; then
    echo "All prerequisites are installed."
else
    echo "Some prerequisites are missing. Please install them and re-run this script."
    exit 1
fi
