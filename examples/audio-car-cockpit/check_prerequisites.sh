#!/usr/bin/env bash
#
# Pre-flight check for the audio car cockpit demo.
# Run this before 'make setup' to catch common environment issues early.
#
# Usage:  bash check_prerequisites.sh
#

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

pass()  { echo -e "  ${GREEN}✔${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; WARNINGS=$((WARNINGS + 1)); }
fail()  { echo -e "  ${RED}✘${NC} $1"; ERRORS=$((ERRORS + 1)); }
info()  { echo -e "  ${CYAN}→${NC} $1"; }
header(){ echo -e "\n${BOLD}$1${NC}"; }

is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

# ─────────────────────────────────────────────────────────
header "System"
# ─────────────────────────────────────────────────────────

UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

if [[ "$UNAME_S" == "Linux" || "$UNAME_S" == "Darwin" ]]; then
    pass "OS: $UNAME_S $UNAME_M"
else
    fail "Unsupported OS: $UNAME_S (need Linux or macOS)"
fi

if is_wsl; then
    info "Running inside WSL2 — extra checks will be performed"
fi

# ─────────────────────────────────────────────────────────
header "Required tools"
# ─────────────────────────────────────────────────────────

# GNU Make >= 4.3
if command -v make &>/dev/null; then
    MAKE_VER=$(make --version | head -1 | grep -oP '\d+\.\d+' | head -1)
    if [[ "$(printf '%s\n' "4.3" "$MAKE_VER" | sort -V | head -1)" == "4.3" ]]; then
        pass "make $MAKE_VER"
    else
        fail "make $MAKE_VER is too old (need >= 4.3)"
        info "Fix: sudo apt install -y make"
    fi
else
    fail "make not found"
    info "Fix: sudo apt install -y make"
fi

# curl
if command -v curl &>/dev/null; then
    pass "curl $(curl --version | head -1 | awk '{print $2}')"
else
    fail "curl not found"
    info "Fix: sudo apt install -y curl"
fi

# git
if command -v git &>/dev/null; then
    pass "git $(git --version | awk '{print $3}')"
else
    fail "git not found"
    info "Fix: sudo apt install -y git"
fi

# cmake
if command -v cmake &>/dev/null; then
    pass "cmake $(cmake --version | head -1 | awk '{print $3}')"
else
    fail "cmake not found (required to build llama-server)"
    info "Fix: sudo apt install -y cmake"
fi

# C++ compiler
if command -v g++ &>/dev/null; then
    pass "g++ $(g++ -dumpversion)"
elif command -v c++ &>/dev/null; then
    pass "c++ found"
else
    fail "No C++ compiler found (required to build llama-server)"
    info "Fix: sudo apt install -y build-essential"
fi

# unzip
if command -v unzip &>/dev/null; then
    pass "unzip"
else
    fail "unzip not found (required to extract pre-built audio runner)"
    info "Fix: sudo apt install -y unzip"
fi

# uv
if command -v uv &>/dev/null; then
    pass "uv $(uv --version 2>/dev/null | awk '{print $2}')"
else
    warn "uv not found (the Makefile will auto-install it, but see PATH note below)"
    info "Install now:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    info "Then run:     export PATH=\"\$HOME/.local/bin:\$PATH\""
    info "Permanent:    echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
fi

# ─────────────────────────────────────────────────────────
header "Libraries (for building llama-server with SSL)"
# ─────────────────────────────────────────────────────────

if [[ "$UNAME_S" == "Linux" ]]; then
    if dpkg -s libssl-dev &>/dev/null 2>&1; then
        pass "libssl-dev installed"
    else
        fail "libssl-dev not installed (llama-server will build without HTTPS support)"
        info "Fix: sudo apt install -y libssl-dev"
    fi

    if dpkg -s libcurl4-openssl-dev &>/dev/null 2>&1; then
        pass "libcurl4-openssl-dev installed"
    else
        warn "libcurl4-openssl-dev not installed"
        info "Fix: sudo apt install -y libcurl4-openssl-dev"
    fi
elif [[ "$UNAME_S" == "Darwin" ]]; then
    pass "macOS — SSL libraries are provided by the system"
fi

# ─────────────────────────────────────────────────────────
header "Network connectivity"
# ─────────────────────────────────────────────────────────

if curl -s --max-time 5 -o /dev/null -w "%{http_code}" https://huggingface.co | grep -q "200"; then
    pass "huggingface.co reachable (HTTPS)"
else
    fail "Cannot reach https://huggingface.co (needed to download models)"
    info "Check your network connection, proxy, or firewall settings"
fi

# ─────────────────────────────────────────────────────────
header "WSL2-specific checks"
# ─────────────────────────────────────────────────────────

if is_wsl; then
    # IPv6 connectivity check
    IPV6_DISABLED=$(cat /proc/sys/net/ipv6/conf/all/disable_ipv6 2>/dev/null || echo "unknown")

    if [[ "$IPV6_DISABLED" == "1" ]]; then
        pass "IPv6 disabled (avoids WSL2 IPv6 routing issues)"
    elif [[ "$IPV6_DISABLED" == "0" ]]; then
        # Actually test if IPv6 works
        if curl -6 --max-time 5 -s -o /dev/null https://huggingface.co 2>/dev/null; then
            pass "IPv6 enabled and working"
        else
            fail "IPv6 enabled but not routable (downloads will hang)"
            info "Fix (temporary):  sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1 net.ipv6.conf.default.disable_ipv6=1 net.ipv6.conf.lo.disable_ipv6=1"
            info "Fix (permanent):  Add the following to /etc/sysctl.conf:"
            info "    net.ipv6.conf.all.disable_ipv6 = 1"
            info "    net.ipv6.conf.default.disable_ipv6 = 1"
            info "    net.ipv6.conf.lo.disable_ipv6 = 1"
        fi
    else
        warn "Could not determine IPv6 status"
    fi

    # Check if ~/.local/bin is in PATH
    if echo "$PATH" | tr ':' '\n' | grep -q "$HOME/.local/bin"; then
        pass "\$HOME/.local/bin is in PATH"
    else
        warn "\$HOME/.local/bin is not in PATH (uv won't be found after install)"
        info "Fix: export PATH=\"\$HOME/.local/bin:\$PATH\""
        info "Permanent: echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    fi
else
    info "Not running in WSL2 — skipping WSL-specific checks"
fi

# ─────────────────────────────────────────────────────────
header "Summary"
# ─────────────────────────────────────────────────────────

echo ""
if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All checks passed.${NC} You are ready to run: make setup"
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}${BOLD}$WARNINGS warning(s), no errors.${NC} You can proceed but review the warnings above."
else
    echo -e "${RED}${BOLD}$ERRORS error(s) and $WARNINGS warning(s).${NC} Fix the errors above before running make setup."
    exit 1
fi
