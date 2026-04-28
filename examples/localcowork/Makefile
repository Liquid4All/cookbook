.PHONY: help all setup serve clean install-deps

all: help

help:  ## Show commands
	@echo "Default to \`help\`. Other targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


# ┌──────────────────────────────────────────────────────────┐
# │              Platform detection and ROCm                 │
# └──────────────────────────────────────────────────────────┘

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifdef CPU
  HAS_ROCM :=
else
  HAS_ROCM := $(shell rocm-smi --showproductname 2>/dev/null | grep "Card Series" >/dev/null && echo 1)
endif

LLAMA_CPP_REPO   := https://github.com/ggml-org/llama.cpp.git
LLAMA_CPP_COMMIT := d03c45c9c56795af8b0e899762bf266c14fd2028


# ┌──────────────────────────────────────────────────────────┐
# │           Build configuration (ROCm vs CPU)              │
# └──────────────────────────────────────────────────────────┘

ifdef HAS_ROCM
  $(info ROCm detected — building with HIP GPU acceleration)

  _DETECTED_HIP_ARCH := $(shell rocm-smi --showproductname 2>/dev/null | grep -oP 'gfx\w+' | head -1)
  HIP_ARCH   ?= $(or $(_DETECTED_HIP_ARCH),gfx1150)
  $(info Using HIP_ARCH=$(HIP_ARCH))
  CMAKE_ARGS := -DGGML_HIP=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_HIP_ARCHITECTURES="$(HIP_ARCH)"
  GPU_FLAGS  := --n-gpu-layers 99

else
  ifdef CPU
    $(info CPU=1 set — forcing CPU-only build)
  else
    $(info ROCm not found — building CPU-only)
  endif

  CMAKE_ARGS := -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON
  GPU_FLAGS  :=

endif


# ┌──────────────────────────────────────────────────────────┐
# │                Build llama.cpp from source                │
# └──────────────────────────────────────────────────────────┘

llama.cpp:
	git clone $(LLAMA_CPP_REPO) && \
		cd llama.cpp && \
		git checkout $(LLAMA_CPP_COMMIT)

install-deps:  ## Install system dependencies required to build llama-server
ifeq ($(UNAME_S),Darwin)
	@echo "On macOS, install Xcode Command Line Tools and cmake via Homebrew:"
	@echo "  xcode-select --install"
	@echo "  brew install cmake"
else ifeq ($(UNAME_S),Linux)
	sudo apt install -y build-essential cmake libssl-dev libcurl4-openssl-dev
else
	$(error install-deps: unsupported OS $(UNAME_S))
endif

llama.cpp/build/bin/llama-server: llama.cpp
ifeq ($(UNAME_S),Linux)
	@dpkg -s libssl-dev >/dev/null 2>&1 || \
		(echo "Error: libssl-dev not found — llama-server would build without HTTPS support." \
		     "Run 'make install-deps' first." && exit 1)
endif
	cd llama.cpp && \
		cmake -B build $(CMAKE_ARGS) && \
		cmake --build build --config Release -t llama-server -j 8

llama-server: llama.cpp/build/bin/llama-server  ## Build llama-server (auto-detects ROCm)
	cp llama.cpp/build/bin/llama-server $@
	@echo ""
	@echo "✅ llama-server built successfully"
ifdef HAS_ROCM
	@echo "   GPU acceleration: HIP ($(HIP_ARCH))"
else
	@echo "   GPU acceleration: none (CPU-only)"
endif
	@echo "   Binary: $(PWD)/llama-server"
	@echo ""
	@echo "   Add to PATH:  ln -s $(PWD)/llama-server ~/.local/bin/llama-server"


# ┌──────────────────────────────────────────────────────────┐
# │                        Utilities                         │
# └──────────────────────────────────────────────────────────┘

clean:  ## Remove build artifacts (llama.cpp clone and binary)
	rm -rf llama.cpp llama-server
