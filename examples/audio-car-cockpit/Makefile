.PHONY: help all \
	setup lint precommit \
	serve audioserver \
	test-search test-quick test-full test-toolcall \
	llama-liquid-audio-runner \
	LFM2-1.2B-Tool-GGUF

all: help

help:  ## Show commands
	@echo "Default to \`help\`. Other targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'


# Make sure the current make version is modern enough
ifneq ($(firstword $(sort $(MAKE_VERSION) 4.3)),4.3)
	$(error Needs GNU Make 4.3 or higher. In MacOS, try using gmake, and maybe add it to the path)
endif

# Override the externally defined value, if any
override VIRTUAL_ENV := $(PWD)/.venv

export PYTHONPATH := $(PWD):$(PYTHONPATH)


# ┌──────────────────────────────────────────────────────────┐
# │                  Environment variables                   │
# └──────────────────────────────────────────────────────────┘

export DEMO_URL ?= http://127.0.0.1:8000
export AUDIO_SERVER_PORT ?= 8142
THREADS ?= 4


# ┌──────────────────────────────────────────────────────────┐
# │                      Initial setup                       │
# └──────────────────────────────────────────────────────────┘

OS := $(shell uname -s)

# Install uv if it does not exist, and create the env
.venv:
ifeq (, $(shell which uv))
	$(info uv is not installed yet, fetching it now)
	$(shell curl -LsSf https://astral.sh/uv/install.sh | sh)
endif
	uv venv --allow-python-downloads --allow-existing

setup: .venv  ## Initial virtual setup through uv
	uv sync --all-groups


# ┌──────────────────────────────────────────────────────────┐
# │        Download audio and function calling models        │
# └──────────────────────────────────────────────────────────┘

LFM2.5-Audio-1.5B-GGUF:  ## Download audio model
	uv run --with "huggingface_hub" hf download "LiquidAI/LFM2.5-Audio-1.5B-GGUF" \
		--local-dir ./LFM2.5-Audio-1.5B-GGUF \
		--include '*Q8_0*'

LFM2-1.2B-Tool-GGUF: llama-server  ## Download tool calling model
	@# Not strictly necessary, but pre-download for smoother first startup
	llama-server -hf "LiquidAI/LFM2-1.2B-Tool-GGUF:Q8_0" --version


# ┌──────────────────────────────────────────────────────────┐
# │       Prepare inference runner, platform specific        │
# │              Required until this is merged:              │
# │     https://github.com/ggml-org/llama.cpp/pull/18641     │
# └──────────────────────────────────────────────────────────┘

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  OS := macos
else ifeq ($(UNAME_S),Linux)
  OS := ubuntu
else
  $(error Unsupported OS: $(UNAME_S))
endif

ifeq ($(UNAME_M),arm64)
  ARCH := arm64
else ifeq ($(UNAME_M),aarch64)
  ARCH := arm64
else ifeq ($(UNAME_M),x86_64)
  ARCH := x64
else ifeq ($(UNAME_M),amd64)
  ARCH := x64
else
  $(error Unsupported arch: $(UNAME_M))
endif

ZIP := runners/llama-liquid-audio-$(OS)-$(ARCH).zip
DIR := $(patsubst runners/%.zip,%,$(ZIP))

llama-liquid-audio-runner: $(ZIP)

$(ZIP):
	@mkdir -p runners
	uv run --with "huggingface_hub" hf download "LiquidAI/LFM2.5-Audio-1.5B-GGUF" --local-dir ./ $@
	unzip -q -o $@

llama-liquid-audio/llama-liquid-audio-server: $(ZIP)  ## Download pre-built llama-liquid-audio-server
	ln -sf $(DIR) llama-liquid-audio
	@# Sanity check
	./llama-liquid-audio/llama-liquid-audio-server --version


# ┌──────────────────────────────────────────────────────────┐
# │                    Build llama-server                    │
# └──────────────────────────────────────────────────────────┘

llama.cpp:
	git clone https://github.com/ggml-org/llama.cpp.git && \
		cd llama.cpp && \
		git checkout d03c45c9c56795af8b0e899762bf266c14fd2028

llama.cpp/build/bin/llama-server: llama.cpp
	cd llama.cpp && \
		export BUILD_TARGETS=llama-server && \
		cmake -B build -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=ON && \
		cmake --build build --config Release -j 8

llama-server:  ## Static build of llama-server
	@# Make doesn't allow non-recursive dependencies, adding the check here instead
	test -e $@ || $(MAKE) llama.cpp/build/bin/llama-server && \
		cp llama.cpp/build/bin/llama-server $@
	touch llama-server


# ┌──────────────────────────────────────────────────────────┐
# │                         Servers                          │
# └──────────────────────────────────────────────────────────┘

serve: llama-server  ## Start FastAPI server
	uv run --frozen server.py

audioserver: llama-liquid-audio/llama-liquid-audio-server LFM2.5-Audio-1.5B-GGUF  ## Start audio server
	$< \
		-m LFM2.5-Audio-1.5B-GGUF/LFM2.5-Audio-1.5B-Q8_0.gguf \
		-mm LFM2.5-Audio-1.5B-GGUF/mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf \
		-mv LFM2.5-Audio-1.5B-GGUF/vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf \
		--tts-speaker-file LFM2.5-Audio-1.5B-GGUF/tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf \
	-t ${THREADS} --host 127.0.0.1 --port ${AUDIO_SERVER_PORT} &>/dev/null


# ┌──────────────────────────────────────────────────────────┐
# │                         Testing                          │
# └──────────────────────────────────────────────────────────┘

BASE_URL = http://localhost:8000

functions.json: static/script.js
	@echo "Fetching all function definitions from server"
	curl -s $(BASE_URL)/functions.json > $@

test-search:  ## Search for functions (usage: make test-search QUERY=media)
	curl -s "$(BASE_URL)/debug/get-functions-matching/$(QUERY)" | jq

test-quick:  ## Run quick system check (basic tests)
	curl -s $(BASE_URL)/checklist/quick-check | jq

test-full:  ## Run full system check (comprehensive tests)
	curl -s $(BASE_URL)/checklist/full-system | jq

test-toolcall:  ## Tool call with the string "play the next song"
	curl -s $(BASE_URL)/toolcall/single/play%20the%20next%20song | jq


# ┌──────────────────────────────────────────────────────────┐
# │                        Utilities                         │
# └──────────────────────────────────────────────────────────┘

UV_FROZEN_DEV = uv run --only-group dev --frozen

lint:  ## Lint and format python code
	@$(UV_FROZEN_DEV) ruff format
	@$(UV_FROZEN_DEV) ruff check --fix --extend-select=I
	@# Type checking
	$(UV_FROZEN_DEV) ty check --exit-zero --respect-ignore-files --color always 2>&1 | grep -v 'ty is pre-release software'
