# Audio and tool calling with LFM2.5-Audio-1.5B and LFM2-1.2B-Tool

[![Discord](https://img.shields.io/discord/1385439864920739850?color=7289da&label=Join%20Discord&logo=discord&logoColor=white)](https://discord.com/invite/liquid-ai)

## What's inside?

Combines [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) in TTS and STT modes with [LFM2-1.2B-Tool](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF) within a mockup of a car cockpit, letting the user control the car functionalities by voice.  
All running locally in real-time.

https://github.com/user-attachments/assets/f9b5a6fd-ed3b-4235-a856-6251441a1ada

[Llama.cpp](https://github.com/ggml-org/llama.cpp) is used for both models inference, with a custom runner for the audio model. The car cockpit (UI) is vanilla js+html+css, and the communication with the backend is through messages over websocket, like a widely simplified [car CAN bus](https://en.wikipedia.org/wiki/CAN_bus).

## Quick start

> [!NOTE]
> **Supported Platforms**
>
> The following platforms are currently supported:
> - macos-arm64
> - ubuntu-arm64
> - ubuntu-x64
> - ubuntu-WSL2

```bash
# Setup python env
make setup

# Optional: if llama-server is already in your PATH, symlink it instead of building
# ln -s $(which llama-server) llama-server

# Prepare the audio and tool calling models
make LFM2.5-Audio-1.5B-GGUF LFM2-1.2B-Tool-GGUF

# Launch demo
make -j2 audioserver serve
```

> [!NOTE]
> **Building llama-server from source**
>
> The `make -j2 audioserver serve` step will build `llama-server` automatically if it is not already present.
> This requires `cmake` and a C++ toolchain. If the build fails, install the missing dependencies first:
>
> | Platform | Command |
> |---|---|
> | macOS | `brew install cmake` (Xcode CLT required: `xcode-select --install`) |
> | Linux / WSL2 | `make install-deps` |
>
> Then re-run `make -j2 audioserver serve`.
