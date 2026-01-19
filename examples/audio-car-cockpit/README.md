> WIP

TODO:
- [ ] readme

## STT + tool calling + TTS

Combines [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) and [LFM2-1.2B-Tool](https://huggingface.co/LiquidAI/LFM2-1.2B-Tool-GGUF) within a mockup of a car cockpit, letting the user control the car functionalities by voice.

Usage:
```bash
# Setup python env
make setup

# Optional, if you have already llama-server in your path, you can
# symlink instead of building it
# ln -s $(which llama-server) llama-server

# Prepare the audio and tool calling models
make LFM2.5-Audio-1.5B-GGUF LFM2-1.2B-Tool-GGUF

# Launch demo
make -j2 audioserver serve
```
