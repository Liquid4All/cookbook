#!/bin/bash

# Function to auto-detect platform and architecture
get_platform_and_architecture() {
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    # Normalize OS name
    case "$OS" in
        darwin) PLATFORM_OS="macos" ;;
        linux) PLATFORM_OS="ubuntu" ;;
        *) PLATFORM_OS="$OS" ;;
    esac

    # Normalize architecture
    case "$ARCH" in
        x86_64|amd64) PLATFORM_ARCH="x64" ;;
        aarch64|arm64) PLATFORM_ARCH="arm64" ;;
        arm*) PLATFORM_ARCH="arm64" ;;
        *) PLATFORM_ARCH="$ARCH" ;;
    esac

    echo "${PLATFORM_OS}-${PLATFORM_ARCH}"
}

# Save current directory
CURRENT_DIR=$(pwd)

export PLATFORM=$(get_platform_and_architecture)

# Change to the directory containing llama-liquid-audio-cli
cd "$CURRENT_DIR/LFM2.5-Audio-1.5B-GGUF/runners/$PLATFORM/llama-liquid-audio-$PLATFORM"

export LLAMA_CPP_BINARY="$CURRENT_DIR/LFM2.5-Audio-1.5B-GGUF/runners/$PLATFORM/llama-liquid-audio-$PLATFORM/llama-liquid-audio-cli"
export CKPT="$CURRENT_DIR/LFM2.5-Audio-1.5B-GGUF"
export QUANT="${QUANT:-Q8_0}"
export INPUT_WAV="$CURRENT_DIR/audio-samples/barackobamafederalplaza.mp3"
export OUTPUT_WAV="${OUTPUT_WAV:-$CURRENT_DIR/audio-samples/output.mp3}"

# Audio to Speech Recognition (ASR)   
./llama-liquid-audio-cli \
-m $CKPT/LFM2.5-Audio-1.5B-$QUANT.gguf \
-mm $CKPT/mmproj-LFM2.5-Audio-1.5B-$QUANT.gguf \
-mv $CKPT/vocoder-LFM2.5-Audio-1.5B-$QUANT.gguf \
--tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-$QUANT.gguf \
-sys "Perform ASR." \
--audio $INPUT_WAV

# Text To Speech (TTS)
./llama-liquid-audio-cli \
-m $CKPT/LFM2.5-Audio-1.5B-$QUANT.gguf \
-mm $CKPT/mmproj-LFM2.5-Audio-1.5B-$QUANT.gguf \
-mv $CKPT/vocoder-LFM2.5-Audio-1.5B-$QUANT.gguf \
--tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-$QUANT.gguf \
-sys "Perform TTS. Use the US male voice." \
-p "My name is Pau Labarta Bajo and I love AI" \
--output $OUTPUT_WAV

# Text To Speech (TTS) with a different voice
# Supported voices: US male, UK male, US female, UK female
./llama-liquid-audio-cli \
-m $CKPT/LFM2.5-Audio-1.5B-$QUANT.gguf \
-mm $CKPT/mmproj-LFM2.5-Audio-1.5B-$QUANT.gguf \
-mv $CKPT/vocoder-LFM2.5-Audio-1.5B-$QUANT.gguf \
--tts-speaker-file $CKPT/tokenizer-LFM2.5-Audio-1.5B-$QUANT.gguf \
-sys "Perform TTS. Use the UK female voice." \
-p "What is your name man?" \
--output $OUTPUT_WAV

# Return to original directory
cd "$CURRENT_DIR"