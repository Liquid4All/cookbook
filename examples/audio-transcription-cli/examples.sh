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

# Change to the directory containing llama-lfm2-audio
cd "$CURRENT_DIR/LFM2-Audio-1.5B-GGUF/runners/$PLATFORM/lfm2-audio-$PLATFORM"

export LLAMA_CPP_BINARY="$CURRENT_DIR/LFM2-Audio-1.5B-GGUF/runners/$PLATFORM/lfm2-audio-$PLATFORM/llama-lfm2-audio"
export CKPT="$CURRENT_DIR/LFM2-Audio-1.5B-GGUF"
export INPUT_WAV="$CURRENT_DIR/audio-samples/barackobamafederalplaza.mp3"
export OUTPUT_WAV="$CURRENT_DIR/audio-samples/output.mp3"

# Audio to Speech Recognition (ASR)   
./llama-lfm2-audio \
-m $CKPT/LFM2-Audio-1.5B-Q8_0.gguf \
--mmproj $CKPT/mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf \
-mv $CKPT/audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf \
-sys "Perform ASR." \
--audio $INPUT_WAV

# Text To Speech (TTS)
./llama-lfm2-audio \
-m $CKPT/LFM2-Audio-1.5B-Q8_0.gguf \
--mmproj $CKPT/mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf \
-mv $CKPT/audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf \
-sys "Perform TTS." \
-p "My name is Pau Labarta Bajo and I love AI" \
--output $OUTPUT_WAV

# Text To Speech (TTS) with custom voice
./llama-lfm2-audio \
-m $CKPT/LFM2-Audio-1.5B-Q8_0.gguf \
--mmproj $CKPT/mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf \
-mv $CKPT/audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf \
-sys "Perform TTS.
Use the following voice: A male speaker delivers a very expressive and animated speech, with a low-pitch voice and a slightly close-sounding tone. The recording carries a slight background noise." \
-p "What is your name man?" \
--output $OUTPUT_WAV

# Return to original directory
cd "$CURRENT_DIR"