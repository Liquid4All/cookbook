# Voice Chat

A voice chat application that uses the LFM2-Audio-1.5B model to generate conversational audio responses.

This application could work 100% locally, but the liquid-audio library requires CUDA. This is why the model is wrapped inside a Modal function and deployed to a serverless GPU environment with CUDA, so you can run it even if you (like me) don't have an NVIDIA GPU at home. Record your voice, send it for processing, and receive an audio response that plays automatically.

## What's in this example?

This example demonstrates how to build an interactive voice chat application using:

- **Audio Recording**: Records audio from your microphone with automatic silence detection
- **Cloud Processing**: Uses Modal to run the LFM2-Audio-1.5B model on GPU in the cloud
- **Audio Playback**: Automatically plays the generated audio response

The application works by:
1. Recording your voice question from the microphone (with auto-stop on silence)
2. Uploading the audio to a Modal volume
3. Processing the audio with LFM2-Audio-1.5B on a GPU instance to generate an interleaved text and audio response
4. Downloading the generated audio response
5. Playing the response through your speakers

The model generates responses that can include both text and audio tokens, creating a natural conversational experience.

## Tools

This example uses the following tools and libraries:

- **[liquid-audio](https://github.com/LiquidAI/liquid-audio)**: Python library for working with LFM2-Audio models
- **[Modal](https://modal.com/)**: Serverless cloud platform for running GPU workloads
- **[PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)**: Cross-platform audio I/O library for recording from microphone
- **[pygame](https://www.pygame.org/)**: Audio playback library
- **[torchaudio](https://pytorch.org/audio/)**: Audio processing utilities for PyTorch
- **[rich](https://rich.readthedocs.io/)**: Terminal formatting library for audio visualization

## How to run it

### Prerequisites

1. Install [uv](https://github.com/astral-sh/uv) if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Set up Modal:
   - Create a Modal account at [modal.com](https://modal.com)
   - Install the Modal CLI: `uv add modal`
   - Authenticate: `uv run modal token new`

3. Ensure microphone permissions are granted to your terminal/IDE

### Running the application

1. **Deploy the server** (first time only, or when server code changes):
   ```bash
   make deploy-server
   ```
   Or directly:
   ```bash
   uv run modal deploy -m src.voice_chat.server
   ```

2. **Run the client**:
   ```bash
   make run
   ```
   Or directly:
   ```bash
   uv run modal run -m src.voice_chat.client
   ```

3. **Follow the prompts**:
   - The application will start recording when you run it
   - Speak your question into the microphone
   - Recording will automatically stop after 2 seconds of silence
   - The audio will be uploaded to Modal and processed
   - The generated audio response will be downloaded and played automatically

The generated audio file will be saved locally as `answer_YYYYMMDD_HHMMSS.wav` for each session.
