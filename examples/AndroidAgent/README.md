# AndroidAgent Example

This is an example Android application that runs a tool-calling agent fully on-device with [llama.cpp](https://github.com/ggml-org/llama.cpp) and LFM2.5-1.2B-Instruct. The app showcases two agents: a calculator and a weather forecaster.

## Features
- A small agent loop in plain Kotlin: the model calls tools until it can answer
- Tools defined as Kotlin classes with a JSON schema for their arguments
- llama.cpp formats the tool list for the model and parses its tool calls

## Getting Started

### Prerequisites
- Android Studio with the Android NDK and CMake installed
- Android device or emulator

### Model Download
On first use, the app downloads [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF) (Q5_K_M, about 850 MB) from Hugging Face and reuses it on later launches.

### Building and Running
1. Clone the cookbook with `git clone --recurse-submodules` to get the pinned llama.cpp source.
2. Open the `AndroidAgent` folder in Android Studio.
3. Build and run the app on your device or emulator.

## License
This project is licensed under the MIT License.
