# On-device Koog agent with LFM3 and llama.cpp

This Android app runs a Koog calculator agent and weather agent with
`LiquidAI/LFM3-1.7B-GGUF`. Koog owns the agent loop and tool execution. The
model runs inside the app through the official llama.cpp Android JNI library;
inference does not need an API key or server. The weather tool itself fetches
live forecast data from Open-Meteo, so that tool needs internet access.

## Requirements

- Android Studio with Android SDK 36, NDK 28.2.13676358, CMake 3.31.6 and JDK 17
- Android 13 (API 33) or later, preferably a device with at least 4 GB RAM
- Git submodules (`git clone --recurse-submodules`)

The llama.cpp submodule is pinned by this cookbook repo. Its Android library
source is copied into `lib/` and adjusted for LFM3 sampling: temperature 0.1,
top-k 50, repeat penalty 1.1. It can be refreshed from the same pinned
llama.cpp commit if the upstream Android binding changes.

## Build and run

```bash
git clone --recurse-submodules https://github.com/Liquid4All/cookbook.git
cd cookbook/examples/android-koog-agent
./gradlew :app:assembleDebug
```

Install the APK from Android Studio or with `adb install
app/build/outputs/apk/debug/app-debug.apk`. Download
[`LFM3-1.7B-Q4_K_M.gguf`](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/blob/main/LFM3-1.7B-Q4_K_M.gguf)
from Hugging Face, tap **Import LFM3-1.7B GGUF** in the app, and select the
downloaded file. The app copies it to app-private external storage so llama.cpp
can memory-map it. The import may take a minute for this 1.1 GB checkpoint.

Try **Calculator** with “What is 15% of 340?” and **Weather Forecast** with a
city name. The calculator requires no network. Both demonstrate Koog tool
calls from an on-device model. LFM3 thinking is enabled; the client removes
`<think>…</think>` from the final answer shown to the user.

## Architecture

- `app/` contains the Compose UI, Koog agent definitions and a local Koog
  `LLMClient`.
- `lib/` is based on the upstream llama.cpp Android JNI module. It loads the
  GGUF in process and resets conversation state between Koog agent steps
  without reloading the model.
- `llama.cpp/` is the pinned upstream native source. `lib` builds it with the
  Android NDK and CMake.

The client presents Koog's tool definitions to LFM3, parses its native XML
tool-call markup, and returns typed tool calls to Koog. Koog then invokes the
Kotlin tool implementation and supplies its result on the next model step.

## Verification

On 2026-11-11, `./gradlew :app:assembleDebug --no-daemon` completed successfully
with Android SDK 36, stable NDK 28.2.13676358, CMake 3.31.6, and JDK 17. The
APK installed on an API 36 ARM64 Pixel 8 emulator. `MainActivity` launched and
rendered the tool-selection screen without a startup crash. The
`LFM3-1.7B-Q4_K_M.gguf` file was checked against Hugging Face's SHA-256 hash,
then imported through Android's file picker.

The calculator prompt `1234 + 4321` produced a Koog `plus` tool call with
`a=1234.0, b=4321.0` and the answer `The result of 1234 + 4321 is 5555.`
[Calculator screenshot](assets/calculator-result.png)

The weather prompt `Berlin` produced a Koog `weather_forecast` tool call with
`location=Berlin, date=2026-11-11, days=1, granularity=DAILY` and displayed a
forecast returned by Open-Meteo. [Weather screenshot](assets/weather-result.png)
