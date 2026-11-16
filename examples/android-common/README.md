# Shared Android runtime

The five Android example projects include this module as `:runtime`. Keep the sibling directories together and clone the cookbook with `--recurse-submodules`. Each app supplies its pinned llama.cpp checkout; all use v0.9.0.

## Model files

`Models.kt` records the download URL, expected byte count, and SHA-256 from Hugging Face LFS metadata. `ModelStore` verifies an existing cached file before using it. New downloads go to `.part`, are verified, then are renamed to the final filename. Failed and cancelled transfers never publish a model; an interrupted transfer restarts on retry. Calls for the same path share a lock.

Changing a checkpoint requires updating its URL, size, and digest together and rerunning real inference tests.

## Native ownership

`NativeTextModel`, `NativeVisionModel`, and `NativeAgentModel` each own a native handle and a `NativeTaskRunner`. The runner queues load, generation, and release on one worker. Closing requests cancellation immediately, then waits in the queue until active native work returns before deleting its handle. Token loops poll cancellation; prefill or model loading must finish its current native operation before release.

Native state belongs to the handle, not process-wide model pointers. Text crosses JNI as standard UTF-8 byte arrays. The agent runtime supplies IDs for parsed tool calls that lack one, keeping assistant calls and tool results consistent.

Some GGUF checkpoints embed older templates that render assistant `content` but ignore structured `tool_calls`. For those templates, the runtime preserves the generated call text in `content` as well as returning llama.cpp's parsed calls. Keep the full assistant reply in conversation history so the model can associate subsequent tool results with its call. Templates that support structured calls use them directly.

Text requests use an 8,192-token context, preserving the slogan and summarizer examples' prior capacity; vision uses 4,096 tokens. Text and vision requests clear their context each time. Agent requests receive an explicit full conversation. A token/context limit is reported as incomplete generation, and callers must not decode partial JSON as a completed document.

`TextTaskViewModel` owns text models across Activity rotation and exposes durable output, busy, and error state. The vision app has an equivalent retained owner for model and projector. Call `close()` when the owning ViewModel is cleared. Do not free native resources independently of the runner.

The APK must extract native libraries: `android:extractNativeLibs="true"` and `packaging.jniLibs.useLegacyPackaging = true`. llama.cpp loads CPU variants from `nativeLibraryDir`.

## Run regression tests

With Android SDK 36, NDK `29.0.13113456`, CMake `3.31.6`, and JDK 17 configured:

```bash
cd examples/android-common
../AndroidAgent/gradlew testDebugUnitTest assembleDebugAndroidTest
```

Unit tests exercise download verification/recovery, concurrent callers, offline cache use, cancellation, and serialized native release. Weather date-range tests live in `AndroidAgent/app/src/test`:

```bash
cd ../AndroidAgent
./gradlew :app:testDebugUnitTest
```

## Real-model Android tests

`NativeInferenceTest` loads actual checkpoints. It is not a mock inference test. Use an ARM64 Android device or emulator with sufficient RAM and storage, and seed these four files into the test app's external files directory:

- `LFM2.5-350M-Q8_0.gguf`
- `LFM2.5-1.2B-Instruct-Q5_K_M.gguf`
- `LFM2.5-VL-1.6B-Q8_0.gguf`
- `mmproj-LFM2.5-VL-1.6b-Q8_0.gguf`

Verify their hashes against `Models.kt` before seeding. From `examples/android-common`:

```bash
adb install -r build/outputs/apk/androidTest/debug/LiquidAndroidRuntime-debug-androidTest.apk
adb shell mkdir -p /sdcard/Android/data/ai.liquid.android.runtime.test/files
adb push /path/to/verified/models/. /sdcard/Android/data/ai.liquid.android.runtime.test/files/
adb shell am instrument -w -r ai.liquid.android.runtime.test/androidx.test.runner.AndroidJUnitRunner
```

The tests cover Unicode JSON, concurrent requests, invalid-schema recovery, close during generation, a replacement model, parsed tool calls with real tool results, and image inference. Inspect the runner's test results; `adb shell am instrument` can exit successfully even when individual tests fail.

Also build and run each app. Check rotation during preparation and generation, leaving during generation, offline first launch, retry, repeated submissions, and ShareAI's subsequent share intents. Unit or native-runtime tests alone do not verify those UI flows.
