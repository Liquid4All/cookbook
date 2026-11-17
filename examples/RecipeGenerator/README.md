# RecipeGenerator

Generate a dinner recipe as five-field JSON constrained by a llama.cpp grammar.

Model: **LFM2.5-1.2B-Instruct Q5_K_M**. These examples intentionally use tested LFM2.5 checkpoints.

## Build and run

1. Install Android Studio with JDK 17, SDK 36, NDK `29.0.13113456`, and CMake `3.31.6`.
2. Clone the cookbook with `git clone --recurse-submodules https://github.com/Liquid4All/cookbook.git`.
3. Open `examples/RecipeGenerator` in Android Studio. Keep the sibling `examples/android-common` module; it contains the shared runtime.
4. Run on Android 12 (API 31) or newer. The app downloads the model on first use.

The app generates automatically on launch using the fixed prompt `A recipe for a dinner dish`. It displays name, ingredients, cookingTime, isVegetarian, and steps. Use Generate another recipe or Retry. There is no ingredient-entry field.

## Runtime behavior

The [shared runtime](../android-common/README.md) verifies model file sizes and SHA-256 hashes before publishing them. Interrupted downloads restart safely, and verified cached files work offline. ViewModels retain models across rotation. A serialized native queue and per-instance handles prevent inference from racing cleanup or another request.

The projects pin Kotlin `2.3.20` and llama.cpp `v0.9.0`. Run the shared runtime's unit and real-model instrumentation tests when changing inference or lifecycle code.
