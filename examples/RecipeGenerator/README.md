Recipe Generator
===

This example demonstrates how to use LeapSDK to generate structured outputs with constraints. See [Leap documentation](https://leap.liquid.ai/docs/edge-sdk/android/constrained-generation) for more details.

## Features

- Generates recipe data as structured JSON output using JSON schema constraints
- Uses `LeapDownloader` to automatically download and cache the LFM2-700M model
- Implements kotlinx serialization for type-safe data parsing
- Demonstrates the `@Generatable` annotation for structured output generation

## Implementation

The main business logic is in [MainActivityViewModel.kt](app/src/main/java/ai/liquid/recipegenerator/MainActivityViewModel.kt).

## Requirements

- LeapSDK 0.10.0
- Android device or emulator with internet connection (for initial model download)

The model will be automatically downloaded and cached on first run via `LeapDownloader`.

## Screenshot
<img src="docs/screenshot.png" width="200">
