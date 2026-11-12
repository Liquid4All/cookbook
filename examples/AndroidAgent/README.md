# LeapKoogAgent Android Example

This is an example Android application demonstrating the integration of the [Koog framework](https://docs.koog.ai) with Leap SDK. The app showcases how to use Koog's AI agent capabilities within an Android environment.

## Features
- Integration with Koog framework for AI agent functionality
- Example usage of Leap SDK
- Demonstrates event handling and agent communication

## Getting Started

### Prerequisites
- Android Studio
- Android device or emulator

### Model Installation
To use the AI agent, you need to push the model file to your device:

```
adb push lfm2-1.2b-tool.bundle /tmp/models
```

This command copies the model bundle to the `/tmp/models` directory on your Android device. Make sure `adb` is installed and your device is connected.

### Building and Running
1. Clone this repository.
2. Open in Android Studio.
3. Build and run the app on your device or emulator.

## Koog Framework
[Koog](https://docs.koog.ai) is an open-source framework for building AI agents. It provides tools for:
- Natural language understanding
- Tool invocation
- Context management
- Extensible agent architecture
- Integration with MCP servers (tools retrieval and execution)

## License
This project is licensed under the MIT License.

