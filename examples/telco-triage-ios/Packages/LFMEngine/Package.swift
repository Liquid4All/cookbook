// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "LFMEngine",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "LFMEngine", targets: ["LFMEngine"]),
    ],
    dependencies: [
        // llama.cpp via prebuilt XCFramework (Metal + Accelerate).
        // Provides `import llama` for the raw C API including LoRA hot-swap.
        // Semver-tagged wrapper around official ggml-org/llama.cpp releases.
        .package(
            url: "https://github.com/mattt/llama.swift.git",
            exact: "2.8851.0"
        ),
        // MLX-Swift for on-device vision-language model inference (LFM2-VL-450M).
        // Provides MLXVLM with first-class LFM2VL architecture support.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            .upToNextMinor(from: "2.30.3")
        ),
    ],
    targets: [
        .target(
            name: "LFMEngine",
            dependencies: [
                .product(name: "LlamaSwift", package: "llama.swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/LFMEngine",
            swiftSettings: [
                .define("LLAMA_CPP_AVAILABLE"),
            ]
        ),
        .testTarget(
            name: "LFMEngineTests",
            dependencies: ["LFMEngine"],
            path: "Tests/LFMEngineTests"
        ),
    ]
)
