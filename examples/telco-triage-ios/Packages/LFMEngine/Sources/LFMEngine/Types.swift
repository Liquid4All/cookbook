import Foundation

// MARK: - Inference Types

/// Result of a single inference call.
public struct InferenceResult: Sendable {
    public let text: String
    public let latencyMs: Double
    public let tokensGenerated: Int
    public let model: String
    public let adapter: String?
    public let ranOn: InferenceLocation
    /// Detailed latency breakdown for observability. Nil for proxy mode.
    public let latencyBreakdown: LatencyBreakdown?
    public let promptTokens: Int?
    public let outputTokens: Int?

    public init(
        text: String,
        latencyMs: Double,
        tokensGenerated: Int,
        model: String,
        adapter: String?,
        ranOn: InferenceLocation,
        latencyBreakdown: LatencyBreakdown? = nil,
        promptTokens: Int? = nil,
        outputTokens: Int? = nil
    ) {
        self.text = text
        self.latencyMs = latencyMs
        self.tokensGenerated = tokensGenerated
        self.model = model
        self.adapter = adapter
        self.ranOn = ranOn
        self.latencyBreakdown = latencyBreakdown
        self.promptTokens = promptTokens
        self.outputTokens = outputTokens
    }
}

/// Per-inference latency breakdown for tracing and observability.
public struct LatencyBreakdown: Sendable {
    /// Time to evaluate the prompt tokens (ms).
    public let promptEvalMs: Double
    /// Time to generate output tokens (ms).
    public let tokenGenerationMs: Double
    /// Total wall-clock inference time (ms).
    public let totalMs: Double

    public init(promptEvalMs: Double, tokenGenerationMs: Double, totalMs: Double) {
        self.promptEvalMs = promptEvalMs
        self.tokenGenerationMs = tokenGenerationMs
        self.totalMs = totalMs
    }
}

/// Where inference was executed.
public enum InferenceLocation: String, Sendable {
    case device = "device"
    case cloud = "cloud"
    case specialist = "specialist"
}

/// How the engine should run inference.
public enum InferenceMode: String, Sendable {
    /// Call a remote inference backend via HTTP (rapid iteration).
    case proxy
    /// Run llama.cpp on-device with Metal (production demo).
    case onDevice
}

// MARK: - Model Configuration

/// Configuration for the base model.
public struct ModelConfig: Sendable {
    public let name: String
    public let fileName: String
    public let contextLength: Int
    public let gpuLayers: Int
    public let defaultTemperature: Float

    public init(
        name: String,
        fileName: String,
        contextLength: Int = 2048,
        gpuLayers: Int = 99,
        defaultTemperature: Float = 0.0
    ) {
        self.name = name
        self.fileName = fileName
        self.contextLength = contextLength
        self.gpuLayers = gpuLayers
        self.defaultTemperature = defaultTemperature
    }
}

/// Configuration for a LoRA adapter.
public struct AdapterConfig: Sendable, Identifiable {
    public let id: String
    public let name: String
    public let fileName: String
    public let capability: String

    public init(id: String, name: String, fileName: String, capability: String) {
        self.id = id
        self.name = name
        self.fileName = fileName
        self.capability = capability
    }
}

// MARK: - Generation Parameters

/// Parameters for a single generation call.
public struct GenerationParams: Sendable {
    public enum OutputMode: Sendable {
        case text
        case jsonObject
    }

    public let prompt: String
    public let maxTokens: Int
    public let temperature: Float
    public let stopSequences: [String]
    /// Whether to clear the KV cache before generation.
    /// Set to `false` for multi-turn conversations. Defaults to `true`.
    public let clearCache: Bool
    public let outputMode: OutputMode

    public init(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.0,
        stopSequences: [String] = [],
        clearCache: Bool = true,
        outputMode: OutputMode = .text
    ) {
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.stopSequences = stopSequences
        self.clearCache = clearCache
        self.outputMode = outputMode
    }
}

// MARK: - Engine State

/// Current state of the inference engine.
public enum EngineState: Sendable, Equatable {
    case unloaded
    case loading(progress: Double)
    case ready
    case inferring
    case error(String)
}

// MARK: - Errors

/// Typed errors for the inference engine.
public enum LFMEngineError: Error, LocalizedError, Sendable {
    case modelNotLoaded
    case engineBusy
    case modelLoadFailed(String)
    case noDownloadURL
    case downloadHTTPError(statusCode: Int)
    case checksumMismatch(fileName: String)
    case adapterLoadFailed(adapter: String, reason: String)
    case inferenceTimeout(durationMs: Double)
    case inferenceFailed(String)
    case invalidPromptFormat(String)
    case insufficientMemory(requiredMB: Int, availableMB: Int)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "Base model is not loaded. Call loadModel() first."
        case .engineBusy:
            return "Engine is busy with another inference request. Wait for it to complete."
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .noDownloadURL:
            return "No base model download URL configured"
        case .downloadHTTPError(let statusCode):
            return "Download failed: HTTP \(statusCode)"
        case .checksumMismatch(let fileName):
            return "Checksum mismatch for \(fileName)"
        case .adapterLoadFailed(let adapter, let reason):
            return "Failed to load adapter '\(adapter)': \(reason)"
        case .inferenceTimeout(let duration):
            return "Inference timed out after \(Int(duration))ms"
        case .inferenceFailed(let reason):
            return "Inference failed: \(reason)"
        case .invalidPromptFormat(let reason):
            return "Invalid prompt format: \(reason)"
        case .insufficientMemory(let required, let available):
            return "Insufficient memory: need \(required)MB, have \(available)MB"
        }
    }
}

// MARK: - Token Streaming

/// A single token emitted during streaming inference.
public struct StreamToken: Sendable {
    public let text: String
    public let index: Int
    public let isFinal: Bool

    public init(text: String, index: Int, isFinal: Bool) {
        self.text = text
        self.index = index
        self.isFinal = isFinal
    }
}

// MARK: - Multimodal Input

/// Input to the inference engine — text, text+image, or text+audio.
///
/// Used by `LFMEngineProxy.infer(capability:input:)` to support
/// vision (MLX-Swift) and audio (LFM2.5-Audio / Apple Speech) modalities alongside text.
public enum MultimodalInput: Sendable {
    /// Text-only input (existing behavior).
    case text(String)
    /// Text prompt with an image (JPEG data). Routed to MLX VLM backend.
    case textWithImage(String, Data)
    /// Text prompt with audio recording URL. Routed to on-device STT first.
    case textWithAudio(String, URL)

    /// Extract the text component from any input variant.
    public var textContent: String {
        switch self {
        case .text(let text): return text
        case .textWithImage(let text, _): return text
        case .textWithAudio(let text, _): return text
        }
    }
}

// MARK: - Vision Configuration

/// Configuration for the on-device vision-language model.
public struct VisionModelConfig: Sendable {
    /// HuggingFace model ID (e.g., "mlx-community/LFM2-VL-450M-5bit").
    public let modelID: String
    /// Maximum tokens to generate from image+text input.
    public let maxTokens: Int
    /// Estimated RAM usage in MB when loaded.
    public let ramMB: Int

    public init(modelID: String, maxTokens: Int = 512, ramMB: Int = 500) {
        self.modelID = modelID
        self.maxTokens = maxTokens
        self.ramMB = ramMB
    }
}

/// Configuration for the on-device speech-to-text engine.
public struct AudioModelConfig: Sendable {
    /// Audio model variant (e.g., "base", "small", "tiny").
    public let modelVariant: String
    /// Estimated RAM usage in MB when loaded.
    public let ramMB: Int

    public init(modelVariant: String = "base", ramMB: Int = 400) {
        self.modelVariant = modelVariant
        self.ramMB = ramMB
    }
}

// MARK: - Model Download

/// Progress of a model download operation.
public struct DownloadProgress: Sendable {
    public let bytesDownloaded: Int64
    public let totalBytes: Int64
    public let fileName: String

    public var fraction: Double {
        guard totalBytes > 0 else { return 0 }
        return Double(bytesDownloaded) / Double(totalBytes)
    }

    public init(bytesDownloaded: Int64, totalBytes: Int64, fileName: String) {
        self.bytesDownloaded = bytesDownloaded
        self.totalBytes = totalBytes
        self.fileName = fileName
    }
}
