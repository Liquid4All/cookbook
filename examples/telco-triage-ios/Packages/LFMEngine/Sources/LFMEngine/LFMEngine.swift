import Foundation
import os.log

/// Central inference engine that manages the base model and LoRA adapters.
///
/// Supports two modes:
/// - `.onDevice`: Direct llama.cpp inference via Metal (production)
/// - `.proxy`: HTTP calls to a remote inference backend (rapid iteration)
///
/// The engine is mode-agnostic to callers — `generate()` returns the same
/// `InferenceResult` regardless of mode.
///
/// UI state (`state`, `activeAdapter`, `loadedAdapters`) is published on
/// `@MainActor`. Heavy inference work dispatches to a background task so
/// the main thread stays responsive during generation.
@MainActor
public final class LFMEngine: ObservableObject {

    // MARK: - Published State

    @Published public internal(set) var state: EngineState = .unloaded
    @Published public private(set) var activeAdapter: AdapterConfig?
    @Published public private(set) var loadedAdapters: [String: AdapterConfig] = [:]

    // MARK: - Configuration

    public let mode: InferenceMode
    let modelConfig: ModelConfig
    private let logger = Logger(subsystem: "ai.liquid.banking", category: "LFMEngine")

    /// Backend URL for proxy mode.
    let proxyBaseURL: URL?

    // MARK: - On-Device Backend

    /// The llama.cpp backend actor. Created lazily on first on-device call.
    private var backend: LlamaBackend?

    /// Local path to the base GGUF model file.
    let modelBasePath: String?

    // MARK: - Init

    public init(
        mode: InferenceMode,
        modelConfig: ModelConfig,
        proxyBaseURL: URL? = nil,
        modelBasePath: String? = nil
    ) {
        self.mode = mode
        self.modelConfig = modelConfig
        self.proxyBaseURL = proxyBaseURL
        self.modelBasePath = modelBasePath
    }

    // MARK: - Backend Access

    /// The underlying LlamaBackend for classifier head embedding extraction.
    /// Only valid after `loadModel()` completes in `.onDevice` mode.
    /// Used by `BankingClassifierBridge` and `ClassifierBackedBridge` to
    /// call `embeddings(prompt:)` for hidden state extraction.
    public var llamaBackend: LlamaBackend? { backend }

    // MARK: - Model Lifecycle

    /// Load the base model. In proxy mode, this validates the backend connection.
    /// In on-device mode, this loads the GGUF model into memory and runs a
    /// warm-up inference to JIT-compile Metal shaders.
    public func loadModel() async throws {
        state = .loading(progress: 0)
        logger.info("Loading model: \(self.modelConfig.name) [mode: \(self.mode.rawValue)]")

        do {
            switch mode {
            case .proxy:
                try await validateProxyConnection()
            case .onDevice:
                try await loadOnDeviceModel()
            }
        } catch {
            state = .error(error.localizedDescription)
            throw error
        }

        state = .ready
        logger.info("Model ready: \(self.modelConfig.name)")
    }

    /// Warm up the on-device backend by running a dummy inference.
    ///
    /// This JIT-compiles Metal shaders and primes the KV cache so the first
    /// real inference avoids a ~200-400ms cold-start penalty.
    /// Safe to call in proxy mode (no-op).
    public func warmUp() async {
        guard mode == .onDevice, let backend else {
            logger.info("Warm-up skipped: not in on-device mode or backend not loaded")
            return
        }
        do {
            let warmUpMs = try await backend.warmUp()
            logger.info("Engine warm-up complete: \(String(format: "%.1f", warmUpMs))ms")
        } catch {
            logger.warning("Engine warm-up failed (non-fatal): \(error.localizedDescription)")
        }
    }

    /// Unload the model and free all resources.
    ///
    /// This is async because the backend must finish releasing GPU resources
    /// before we nil the reference. Fire-and-forget causes races when a new
    /// model is loaded immediately after eviction.
    public func unloadModel() async {
        if let backend {
            await backend.unload()
            self.backend = nil
        }
        activeAdapter = nil
        loadedAdapters.removeAll()
        state = .unloaded
        logger.info("Model unloaded")
    }

    // MARK: - LoRA Adapter Management

    /// Swap to a different LoRA adapter. In proxy mode, this sets the capability
    /// for the next API call. In on-device mode, this hot-swaps the adapter.
    ///
    /// - Parameters:
    ///   - adapter: The adapter configuration.
    ///   - resolvedPath: Pre-resolved absolute path to the adapter file. When provided,
    ///     this path is used directly instead of constructing one from `modelBasePath`.
    ///     Callers (e.g. `OnDeviceEngineProxy`) should resolve the path using their
    ///     own lookup logic (bundle → Documents) and pass it here.
    public func setAdapter(_ adapter: AdapterConfig, resolvedPath: String? = nil) async throws {
        guard state == .ready else {
            throw LFMEngineError.modelNotLoaded
        }

        let start = CFAbsoluteTimeGetCurrent()

        switch mode {
        case .proxy:
            // Proxy mode: just track the adapter for capability routing
            activeAdapter = adapter
        case .onDevice:
            try await swapOnDeviceAdapter(adapter, resolvedPath: resolvedPath)
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        activeAdapter = adapter
        loadedAdapters[adapter.id] = adapter
        logger.info("Adapter set: \(adapter.name) [\(String(format: "%.1f", elapsed))ms]")
    }

    /// Remove the current adapter, reverting to base model behavior.
    public func removeAdapter() async {
        switch mode {
        case .proxy:
            break
        case .onDevice:
            await removeOnDeviceAdapter()
        }
        activeAdapter = nil
        logger.info("Adapter removed, using base model")
    }

    // MARK: - Inference

    /// Generate a completion for the given parameters.
    ///
    /// On-device inference runs on the `LlamaBackend` actor's executor, not
    /// the main thread. State transitions (`inferring` → `ready`) happen on
    /// `@MainActor` but the heavy token generation is fully off-main-thread.
    public func generate(_ params: GenerationParams) async throws -> InferenceResult {
        guard state != .unloaded else {
            throw LFMEngineError.modelNotLoaded
        }
        if case .error(let reason) = state {
            throw LFMEngineError.modelLoadFailed(reason)
        }
        guard state == .ready else {
            throw LFMEngineError.engineBusy
        }

        state = .inferring

        let result: InferenceResult
        do {
            switch mode {
            case .proxy:
                result = try await generateViaProxy(params)
            case .onDevice:
                result = try await generateOnDevice(params)
            }
        } catch {
            state = .ready
            throw error
        }

        state = .ready

        logger.info("Generated \(result.tokensGenerated) tokens in \(String(format: "%.1f", result.latencyMs))ms [\(result.ranOn.rawValue)]")

        return result
    }

    // MARK: - On-Device Mode (llama.cpp via LlamaBackend)

    private func loadOnDeviceModel() async throws {
        guard let path = modelBasePath else {
            throw LFMEngineError.modelLoadFailed("modelBasePath is required for on-device mode")
        }

        let exists = FileManager.default.fileExists(atPath: path)
        let size = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0
        logger.info("loadOnDeviceModel: path=\(path, privacy: .public) exists=\(exists) size=\(size)bytes ctx=\(self.modelConfig.contextLength) gpu=\(self.modelConfig.gpuLayers)")

        let llamaBackend = LlamaBackend()
        do {
            try await llamaBackend.loadModel(
                path: path,
                contextLength: UInt32(modelConfig.contextLength),
                gpuLayers: Int32(modelConfig.gpuLayers),
                temperature: modelConfig.defaultTemperature
            )
        } catch {
            logger.error("loadOnDeviceModel FAILED: \(error.localizedDescription, privacy: .public)")
            throw error
        }
        self.backend = llamaBackend
    }

    private func swapOnDeviceAdapter(_ adapter: AdapterConfig, resolvedPath: String? = nil) async throws {
        guard let backend else { throw LFMEngineError.modelNotLoaded }

        let adapterPath: String
        if let resolved = resolvedPath {
            adapterPath = resolved
        } else {
            // Fallback: construct path relative to model directory (specialist packs)
            guard let basePath = modelBasePath else {
                throw LFMEngineError.adapterLoadFailed(
                    adapter: adapter.name, reason: "No model base path configured")
            }
            let modelDir = (basePath as NSString).deletingLastPathComponent
            adapterPath = (modelDir as NSString).appendingPathComponent(adapter.fileName)
        }

        try await backend.setAdapter(path: adapterPath)
    }

    private func removeOnDeviceAdapter() async {
        guard let backend else { return }
        await backend.removeAdapter()
    }

    /// Run inference on the llama.cpp backend with real latency instrumentation.
    ///
    /// Although this method is `@MainActor`-isolated, the heavy work happens
    /// inside `backend.generate()` which runs on the `LlamaBackend` actor's
    /// executor. The `await` suspends the main actor, keeping the UI responsive.
    func generateOnDevice(_ params: GenerationParams) async throws -> InferenceResult {
        guard let backend else { throw LFMEngineError.modelNotLoaded }

        let totalStart = CFAbsoluteTimeGetCurrent()
        let (text, tokenCount, timing) = try await backend.generate(
            prompt: params.prompt,
            maxTokens: params.maxTokens,
            temperature: params.temperature,
            stopSequences: params.stopSequences,
            clearCache: params.clearCache,
            outputMode: params.outputMode
        )
        let totalMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000

        let breakdown = LatencyBreakdown(
            promptEvalMs: timing.promptEvalMs,
            tokenGenerationMs: timing.tokenGenerationMs,
            totalMs: totalMs
        )

        return InferenceResult(
            text: text,
            latencyMs: totalMs,
            tokensGenerated: tokenCount,
            model: modelConfig.name,
            adapter: activeAdapter?.name,
            ranOn: .device,
            latencyBreakdown: breakdown,
            promptTokens: timing.promptTokens,
            outputTokens: timing.outputTokens
        )
    }

}
