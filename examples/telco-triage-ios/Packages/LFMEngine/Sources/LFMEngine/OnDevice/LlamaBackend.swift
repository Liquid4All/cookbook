import Foundation
import os.log

#if LLAMA_CPP_AVAILABLE
import llama
#endif

/// Thread-safe llama.cpp C API wrapper. All C pointer operations serialize on
/// this actor's executor, keeping the main thread free for UI.
///
/// Lifecycle: `loadModel()` → `setAdapter()` / `generate()` → `unload()`
public actor LlamaBackend {

    // MARK: - State

    /// Access level is internal so extensions (LlamaTokenization) can reach these.
    /// llama_model, llama_context, llama_adapter_lora are opaque C types → OpaquePointer.
    /// llama_sampler has a full C definition → typed pointer (only when llama is available).
    var model: OpaquePointer?
    var context: OpaquePointer?
    /// Currently active LoRA adapter pointer.
    var loraAdapter: OpaquePointer?
    /// Cache of loaded adapter pointers keyed by file path.
    /// Avoids re-reading .gguf from flash on every swap (~100-300ms saved per hit).
    var adapterCache: [String: OpaquePointer] = [:]
    /// Path of the currently active adapter (for cache lookup).
    var activeAdapterPath: String?
    #if LLAMA_CPP_AVAILABLE
    var sampler: UnsafeMutablePointer<llama_sampler>?
    #else
    var sampler: OpaquePointer?
    #endif

    private let logger = Logger(subsystem: "ai.liquid.banking", category: "LlamaBackend")

    /// Tracks the current sampler temperature to avoid unnecessary recreation.
    var samplerTemperature: Float?

    /// Reference count for `llama_backend_init/free`. The C backend is process-global,
    /// so we must only init once and free when the last instance unloads.
    /// Protected by a lock because static properties on actors are not actor-isolated.
    private static let backendRefLock = OSAllocatedUnfairLock(initialState: 0)

    /// Number of threads for llama.cpp batch decode.
    private let threadCount: Int

    /// Tracks whether this instance has been properly unloaded.
    /// `nonisolated(unsafe)` allows safe read in `deinit` without actor isolation.
    nonisolated(unsafe) private var hasBeenUnloaded = false

    // MARK: - Init

    public init(threadCount: Int? = nil) {
        self.threadCount = threadCount ?? max(ProcessInfo.processInfo.processorCount - 2, 1)
    }

    // MARK: - Model Loading

    /// Load a GGUF model from disk. Initializes the llama backend, model, context, and sampler.
    public func loadModel(
        path: String,
        contextLength: UInt32 = 2048,
        gpuLayers: Int32 = 99,
        temperature: Float = 0.0
    ) throws {
        #if LLAMA_CPP_AVAILABLE
        logger.info("Loading model from \(path, privacy: .public)")

        Self.backendRefLock.withLock { count in
            if count == 0 { llama_backend_init() }
            count += 1
        }

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = gpuLayers

        logger.info("Calling llama_model_load_from_file: gpu_layers=\(gpuLayers), path_length=\(path.count)")
        guard let loadedModel = llama_model_load_from_file(path, modelParams) else {
            Self.backendRefLock.withLock { count in
                count -= 1
                if count == 0 { llama_backend_free() }
            }
            logger.error("llama_model_load_from_file returned nil — check device console for llama.cpp stderr (architecture mismatch, corrupt file, or OOM)")
            throw LFMEngineError.modelLoadFailed("llama_model_load_from_file returned nil for \(path)")
        }
        self.model = loadedModel

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = contextLength
        ctxParams.n_threads = Int32(threadCount)
        ctxParams.n_threads_batch = Int32(threadCount)

        guard let loadedContext = llama_init_from_model(loadedModel, ctxParams) else {
            llama_model_free(loadedModel)
            self.model = nil
            Self.backendRefLock.withLock { count in
                count -= 1
                if count == 0 { llama_backend_free() }
            }
            throw LFMEngineError.modelLoadFailed("llama_init_from_model returned nil")
        }
        self.context = loadedContext

        applySampler(temperature: temperature)

        logger.info("Model loaded: ctx=\(contextLength), gpu_layers=\(gpuLayers), threads=\(self.threadCount)")
        #else
        throw LFMEngineError.modelLoadFailed("llama.cpp not available on this platform")
        #endif
    }

    // MARK: - Inference

    /// Run greedy (or sampled) token generation and return the decoded text with timing.
    ///
    /// - Parameters:
    ///   - prompt: The full prompt to process.
    ///   - maxTokens: Maximum number of tokens to generate.
    ///   - temperature: Sampling temperature (0.0 = greedy).
    ///   - stopSequences: Sequences that halt generation when encountered.
    ///   - clearCache: Whether to clear the KV cache before generation.
    ///     Set to `false` for multi-turn conversations where prior context
    ///     should be preserved. Defaults to `true` for single-shot inference.
    /// - Returns: Generated text, token count, and per-phase timing breakdown.
    public func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float = 0.0,
        stopSequences: [String] = [],
        clearCache: Bool = true,
        outputMode: GenerationParams.OutputMode = .text
    ) throws -> (text: String, tokenCount: Int, timing: GenerationTiming) {
        #if LLAMA_CPP_AVAILABLE
        guard let context, let model else { throw LFMEngineError.modelNotLoaded }

        applySampler(temperature: temperature)

        let tokens = tokenize(prompt, addBos: true)
        guard !tokens.isEmpty else {
            throw LFMEngineError.invalidPromptFormat("Tokenization produced empty result")
        }

        // Clear KV cache unless multi-turn context should be preserved
        if clearCache {
            let memory = llama_get_memory(context)
            llama_memory_clear(memory, true)
        }

        // Evaluate prompt tokens using batch
        let promptEvalStart = CFAbsoluteTimeGetCurrent()
        var batch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(batch) }

        for (i, token) in tokens.enumerated() {
            let pos = Int32(i)
            batch.token[Int(pos)] = token
            batch.pos[Int(pos)] = llama_pos(pos)
            batch.n_seq_id[Int(pos)] = 1
            batch.seq_id[Int(pos)]![0] = 0
            batch.logits[Int(pos)] = (i == tokens.count - 1) ? 1 : 0
        }
        batch.n_tokens = Int32(tokens.count)

        var status = llama_decode(context, batch)
        guard status == 0 else {
            throw LFMEngineError.inferenceFailed("Prompt decode failed with code \(status)")
        }
        let promptEvalMs = (CFAbsoluteTimeGetCurrent() - promptEvalStart) * 1000
        let generationStart = CFAbsoluteTimeGetCurrent()

        // Autoregressive generation loop
        // Optimizations vs naive approach:
        // 1. Incremental detokenization — O(1) per token instead of O(N)
        // 2. Batch reuse — single allocation instead of per-token alloc/free
        var generated: [llama_token] = []
        generated.reserveCapacity(maxTokens)
        let eosToken = llama_vocab_eos(llama_model_get_vocab(model))
        var pos = Int32(tokens.count)
        var textSoFar = ""

        // Pre-compute max stop sequence length for suffix window check
        let maxStopLen = stopSequences.map(\.count).max() ?? 0

        // Allocate a single-token batch once, reused every iteration
        var nextBatch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(nextBatch) }

        for _ in 0..<maxTokens {
            guard let sampler else { break }

            let newToken = llama_sampler_sample(sampler, context, -1)
            if newToken == eosToken { break }

            generated.append(newToken)

            // Incremental detokenization — only decode the new token
            let piece = detokenizeSingle(newToken)
            textSoFar += piece

            // Check stop sequences against the tail of generated text.
            // +8 margin accounts for multi-byte UTF-8 chars that could straddle the window boundary.
            if maxStopLen > 0 {
                let tailStart = textSoFar.index(
                    textSoFar.endIndex,
                    offsetBy: -min(maxStopLen + 8, textSoFar.count),
                    limitedBy: textSoFar.startIndex
                ) ?? textSoFar.startIndex
                let tail = String(textSoFar[tailStart...])
                if stopSequences.contains(where: { tail.hasSuffix($0) }) {
                    break
                }
            }

            if shouldStop(for: textSoFar, mode: outputMode) {
                break
            }

            // Bail early if model is producing prose instead of JSON.
            // After 16 tokens with no '{', stop to avoid wasting time.
            if shouldBailNoJSON(for: textSoFar, tokenCount: generated.count, mode: outputMode) {
                break
            }

            // Reuse the pre-allocated batch
            nextBatch.token[0] = newToken
            nextBatch.pos[0] = llama_pos(pos)
            nextBatch.n_seq_id[0] = 1
            nextBatch.seq_id[0]![0] = 0
            nextBatch.logits[0] = 1
            nextBatch.n_tokens = 1

            status = llama_decode(context, nextBatch)

            guard status == 0 else {
                throw LFMEngineError.inferenceFailed("Decode step failed with code \(status)")
            }
            pos += 1
        }

        let tokenGenerationMs = (CFAbsoluteTimeGetCurrent() - generationStart) * 1000
        let timing = GenerationTiming(
            promptEvalMs: promptEvalMs,
            tokenGenerationMs: tokenGenerationMs,
            promptTokens: tokens.count,
            outputTokens: generated.count
        )

        logger.info("Inference: prompt_eval=\(String(format: "%.1f", promptEvalMs))ms, generation=\(String(format: "%.1f", tokenGenerationMs))ms, tokens=\(generated.count)")

        return (text: textSoFar, tokenCount: generated.count, timing: timing)
        #else
        throw LFMEngineError.inferenceFailed("llama.cpp not available on this platform")
        #endif
    }

    // MARK: - Cleanup

    /// Free all resources. Safe to call even if model was never loaded.
    public func unload() {
        #if LLAMA_CPP_AVAILABLE
        let wasLoaded = model != nil

        if let s = sampler { llama_sampler_free(s) }
        // Detach adapters from context before freeing
        if let ctx = context {
            llama_set_adapters_lora(ctx, nil, 0, nil)
        }
        // Free all cached adapters (includes the active one)
        for (_, cached) in adapterCache {
            llama_adapter_lora_free(cached)
        }
        adapterCache.removeAll()
        if let c = context { llama_free(c) }
        if let m = model { llama_model_free(m) }

        sampler = nil
        samplerTemperature = nil
        loraAdapter = nil
        activeAdapterPath = nil
        context = nil
        model = nil

        if wasLoaded {
            Self.backendRefLock.withLock { count in
                count -= 1
                if count == 0 { llama_backend_free() }
            }
        }
        hasBeenUnloaded = true
        logger.info("Backend unloaded")
        #endif
    }

    deinit {
        // Uses `hasBeenUnloaded` (nonisolated(unsafe)) to avoid accessing
        // actor-isolated `model` property — safe for Swift 6 strict concurrency.
        if !hasBeenUnloaded {
            Logger(subsystem: "ai.liquid.banking", category: "LlamaBackend")
                .warning("LlamaBackend deallocated without unload() — potential resource leak")
        }
    }
}
