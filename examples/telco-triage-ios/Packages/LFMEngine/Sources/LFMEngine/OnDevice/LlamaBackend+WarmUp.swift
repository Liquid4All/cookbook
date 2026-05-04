import Foundation
import os.log

/// Warm-up and timing utilities for LlamaBackend.
///
/// Extracted from LlamaBackend.swift to keep the main file under 300 lines.
extension LlamaBackend {

    // MARK: - Generation Timing

    /// Detailed timing from a single generate call.
    public struct GenerationTiming: Sendable {
        /// Time to evaluate the prompt batch (ms).
        public let promptEvalMs: Double
        /// Time for the autoregressive generation loop (ms).
        public let tokenGenerationMs: Double
        /// Number of prompt tokens evaluated.
        public let promptTokens: Int
        /// Number of output tokens generated.
        public let outputTokens: Int

        public init(
            promptEvalMs: Double,
            tokenGenerationMs: Double,
            promptTokens: Int,
            outputTokens: Int
        ) {
            self.promptEvalMs = promptEvalMs
            self.tokenGenerationMs = tokenGenerationMs
            self.promptTokens = promptTokens
            self.outputTokens = outputTokens
        }
    }

    // MARK: - Warm-Up

    /// Run a minimal dummy inference to JIT-compile Metal shaders and prime the KV cache.
    ///
    /// Call once after `loadModel()` completes. The first inference after model load
    /// incurs a ~200-400ms Metal shader compilation penalty. This method absorbs that
    /// cost during startup rather than on the first user-facing request.
    ///
    /// - Returns: The warm-up latency in milliseconds.
    @discardableResult
    public func warmUp() throws -> Double {
        #if LLAMA_CPP_AVAILABLE
        guard model != nil, context != nil else {
            throw LFMEngineError.modelNotLoaded
        }

        let start = CFAbsoluteTimeGetCurrent()
        // Single-token generation triggers Metal shader compilation + KV cache init.
        // Discard the result — we only care about the side effect.
        let (_, _, _) = try generate(prompt: "hi", maxTokens: 1, temperature: 0.0, clearCache: true)
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000

        let logger = Logger(subsystem: "ai.liquid.banking", category: "LlamaBackend")
        logger.info("Warm-up complete: \(String(format: "%.1f", elapsedMs))ms")
        return elapsedMs
        #else
        let logger = Logger(subsystem: "ai.liquid.banking", category: "LlamaBackend")
        logger.info("Warm-up skipped: llama.cpp not available")
        return 0
        #endif
    }
}
