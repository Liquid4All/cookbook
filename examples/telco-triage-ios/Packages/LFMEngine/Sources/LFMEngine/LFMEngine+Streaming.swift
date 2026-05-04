import Foundation
import os.log

/// Streaming inference methods for LFMEngine.
///
/// Extracted from LFMEngine.swift to keep the main file under 300 lines.
extension LFMEngine {

    // MARK: - Stream Entry Point

    /// Stream tokens for the given parameters.
    public func generateStream(
        _ params: GenerationParams
    ) -> AsyncThrowingStream<StreamToken, Error> {
        AsyncThrowingStream { continuation in
            Task { @MainActor in
                do {
                    guard self.state != .unloaded else {
                        throw LFMEngineError.modelNotLoaded
                    }
                    guard self.state == .ready else {
                        throw LFMEngineError.engineBusy
                    }
                    self.state = .inferring

                    switch self.mode {
                    case .proxy:
                        try await self.streamViaProxy(params, continuation: continuation)
                    case .onDevice:
                        try await self.streamOnDevice(params, continuation: continuation)
                    }

                    self.state = .ready
                    continuation.finish()
                } catch {
                    self.state = .ready
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - On-Device Streaming

    func streamOnDevice(
        _ params: GenerationParams,
        continuation: AsyncThrowingStream<StreamToken, Error>.Continuation
    ) async throws {
        // Generate full result then emit as stream (true token-by-token streaming
        // requires a callback-based llama_decode loop — deferred to Phase 3)
        let result = try await generateOnDevice(params)
        let words = result.text.components(separatedBy: " ")

        for (index, word) in words.enumerated() {
            let token = StreamToken(
                text: index == 0 ? word : " " + word,
                index: index,
                isFinal: index == words.count - 1
            )
            continuation.yield(token)
        }
    }
}
