import Foundation

/// Protocol for on-device speech-to-text. Defaults to Apple's
/// SFSpeechRecognizer (built in) so voice works before any pack is
/// downloaded — after the audio pack is installed, a richer LFM-based
/// transcriber takes over for accented / noisy audio.
public protocol VoiceTranscriber: Sendable {
    /// Start listening. Returns a stream of partial and final transcripts.
    /// Implementations should emit at least one `.final` before finishing.
    func startListening() async throws -> AsyncStream<TranscriptionEvent>

    /// Stop the current recording / recognition and finalize.
    func stopListening() async

    /// Release heavyweight resources (model runners, audio buffers) when
    /// the transcriber will not be reused. Default implementation is a
    /// no-op — only override in transcribers that hold expensive state
    /// (e.g., LFMAudioTranscriber's LEAP model runner).
    func releaseResources() async
}

public extension VoiceTranscriber {
    func releaseResources() async { /* no-op by default */ }
}

public enum TranscriptionEvent: Sendable {
    case partial(String)
    case final(String)
    case error(String)
}

public enum TranscriptionError: Error {
    case permissionDenied
    case unavailable
    case recognitionFailed(String)
}
