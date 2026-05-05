import Foundation
import AVFoundation

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

public enum TranscriptionError: LocalizedError {
    case permissionDenied
    case unavailable
    case recognitionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .permissionDenied:
            return "Microphone or speech recognition permission was denied."
        case .unavailable:
            return "No valid microphone input route is available."
        case .recognitionFailed(let message):
            return message
        }
    }
}

enum AudioTapInstaller {
    /// AVFAudio reports some microphone-route failures by throwing an
    /// Objective-C `NSException`, which Swift cannot catch. Install taps only
    /// through the Obj-C bridge so bad simulator routes become recoverable
    /// voice errors instead of process-ending crashes.
    static func install(
        on node: AVAudioNode,
        bus: AVAudioNodeBus = 0,
        bufferSize: AVAudioFrameCount,
        block: @escaping (AVAudioPCMBuffer, AVAudioTime) -> Void
    ) throws {
        var error: NSError?
        let installed = LFMInstallAudioTapSafely(
            node,
            UInt(bus),
            bufferSize,
            nil,
            block,
            &error
        )
        guard installed else {
            throw TranscriptionError.recognitionFailed(
                error?.localizedDescription ?? "Microphone input route could not be opened."
            )
        }
    }

    static func isValid(_ format: AVAudioFormat) -> Bool {
        isValid(sampleRate: format.sampleRate, channelCount: format.channelCount)
    }

    static func isValid(sampleRate: Double, channelCount: AVAudioChannelCount) -> Bool {
        sampleRate.isFinite && sampleRate > 0 && channelCount > 0
    }
}
