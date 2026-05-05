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

enum AudioInputTapFormat {
    static func resolve(
        for input: AVAudioInputNode,
        session: AVAudioSession = .sharedInstance()
    ) throws -> AVAudioFormat {
        let inputFormat = input.inputFormat(forBus: 0)
        if isValid(inputFormat) {
            return inputFormat
        }

        let outputFormat = input.outputFormat(forBus: 0)
        if isValid(outputFormat) {
            return outputFormat
        }

        let sessionSampleRate = session.sampleRate
        let sessionChannels = AVAudioChannelCount(max(session.inputNumberOfChannels, 0))
        if isValid(sampleRate: sessionSampleRate, channelCount: sessionChannels),
           let fallback = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sessionSampleRate,
            channels: sessionChannels,
            interleaved: false
           ) {
            return fallback
        }

        throw TranscriptionError.unavailable
    }

    static func isValid(_ format: AVAudioFormat) -> Bool {
        isValid(sampleRate: format.sampleRate, channelCount: format.channelCount)
    }

    static func isValid(sampleRate: Double, channelCount: AVAudioChannelCount) -> Bool {
        sampleRate.isFinite && sampleRate > 0 && channelCount > 0
    }
}
