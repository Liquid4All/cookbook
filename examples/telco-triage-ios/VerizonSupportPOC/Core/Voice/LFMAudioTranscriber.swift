import Foundation
import AVFoundation
import LeapSDK
import os.log

/// On-device STT transcriber backed by LFM2.5-Audio-1.5B via LEAP SDK.
///
/// Activated when the "Voice Support Pack" is installed. Before install,
/// `VoiceCoordinator` falls back to `AppleSpeechTranscriber` so the mic
/// works without the ~1 GB audio bundle on disk.
///
/// The model is loaded lazily on first `startListening()`. LEAP's model
/// runner is cached across recording sessions for the lifetime of this
/// transcriber — a 1 GB model load is not something we want to repeat
/// between utterances.
///
/// ASR mode is driven entirely by the system prompt `"Perform ASR."`,
/// which is the documented transcription-only mode for LFM2.5-Audio
/// (sequential generation: audio-in, text-out, no TTS response). Any
/// `.audioSample` chunks that leak through are dropped defensively.
///
/// Not thread-safe across concurrent recording sessions — the
/// AVAudioEngine and captured-sample buffer are single-recording-at-a-time.
/// `VoiceCoordinator` serializes start/stop so this is fine at the UI
/// layer. `@MainActor` pins all state to the main queue, matching the
/// existing `VoiceCoordinator` isolation model.
@MainActor
public final class LFMAudioTranscriber: VoiceTranscriber {
    // MARK: - Constants

    /// HuggingFace repo hosting the LEAP bundle manifest + GGUFs.
    /// `Leap.load(model:quantization:)` does NOT resolve this model — the
    /// LEAP registry's `/api/edge-sdk/model-manifest` endpoint returns
    /// "Manifest does not exist" for every LFM2.5-family slug as of SDK
    /// 0.9.4 (their `gguf_repo_url` field is empty in the models index).
    /// We use the manifest-URL overload instead and point directly at HF.
    public static let huggingFaceRepo = "LiquidAI/LFM2.5-Audio-1.5B-GGUF-LEAP"

    /// Q4_0 balances disk (~1 GB) and quality. LibriSpeech-clean WER is
    /// 1.95% at Q4_0 per the model card — meaningfully better than
    /// Apple Speech on clean audio. Repo exposes manifests at
    /// `leap/{Q4_0,Q8_0,F16}.json`.
    public static let quantization = "Q4_0"

    /// Full URL of the LEAP manifest JSON on HF. The manifest lists the
    /// main GGUF + mmproj + vocoder via relative paths (`../*.gguf`),
    /// which the SDK resolves against this URL's directory.
    ///
    /// The force-unwrap is justified: all interpolated components are
    /// compile-time constants containing only URL-safe characters
    /// ([A-Za-z0-9._/-]). A failing `URL(string:)` here is not a
    /// runtime possibility — it would indicate a source-level typo in
    /// `huggingFaceRepo` or `quantization` and is caught by the
    /// `test_manifestURL_pointsAtHFBundle` drift guard.
    public static let manifestURL: URL = URL(
        string: "https://huggingface.co/\(huggingFaceRepo)/resolve/main/leap/\(quantization).json"
    )!

    /// Canonical ASR system prompt from the LFM2.5-Audio-1.5B-GGUF model
    /// card. Puts the model in sequential-generation mode — text-only
    /// output, no TTS response. Paired prompts are "Perform TTS." and
    /// "Respond with interleaved text and audio." which we do NOT want
    /// here.
    public static let systemPrompt = "Perform ASR."

    /// LFM2.5-Audio was trained on 16 kHz mono audio. AVAudioEngine's
    /// native input format is typically 44.1 or 48 kHz; we resample
    /// linearly before feeding the model. Linear is fine — the model
    /// is robust to basic decimation and this is a mobile STT path,
    /// not an audiophile recording.
    private static let targetSampleRate: Double = 16_000

    /// Short-utterance cap. Transcriptions rarely exceed ~100 tokens
    /// for phrase-length commands; 256 gives comfortable headroom for
    /// multi-sentence inputs without letting the model ramble.
    private static let maxOutputTokens: UInt32 = 256

    // MARK: - Dependencies

    private let logger = Logger(subsystem: "ai.liquid.demos.telcotriage", category: "LFMAudioTranscriber")
    private let audioEngine = AVAudioEngine()

    // MARK: - State

    private var modelRunner: (any ModelRunner)?
    private var continuation: AsyncStream<TranscriptionEvent>.Continuation?
    private var capturedSamples: [Float] = []
    private var captureSampleRate: Double = LFMAudioTranscriber.targetSampleRate
    private var sessionActive: Bool = false
    private var generationTask: Task<Void, Never>?

    // MARK: - Init

    public init() {}

    /// Release the LEAP model runner and any in-flight generation.
    /// Called when the audio specialist pack is uninstalled so the
    /// ~1 GB model is freed from memory rather than lingering until
    /// the transcriber instance is deallocated.
    ///
    /// CRITICAL: `ModelRunner.unload()` MUST be called before dropping
    /// the reference. The inference engine (C++ + Metal) allocates GPU
    /// command queues, memory-mapped GGUF files, and KV cache buffers
    /// that require ordered async teardown. Setting `modelRunner = nil`
    /// without `unload()` triggers use-after-free in the native deinit.
    public func releaseResources() async {
        generationTask?.cancel()
        generationTask = nil
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        deactivateSession()
        await modelRunner?.unload()
        modelRunner = nil
        continuation?.finish()
        continuation = nil
        capturedSamples.removeAll(keepingCapacity: false)
    }

    // MARK: - VoiceTranscriber

    public func startListening() async throws -> AsyncStream<TranscriptionEvent> {
        // Lazy-load on first use. Cache hit after SpecialistPackManager
        // install is near-instant; first-ever load (if the user
        // somehow bypassed the pack UI) triggers the 1 GB download,
        // which the progress UI would normally surface. We don't wire
        // a progress handler here because this path is not the user's
        // install path — the pack-gated `VoiceCoordinator` gates us.
        if modelRunner == nil {
            logger.info("Loading LFM2.5-Audio-1.5B (cache hit expected)")
            modelRunner = try await Leap.load(
                manifestURL: Self.manifestURL,
                options: Self.loadOptions()
            )
            logger.info("LFM2.5-Audio loaded")
        }

        try await requestMicrophonePermission()
        try configureAudioSession()

        capturedSamples.removeAll(keepingCapacity: true)

        let stream = AsyncStream<TranscriptionEvent> { cont in
            self.continuation = cont
        }

        let input = audioEngine.inputNode
        let recordingFormat = input.outputFormat(forBus: 0)
        captureSampleRate = recordingFormat.sampleRate
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 4_096, format: recordingFormat) { [weak self] buffer, _ in
            // Buffer callbacks fire off the main actor — hop back before
            // touching `capturedSamples`.
            Task { @MainActor [weak self] in
                self?.appendSamples(buffer)
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
        logger.info("Recording started (rate=\(recordingFormat.sampleRate))")

        return stream
    }

    public func stopListening() async {
        guard audioEngine.isRunning || !capturedSamples.isEmpty else {
            continuation?.finish()
            continuation = nil
            deactivateSession()
            return
        }

        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        deactivateSession()

        let samples = capturedSamples
        capturedSamples.removeAll(keepingCapacity: false)

        guard !samples.isEmpty else {
            logger.warning("stopListening called with zero samples")
            continuation?.yield(.final(""))
            continuation?.finish()
            continuation = nil
            return
        }

        guard let runner = modelRunner else {
            continuation?.yield(.error("audio model not loaded"))
            continuation?.finish()
            continuation = nil
            return
        }

        // Resample + wrap in a ChatMessage + kick off streaming generation.
        // Run on a detached task so we can surface partial text as chunks
        // arrive without blocking the caller of stopListening().
        let samples16k = Self.resample(samples, fromRate: captureSampleRate, toRate: Self.targetSampleRate)
        generationTask = Task { [weak self, logger] in
            guard let self else { return }
            await self.runTranscription(runner: runner, samples16k: samples16k, logger: logger)
        }
    }

    // MARK: - Transcription

    private func runTranscription(runner: any ModelRunner, samples16k: [Float], logger: Logger) async {
        let audioContent = ChatMessageContent.fromFloatSamples(
            samples16k,
            sampleRate: Int(Self.targetSampleRate)
        )
        // Fully-qualified — `VerizonSupportPOC.Features.Chat.ChatMessage`
        // is a local display type that otherwise shadows this one.
        let message = LeapSDK.ChatMessage(role: .user, content: [audioContent])

        let conversation = runner.createConversation(systemPrompt: Self.systemPrompt)
        let options = GenerationOptions(
            temperature: 0.0,
            resetHistory: true,
            maxOutputTokens: Self.maxOutputTokens
        )

        let stream = conversation.generateResponse(message: message, generationOptions: options)

        var accumulated = ""
        do {
            for try await response in stream {
                switch response {
                case .chunk(let delta):
                    accumulated += delta
                    continuation?.yield(.partial(accumulated.trimmingCharacters(in: .whitespaces)))
                case .complete:
                    let finalText = accumulated.trimmingCharacters(in: .whitespacesAndNewlines)
                    logger.info("Transcription complete (\(finalText.count) chars)")
                    continuation?.yield(.final(finalText))
                case .reasoningChunk, .audioSample, .functionCall:
                    // ASR mode should not produce these; defensively drop
                    // so a prompt drift (e.g. user swaps system prompt)
                    // doesn't leak thinking traces into the transcript.
                    continue
                }
            }
        } catch {
            logger.error("Transcription failed: \(error.localizedDescription)")
            continuation?.yield(.error("transcription failed: \(error.localizedDescription)"))
        }

        continuation?.finish()
        continuation = nil
    }

    // MARK: - Audio capture

    private func appendSamples(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameCount = Int(buffer.frameLength)
        let bufferPointer = UnsafeBufferPointer(start: channelData, count: frameCount)
        capturedSamples.append(contentsOf: bufferPointer)
    }

    private func requestMicrophonePermission() async throws {
        let granted = await AVAudioApplication.requestRecordPermission()
        guard granted else { throw TranscriptionError.permissionDenied }
    }

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        sessionActive = true
    }

    private func deactivateSession() {
        guard sessionActive else { return }
        sessionActive = false
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
    }

    // MARK: - Helpers

    /// Linear resample. Good enough for mobile STT; LFM2.5-Audio is
    /// robust to basic decimation (the `llama-liquid-audio-cli` WAV
    /// path does essentially the same thing internally for non-16k
    /// inputs).
    ///
    /// `internal` + `nonisolated` so unit tests can exercise it
    /// without spinning up AVAudioEngine or LEAP, and without needing
    /// MainActor isolation (pure function over `[Float]`).
    internal nonisolated static func resample(_ samples: [Float], fromRate: Double, toRate: Double) -> [Float] {
        guard fromRate != toRate, !samples.isEmpty else { return samples }
        let ratio = fromRate / toRate
        let outputCount = max(1, Int(Double(samples.count) / ratio))
        var output = [Float](repeating: 0, count: outputCount)
        let lastIdx = samples.count - 1
        for i in 0..<outputCount {
            let srcIdx = Double(i) * ratio
            let lo = min(Int(srcIdx), lastIdx)
            let hi = min(lo + 1, lastIdx)
            let frac = Float(srcIdx - Double(lo))
            output[i] = samples[lo] * (1 - frac) + samples[hi] * frac
        }
        return output
    }

    /// LEAP load options. Forces CPU inference on the simulator — iOS
    /// Simulator's MTL0 reports "0 MiB free" for GGUF workloads and
    /// GPU offload silently produces garbage tokens (token id 0 every
    /// sample → all `<|pad|>`). Same root cause as BUG-022 in the
    /// mattt/llama.swift path; same fix.
    private static func loadOptions() -> LiquidInferenceEngineManifestOptions {
        #if targetEnvironment(simulator)
        return LiquidInferenceEngineManifestOptions(nGpuLayers: 0)
        #else
        return LiquidInferenceEngineManifestOptions()
        #endif
    }
}
