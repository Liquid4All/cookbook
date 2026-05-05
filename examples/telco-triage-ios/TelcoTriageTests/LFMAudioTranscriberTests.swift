import XCTest
@testable import TelcoTriage

/// Unit tests for pure helpers on `LFMAudioTranscriber`.
///
/// Full on-device inference is covered by a separate, opt-in smoke
/// test (gated on the audio pack being downloaded) — those are too
/// slow / too network-heavy to run in the default test plan.
final class LFMAudioTranscriberTests: XCTestCase {

    func test_resample_passesThroughWhenRateMatches() {
        let input: [Float] = [0.1, 0.2, 0.3, 0.4]
        let output = LFMAudioTranscriber.resample(input, fromRate: 16_000, toRate: 16_000)
        XCTAssertEqual(output, input)
    }

    func test_resample_downsamples48kTo16k_preservesLength_ratio() {
        // 48 kHz → 16 kHz = 3:1 decimation. A 3-second buffer of
        // 48k samples (144k floats) should land near 48k samples
        // at 16 kHz.
        let sampleCount = 48_000 * 3
        let input = (0..<sampleCount).map { Float($0) / Float(sampleCount) }
        let output = LFMAudioTranscriber.resample(input, fromRate: 48_000, toRate: 16_000)
        XCTAssertEqual(output.count, 48_000, "expected 16k samples × 3s = 48k, got \(output.count)")
    }

    func test_resample_upsamples8kTo16k_preservesLength_ratio() {
        // Defensive: even though we never upsample in production
        // (AVAudioEngine always gives us ≥ 16 kHz on iOS), the
        // algorithm should behave reasonably.
        let sampleCount = 8_000
        let input = [Float](repeating: 0.5, count: sampleCount)
        let output = LFMAudioTranscriber.resample(input, fromRate: 8_000, toRate: 16_000)
        XCTAssertEqual(output.count, 16_000)
        // All constant → all output should equal the constant after
        // linear interpolation between two identical neighbors.
        XCTAssertTrue(output.allSatisfy { abs($0 - 0.5) < 1e-6 })
    }

    func test_resample_emptyInput_returnsEmpty() {
        XCTAssertTrue(LFMAudioTranscriber.resample([], fromRate: 48_000, toRate: 16_000).isEmpty)
    }

    func test_systemPrompt_isPerformASR_verbatim() {
        // Gate: changing the ASR prompt away from the documented
        // LFM2.5-Audio-1.5B-GGUF model-card value ("Perform ASR.") will
        // flip the model out of transcription-only mode and start
        // emitting audio chunks. Guard the exact string.
        XCTAssertEqual(LFMAudioTranscriber.systemPrompt, "Perform ASR.")
    }

    func test_manifestURL_pointsAtHFBundle() {
        // Drift guard: the LEAP registry slug path (model:quantization:)
        // does NOT resolve for LFM2.5-Audio — Leap.load throws
        // loadModelFailure (case 0) because the registry has an empty
        // gguf_repo_url for it. We must stay on the manifest-URL path
        // pointing directly at the HF-hosted LEAP bundle.
        XCTAssertEqual(LFMAudioTranscriber.huggingFaceRepo, "LiquidAI/LFM2.5-Audio-1.5B-GGUF-LEAP")
        XCTAssertEqual(LFMAudioTranscriber.quantization, "Q4_0")
        XCTAssertEqual(
            LFMAudioTranscriber.manifestURL.absoluteString,
            "https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF-LEAP/resolve/main/leap/Q4_0.json"
        )
    }

    func test_audioInputTapFormat_rejectsZeroRateOrChannels() {
        XCTAssertFalse(AudioTapInstaller.isValid(sampleRate: 0, channelCount: 1))
        XCTAssertFalse(AudioTapInstaller.isValid(sampleRate: 16_000, channelCount: 0))
        XCTAssertTrue(AudioTapInstaller.isValid(sampleRate: 16_000, channelCount: 1))
    }
}
