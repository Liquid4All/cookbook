import XCTest
@testable import VerizonSupportPOC

/// Deterministic in-memory transcriber. Emits a canned script of
/// TranscriptionEvents on start.
private actor ScriptedTranscriber: VoiceTranscriber {
    private let events: [TranscriptionEvent]
    private var stopped = false

    init(events: [TranscriptionEvent]) {
        self.events = events
    }

    func startListening() async throws -> AsyncStream<TranscriptionEvent> {
        AsyncStream { continuation in
            Task {
                for event in events {
                    try? await Task.sleep(nanoseconds: 10_000_000)  // 10ms between events
                    continuation.yield(event)
                }
                continuation.finish()
            }
        }
    }

    func stopListening() async {
        stopped = true
    }
}

/// Transcriber that stays "listening" indefinitely — the stream never
/// yields and never finishes. Used to test mid-session lifecycle events
/// like pack uninstall while voice is active.
private actor NeverEndingTranscriber: VoiceTranscriber {
    func startListening() async throws -> AsyncStream<TranscriptionEvent> {
        AsyncStream { _ in /* never yields, never finishes */ }
    }

    func stopListening() async {}
}

@MainActor
final class VoiceCoordinatorTests: XCTestCase {
    private var packManager: SpecialistPackManager!

    override func setUp() {
        super.setUp()
        let defaults = UserDefaults(suiteName: "VoiceCoordinatorTests-\(UUID().uuidString)")!
        packManager = SpecialistPackManager(defaults: defaults)
    }

    func test_start_emitsPartialThenFinal_andUpdatesState() async throws {
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in
                ScriptedTranscriber(events: [
                    .partial("restart"),
                    .partial("restart my"),
                    .final("restart my router"),
                ])
            }
        )

        XCTAssertEqual(coordinator.state, .idle)
        coordinator.start()
        XCTAssertTrue(coordinator.isListening)

        // Wait for the scripted events to drain
        try await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertFalse(coordinator.isListening)
        if case .finalized(let text) = coordinator.state {
            XCTAssertEqual(text, "restart my router")
        } else {
            XCTFail("expected .finalized, got \(coordinator.state)")
        }
    }

    func test_error_movesToErrorState() async throws {
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in
                ScriptedTranscriber(events: [.error("mic busy")])
            }
        )
        coordinator.start()
        try await Task.sleep(nanoseconds: 100_000_000)

        if case .error(let msg) = coordinator.state {
            XCTAssertEqual(msg, "mic busy")
        } else {
            XCTFail("expected .error, got \(coordinator.state)")
        }
        XCTAssertFalse(coordinator.isListening)
    }

    func test_consumeFinal_returnsTextAndResetsState() async throws {
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in
                ScriptedTranscriber(events: [.final("test")])
            }
        )
        coordinator.start()
        try await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(coordinator.consumeFinal(), "test")
        XCTAssertEqual(coordinator.state, .idle)
    }

    func test_consumeFinal_whenNotFinalized_returnsNil() {
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in
                ScriptedTranscriber(events: [])
            }
        )
        XCTAssertNil(coordinator.consumeFinal())
    }

    func test_usingPack_reflectsInstalledState() {
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in ScriptedTranscriber(events: []) }
        )
        XCTAssertFalse(coordinator.usingPack)

        coordinator.start()
        XCTAssertFalse(coordinator.usingPack, "pack not installed → false")
    }

    func test_start_whileListening_isNoop() {
        var factoryCalls = 0
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in
                factoryCalls += 1
                return ScriptedTranscriber(events: [])
            }
        )
        coordinator.start()
        coordinator.start()
        XCTAssertEqual(factoryCalls, 1)
    }

    func test_factory_receivesInstalledFalse_whenPackNotInstalled() {
        var sawFlag: Bool?
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { installed in
                sawFlag = installed
                return ScriptedTranscriber(events: [])
            }
        )
        coordinator.start()
        XCTAssertEqual(sawFlag, false)
    }

    /// Audio pack forced `.available` so lifecycle tests can run. The
    /// production pack is coming-soon today (LEAP 0.9.4 ie_llamacpp
    /// crash on LFM2.5-Audio) — voice input falls back to Apple Speech.
    /// These tests exercise the install/auto-stop machinery for when
    /// the pack is eventually re-enabled.
    private static let availableAudio: SpecialistPack =
        SpecialistPack.audio.with(availability: .available)

    func test_uninstallAudioPack_whileListening_autoStopsVoice() async throws {
        // Install audio pack via fake installer
        let defaults = UserDefaults(suiteName: "uninstall-auto-stop-\(UUID().uuidString)")!
        let mgr = SpecialistPackManager(
            packs: [Self.availableAudio, SpecialistPack.vision],
            defaults: defaults,
            audioInstaller: { p in p(1.0) },
            audioRemover: { /* no-op: disk cleanup not under test */ }
        )
        mgr.install(Self.availableAudio)
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(mgr.isInstalled(Self.availableAudio.id), "precondition")

        let coordinator = VoiceCoordinator(
            packManager: mgr,
            transcriberFactory: { _ in
                // Stream that never finishes — simulates active listening
                NeverEndingTranscriber()
            }
        )

        coordinator.start()
        XCTAssertTrue(coordinator.isListening)
        XCTAssertTrue(coordinator.usingPack)

        // Remove the audio pack while listening
        await mgr.uninstall(Self.availableAudio)

        // Give the Combine sink + MainActor hop time to fire
        try await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertFalse(coordinator.isListening, "voice should auto-stop when audio pack is removed")
        XCTAssertFalse(coordinator.usingPack)
    }

    func test_factory_receivesInstalledTrue_whenPackInstalled() async {
        // Inject a fake audio installer so install() doesn't hit LEAP.
        let defaults = UserDefaults(suiteName: "factory-installed-\(UUID().uuidString)")!
        let installedManager = SpecialistPackManager(
            packs: [Self.availableAudio, SpecialistPack.vision],
            defaults: defaults,
            audioInstaller: { progress in
                progress(0.5)
                progress(1.0)
            }
        )
        installedManager.install(Self.availableAudio)
        // Drain the install task — the fake installer returns almost
        // immediately, but still crosses an actor hop.
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(installedManager.isInstalled(Self.availableAudio.id),
                      "precondition: audio pack marked installed")

        var sawFlag: Bool?
        let coordinator = VoiceCoordinator(
            packManager: installedManager,
            transcriberFactory: { installed in
                sawFlag = installed
                return ScriptedTranscriber(events: [])
            }
        )
        coordinator.start()
        XCTAssertEqual(sawFlag, true)
        XCTAssertTrue(coordinator.usingPack)
    }

    func test_stop_preservesPartialAsFinalized() async throws {
        // Bug this guards: `AppleSpeechTranscriber.stopListening()`
        // finishes the AsyncStream continuation immediately, so any
        // in-progress partial is lost — `.final` never fires, ChatView's
        // onChange never populates the text field, user loses their
        // transcription on Stop. Fix: capture the partial in stop()
        // before teardown and transition state to .finalized.
        //
        // Uses NeverEndingTranscriber so the stream stays open with the
        // last partial held in state. Then we manually set state to
        // simulate a partial having been received, and stop().
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in
                ScriptedTranscriber(events: [
                    .partial("restart my"),
                    .partial("restart my router"),
                ])
            }
        )

        coordinator.start()
        // Let the partials propagate
        try await Task.sleep(nanoseconds: 60_000_000)

        guard case .listening(let partial) = coordinator.state else {
            XCTFail("expected .listening state, got \(coordinator.state)")
            return
        }
        XCTAssertFalse(partial.isEmpty, "partial should have accumulated from scripted events")

        // User taps Stop — must transition to .finalized(partial)
        await coordinator.stop()

        guard case .finalized(let finalText) = coordinator.state else {
            XCTFail("stop() must transition .listening(partial) → .finalized(partial); got \(coordinator.state)")
            return
        }
        XCTAssertFalse(finalText.isEmpty, "finalized text must not be empty")
        XCTAssertFalse(coordinator.isListening)
    }

    func test_stop_whenSilent_goesIdle_notFinalized() async throws {
        // User taps mic, says nothing, taps Stop → don't send an empty
        // message. Stop() should transition .listening("") → .idle.
        let coordinator = VoiceCoordinator(
            packManager: packManager,
            transcriberFactory: { _ in
                NeverEndingTranscriber()
            }
        )

        coordinator.start()
        try await Task.sleep(nanoseconds: 30_000_000)
        XCTAssertEqual(coordinator.state, .listening(partial: ""))

        await coordinator.stop()
        XCTAssertEqual(coordinator.state, .idle,
                       "empty-partial stop must NOT transition to .finalized('') — that would send an empty message")
    }

    func test_comingSoonAudioPack_routesThroughAppleSpeechFallback() async {
        // End-to-end: with the production (coming-soon) pack config,
        // VoiceCoordinator.start() must see isPackInstalled == false
        // and route through the Apple Speech transcriber. This is the
        // contract that lets voice input keep working while the LFM
        // audio pack is blocked.
        let defaults = UserDefaults(suiteName: "apple-fallback-\(UUID().uuidString)")!
        let productionMgr = SpecialistPackManager(defaults: defaults)

        var sawFlag: Bool?
        let coordinator = VoiceCoordinator(
            packManager: productionMgr,
            transcriberFactory: { installed in
                sawFlag = installed
                return ScriptedTranscriber(events: [])
            }
        )
        coordinator.start()
        XCTAssertEqual(sawFlag, false,
                       "coming-soon pack must present as not-installed so the factory picks Apple Speech")
        XCTAssertFalse(coordinator.usingPack,
                       "usingPack must be false when routing through Apple Speech fallback")
    }
}
