import XCTest
@testable import TelcoTriage

@MainActor
final class SpecialistPackManagerTests: XCTestCase {
    private var manager: SpecialistPackManager!
    private var defaults: UserDefaults!

    /// Audio pack forced `.available` so lifecycle tests can run. The
    /// production pack is coming-soon today (LEAP 0.9.4 ie_llamacpp
    /// crash) — but the install/uninstall machinery still needs
    /// coverage for when the pack is eventually re-enabled.
    private let availableAudio: SpecialistPack =
        SpecialistPack.audio.with(availability: .available)

    override func setUp() {
        super.setUp()
        defaults = UserDefaults(suiteName: "SpecialistPackManagerTests-\(UUID().uuidString)")!
        manager = SpecialistPackManager(defaults: defaults)
    }

    /// Build a manager with audio forced `.available`. Use for tests
    /// that exercise the install lifecycle.
    private func makeLifecycleManager(
        defaults: UserDefaults,
        audioInstaller: @escaping SpecialistPackManager.AudioInstaller,
        audioRemover: @escaping SpecialistPackManager.AudioRemover = { }
    ) -> SpecialistPackManager {
        SpecialistPackManager(
            packs: [availableAudio, SpecialistPack.vision],
            defaults: defaults,
            audioInstaller: audioInstaller,
            audioRemover: audioRemover
        )
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: defaults.dictionaryRepresentation().description)
        super.tearDown()
    }

    func test_initialState_allPacksNotInstalled() {
        for pack in SpecialistPack.all {
            XCTAssertFalse(manager.isInstalled(pack.id))
        }
    }

    func test_install_refusesComingSoonPacks() {
        // Both audio and vision are coming-soon today. install() must
        // be a no-op so UI deep links, debug menus, or future
        // automation can't accidentally trigger a crash-prone download.
        for pack in SpecialistPack.all {
            XCTAssertFalse(pack.isAvailable, "precondition: all packs currently coming-soon")
            manager.install(pack)
            XCTAssertEqual(manager.state(for: pack.id), .notInstalled,
                           "install() on coming-soon \(pack.id) must not flip to downloading")
        }
    }

    func test_comingSoonPack_cannotStartInstalled_evenWithStalePersistence() {
        // A prior build may have persisted audio as installed. When
        // audio is marked coming-soon, manager MUST force it back to
        // .notInstalled AND scrub the persisted flag so the downgrade
        // survives relaunches. Without this, VoiceCoordinator would
        // instantiate LFMAudioTranscriber and crash on first use.
        let staleDefaults = UserDefaults(suiteName: "stale-\(UUID().uuidString)")!
        staleDefaults.set(["audio", "vision"], forKey: "installedSpecialistPackIDs")

        let recovered = SpecialistPackManager(defaults: staleDefaults)
        XCTAssertFalse(recovered.isInstalled("audio"))
        XCTAssertFalse(recovered.isInstalled("vision"))

        // Persistence must also be scrubbed, not just in-memory state —
        // a second construction without the scrub would re-surface the
        // stale .installed flag.
        let second = SpecialistPackManager(defaults: staleDefaults)
        XCTAssertFalse(second.isInstalled("audio"))
        XCTAssertFalse(second.isInstalled("vision"))
        let persisted = staleDefaults.stringArray(forKey: "installedSpecialistPackIDs") ?? []
        XCTAssertTrue(persisted.isEmpty, "coming-soon packs must be scrubbed from persistence")
    }

    func test_packCatalog_includesAudioAndVision() {
        XCTAssertNotNil(manager.pack(for: .voice))
        XCTAssertNotNil(manager.pack(for: .vision))
        XCTAssertEqual(manager.pack(for: .voice)?.id, "audio")
        XCTAssertEqual(manager.pack(for: .vision)?.id, "vision")
    }

    func test_audioInstall_bridgesInjectedProgress_andPersistsInstalled() async {
        // A fake installer replaces the LEAP call so tests are
        // hermetic. We drive a deterministic progress curve and assert
        // it's faithfully mirrored into pack state.
        var observedProgress: [Double] = []
        let fakeManager = makeLifecycleManager(
            defaults: defaults,
            audioInstaller: { progress in
                for step in [0.25, 0.5, 0.75, 1.0] {
                    progress(step)
                    observedProgress.append(step)
                    try? await Task.sleep(nanoseconds: 5_000_000)
                }
            }
        )

        fakeManager.install(availableAudio)

        // Poll until the installer's scripted progress drains.
        for _ in 0..<50 {
            if case .installed = fakeManager.state(for: availableAudio.id) { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }

        XCTAssertEqual(observedProgress, [0.25, 0.5, 0.75, 1.0])
        if case .installed = fakeManager.state(for: availableAudio.id) {
            // expected
        } else {
            XCTFail("audio pack did not reach .installed, got \(fakeManager.state(for: availableAudio.id))")
        }

        // Persisted across manager instances via UserDefaults — but
        // only when the pack is still .available. If we constructed
        // with the production (coming-soon) pack, the init-time
        // scrub would wipe the flag — that's the coming-soon test's
        // job. Here we re-inject the .available pack to verify
        // persistence per se.
        let second = SpecialistPackManager(
            packs: [availableAudio, SpecialistPack.vision],
            defaults: defaults
        )
        XCTAssertTrue(second.isInstalled(availableAudio.id))
    }

    func test_audioInstall_surfacesErrorState_onInstallerThrow() async {
        struct FakeError: Error, LocalizedError {
            var errorDescription: String? { "fake leap failure" }
        }

        let fakeManager = makeLifecycleManager(
            defaults: defaults,
            audioInstaller: { _ in throw FakeError() }
        )
        fakeManager.install(availableAudio)

        for _ in 0..<50 {
            if case .error = fakeManager.state(for: availableAudio.id) { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }

        if case .error(let msg) = fakeManager.state(for: availableAudio.id) {
            XCTAssertTrue(msg.contains("fake leap failure"))
        } else {
            XCTFail("expected .error, got \(fakeManager.state(for: availableAudio.id))")
        }
    }

    func test_uninstall_cancelsActiveDownload() async throws {
        var installerStarted = false
        let fakeManager = makeLifecycleManager(
            defaults: defaults,
            audioInstaller: { progress in
                installerStarted = true
                progress(0.25)
                // Simulate a slow download — long enough for uninstall to fire
                try await Task.sleep(nanoseconds: 2_000_000_000)
                progress(1.0)
            }
        )

        fakeManager.install(availableAudio)
        // Let the download start
        try await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertTrue(installerStarted, "installer should have started")

        // Uninstall mid-download
        await fakeManager.uninstall(availableAudio)
        XCTAssertEqual(fakeManager.state(for: availableAudio.id), .notInstalled,
                       "state should be .notInstalled immediately after uninstall")

        // Wait past where the download would have finished
        try await Task.sleep(nanoseconds: 300_000_000)

        // Verify state wasn't overwritten back to .installed by the cancelled task
        XCTAssertEqual(fakeManager.state(for: availableAudio.id), .notInstalled,
                       "cancelled download must not overwrite state back to .installed")
    }

    func test_uninstall_callsAudioRemover_afterInstall() async {
        var removerCallCount = 0
        let fakeManager = makeLifecycleManager(
            defaults: defaults,
            audioInstaller: { progress in progress(1.0) },
            audioRemover: { removerCallCount += 1 }
        )

        fakeManager.install(availableAudio)
        for _ in 0..<50 {
            if case .installed = fakeManager.state(for: availableAudio.id) { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }

        await fakeManager.uninstall(availableAudio)
        XCTAssertEqual(removerCallCount, 1,
                       "uninstall must actually call the disk-reclaiming remover — not just flip state")
        XCTAssertEqual(fakeManager.state(for: availableAudio.id), .notInstalled)
    }

    func test_uninstall_releasesRunnerBeforeDeletingBytes() async {
        // Order guard: LEAP mmaps GGUFs, so we must unload the live
        // runner BEFORE removing files. This test pins the ordering
        // via a shared event log — any future refactor that reorders
        // these steps (e.g. "nuke bytes then stop") fails here.
        var events: [String] = []
        let fakeManager = makeLifecycleManager(
            defaults: defaults,
            audioInstaller: { progress in progress(1.0) },
            audioRemover: { events.append("remove") }
        )
        fakeManager.setPreUninstallHook { _ in
            // Simulate voice teardown taking a real async hop — the
            // test must still observe "release" before "remove".
            try? await Task.sleep(nanoseconds: 20_000_000)
            events.append("release")
        }

        fakeManager.install(availableAudio)
        for _ in 0..<50 {
            if case .installed = fakeManager.state(for: availableAudio.id) { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }

        await fakeManager.uninstall(availableAudio)
        XCTAssertEqual(events, ["release", "remove"],
                       "pre-uninstall hook must complete before audio remover runs")
    }

    func test_install_cancelReinstall_doesNotStompNewerState() async throws {
        // Race guard (code review CRITICAL #2): a slow installer for
        // install-A is cancelled by install-B starting. If A's task
        // eventually completes AFTER B has taken over, A must NOT
        // transition state back to .installed — otherwise B's in-flight
        // .downloading (or eventual .error/.notInstalled) gets stomped.
        //
        // The generation-token check gates every state write on the
        // captured generation still being current. Verified here with
        // a continuation that gives us precise control over when A
        // "finishes" its installer work.
        //
        // Uses `@unchecked Sendable` box to pass a continuation across
        // the MainActor/closure boundary — the test serializes access
        // so the box is safe.
        final class ContinuationBox: @unchecked Sendable {
            var cont: CheckedContinuation<Void, Never>?
        }
        let box = ContinuationBox()

        var callNumber = 0
        var bStarted = false
        let fakeManager = makeLifecycleManager(
            defaults: defaults,
            audioInstaller: { progress in
                callNumber += 1
                if callNumber == 1 {
                    // Installer A: suspend until the test releases us.
                    progress(0.5)
                    await withCheckedContinuation { cont in
                        box.cont = cont
                    }
                    progress(1.0)
                } else {
                    // Installer B: report progress, then hang so the
                    // state stays in .downloading while we release A.
                    bStarted = true
                    progress(0.3)
                    try? await Task.sleep(nanoseconds: 500_000_000)
                }
            }
        )

        // 1. install A, wait for it to suspend inside the continuation
        fakeManager.install(availableAudio)
        for _ in 0..<50 {
            if box.cont != nil { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertNotNil(box.cont, "installer A should have reached the suspension point")

        // 2. cancel A (nils its generation), then install B
        fakeManager.cancelDownload(availableAudio)
        fakeManager.install(availableAudio)

        // 3. wait for B to report progress (.downloading(0.3))
        for _ in 0..<50 {
            if bStarted { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(bStarted, "installer B should have run")
        if case .downloading(let p) = fakeManager.state(for: availableAudio.id) {
            XCTAssertEqual(p, 0.3, accuracy: 0.01)
        } else {
            XCTFail("expected B to be .downloading(0.3), got \(fakeManager.state(for: availableAudio.id))")
        }

        // 4. release A. Without the generation guard, A's installAudio
        //    wakes up, checks Task.isCancelled (true → sets .notInstalled)
        //    OR if not cancelled sets .installed. Either way stomps B.
        //    With the guard, A's post-await continuation bails silently.
        box.cont?.resume()
        box.cont = nil

        // Give A's continuation time to wake and (ideally) bail
        try await Task.sleep(nanoseconds: 100_000_000)

        // 5. state must still reflect B's progress, not A's completion
        if case .downloading(let p) = fakeManager.state(for: availableAudio.id) {
            XCTAssertEqual(p, 0.3, accuracy: 0.01,
                           "B's .downloading progress must survive A's stale completion")
        } else {
            XCTFail("stale installer A clobbered B — got \(fakeManager.state(for: availableAudio.id))")
        }
    }

    func test_uninstall_toleratesRemoverThrow_stateStillFlips() async {
        // If LEAP's remover errors (e.g. a sibling was already gc'd by
        // the OS), we still want the user's intent — "I tapped Remove"
        // — to land in state. Otherwise the pack is permanently stuck
        // showing "Installed" with no way to retry.
        struct FakeRemoveError: Error {}
        let fakeManager = makeLifecycleManager(
            defaults: defaults,
            audioInstaller: { progress in progress(1.0) },
            audioRemover: { throw FakeRemoveError() }
        )

        fakeManager.install(availableAudio)
        for _ in 0..<50 {
            if case .installed = fakeManager.state(for: availableAudio.id) { break }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }

        await fakeManager.uninstall(availableAudio)
        XCTAssertEqual(fakeManager.state(for: availableAudio.id), .notInstalled)
    }

    func test_audioPack_metadata_reflectsLeapBundle() {
        // Guardrail: the LEAP-bundled size + repo are what ships to
        // users. If someone reverts to the pre-LEAP 1.5 GB claim or
        // the non-LEAP HF repo, this test fires.
        let pack = SpecialistPack.audio
        XCTAssertEqual(pack.modelRepo, "LiquidAI/LFM2.5-Audio-1.5B-GGUF-LEAP")
        XCTAssertLessThan(pack.downloadSizeBytes, 1_500_000_000,
                          "Q4_0 LEAP bundle is ~1.065 GB; 1.5 GB claim was pre-LEAP")
        XCTAssertGreaterThan(pack.downloadSizeBytes, 900_000_000)
    }
}
