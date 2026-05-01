import Foundation
import Combine
import LeapSDK
import LeapModelDownloader
import os.log

/// Manages the lifecycle of optional specialist packs — tracks install
/// state, drives downloads, and persists which packs are installed so
/// the UI survives relaunches.
///
/// Audio pack (LFM2.5-Audio-1.5B) installs for real via the LEAP SDK:
/// `Leap.load(manifestURL:downloadProgressHandler:)` handles the GGUF +
/// mmproj + vocoder download and cache. We use the manifest-URL overload
/// rather than the model-slug overload because the LEAP registry returns
/// "Manifest does not exist" for LFM2.5-family slugs as of SDK 0.9.4 —
/// the canonical manifest lives at HF (`LFM2.5-Audio-1.5B-GGUF-LEAP`).
/// Progress is bridged into our existing `.downloading(progress:)` state
/// so the pack card UI works unchanged. Vision pack still simulates
/// pending its own real integration (Phase B).
@MainActor
public final class SpecialistPackManager: ObservableObject {
    @Published public private(set) var states: [String: SpecialistPackState]
    public let packs: [SpecialistPack]

    private let persistenceKey = "installedSpecialistPackIDs"
    private let defaults: UserDefaults
    private let logger = Logger(subsystem: "ai.liquid.demos.telcotriage", category: "SpecialistPacks")
    private var downloadTasks: [String: Task<Void, Never>] = [:]

    /// Hook so tests can swap the real LEAP loader for a fake that
    /// plays out a scripted progress curve without hitting the network.
    /// The hook returns `Void` because we discard the runner immediately
    /// — `LFMAudioTranscriber` re-loads from cache when voice input
    /// starts, keeping install and use concerns separate.
    public typealias AudioInstaller = @MainActor (
        _ progress: @escaping @MainActor (Double) -> Void
    ) async throws -> Void

    /// Hook called BEFORE disk deletion so the app can release any
    /// in-memory resources (LEAP ModelRunner, Metal command queues,
    /// mmap'd GGUFs) that point at the cached bundle. LEAP mmaps the
    /// GGUFs — deleting them while a runner is live is use-after-free.
    /// Wired in `AppState` to await `voice.stop()`.
    public typealias PreUninstallHook = @MainActor (SpecialistPack) async -> Void

    /// Hook that actually reclaims disk for a given pack. Default uses
    /// `LeapModelDownloader.ModelDownloader.removeModel(fromManifestURL:)`
    /// which walks the manifest and unlinks every downloaded sibling.
    /// Injectable so tests can observe that it was called without
    /// needing real bytes on disk.
    public typealias AudioRemover = @MainActor () async throws -> Void

    private let audioInstaller: AudioInstaller
    private let audioRemover: AudioRemover
    private var preUninstallHook: PreUninstallHook?

    /// Per-install ownership token. `install()` stamps a fresh UUID
    /// before spawning the task; `installAudio`/`simulateDownload`
    /// only transition state if their captured token still matches.
    /// Protects against install → cancel → re-install races where a
    /// stale task would otherwise stomp a newer one's state. @MainActor
    /// serialization makes the check-then-act atomic.
    private var installGeneration: [String: UUID] = [:]

    public init(
        packs: [SpecialistPack] = SpecialistPack.all,
        defaults: UserDefaults = .standard,
        audioInstaller: AudioInstaller? = nil,
        audioRemover: AudioRemover? = nil
    ) {
        self.packs = packs
        self.defaults = defaults
        self.audioInstaller = audioInstaller ?? Self.defaultLeapInstaller
        self.audioRemover = audioRemover ?? Self.defaultLeapRemover

        let persistedInstalled = Set(defaults.stringArray(forKey: persistenceKey) ?? [])
        var initialStates: [String: SpecialistPackState] = [:]
        var prunedPersisted = persistedInstalled
        for pack in packs {
            // Coming-soon packs must never start "installed" — even if
            // a prior build persisted that state, the on-disk bundle
            // may be the one that crashes ie_llamacpp during load.
            // Force .notInstalled AND scrub the UserDefaults entry so
            // the downgrade is permanent until the pack is flipped
            // back to .available.
            if !pack.isAvailable {
                initialStates[pack.id] = .notInstalled
                prunedPersisted.remove(pack.id)
                continue
            }
            if persistedInstalled.contains(pack.id) {
                initialStates[pack.id] = .installed(installedAt: Date())
            } else {
                initialStates[pack.id] = .notInstalled
            }
        }
        self.states = initialStates
        if prunedPersisted != persistedInstalled {
            defaults.set(Array(prunedPersisted), forKey: persistenceKey)
        }
    }

    /// Register a pre-uninstall hook after construction. Needed because
    /// `VoiceCoordinator` requires `packManager` in its init, so we
    /// can't pass a voice-releasing closure through `init`.
    public func setPreUninstallHook(_ hook: @escaping PreUninstallHook) {
        self.preUninstallHook = hook
    }

    // MARK: - Queries

    public func state(for packID: String) -> SpecialistPackState {
        states[packID] ?? .notInstalled
    }

    public func isInstalled(_ packID: String) -> Bool {
        if case .installed = state(for: packID) { return true }
        return false
    }

    public func pack(for capability: SpecialistPack.Capability) -> SpecialistPack? {
        packs.first { $0.capability == capability }
    }

    /// Whether LEAP reports a complete cache for this pack. Used by the
    /// UI to distinguish a fresh install (download + verify) from a
    /// warm reinstall (instant cache hit). Voice only for now — the
    /// vision pack is still simulated so there is no cache to probe.
    public func isCached(_ pack: SpecialistPack) -> Bool {
        guard pack.capability == .voice else { return false }
        let downloader = ModelDownloader()
        return downloader.queryStatus(LFMAudioTranscriber.manifestURL) == .downloaded
    }

    // MARK: - Lifecycle

    public func install(_ pack: SpecialistPack) {
        guard downloadTasks[pack.id] == nil else { return }
        // Coming-soon packs can't be installed. The UI hides the
        // Download button, but guard here too so a programmatic call
        // (deep link, debug menu, future automation) can't bypass it.
        guard pack.isAvailable else {
            logger.info("install() ignored — pack \(pack.id, privacy: .public) is not yet available")
            return
        }
        states[pack.id] = .downloading(progress: 0.0)

        let generation = UUID()
        installGeneration[pack.id] = generation
        let task = Task { [weak self] in
            guard let self else { return }
            switch pack.capability {
            case .voice:
                await self.installAudio(pack, generation: generation)
            case .vision:
                await self.simulateDownload(pack, generation: generation)
            }
        }
        downloadTasks[pack.id] = task
    }

    public func cancelDownload(_ pack: SpecialistPack) {
        downloadTasks[pack.id]?.cancel()
        downloadTasks[pack.id] = nil
        installGeneration[pack.id] = nil
        states[pack.id] = .notInstalled
    }

    public func uninstall(_ pack: SpecialistPack) async {
        // 1. Cancel any in-flight download — without this, removing a pack
        //    mid-download leaves an orphaned Task that overwrites state back
        //    to .installed when the download finishes.
        downloadTasks[pack.id]?.cancel()
        downloadTasks[pack.id] = nil
        installGeneration[pack.id] = nil

        // 2. Release any in-memory LEAP resources BEFORE we unlink the
        //    cached GGUFs. The engine mmaps the model — deleting the
        //    file out from under a live runner is undefined behavior.
        //    The hook awaits voice teardown (`runner.unload()` + drop).
        if let hook = preUninstallHook {
            await hook(pack)
        }

        // 3. Reclaim disk. Only the voice pack has a real LEAP cache to
        //    clean up today; vision is still simulated. Errors are
        //    logged but do NOT block the state transition — the user
        //    expressed intent to remove, and a stuck `.installed` state
        //    would be more confusing than a leaked file on disk.
        if pack.capability == .voice {
            do {
                try await audioRemover()
                logger.info("Audio pack cache deleted")
            } catch {
                logger.error("Audio pack cache cleanup failed: \(error.localizedDescription)")
            }
        }

        // 4. Flip state + persist.
        states[pack.id] = .notInstalled
        persistInstalled()
    }

    // MARK: - Audio install (real LEAP download)

    private func installAudio(_ pack: SpecialistPack, generation: UUID) async {
        logger.info("Installing audio pack via LEAP SDK")
        do {
            try await audioInstaller { [weak self] progress in
                guard let self else { return }
                // Stale generation = someone cancelled or re-installed.
                // Drop the progress update so we don't repaint a stale
                // progress bar over the new task's state.
                guard self.installGeneration[pack.id] == generation else { return }
                self.states[pack.id] = .downloading(progress: progress)
            }
            // Ownership check: if generation drifted, a concurrent
            // cancel/uninstall/re-install already transitioned state.
            // Bail without writing — otherwise we could stomp
            // .notInstalled back to .installed.
            guard installGeneration[pack.id] == generation else { return }
            if Task.isCancelled {
                states[pack.id] = .notInstalled
            } else {
                states[pack.id] = .installed(installedAt: Date())
                persistInstalled()
                logger.info("Audio pack installed")
            }
        } catch is CancellationError {
            guard installGeneration[pack.id] == generation else { return }
            states[pack.id] = .notInstalled
        } catch {
            guard installGeneration[pack.id] == generation else { return }
            let detail = Self.describe(error)
            logger.error("Audio pack install failed: \(detail, privacy: .public)")
            states[pack.id] = .error(message: detail)
        }
        // Only clear the slot if we still own it. A concurrent re-install
        // stamps a new generation + task; clearing here unconditionally
        // would orphan the new task (no reference to cancel).
        if installGeneration[pack.id] == generation {
            downloadTasks[pack.id] = nil
            installGeneration[pack.id] = nil
        }
    }

    /// Production LEAP installer. Calls `Leap.load` to trigger the
    /// bundled download + cache, then properly tears down the runner
    /// so the transcriber can re-load from cache on first voice use.
    ///
    /// CRITICAL: `ModelRunner.unload()` MUST be called before dropping
    /// the reference. The inference engine (C++ + Metal) allocates GPU
    /// command queues, memory-mapped GGUF files, and KV cache buffers
    /// that require ordered async teardown. Dropping the runner without
    /// `unload()` triggers use-after-free in the native deinit path.
    private static let defaultLeapInstaller: AudioInstaller = { progress in
        let runner = try await Leap.load(
            manifestURL: LFMAudioTranscriber.manifestURL,
            downloadProgressHandler: { fraction, _ in
                Task { @MainActor in progress(fraction) }
            }
        )
        await runner.unload()
    }

    /// Production LEAP cache cleaner. Deletes the main GGUF + mmproj +
    /// vocoder that `defaultLeapInstaller` downloaded. Idempotent —
    /// safe to call when nothing is cached (the downloader returns
    /// silently for already-absent files).
    private static let defaultLeapRemover: AudioRemover = {
        let downloader = ModelDownloader()
        try downloader.removeModel(fromManifestURL: LFMAudioTranscriber.manifestURL)
    }

    /// Unwraps `LeapError` and `ModelDownloadError` associated values so
    /// install failures surface something actionable in logs and in the
    /// error card ("HTTP 403 from huggingface.co", "disk full, need 1.1
    /// GB", "SHA mismatch") instead of the generic NSError
    /// "The operation couldn't be completed" that `localizedDescription`
    /// gives you for a bare `LeapError` case.
    static func describe(_ error: Error) -> String {
        describe(error, depth: 0)
    }

    /// Max depth for nested error unwrap. Three levels is enough for
    /// LEAP → ModelDownloadError → URLError without truncating anything
    /// realistic, and guards against a buggy SDK building a cycle.
    private static let maxErrorUnwrapDepth = 3

    private static func describe(_ error: Error, depth: Int) -> String {
        guard depth < maxErrorUnwrapDepth else {
            return "(truncated — error chain deeper than \(maxErrorUnwrapDepth) levels)"
        }
        if let leap = error as? LeapError {
            switch leap {
            case .loadModelFailure:
                return "LEAP.loadModelFailure (manifest unreachable or malformed)"
            case .modelLoadingFailure(let reason, let inner):
                return "LEAP.modelLoadingFailure: \(reason)\(inner.map { " — \(describe($0, depth: depth + 1))" } ?? "")"
            case .generationFailure(let reason, let inner):
                return "LEAP.generationFailure: \(reason)\(inner.map { " — \(describe($0, depth: depth + 1))" } ?? "")"
            case .serializationFailure(let reason, let inner):
                return "LEAP.serializationFailure: \(reason)\(inner.map { " — \(describe($0, depth: depth + 1))" } ?? "")"
            case .promptExceedContextLengthFailure(let reason, let inner):
                return "LEAP.promptExceedContextLengthFailure: \(reason)\(inner.map { " — \(describe($0, depth: depth + 1))" } ?? "")"
            @unknown default:
                // SDK 0.10+ may add cases. Fall through to NSError so
                // we still surface something actionable.
                break
            }
        }
        if let dl = error as? ModelDownloadError {
            switch dl {
            case .downloadFailed(let msg, let inner):
                return "download failed: \(msg)\(inner.map { " — \(describe($0, depth: depth + 1))" } ?? "")"
            case .downloadNetworkError(let msg, let inner):
                return "network error: \(msg) — \(describe(inner, depth: depth + 1))"
            case .downloadFileIOError(let msg, let inner):
                return "file I/O: \(msg) — \(describe(inner, depth: depth + 1))"
            case .downloadCancelled(let msg):
                return "cancelled: \(msg)"
            case .downloadInsufficientSpace(let need, let have):
                let fmt = ByteCountFormatter()
                return "insufficient disk: need \(fmt.string(fromByteCount: need)), have \(fmt.string(fromByteCount: have))"
            case .sizeMismatch(let expected, let actual):
                return "size mismatch: expected \(expected) bytes, got \(actual)"
            case .sha256Mismatch(let expected, let actual):
                return "SHA-256 mismatch: expected \(expected.prefix(12))…, got \(actual.prefix(12))…"
            @unknown default:
                break
            }
        }
        let ns = error as NSError
        return "\(ns.domain) \(ns.code): \(ns.localizedDescription)"
    }

    // MARK: - Vision install (still simulated)

    /// Simulate a download with a realistic progress curve. Still a
    /// placeholder for the vision pack; replaced by a real MLXVLM (or
    /// LEAP VL) install in Phase B.
    private func simulateDownload(_ pack: SpecialistPack, generation: UUID) async {
        let totalSteps = 40
        for step in 1...totalSteps {
            try? await Task.sleep(nanoseconds: 200_000_000)
            // Same ownership check as `installAudio`. A cancel/uninstall
            // or re-install mid-curve stamps a new generation; we bail
            // silently and let the new task drive state.
            guard installGeneration[pack.id] == generation else { return }
            if Task.isCancelled {
                downloadTasks[pack.id] = nil
                installGeneration[pack.id] = nil
                return
            }
            let progress = Double(step) / Double(totalSteps)
            states[pack.id] = .downloading(progress: progress)
        }
        guard installGeneration[pack.id] == generation else { return }
        states[pack.id] = .installed(installedAt: Date())
        persistInstalled()
        downloadTasks[pack.id] = nil
        installGeneration[pack.id] = nil
    }

    private func persistInstalled() {
        let installed = states.compactMap { id, state -> String? in
            if case .installed = state { return id }
            return nil
        }
        defaults.set(installed, forKey: persistenceKey)
    }
}
