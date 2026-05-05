import Foundation
import Combine

/// Chooses the right VoiceTranscriber based on pack state and coordinates
/// start/stop for the chat input bar. Views observe `state` for the
/// mic-button UI.
@MainActor
public final class VoiceCoordinator: ObservableObject {
    public enum State: Equatable {
        case idle
        case listening(partial: String)
        case transcribing(partial: String)
        case finalized(String)
        case error(String)
    }

    @Published public private(set) var state: State = .idle
    @Published public private(set) var isListening: Bool = false
    @Published public private(set) var usingPack: Bool = false

    public var isTranscribing: Bool {
        if case .transcribing = state { return true }
        return false
    }

    /// Factory receives `isPackInstalled` so the coordinator can choose
    /// between Apple Speech (no pack) and LFM2.5-Audio via LEAP (pack
    /// installed). Tests inject fakes without touching either real
    /// backend.
    public typealias TranscriberFactory = @MainActor (_ isPackInstalled: Bool) -> VoiceTranscriber

    private let packManager: SpecialistPackManager
    private let transcriberFactory: TranscriberFactory
    private var transcriber: VoiceTranscriber?
    private var streamTask: Task<Void, Never>?
    private var packObserver: AnyCancellable?

    public init(
        packManager: SpecialistPackManager,
        transcriberFactory: @escaping TranscriberFactory = VoiceCoordinator.defaultFactory
    ) {
        self.packManager = packManager
        self.transcriberFactory = transcriberFactory

        // Auto-stop voice when the audio pack is removed mid-session.
        // Without this, removing the pack while listening leaves a
        // dangling LFMAudioTranscriber with a stale LEAP model runner
        // and an orphaned AVAudioEngine tap — guaranteed crash or
        // silent audio-session leak.
        self.packObserver = packManager.$states
            .sink { [weak self] states in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    let audioState = states[SpecialistPack.audio.id] ?? .notInstalled
                    if case .notInstalled = audioState, self.isListening, self.usingPack {
                        await self.stop()
                    }
                }
            }
    }

    /// Pack installed ⇒ LFM2.5-Audio STT via LEAP. Otherwise Apple
    /// Speech as the zero-download fallback so the mic still works on
    /// a fresh install. The factory re-runs on every `start()` so the
    /// user's install-while-idle state change takes effect on the very
    /// next mic tap — no app restart.
    public static let defaultFactory: TranscriberFactory = { isPackInstalled in
        isPackInstalled ? LFMAudioTranscriber() : AppleSpeechTranscriber()
    }

    public func start(localRuntimeBusy: Bool = false) {
        guard !isListening else { return }
        guard !localRuntimeBusy else {
            state = .error("Voice input is available after the current on-device answer finishes.")
            isListening = false
            usingPack = false
            return
        }
        state = .listening(partial: "")
        isListening = true
        let packInstalled = packManager.isInstalled(SpecialistPack.audio.id)
        usingPack = packInstalled
        let chosen = transcriberFactory(packInstalled)
        transcriber = chosen

        streamTask = Task { [weak self] in
            guard let self else { return }
            do {
                let stream = try await chosen.startListening()
                for await event in stream {
                    switch event {
                    case .partial(let text):
                        await MainActor.run {
                            self.state = .listening(partial: text)
                        }
                    case .final(let text):
                        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                        await self.finishStream(
                            finalState: trimmed.isEmpty ? .idle : .finalized(trimmed),
                            transcriber: chosen
                        )
                        return
                    case .error(let msg):
                        await self.finishStream(
                            finalState: .error(msg),
                            transcriber: chosen
                        )
                        return
                    }
                }
                await self.finishStream(finalState: .idle, transcriber: chosen)
            } catch {
                await self.finishStream(
                    finalState: .error(error.localizedDescription),
                    transcriber: chosen
                )
            }
        }
    }

    public func stop() async {
        // Capture the in-progress partial BEFORE tearing anything
        // down. `stopListening()` finishes the AsyncStream's
        // continuation immediately — Apple Speech's final-result
        // callback never fires, so .final is never emitted and the
        // accumulated transcription was being dropped on the floor.
        let capturedPartial: String
        if case .listening(let partial) = state { capturedPartial = partial } else { capturedPartial = "" }

        guard let activeTranscriber = transcriber else {
            isListening = false
            usingPack = false
            state = .idle
            return
        }

        if activeTranscriber.stopBehavior == .awaitFinalEventAfterStop {
            await activeTranscriber.stopListening()
            await Self.waitForAudioRouteToSettle()
            isListening = false
            usingPack = false

            // LFM audio runs ASR after capture stops. Leave the stream
            // consumer alive so the model's final transcript can flow
            // back into ChatView. If the stream already produced a
            // terminal state while stopListening() was suspended, keep it.
            if !state.isTerminal {
                let trimmed = capturedPartial.trimmingCharacters(in: .whitespacesAndNewlines)
                state = .transcribing(partial: trimmed)
            }
            return
        }

        // Apple Speech often finishes its AsyncStream before sending a
        // final callback. For that backend the latest partial is the safest
        // user-visible final text.
        streamTask?.cancel()
        streamTask = nil

        await activeTranscriber.stopListening()
        await activeTranscriber.releaseResources()
        transcriber = nil
        await Self.waitForAudioRouteToSettle()

        isListening = false
        usingPack = false

        // Transition to .finalized so ChatView's onChange populates
        // the text field. Empty partial = user tapped mic and stop
        // without speaking; go idle silently.
        let trimmed = capturedPartial.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            state = .finalized(trimmed)
        } else {
            state = .idle
        }
    }

    public func consumeFinal() -> String? {
        if case .finalized(let text) = state {
            state = .idle
            return text
        }
        return nil
    }

    public func reset() {
        state = .idle
        isListening = false
    }

    private func finishStream(finalState: State, transcriber completedTranscriber: VoiceTranscriber) async {
        await completedTranscriber.stopListening()
        await completedTranscriber.releaseResources()
        transcriber = nil
        streamTask = nil
        await Self.waitForAudioRouteToSettle()
        isListening = false
        usingPack = false
        state = finalState
    }

    private static func waitForAudioRouteToSettle() async {
        #if targetEnvironment(simulator)
        try? await Task.sleep(nanoseconds: 250_000_000)
        #endif
    }
}

private extension VoiceCoordinator.State {
    var isTerminal: Bool {
        switch self {
        case .finalized, .error, .idle:
            return true
        case .listening, .transcribing:
            return false
        }
    }
}
