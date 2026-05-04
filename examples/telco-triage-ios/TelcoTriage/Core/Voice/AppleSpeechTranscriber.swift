import Foundation
import Speech
import AVFoundation

/// Default voice transcriber using Apple's SFSpeechRecognizer + AVAudioEngine.
/// Works before any specialist pack is installed. Produces partial results
/// as the user speaks; emits a `.final` when `stopListening()` is called
/// or the recognizer finishes.
///
/// Implemented as an `actor` so the recognition-task callback (which fires
/// on an arbitrary dispatch queue) cannot race with the caller's
/// `start` / `stop`. All mutable state lives inside the actor; the
/// callback hops back via `Task { await actor.handle(...) }`.
public actor AppleSpeechTranscriber: VoiceTranscriber {
    private let recognizer: SFSpeechRecognizer
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var continuation: AsyncStream<TranscriptionEvent>.Continuation?
    private var sessionActive: Bool = false

    public init(locale: Locale = Locale(identifier: "en-US")) {
        // Prefer the requested locale, fall back to the system default, fall
        // back again to constructing a fresh recognizer with no locale.
        // If all three fail there is no speech support on-device, so a
        // deliberate crash is better than a confusing runtime error later.
        if let configured = SFSpeechRecognizer(locale: locale) {
            self.recognizer = configured
        } else if let system = SFSpeechRecognizer() {
            self.recognizer = system
        } else {
            fatalError("No SFSpeechRecognizer available on this device")
        }
    }

    public func startListening() async throws -> AsyncStream<TranscriptionEvent> {
        try await requestPermissions()
        try configureSession()

        let newRequest = SFSpeechAudioBufferRecognitionRequest()
        newRequest.shouldReportPartialResults = true
        self.request = newRequest

        let stream = AsyncStream<TranscriptionEvent> { continuation in
            self.continuation = continuation
        }

        self.task = recognizer.recognitionTask(with: newRequest) { [weak self] result, error in
            // Callback fires on an arbitrary queue — hop back into the actor
            // before touching any mutable state.
            guard let self else { return }
            Task { await self.handleRecognition(result: result, error: error) }
        }

        let input = audioEngine.inputNode
        let recordingFormat = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            // Buffer callbacks also fire off-actor. Ferry the buffer in.
            Task { await self?.append(buffer: buffer) }
        }

        audioEngine.prepare()
        try audioEngine.start()

        return stream
    }

    public func stopListening() async {
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.finish()
        task = nil
        request = nil
        continuation?.finish()
        continuation = nil
        deactivateSession()
    }

    // MARK: - Actor-isolated handlers

    private func append(buffer: AVAudioPCMBuffer) {
        request?.append(buffer)
    }

    private func handleRecognition(result: SFSpeechRecognitionResult?, error: Error?) {
        if let error {
            continuation?.yield(.error(error.localizedDescription))
            continuation?.finish()
            continuation = nil
            return
        }
        guard let result else { return }
        let text = result.bestTranscription.formattedString
        if result.isFinal {
            continuation?.yield(.final(text))
            continuation?.finish()
            continuation = nil
        } else {
            continuation?.yield(.partial(text))
        }
    }

    // MARK: - Permissions & session

    private func requestPermissions() async throws {
        let speechStatus = await withCheckedContinuation { (cont: CheckedContinuation<SFSpeechRecognizerAuthorizationStatus, Never>) in
            SFSpeechRecognizer.requestAuthorization { cont.resume(returning: $0) }
        }
        guard speechStatus == .authorized else {
            throw TranscriptionError.permissionDenied
        }

        // iOS 17+: `AVAudioSession.sharedInstance().requestRecordPermission`
        // is deprecated and can silently return false without prompting on
        // iOS 18 — the replacement lives on `AVAudioApplication`. Using the
        // deprecated API is the most common "mic button does nothing"
        // root cause on recent iOS.
        let micGranted = await AVAudioApplication.requestRecordPermission()
        guard micGranted else {
            throw TranscriptionError.permissionDenied
        }
    }

    private func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        sessionActive = true
    }

    /// Deactivate on stop — critical so the app doesn't silently hold the
    /// audio route after the user stops talking. Without this, other apps
    /// (Music, Podcasts, the foreground call) stay ducked.
    private func deactivateSession() {
        guard sessionActive else { return }
        sessionActive = false
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
    }
}
