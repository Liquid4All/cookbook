import Foundation

/// A downloadable specialist pack — an LFM variant that lights up a
/// capability (voice, vision) beyond the bundled 350M base. Packs ship
/// separately so the app stays under the App Store size guidance; users
/// only download the ones relevant to their use case.
public struct SpecialistPack: Sendable, Identifiable, Equatable {
    public enum Capability: String, Sendable, Codable { case voice, vision }

    /// Whether the pack can be installed today. "Coming soon" packs
    /// keep the product story visible on the Packs screen but block
    /// install — the associated capability falls back to a
    /// platform-native alternative (Apple Speech for voice) wired in
    /// `VoiceCoordinator.defaultFactory`.
    public enum Availability: Sendable, Equatable {
        case available
        /// Human-readable explanation shown on the card. Should tell the
        /// user what IS working in the meantime (e.g. "Using iOS Speech
        /// Recognition until the on-device LFM audio pack lands").
        case comingSoon(reason: String)
    }

    public let id: String
    public let displayName: String
    public let summary: String            // one-line for the card
    public let detailedDescription: String
    public let capability: Capability
    public let modelRepo: String          // Hugging Face repo reference for provenance
    public let downloadSizeBytes: Int64
    public let filesToFetch: [String]     // file names on disk under `localDirectory`
    public let localDirectory: String     // under Application Support
    public let icon: String               // SF Symbol
    public let valueProps: [String]       // bullet list shown on the card
    public let availability: Availability

    public static let audio = SpecialistPack(
        id: "audio",
        displayName: "Voice Support Pack",
        summary: "On-device LFM2.5-Audio transcription for accented, noisy, hands-free help.",
        detailedDescription: "Adds LFM2.5-Audio-1.5B for on-device speech-to-text via the LEAP Edge SDK. Apple Speech handles voice input before install as a zero-download fallback; after install, LFM2.5-Audio takes over with meaningfully higher accuracy on clean audio (LibriSpeech-clean WER 1.95% vs. Apple Speech's server fallback behavior) and fully offline operation.",
        capability: .voice,
        // LEAP-bundled variant. The LEAP SDK resolves this identifier
        // to the main GGUF + mmproj + vocoder trio and caches them in
        // its own application-support directory — our `filesToFetch`
        // list is informational only for the pack-card UI.
        modelRepo: "LiquidAI/LFM2.5-Audio-1.5B-GGUF-LEAP",
        // Q4_0 bundle total: main 696 MB + mmproj 220 MB + vocoder
        // 149 MB ≈ 1.065 GB. Earlier 1.5 GB estimate was based on the
        // pre-GGUF disk footprint and was inaccurate.
        downloadSizeBytes: 1_065_000_000,
        filesToFetch: [
            "LFM2.5-Audio-1.5B-Q4_0.gguf",
            "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf",
            "vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf",
        ],
        localDirectory: "packs/audio",
        icon: "waveform",
        valueProps: [
            "Hands-free support while driving or cooking",
            "Works offline in noisy environments",
            "Handles accented English out of the box",
            "Voice data never leaves the device",
        ],
        // LEAP SDK 0.9.4 crashes in ie_llamacpp's gguf.cpp parser on
        // the published LFM2.5-Audio-1.5B-GGUF-LEAP bundle
        // (EXC_BAD_ACCESS on a null std::string during KV metadata
        // read). Voice input falls back to AVFoundation's on-device
        // Apple Speech until LiquidAI ships a compatible SDK build.
        availability: .comingSoon(
            reason: "Using iOS Speech Recognition until the on-device LFM audio pack lands."
        )
    )

    public static let vision = SpecialistPack(
        id: "vision",
        displayName: "Visual Troubleshoot Pack",
        summary: "Snap a photo — router lights, error screen, or a bill — and get an instant on-device explanation.",
        detailedDescription: "Adds LFM2.5-VL-450M, a vision-language model that reads router LED patterns, bill line items, and device error screens. Everything stays on-device; no photos are uploaded.",
        capability: .vision,
        modelRepo: "LiquidAI/LFM2-VL-450M",
        downloadSizeBytes: 287_000_000,
        filesToFetch: [
            "LFM2_VL-450M-Q4_0.gguf",
            "mmproj-LFM2_VL-450M-Q8_0.gguf",
        ],
        localDirectory: "packs/vision",
        icon: "camera.viewfinder",
        valueProps: [
            "Diagnose router lights in one snapshot",
            "Explain unfamiliar bill charges",
            "OCR error screens for self-service",
            "Photos never leave the phone",
        ],
        // Vision analyzer is still backed by `MockVisionAnalyzer`
        // pending real LFM-VL integration. Keep the card to tell the
        // product story, but block install until a real model lands.
        availability: .comingSoon(
            reason: "Visual troubleshooting is in closed beta — card preview only for now."
        )
    )

    public static let all: [SpecialistPack] = [.audio, .vision]

    public var isAvailable: Bool {
        if case .available = availability { return true }
        return false
    }

    /// Returns a copy with `availability` overridden. Primarily for
    /// tests that exercise the install/uninstall lifecycle — the
    /// production config marks all packs coming-soon today.
    public func with(availability: Availability) -> SpecialistPack {
        SpecialistPack(
            id: id,
            displayName: displayName,
            summary: summary,
            detailedDescription: detailedDescription,
            capability: capability,
            modelRepo: modelRepo,
            downloadSizeBytes: downloadSizeBytes,
            filesToFetch: filesToFetch,
            localDirectory: localDirectory,
            icon: icon,
            valueProps: valueProps,
            availability: availability
        )
    }

    public var capabilityDisplayName: String {
        switch capability {
        case .voice: return "Voice"
        case .vision: return "Vision"
        }
    }

    public var formattedSize: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: downloadSizeBytes)
    }
}

public enum SpecialistPackState: Equatable, Sendable {
    case notInstalled
    case downloading(progress: Double)
    case installed(installedAt: Date)
    case error(message: String)
}
