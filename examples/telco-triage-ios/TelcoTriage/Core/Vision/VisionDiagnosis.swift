import Foundation

/// Typed output of the photo-to-fix flow: a natural-language
/// explanation plus, when applicable, a concrete tool proposal the
/// customer can accept with one tap.
///
/// The narrower `kind` enum lets the UI render domain-specific cards
/// (router lights get LED-pattern chips; bill gets the charge
/// breakdown) while keeping the shared `VisionResult` metadata (headline,
/// detail, hints) for general rendering.
///
/// When `proposedToolID` is non-nil, the chat surfaces a one-tap
/// execute button below the diagnosis card. The tool still runs
/// through the usual confirmation flow — this isn't a bypass.
public struct VisionDiagnosis: Sendable, Equatable {
    public enum Kind: String, Sendable {
        case routerLights
        case billScreenshot
        case errorScreen
        case unknown
    }

    public enum Severity: String, Sendable {
        case normal, degraded, critical, unclear
    }

    public let kind: Kind
    public let severity: Severity
    public let headline: String
    public let explanation: String
    public let proposedToolID: String?
    public let proposedArguments: [String: String]
    public let confidence: Double
    public let latencyMS: Int
    public let usedPack: Bool

    public init(
        kind: Kind,
        severity: Severity,
        headline: String,
        explanation: String,
        proposedToolID: String? = nil,
        proposedArguments: [String: String] = [:],
        confidence: Double,
        latencyMS: Int,
        usedPack: Bool
    ) {
        self.kind = kind
        self.severity = severity
        self.headline = headline
        self.explanation = explanation
        self.proposedToolID = proposedToolID
        self.proposedArguments = proposedArguments
        self.confidence = confidence
        self.latencyMS = latencyMS
        self.usedPack = usedPack
    }
}

public extension VisionDiagnosis {
    /// Converts an existing `VisionResult` into a `VisionDiagnosis` by
    /// mapping the category to a domain kind and inferring a tool
    /// proposal when the confidence is high enough. Keeps the heuristic
    /// analyzer and the future LFM2.5-VL path compatible at the call
    /// site.
    static func from(_ result: VisionResult) -> VisionDiagnosis {
        let kind: Kind
        let severity: Severity
        let toolID: String?
        switch result.category {
        case .router:
            kind = .routerLights
            // When the visual-troubleshoot pack is installed we have
            // a real diagnosis; otherwise we're on the heuristic
            // preview and shouldn't push a one-tap restart.
            severity = result.usedPack ? .normal : .unclear
            toolID = result.usedPack ? "restart-router" : nil
        case .bill:
            kind = .billScreenshot
            severity = .normal
            toolID = nil
        case .errorScreen:
            kind = .errorScreen
            severity = result.usedPack ? .degraded : .unclear
            toolID = result.usedPack ? "restart-router" : nil
        case .other:
            kind = .unknown
            severity = .unclear
            toolID = nil
        }

        return VisionDiagnosis(
            kind: kind,
            severity: severity,
            headline: result.headline,
            explanation: result.detail,
            proposedToolID: toolID,
            proposedArguments: [:],
            confidence: result.usedPack ? 0.85 : 0.45,
            latencyMS: result.latencyMS,
            usedPack: result.usedPack
        )
    }
}
