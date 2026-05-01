import Foundation
import UIKit

/// On-device image understanding. Gated by the Vision specialist pack —
/// before the pack is installed the `MockVisionAnalyzer` returns a clear
/// "install the pack to enable this" response rather than failing silently.
public protocol VisionAnalyzer: Sendable {
    func analyze(image: UIImage, prompt: String) async throws -> VisionResult
}

public extension VisionAnalyzer {
    /// Convenience wrapper that composes a typed `VisionDiagnosis`
    /// from `analyze`. Concrete analyzers can override this for
    /// tighter tool-proposal logic (e.g., the real LFM2.5-VL pack
    /// would return a proposal with per-LED reasoning), but the
    /// default is good enough to demo Photo-to-Fix end-to-end with
    /// the mock analyzer.
    func diagnose(image: UIImage, prompt: String) async throws -> VisionDiagnosis {
        let result = try await analyze(image: image, prompt: prompt)
        return .from(result)
    }
}

public struct VisionResult: Sendable {
    public enum Category: String, Sendable { case router, bill, errorScreen, other }

    public let category: Category
    public let headline: String
    public let detail: String
    public let actionableHints: [String]
    public let latencyMS: Int
    public let usedPack: Bool

    public init(
        category: Category,
        headline: String,
        detail: String,
        actionableHints: [String],
        latencyMS: Int,
        usedPack: Bool
    ) {
        self.category = category
        self.headline = headline
        self.detail = detail
        self.actionableHints = actionableHints
        self.latencyMS = latencyMS
        self.usedPack = usedPack
    }
}

public enum VisionError: Error {
    case packNotInstalled
    case analysisFailed(String)
}
