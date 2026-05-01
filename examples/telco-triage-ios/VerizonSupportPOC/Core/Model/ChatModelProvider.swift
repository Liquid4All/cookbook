import Foundation

/// Abstraction over the model that generates a user-facing response.
/// Production implementation is `LFMChatProvider` — an on-device base
/// LFM2.5-350M generation with five prompt templates (grounded QA,
/// tool proposal, tool confirmation, profile summary, out-of-scope).
public protocol ChatModelProvider: Sendable {
    func respond(
        to message: String,
        context: [KBEntry],
        history: [ChatTurn]
    ) async throws -> ChatModelResponse
}

/// A turn in the conversation passed to the provider as context. Keeping this
/// intentionally minimal so provider implementations can't leak view-layer
/// concerns back into themselves.
public struct ChatTurn: Sendable, Equatable {
    /// Shared role enum — `ChatMessage.Role` is a typealias to this
    /// so the chat UI and the provider protocol agree on the same
    /// closed set of roles (`.user`, `.assistant`, `.system`).
    public enum Role: String, Sendable, Equatable {
        case user
        case assistant
        case system
    }

    public let role: Role
    public let text: String

    public init(role: Role, text: String) {
        self.role = role
        self.text = text
    }
}

public struct ChatModelResponse: Sendable {
    public let text: String                    // Markdown; supports **Step N:** and inline links
    public let confidence: Double              // 0.0...1.0
    public let latencyMS: Int
    public let usedContextIDs: [String]        // KBEntry.id values the answer drew from
    public let deepLinks: [DeepLink]
    public let inputTokens: Int                // Estimated — for TokenLedger
    public let outputTokens: Int

    public init(
        text: String,
        confidence: Double,
        latencyMS: Int,
        usedContextIDs: [String],
        deepLinks: [DeepLink],
        inputTokens: Int,
        outputTokens: Int
    ) {
        self.text = text
        self.confidence = confidence
        self.latencyMS = latencyMS
        self.usedContextIDs = usedContextIDs
        self.deepLinks = deepLinks
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
    }
}

public enum ChatModelError: Error {
    case providerUnavailable
}

/// Rough token estimate — whitespace split × 1.3 — good enough for a dashboard.
public enum TokenEstimator {
    public static func estimate(_ text: String) -> Int {
        let words = text.split { $0.isWhitespace }.count
        return Int(ceil(Double(words) * 1.3))
    }
}
