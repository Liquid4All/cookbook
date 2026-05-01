import Foundation

/// A single knowledge-base entry, matching the shape carriers use in support FAQ sheets.
///
/// Each entry has a canonical topic plus `/`-separated aliases, a numbered-step
/// answer in Markdown, and zero or more `telco://` deep links that surface
/// in the UI as "where in the app" capsules. `requiresToolExecution = true`
/// means the answer describes a UI flow that the LFM tool selector is
/// expected to surface as a concrete tool call — the router uses this flag
/// to prefer `.toolCall` over `.answerWithRAG` for the query.
public struct KBEntry: Codable, Identifiable, Sendable, Equatable {
    public let id: String
    public let topic: String
    public let aliases: [String]
    public let category: String
    public let answer: String
    public let deepLinks: [DeepLink]
    public let tags: [String]
    public let requiresToolExecution: Bool

    public init(
        id: String,
        topic: String,
        aliases: [String],
        category: String,
        answer: String,
        deepLinks: [DeepLink],
        tags: [String],
        requiresToolExecution: Bool
    ) {
        self.id = id
        self.topic = topic
        self.aliases = aliases
        self.category = category
        self.answer = answer
        self.deepLinks = deepLinks
        self.tags = tags
        self.requiresToolExecution = requiresToolExecution
    }

    /// Every string the retriever should consider part of this entry's searchable
    /// surface. Topic + aliases + tags weighted equally.
    public var searchableTerms: [String] {
        [topic] + aliases + tags
    }
}

public struct DeepLink: Codable, Sendable, Equatable, Hashable {
    public let label: String
    public let url: String

    public init(label: String, url: String) {
        self.label = label
        self.url = url
    }
}
