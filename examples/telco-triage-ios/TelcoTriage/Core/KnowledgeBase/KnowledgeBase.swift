import Foundation

/// The loaded knowledge base. Immutable at runtime; reload via `loadFromDocuments`
/// if a carrier ships a fresh KB dump without rebuilding the app.
public struct KnowledgeBase: Codable, Sendable {
    public let version: String
    public let entries: [KBEntry]

    public init(version: String, entries: [KBEntry]) {
        self.version = version
        self.entries = entries
    }

    /// Load the KB bundled with the app. Crashes in debug if the resource is
    /// missing — that's a build bug, not a runtime condition worth recovering.
    public static func loadFromBundle() -> KnowledgeBase {
        guard let url = Bundle.main.url(forResource: "knowledge-base", withExtension: "json") else {
            assertionFailure("knowledge-base.json missing from bundle")
            return KnowledgeBase(version: "empty", entries: [])
        }
        return (try? load(from: url)) ?? KnowledgeBase(version: "empty", entries: [])
    }

    /// If a newer KB has been dropped into the app's Documents directory
    /// (Settings → "Reload KB"), prefer it. Otherwise fall back to bundle.
    public static func loadPreferringDocuments() -> KnowledgeBase {
        let fm = FileManager.default
        if let docs = fm.urls(for: .documentDirectory, in: .userDomainMask).first {
            let override = docs.appendingPathComponent("knowledge-base.json")
            if fm.fileExists(atPath: override.path),
               let kb = try? load(from: override) {
                return kb
            }
        }
        return loadFromBundle()
    }

    private static func load(from url: URL) throws -> KnowledgeBase {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(KnowledgeBase.self, from: data)
    }
}
