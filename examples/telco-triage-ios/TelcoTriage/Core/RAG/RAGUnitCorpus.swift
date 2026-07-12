import Foundation
import os.log

/// In-memory canonical RAG corpus loaded once at app start from
/// `Resources/rag-units-v1.json`. Mirrors
/// `data/finetune/telco-home-internet/rag_units.json`.
///
/// The corpus is the single source of truth for:
///
/// * what `telcohome://` URLs the composer is allowed to render,
/// * citation labels + steps per page,
/// * response-ready summaries/facts, when the ingestion pipeline supplied them,
/// * link_id → page_id index used by the BM25HierarchyRetriever history hints,
/// * the universe of "valid" deep links for grading / debug assertions.
///
/// Raw body content remains available for retrieval and legacy corpus fallback.
/// New or updated units should provide `answerSummary` / `answerFacts` so the
/// composer renders reviewable evidence instead of mining arbitrary page prose.
public final class RAGUnitCorpus: Sendable {
    public enum LoadError: Error, CustomStringConvertible, Equatable {
        case bundleResourceMissing(name: String, ext: String)
        case decodeFailed(String)

        public var description: String {
            switch self {
            case let .bundleResourceMissing(name, ext):
                return "RAG corpus missing from bundle: \(name).\(ext)"
            case let .decodeFailed(detail):
                return "RAG corpus decode failed: \(detail)"
            }
        }
    }

    /// JSON wrapper around the `units` dict + metadata. We only care
    /// about `units` at runtime; the rest is provenance the Python
    /// pipeline writes for debugging.
    private struct CorpusFile: Decodable {
        let units: [String: RAGUnit]
    }

    private let unitsByPageID: [String: RAGUnit]
    private let unitsByLinkID: [String: [RAGUnit]]
    private let allCanonicalURLsSet: Set<String>

    private static let logger = Logger(
        subsystem: "ai.liquid.demos.telcotriage",
        category: "RAGUnitCorpus"
    )

    public init(units: [String: RAGUnit]) {
        self.unitsByPageID = units
        var byLink: [String: [RAGUnit]] = [:]
        var urls = Set<String>()
        for unit in units.values {
            byLink[unit.linkID, default: []].append(unit)
            urls.insert(unit.canonicalURL)
            urls.insert(unit.canonicalURLBare)
        }
        self.unitsByLinkID = byLink
        self.allCanonicalURLsSet = urls
    }

    // MARK: - Loading

    /// Load the bundled corpus. Crashes only in DEBUG when the
    /// resource is missing — release builds throw so the call site
    /// can degrade gracefully (the app already has multiple corpus
    /// fallbacks).
    ///
    /// Default `bundle` is `Bundle(for: RAGUnitCorpus.self)` so the
    /// loader works correctly in both the main app (where the class
    /// and the resource both live in the main bundle) and in unit
    /// tests (where `@testable import` exposes the type but the
    /// resource is still in the main app bundle, NOT the test bundle).
    /// Passing `Bundle(for: type(of: self))` from a test class would
    /// resolve to the test bundle and fail to find the resource.
    public static func loadFromBundle(
        bundle: Bundle? = nil,
        resource: String = "rag-units-v1",
        extension ext: String = "json"
    ) throws -> RAGUnitCorpus {
        let bundle = bundle ?? Bundle(for: RAGUnitCorpus.self)
        guard let url = bundle.url(forResource: resource, withExtension: ext) else {
            logger.error("rag-units-v1.json missing from bundle")
            #if DEBUG
            assertionFailure("rag-units-v1.json missing from main bundle — add it via Build Phases > Copy Bundle Resources")
            #endif
            throw LoadError.bundleResourceMissing(name: resource, ext: ext)
        }
        let data = try Data(contentsOf: url)
        do {
            let decoded = try JSONDecoder().decode(CorpusFile.self, from: data)
            logger.info("loaded \(decoded.units.count, privacy: .public) RAG units")
            return RAGUnitCorpus(units: decoded.units)
        } catch {
            throw LoadError.decodeFailed(String(describing: error))
        }
    }

    // MARK: - Lookups

    /// Total number of canonical units in the corpus.
    public var count: Int { unitsByPageID.count }

    /// All units, in stable page_id order.
    public var allUnits: [RAGUnit] {
        unitsByPageID.values.sorted { $0.pageID < $1.pageID }
    }

    /// Lookup by page_id (e.g. `"02.07"`).
    public func unit(forPageID pageID: String) -> RAGUnit? {
        unitsByPageID[pageID]
    }

    /// Lookup by link_id. Multiple pages can share a link_id (parent
    /// chain — e.g. several Network sub-pages share `network`); we
    /// return all matches in stable page_id order, smallest first.
    public func units(forLinkID linkID: String) -> [RAGUnit] {
        (unitsByLinkID[linkID] ?? []).sorted { $0.pageID < $1.pageID }
    }

    /// `link_id → [page_id]` index, the same shape
    /// `retriever_ablation._build_link_index` produces in Python.
    /// Used by the BM25HierarchyRetriever's history-hint mechanism.
    public var linkIndex: [String: [String]] {
        unitsByLinkID.mapValues { units in
            units.map(\.pageID).sorted()
        }
    }

    /// All canonical URLs the composer is allowed to emit. Includes
    /// both the full URL (with `?launchPoint=...` suffix) and the bare
    /// form. Used by `ComposerGrading.isLinkValid` to detect
    /// hallucinated links in DEBUG self-audit.
    public var allCanonicalURLs: Set<String> {
        allCanonicalURLsSet
    }
}
