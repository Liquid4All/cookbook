import Foundation

/// Deterministic pre-filter that decides whether a query is in scope
/// for the telco support assistant.
///
/// Why this exists:
///   The 8-class `support_intent` head has no `out_of_scope` class —
///   every query MUST be classified as one of {troubleshooting, outage,
///   billing, appointment, device_setup, plan_account, equipment_return,
///   agent_handoff}. So "what is the weather in new york" lands on
///   `outage` (because of vague service-disruption co-occurrence in
///   training data) and the lane router sends it to cloud_assist where
///   the LFM hallucinates an answer.
///
///   Adding an `out_of_scope` class to the head would require retraining
///   with a balanced negative set — feasible but heavy. For a closed
///   support domain (34 KB entries + a handful of tools), a vocabulary
///   gate built from the KB itself is faster, perfectly accurate within
///   its scope, and impossible to silently regress on.
///
///   This is exactly how production RAG systems work: BM25/keyword
///   gates the retrieval space first, embeddings handle paraphrase
///   fallback only inside that gated space.
public struct TelcoTopicGate: Sendable {
    public enum Decision: Sendable, Equatable {
        case inScope
        case outOfScope(reason: String)
    }

    /// Hand-curated telco vocabulary. Built from the KB topics / aliases
    /// plus the standard support categories. Lowercased for matching.
    private static let coreVocabulary: Set<String> = [
        // Connectivity / network
        "wifi", "wi-fi", "wireless", "internet", "ethernet", "connection",
        "connectivity", "network", "online", "offline", "signal", "speed",
        "bandwidth", "throughput", "ssid", "ip", "ipv4", "ipv6", "mac",
        "dns", "vpn", "extender", "mesh", "ping", "latency",
        // Equipment
        "router", "modem", "gateway", "fios", "fiber", "cable", "device",
        "equipment", "firmware", "hardware", "led", "wps", "tv", "stream",
        "set-top", "set top", "ont", "battery",
        // Service / billing / account
        "service", "outage", "down", "bill", "billing", "charge", "charged",
        "payment", "autopay", "plan", "subscription", "account", "password",
        "email", "username", "login", "renew", "upgrade", "downgrade",
        "cancel", "switch", "transfer", "move", "address", "appointment",
        "technician", "tech", "install", "installation", "schedule",
        "agent", "manager", "representative", "rep", "support",
        // Diagnostics / actions
        "restart", "reboot", "reset", "diagnose", "diagnostic", "troubleshoot",
        "fix", "broken", "not working", "buffer", "buffering", "lag",
        "frozen", "crash", "fail", "failing", "drop", "dropping",
        // Parental / device control
        "parental", "block", "pause", "kid", "child", "tablet", "phone",
        "laptop", "computer", "console", "gaming", "alexa", "echo",
        // Returns / equipment
        "return", "send back", "ship", "shipping", "tracking", "label",
        // Common verbs that imply telco context
        "configure", "setup", "set up", "set-up", "activate", "deactivate",
        "enable", "disable", "share",
    ]

    /// Generic English stopwords stripped before vocabulary check.
    private static let stopwords: Set<String> = [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "doing", "have", "has", "had", "having",
        "i", "you", "he", "she", "we", "they", "it", "my", "your", "his",
        "her", "our", "their", "its", "this", "that", "these", "those",
        "what", "where", "when", "why", "how", "who", "which",
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "about",
        "and", "or", "but", "if", "then", "else", "so", "than", "as",
        "can", "could", "should", "would", "will", "might", "may", "must",
        "please", "thanks", "thank", "ok", "okay", "yes", "no", "not",
        "me", "us", "them", "myself", "yourself", "themselves",
        "very", "really", "just", "also", "too", "only", "here", "there",
        "now", "today", "tomorrow", "yesterday", "always", "never", "again",
        "more", "less", "most", "least", "some", "any", "all", "both",
        "much", "many", "few", "several",
    ]

    public init() {}

    public func decide(_ query: String) -> Decision {
        let tokens = Self.tokenize(query)
        // Empty / pure stopwords / sub-3-character → ambiguous; let
        // it through so the classifier can ask for clarification.
        let content = tokens.filter { !Self.stopwords.contains($0) && $0.count >= 2 }
        if content.isEmpty {
            return .inScope
        }

        // Hit if ANY content token matches the telco vocabulary,
        // either exactly or as a substring (e.g., "buffering" hits
        // because "buffer" is in vocab).
        for token in content {
            if Self.coreVocabulary.contains(token) { return .inScope }
            for term in Self.coreVocabulary where token.contains(term) {
                return .inScope
            }
        }

        // No content token overlapped with telco vocab — this is an
        // out-of-domain query (weather, sports, general chat, etc).
        return .outOfScope(reason: "no telco vocabulary in query")
    }

    static func tokenize(_ text: String) -> [String] {
        text
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }
}
