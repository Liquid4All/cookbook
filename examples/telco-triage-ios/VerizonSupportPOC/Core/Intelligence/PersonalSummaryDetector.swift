import Foundation

/// Deterministic detector for "show me my own data" requests
/// (`personalSummary` lane). Fixes a real classifier mis-route:
///
///   "summarize my home network" → classifier picks intent=
///   troubleshooting + tool=run_diagnostics → user gets a fix-it
///   tool action when they actually wanted a status read.
///
/// The classifier vocabulary doesn't have a "show my data" class —
/// it has tool actions (run_diagnostics, speed_test) and KB lookups.
/// "Summarize" / "show me my X" / "list my X" are personal-context
/// reads that should hit `runPersonalizedSummary` and surface the
/// customer's profile (devices, plan, bill).
///
/// Same pattern as `ImperativeToolDetector`: deterministic post-
/// classifier override, runs only on unambiguous patterns, leaves
/// the LFM in charge of everything else.
public enum PersonalSummaryDetector {
    /// Triggers that signal "give me a read of my own data". Each
    /// must be paired with a personal-context noun (network, devices,
    /// bill, account, plan) before we override the classifier — the
    /// noun constraint prevents "summarize Netflix" from falsely
    /// firing personalSummary.
    private static let summaryTriggers: [String] = [
        "summarize", "summary of", "summary about",
        "show me my", "show my",
        "what's on my", "what is on my", "whats on my",
        "what's connected to my", "whats connected to my",
        "tell me about my",
        "list my", "list all my",
        "give me a summary", "give me an overview",
        "overview of my", "overview about my",
        "rundown of my", "rundown on my",
        "what devices", "which devices",
        "status of my",
    ]

    /// Personal-data nouns that must co-occur with a trigger.
    /// "summarize my home network" hits ("summarize" + "network");
    /// "summarize this article" misses (no personal noun).
    private static let personalNouns: [String] = [
        "network", "wifi", "wi-fi", "internet", "connection",
        "device", "devices", "router", "modem", "extender",
        "bill", "billing", "account", "plan", "subscription",
        "usage", "data", "home", "household", "service",
    ]

    public static func detect(_ query: String) -> Bool {
        let q = query.lowercased()

        // Trigger present?
        let hit = summaryTriggers.first { q.contains($0) }
        guard let trigger = hit else { return false }

        // Personal-data noun present?
        // Special case: "what devices" / "which devices" already
        // contains its own personal noun, so it always qualifies.
        if trigger.hasSuffix("devices") {
            return true
        }
        return personalNouns.contains { q.contains($0) }
    }
}
