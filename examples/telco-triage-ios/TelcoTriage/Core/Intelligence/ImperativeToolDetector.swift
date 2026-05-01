import Foundation

/// Deterministic post-classifier override for iOS tools the
/// `required_tool` head can't represent.
///
/// Why this exists (first-principles):
///   The ADR-015 `required_tool` head has 6 classes:
///   {restart_gateway, run_diagnostics, speed_test, schedule_technician,
///   no_tool, cloud_only}. The iOS app has 8 tools — the schema is
///   missing `toggle_parental_controls` and `reboot_extender`. When the
///   user says "pause internet for my son's tablet" or "restart the
///   wifi extender upstairs", the classifier MUST route them to
///   `local_answer` (no tool slot can hold them) — and the user sees
///   a KB lookup instead of a tool action.
///
///   Two ways out:
///     1. Extend the head schema and retrain. Right answer for
///        production; ~30 min per cycle.
///     2. Deterministic pattern override post-classifier. Right
///        answer for the demo: clear, debuggable, never silently
///        regresses on retraining.
///
///   This file is option 2. It runs AFTER the classifier and only
///   fires when the query is unambiguously imperative for one of the
///   missing tools. Question forms ("how do I pause internet") are
///   explicitly rejected so we don't override KB lookups.
public enum ImperativeToolDetector {
    /// Returns a tool intent if the query is a clear imperative for
    /// one of the iOS-only tools (toggleParentalControls, rebootExtender).
    /// Returns nil otherwise — classifier output is honored.
    public static func detect(_ query: String) -> ToolIntent? {
        let q = query.lowercased()

        // Question forms are NEVER overridden — those are real KB
        // lookups even when they mention tool topics.
        let questionStarters = [
            "how do", "how can", "how should", "how to",
            "what is", "what's", "what are", "what does", "what happens",
            "where", "why", "should i", "should we", "when",
            "can you tell", "can you explain", "could you tell",
        ]
        for starter in questionStarters where q.hasPrefix(starter) {
            return nil
        }

        if matchesParentalControls(q) { return .toggleParentalControls }
        if matchesRebootExtender(q) { return .rebootExtender }
        return nil
    }

    /// `pause/block/stop internet for <device>` patterns.
    /// Requires: an action verb + an internet/wifi noun + a device or
    /// person noun. Two of the three is not enough — "block this site"
    /// is parental-controls-adjacent but isn't unambiguous.
    private static func matchesParentalControls(_ q: String) -> Bool {
        let pauseBlock = ["pause", "block", "stop", "shut off", "kill", "disable"]
        let internetTerms = ["internet", "wifi", "wi-fi", "wi fi", "network", "connection"]
        let deviceOrPerson = [
            "tablet", "phone", "laptop", "computer", "console", "playstation",
            "ps4", "ps5", "xbox", "switch", "ipad", "iphone", "device",
            "kid", "kids", "son", "daughter", "child", "children", "teen",
            "his", "her", "their", "youtube", "tiktok",
        ]
        let hasAction = pauseBlock.contains { q.contains($0) }
        let hasInternet = internetTerms.contains { q.contains($0) }
        let hasTarget = deviceOrPerson.contains { q.contains($0) }
        return hasAction && hasInternet && hasTarget
    }

    /// `restart/reboot/reset` + `extender/mesh/satellite/node` and
    /// optional location qualifier (`upstairs`, `bedroom`, etc.).
    /// The mention of "extender" alone with a restart verb is enough
    /// — there's no benign question reading of "reset the extender".
    private static func matchesRebootExtender(_ q: String) -> Bool {
        let restartVerbs = ["restart", "reboot", "reset", "cycle", "power cycle", "kick"]
        let extenderTerms = ["extender", "mesh node", "mesh point", "satellite", "wifi point"]
        let hasVerb = restartVerbs.contains { q.contains($0) }
        let hasExtender = extenderTerms.contains { q.contains($0) }
        return hasVerb && hasExtender
    }
}
