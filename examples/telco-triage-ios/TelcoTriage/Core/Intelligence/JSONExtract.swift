import Foundation

/// Shared JSON extraction utilities used by the LFM-backed intent
/// classifier and tool selector. Both parse the same emission styles
/// (plain JSON, ```json fences, JSON trailed by prose), so the logic
/// lives once here and is imported by both.
///
/// Extracted per the CLAUDE.local.md "Code Reusability" principle —
/// the two classifiers previously inlined identical parsers.
enum JSONExtract {
    /// Drops leading/trailing ``` or ```json fences. Returns the inner
    /// content trimmed of whitespace.
    static func stripFences(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            if let nl = s.firstIndex(of: "\n") {
                s = String(s[s.index(after: nl)...])
            } else {
                s = String(s.dropFirst(3))
            }
            if s.hasSuffix("```") {
                s = String(s.dropLast(3))
            }
        }
        return s
    }

    /// Returns the first balanced `{…}` substring in `s`, accounting for
    /// strings and escape sequences. The model sometimes emits extra
    /// prose after the JSON; truncating at the matching close-brace
    /// keeps JSONSerialization happy.
    static func firstJSONObject(in s: String) -> String? {
        var depth = 0
        var inString = false
        var escape = false
        var startIdx: String.Index?
        for idx in s.indices {
            let c = s[idx]
            if escape { escape = false; continue }
            if c == "\\" && inString { escape = true; continue }
            if c == "\"" { inString.toggle(); continue }
            if inString { continue }
            if c == "{" {
                if depth == 0 { startIdx = idx }
                depth += 1
            } else if c == "}" {
                depth -= 1
                if depth == 0, let start = startIdx {
                    return String(s[start...idx])
                }
            }
        }
        return nil
    }
}
