import Foundation

/// Placeholder extractor using regex + keyword patterns over a carrier
/// product catalog. Good enough to populate the call-trace demo and feed
/// the downstream tool selector. Swapped out for the fine-tuned Telco
/// Extraction LFM when it lands.
public struct RegexQueryExtractor: QueryExtractor {
    public init() {}

    public func extract(from query: String) -> ExtractionResult {
        let start = Date()
        let q = query.lowercased()

        let device = Self.findFirstMatch(in: q, candidates: Self.devicePatterns)
        let errorCode = Self.findErrorCode(in: q)
        let planName = Self.findFirstMatch(in: q, candidates: Self.planPatterns)
        let requestedAction = Self.findFirstMatch(in: q, candidates: Self.actionPatterns)
        let targetDevice = Self.findTargetDevice(in: q)
        let locationHint = Self.findFirstMatch(in: q, candidates: Self.locationPatterns)
        let requestedTime = Self.findFirstMatch(in: q, candidates: Self.timePatterns)
        let urgency = Self.inferUrgency(from: q)

        return ExtractionResult(
            device: device,
            errorCode: errorCode,
            planName: planName,
            requestedAction: requestedAction,
            targetDevice: targetDevice,
            locationHint: locationHint,
            requestedTime: requestedTime,
            urgency: urgency,
            runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }

    // MARK: - Device / plan / action patterns

    /// (lowercased regex pattern, canonical display name).
    private static let devicePatterns: [(String, String)] = [
        (#"\bg3100\b"#, "Fiber Router G3100"),
        (#"\be3200\b"#, "Mesh Extender E3200"),
        (#"fios\s+router|fiber\s+router"#, "Fiber Router"),
        (#"\bmesh\b"#, "Fiber Mesh"),
        (#"\bextender\b"#, "Mesh Extender"),
        (#"stream\s*tv|set[\s-]?top\s*box"#, "Stream TV"),
        (#"\brouter\b"#, "router"),
    ]

    // Ordering matters — `findFirstMatch` returns the first pattern that
    // hits, so specific named plans come before the generic speed-tier
    // shorthand. Otherwise "I'm on More Premium and want gigabit" would
    // resolve to "Fiber Gigabit" (the query the customer is asking about)
    // instead of the plan they're currently on.
    private static let planPatterns: [(String, String)] = [
        (#"\bmore\s+premium\b"#, "More Premium"),
        (#"500\s*/\s*500|500\s*mbps"#, "500/500"),
        (#"300\s*/\s*300|300\s*mbps"#, "300/300"),
        (#"\bgigabit\b|\bgig\b"#, "Fiber Gigabit"),
    ]

    private static let actionPatterns: [(String, String)] = [
        (#"\brestart\b|\breboot\b|power\s*cycle"#, "restart"),
        (#"speed\s*test"#, "speed_test"),
        (#"\bupgrade\b"#, "upgrade"),
        (#"\bdowngrade\b"#, "downgrade"),
        (#"\bcancel\b"#, "cancel"),
        (#"\bpair\b|\bwps\b"#, "pair"),
        (#"\bblock\b|pause\s+internet|bedtime"#, "pause_internet"),
        (#"\bresume\b|unblock"#, "resume_internet"),
        (#"schedule\s+(a\s+)?tech|send\s+(a\s+)?tech|technician"#, "schedule_visit"),
        (#"\bdiagnostic\b|\bdiagnostics\b"#, "run_diagnostics"),
        (#"\breset\b"#, "reset"),
    ]

    private static let locationPatterns: [(String, String)] = [
        (#"\bupstairs\b"#, "upstairs"),
        (#"\bdownstairs\b"#, "downstairs"),
        (#"\bbasement\b"#, "basement"),
        (#"\bgarage\b"#, "garage"),
        (#"\bliving room\b"#, "living room"),
        (#"\bbedroom\b"#, "bedroom"),
    ]

    // Time patterns are ordered so specific phrases ("tomorrow morning")
    // match before their generic cousins ("tomorrow"). The display string
    // feeds the call-trace UI; concrete Date resolution lives in
    // TimeExpressionParser, which the SetDowntime tool uses to get an
    // absolute timestamp.
    private static let timePatterns: [(String, String)] = [
        (#"\buntil\s+bedtime\b"#, "until bedtime"),
        (#"\bfor\s+(?:an?\s+)?hour\b"#, "for an hour"),
        (#"\bfor\s+\d+\s+hours?\b"#, "for a few hours"),
        (#"\buntil\s+tomorrow\s+morning\b"#, "until tomorrow morning"),
        (#"\buntil\s+tomorrow\b"#, "until tomorrow"),
        (#"\buntil\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b"#, "until specific time"),
        (#"\bnext week\b"#, "next week"),
        (#"\btomorrow morning\b"#, "tomorrow morning"),
        (#"\btomorrow afternoon\b"#, "tomorrow afternoon"),
        (#"\bthis weekend\b"#, "this weekend"),
        (#"\btonight\b"#, "tonight"),
    ]

    private static let urgencyHighKeywords = [
        "asap", "urgent", "emergency", "can't work", "broken", "totally down",
        "nothing works", "completely offline",
    ]
    private static let urgencyMediumKeywords = [
        "slow", "not working", "issue", "problem", "frustrated",
    ]

    // MARK: - Helpers

    private static func findFirstMatch(in text: String, candidates: [(String, String)]) -> String? {
        for (pattern, canonical) in candidates {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return canonical
            }
        }
        return nil
    }

    private static func findErrorCode(in text: String) -> String? {
        // Explicit "error XXX" / "code XXXX" patterns
        if let range = text.range(
            of: #"(?:error|code)\s*[:#]?\s*(\w{2,6})"#,
            options: .regularExpression
        ) {
            return String(text[range])
        }
        // LED state descriptions (router-light queries)
        let ledPatterns = [
            "blinking orange", "solid red", "blinking red", "solid white",
            "blinking white", "solid yellow", "blinking yellow", "no light",
        ]
        return ledPatterns.first { text.contains($0) }
    }

    private static func inferUrgency(from text: String) -> ExtractionResult.Urgency {
        if urgencyHighKeywords.contains(where: { text.contains($0) }) {
            return .high
        }
        if urgencyMediumKeywords.contains(where: { text.contains($0) }) {
            return .medium
        }
        return .low
    }

    private static func findTargetDevice(in text: String) -> String? {
        // Terminator set includes `until|for` so "pause my son's tablet
        // until 7" doesn't capture the time phrase into the target
        // device slot. Also includes `tonight` so "pause the tablet
        // tonight" resolves cleanly.
        let patterns = [
            #"(?:block|pause|resume|unblock|stop)\s+my\s+(.+?)(?:\s+(?:from|on)\s+the\s+internet|\s+(?:until|for|tonight)\b|\s+and\b|[,.!?]|$)"#,
            #"(?:block|pause|resume|unblock|stop)\s+(.+?)(?:\s+(?:from|on)\s+the\s+internet|\s+(?:until|for|tonight)\b|\s+and\b|[,.!?]|$)"#,
            #"(?:set up|create)\s+a\s+bedtime\s+for\s+([^,.!?]+)"#,
        ]

        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern) else { continue }
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            guard let match = regex.firstMatch(in: text, range: range),
                  match.numberOfRanges > 1,
                  let capture = Range(match.range(at: 1), in: text)
            else { continue }

            let value = text[capture]
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: #"^(?:the|my)\s+"#, with: "", options: .regularExpression)
            if !value.isEmpty {
                return value
            }
        }
        return nil
    }
}

/// Stub for the future real extractor. Drop in a fine-tuned
/// LFM2.5-350M-Telco-Extract here once trained.
public struct LFMQueryExtractor: QueryExtractor {
    public init() {}

    public func extract(from query: String) -> ExtractionResult {
        // TELCO EXTRACTION MODEL SWAP POINT.
        // Load `LFM2.5-350M-Telco-Extract` via LFMEngine here when trained.
        // Production uses `RegexQueryExtractor` until then.
        return ExtractionResult(runtimeMS: 0)
    }
}
