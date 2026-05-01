import Foundation

/// A span of detected PII in the user's message. Indexed into the raw query
/// string so the Privacy Shield view can render inline highlights.
public struct PIISpan: Sendable, Equatable {
    public enum Kind: String, Sendable, Codable {
        case ssn
        case email
        case phone
        case accountNumber
        case creditCard
    }

    public let kind: Kind
    public let range: Range<String.Index>
    public let matched: String

    public init(kind: Kind, range: Range<String.Index>, matched: String) {
        self.kind = kind
        self.range = range
        self.matched = matched
    }
}

/// Stateless regex-based PII detector. The banking POC has a more elaborate
/// filter; for the telco alpha we focus on the five categories most likely
/// to appear in a home-internet support flow: account #, phone, email, plus
/// SSN / credit-card as shield-demo sentinels.
///
/// Detection runs entirely on-device. Phase 2 cloud escalation pre-strips via
/// `redact(_:)` so the flagged values never leave the phone.
public final class PIIAnalyzer: Sendable {
    public init() {}

    private static let patterns: [(PIISpan.Kind, String)] = [
        (.email,         #"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"#),
        (.ssn,           #"\b\d{3}-\d{2}-\d{4}\b"#),
        (.creditCard,    #"\b(?:\d[ -]?){13,19}\b"#),
        // Account number before phone so "account number is 1234567890" wins
        // over the 10-digit phone pattern.
        (.accountNumber, #"\b(?:acct|account)(?:\s*#|\s+number)?\s+(?:is\s+|of\s+|[:=]\s*)?\d{6,16}\b"#),
        (.phone,         #"\b(?:\+?1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"#),
    ]

    public func scan(_ text: String) -> [PIISpan] {
        var spans: [PIISpan] = []
        var usedRanges: [Range<String.Index>] = []

        for (kind, pattern) in Self.patterns {
            guard let regex = try? NSRegularExpression(
                pattern: pattern,
                options: [.caseInsensitive]
            ) else { continue }

            let nsText = text as NSString
            let full = NSRange(location: 0, length: nsText.length)
            regex.enumerateMatches(in: text, range: full) { match, _, _ in
                guard
                    let nsRange = match?.range,
                    let range = Range(nsRange, in: text)
                else { return }

                // Skip overlaps with earlier, higher-priority matches.
                if usedRanges.contains(where: { $0.overlaps(range) }) {
                    return
                }
                usedRanges.append(range)
                spans.append(PIISpan(
                    kind: kind,
                    range: range,
                    matched: String(text[range])
                ))
            }
        }
        return spans.sorted { $0.range.lowerBound < $1.range.lowerBound }
    }

    /// Produce a sanitized copy with PII replaced by `[REDACTED-<kind>]`.
    public func redact(_ text: String, spans: [PIISpan]? = nil) -> String {
        let targets = spans ?? scan(text)
        guard !targets.isEmpty else { return text }

        // Apply replacements from the back so indices stay valid.
        var result = text
        for span in targets.reversed() {
            let replacement = "[REDACTED-\(span.kind.rawValue)]"
            result.replaceSubrange(span.range, with: replacement)
        }
        return result
    }
}
