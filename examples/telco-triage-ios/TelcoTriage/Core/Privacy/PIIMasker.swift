import Foundation

/// Kind-aware redaction for PII values surfaced in UI (inspect sheets,
/// screenshots, demo recordings). Keeps enough of the original visible
/// for the demo to prove "the scanner caught this" while preventing
/// the raw PII from leaking if a screenshot is shared.
///
/// Philosophy: never mask more than necessary. For every kind we keep
/// the trailing digits / domain / first letter so the demoer can
/// visually match the redacted form against the raw query shown in
/// the "Raw query" section of the inspect sheet.
///
/// Pure function — no state, no dependencies — so it's trivially
/// reusable by any future PII-display surface (Savings dashboard,
/// escalation preview, etc.).
public enum PIIMasker {
    /// Returns a display-safe redacted form of the matched PII string.
    /// Falls back to a neutral `***` mask for inputs too short to
    /// safely partially-mask.
    public static func masked(_ span: PIISpan) -> String {
        let raw = span.matched
        switch span.kind {
        case .ssn:           return maskSSN(raw)
        case .creditCard:    return maskCard(raw)
        case .phone:         return maskPhone(raw)
        case .email:         return maskEmail(raw)
        case .accountNumber: return maskTrailing(raw, keep: 4)
        }
    }

    // MARK: - Kind-specific maskers

    /// SSN → `***-**-1234`. Expects a 9-digit value with or without
    /// dashes; falls back to the generic mask on unusual input.
    static func maskSSN(_ raw: String) -> String {
        let digits = raw.filter(\.isNumber)
        guard digits.count == 9 else { return genericMask(raw) }
        let last4 = digits.suffix(4)
        return "***-**-\(last4)"
    }

    /// Credit card → `**** **** **** 1234`. Works for 13-19 digit cards.
    static func maskCard(_ raw: String) -> String {
        let digits = raw.filter(\.isNumber)
        guard digits.count >= 12 else { return genericMask(raw) }
        return "**** **** **** \(digits.suffix(4))"
    }

    /// Phone → `(***) ***-1234` for US-style numbers. Keeps the
    /// trailing 4. For shorter/longer or non-US grouped inputs, fall
    /// back to the generic `***1234` pattern so international formats
    /// still mask.
    static func maskPhone(_ raw: String) -> String {
        let digits = raw.filter(\.isNumber)
        guard digits.count >= 4 else { return genericMask(raw) }
        if isUSPhoneShape(raw, digitCount: digits.count) {
            return "(***) ***-\(digits.suffix(4))"
        }
        return "***\(digits.suffix(4))"
    }

    /// Email → `f***@domain.com`. Keeps the first character of the
    /// local-part and the full domain, so the demoer can still match
    /// "that's my work address" without revealing the local-part.
    static func maskEmail(_ raw: String) -> String {
        guard let atIdx = raw.firstIndex(of: "@") else { return genericMask(raw) }
        let local = raw[..<atIdx]
        let domain = raw[raw.index(after: atIdx)...]
        guard let first = local.first else { return "***@\(domain)" }
        return "\(first)***@\(domain)"
    }

    // MARK: - Shared helpers

    /// Keep only the last `keep` characters; mask the rest. Used for
    /// account numbers where format is free-form.
    static func maskTrailing(_ raw: String, keep: Int) -> String {
        guard raw.count > keep else { return genericMask(raw) }
        let visible = raw.suffix(keep)
        return "***\(visible)"
    }

    private static func isUSPhoneShape(_ raw: String, digitCount: Int) -> Bool {
        guard digitCount == 10 || digitCount == 11 else { return false }
        let pattern = #"^\s*(?:\+?1[\s.-]*)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\s*$"#
        return raw.range(of: pattern, options: .regularExpression) != nil
    }

    /// Opaque fallback for inputs that don't fit a known format. Always
    /// safe — never leaks any part of the original.
    static func genericMask(_ raw: String) -> String {
        "***"
    }
}
