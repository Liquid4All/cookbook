import Foundation

/// Resolves common time-until expressions to a concrete `Date` in the
/// user's current locale/calendar.
///
/// Scope is deliberately narrow — 6 phrase families that cover the
/// overwhelming majority of "pause this device for / until X" requests
/// in customer support chat. When the extractor can't confidently
/// resolve a phrase, it returns `nil` so the caller can ask a
/// follow-up ("Didn't catch the time — can you rephrase?") rather than
/// silently guessing.
///
/// Patterns supported (case-insensitive, leading/trailing whitespace
/// tolerated):
/// 1. "until 7" / "until 7 pm" / "until 7:30 pm" — nearest future
///    occurrence of that clock time today; tomorrow if already past.
/// 2. "until tomorrow morning" — 8:00 AM next day.
/// 3. "until tomorrow" — 8:00 AM next day (same default as "morning").
/// 4. "until bedtime" / "tonight" — 9:00 PM today (or tomorrow if
///    already past).
/// 5. "for an hour" / "for 2 hours" — now + duration.
///
/// Non-goals: ambiguous natural-language ("in a while", "later"), past
/// timestamps, recurring windows, timezone-aware DST edge cases —
/// these hand off to an LFM2.5-350M-Telco-Extract fine-tune when it
/// ships.
public struct TimeExpressionParser: Sendable {
    private let calendar: Calendar
    private let now: @Sendable () -> Date

    public init(
        calendar: Calendar = .current,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.calendar = calendar
        self.now = now
    }

    public func resolve(_ query: String) -> Date? {
        let text = query.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }

        if let date = matchForDuration(in: text) { return date }
        if let date = matchUntilTomorrow(in: text) { return date }
        if let date = matchBedtimeOrTonight(in: text) { return date }
        if let date = matchUntilClockTime(in: text) { return date }
        return nil
    }

    // MARK: - "for an hour" / "for 2 hours"

    private func matchForDuration(in text: String) -> Date? {
        if text.range(of: #"\bfor\s+an?\s+hour\b"#, options: .regularExpression) != nil {
            return calendar.date(byAdding: .hour, value: 1, to: now())
        }
        guard let range = text.range(
            of: #"\bfor\s+(\d+)\s+hours?\b"#,
            options: .regularExpression
        ) else { return nil }

        let phrase = text[range]
        guard let digits = phrase.split(whereSeparator: { !$0.isNumber }).first,
              let hours = Int(digits),
              hours > 0, hours <= 24
        else { return nil }
        return calendar.date(byAdding: .hour, value: hours, to: now())
    }

    // MARK: - "until tomorrow [morning]"

    private func matchUntilTomorrow(in text: String) -> Date? {
        guard text.range(of: #"\buntil\s+tomorrow\b"#, options: .regularExpression) != nil
        else { return nil }
        // Treat both "until tomorrow" and "until tomorrow morning" as 8 AM
        // next day — the canonical "morning" anchor.
        let tomorrow = calendar.date(byAdding: .day, value: 1, to: startOfDay()) ?? now()
        return calendar.date(bySettingHour: 8, minute: 0, second: 0, of: tomorrow)
    }

    // MARK: - "until bedtime" / "tonight"

    private func matchBedtimeOrTonight(in text: String) -> Date? {
        let bedtimeKeywords = [
            #"\buntil\s+bedtime\b"#,
            #"\btonight\b"#,
        ]
        guard bedtimeKeywords.contains(where: {
            text.range(of: $0, options: .regularExpression) != nil
        }) else { return nil }
        return nextOccurrence(hour: 21, minute: 0)
    }

    // MARK: - "until 7" / "until 7 pm" / "until 7:30 pm"

    private func matchUntilClockTime(in text: String) -> Date? {
        let pattern = #"\buntil\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(
                in: text,
                range: NSRange(text.startIndex..<text.endIndex, in: text)
              ),
              match.numberOfRanges >= 4,
              let hourRange = Range(match.range(at: 1), in: text),
              let hour = Int(text[hourRange])
        else { return nil }

        let minute: Int = {
            guard let minRange = Range(match.range(at: 2), in: text) else { return 0 }
            return Int(text[minRange]) ?? 0
        }()

        let suffix: String? = {
            guard let sRange = Range(match.range(at: 3), in: text) else { return nil }
            return String(text[sRange])
        }()

        let resolvedHour = resolve24Hour(hour: hour, suffix: suffix)
        guard let resolvedHour else { return nil }
        return nextOccurrence(hour: resolvedHour, minute: minute)
    }

    private func resolve24Hour(hour: Int, suffix: String?) -> Int? {
        guard (0...23).contains(hour) else { return nil }
        switch suffix {
        case "am":
            // 12 AM = midnight
            return hour == 12 ? 0 : hour
        case "pm":
            // 12 PM = noon; 1 PM–11 PM add 12
            return hour == 12 ? 12 : hour + 12
        default:
            // No suffix: assume PM for 1..11 (the normal evening-hour
            // range customers use when saying "pause until 7" or
            // "pause until 9"). "Until 12" stays literal (noon) —
            // callers who want midnight should say "until 12 am".
            // "Until 10" and "until 11" map to 10 PM / 11 PM, which
            // is the dominant downtime scheduling intent.
            if hour >= 1 && hour <= 11 {
                return hour + 12
            }
            return hour
        }
    }

    // MARK: - Helpers

    private func startOfDay() -> Date {
        calendar.startOfDay(for: now())
    }

    /// Returns the next future occurrence of the given hour/minute. If
    /// the time has already passed today, returns tomorrow's occurrence.
    private func nextOccurrence(hour: Int, minute: Int) -> Date? {
        guard let today = calendar.date(
            bySettingHour: hour,
            minute: minute,
            second: 0,
            of: startOfDay()
        ) else { return nil }

        if today > now() {
            return today
        }
        return calendar.date(byAdding: .day, value: 1, to: today)
    }
}
