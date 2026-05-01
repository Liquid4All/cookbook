import XCTest
@testable import TelcoTriage

final class TimeExpressionParserTests: XCTestCase {

    /// Fixed "now" anchor for deterministic tests: 2026-04-18 at 14:00
    /// local time (mid-afternoon, far from any day boundary so we
    /// don't accidentally cross a 9pm bedtime window in the middle of
    /// a run).
    private let anchorHour = 14
    private let anchorMinute = 0
    private var calendar: Calendar!
    private var anchorNow: Date!
    private var parser: TimeExpressionParser!

    override func setUp() {
        super.setUp()
        calendar = Calendar(identifier: .gregorian)
        var comps = DateComponents()
        comps.year = 2026
        comps.month = 4
        comps.day = 18
        comps.hour = anchorHour
        comps.minute = anchorMinute
        anchorNow = calendar.date(from: comps)!
        let captured = anchorNow!
        parser = TimeExpressionParser(calendar: calendar, now: { captured })
    }

    // MARK: - "until N" / "until N pm"

    func test_untilClockTime_defaultsToPMForSmallHours() {
        // "pause until 7" at 2 PM → same-day 7 PM.
        let result = parser.resolve("pause my son's tablet until 7")
        XCTAssertNotNil(result)
        let components = calendar.dateComponents([.year, .month, .day, .hour], from: result!)
        XCTAssertEqual(components.day, 18)
        XCTAssertEqual(components.hour, 19)
    }

    func test_untilClockTime_withPMSuffix() {
        let result = parser.resolve("pause until 7pm")
        XCTAssertEqual(calendar.component(.hour, from: result!), 19)
    }

    func test_untilClockTime_withAMSuffix_nextDay() {
        // "until 6 am" at 2 PM today → tomorrow 6 AM.
        let result = parser.resolve("block it until 6 am")
        XCTAssertNotNil(result)
        let day = calendar.component(.day, from: result!)
        XCTAssertEqual(day, 19)
        XCTAssertEqual(calendar.component(.hour, from: result!), 6)
    }

    func test_untilClockTime_withMinutes() {
        let result = parser.resolve("pause until 7:30 pm")
        XCTAssertEqual(calendar.component(.hour, from: result!), 19)
        XCTAssertEqual(calendar.component(.minute, from: result!), 30)
    }

    // MARK: - "until tomorrow"

    func test_untilTomorrow_resolvesTo8AMNextDay() {
        let result = parser.resolve("stop it until tomorrow")
        XCTAssertNotNil(result)
        XCTAssertEqual(calendar.component(.day, from: result!), 19)
        XCTAssertEqual(calendar.component(.hour, from: result!), 8)
    }

    func test_untilTomorrowMorning_sameAsUntilTomorrow() {
        let a = parser.resolve("until tomorrow morning")
        let b = parser.resolve("until tomorrow")
        XCTAssertEqual(a, b)
    }

    // MARK: - "for N hours"

    func test_forHours_addsDuration() {
        let result = parser.resolve("pause for 2 hours")
        XCTAssertNotNil(result)
        let expected = calendar.date(byAdding: .hour, value: 2, to: anchorNow)!
        XCTAssertEqual(result, expected)
    }

    func test_forAnHour() {
        let result = parser.resolve("pause for an hour")
        XCTAssertNotNil(result)
        let expected = calendar.date(byAdding: .hour, value: 1, to: anchorNow)!
        XCTAssertEqual(result, expected)
    }

    // MARK: - bedtime / tonight

    func test_untilBedtime_resolvesTo9PM() {
        let result = parser.resolve("pause it until bedtime")
        XCTAssertEqual(calendar.component(.hour, from: result!), 21)
    }

    func test_tonight_resolvesTo9PM() {
        let result = parser.resolve("pause it tonight")
        XCTAssertEqual(calendar.component(.hour, from: result!), 21)
    }

    // MARK: - Unresolvable

    func test_vague_returnsNil() {
        XCTAssertNil(parser.resolve("pause it for a while"))
        XCTAssertNil(parser.resolve("pause it later"))
        XCTAssertNil(parser.resolve(""))
    }
}
