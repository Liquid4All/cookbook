import XCTest
@testable import TelcoTriage

/// Pure logic tests for the `TraceRow` confidence-band decision.
/// Exercises `ConfidenceBand.classify` directly so the thresholds are
/// locked without depending on SwiftUI `Color` equality (which is
/// platform-dependent).
final class TraceRowTests: XCTestCase {

    func test_highConfidence_mapsToHighBand() {
        XCTAssertEqual(ConfidenceBand.classify(0.9), .high)
        XCTAssertEqual(ConfidenceBand.classify(0.95), .high)
        XCTAssertEqual(ConfidenceBand.classify(1.0), .high)
    }

    func test_boundary_exactly080_mapsToHigh() {
        XCTAssertEqual(ConfidenceBand.classify(0.8), .high)
    }

    func test_mediumConfidence_mapsToMediumBand() {
        XCTAssertEqual(ConfidenceBand.classify(0.65), .medium)
        XCTAssertEqual(ConfidenceBand.classify(0.79), .medium)
    }

    func test_boundary_exactly050_mapsToMedium() {
        XCTAssertEqual(ConfidenceBand.classify(0.5), .medium)
    }

    func test_boundary_justBelow050_mapsToLow() {
        XCTAssertEqual(ConfidenceBand.classify(0.499), .low)
    }

    func test_lowConfidence_mapsToLow() {
        XCTAssertEqual(ConfidenceBand.classify(0.3), .low)
        XCTAssertEqual(ConfidenceBand.classify(0.01), .low)
    }

    func test_nilConfidence_mapsToNeutral() {
        XCTAssertEqual(ConfidenceBand.classify(nil), .neutral)
    }

    func test_zeroConfidence_mapsToNeutral() {
        // LFMChatModeRouter returns confidence=0 on backend failure.
        // We should render neutral, not red — 0 means "no signal," not
        // "low confidence."
        XCTAssertEqual(ConfidenceBand.classify(0.0), .neutral)
    }

    func test_negativeConfidence_mapsToNeutral() {
        // Defensive — models shouldn't emit negative confidence but if
        // a buggy adapter ever does, treat as neutral rather than low.
        XCTAssertEqual(ConfidenceBand.classify(-0.1), .neutral)
    }
}
