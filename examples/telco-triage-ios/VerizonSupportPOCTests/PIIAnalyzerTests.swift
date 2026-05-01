import XCTest
@testable import VerizonSupportPOC

final class PIIAnalyzerTests: XCTestCase {
    private let analyzer = PIIAnalyzer()

    func test_scan_detectsSSN() {
        let spans = analyzer.scan("My SSN is 123-45-6789 please help")
        XCTAssertEqual(spans.count, 1)
        XCTAssertEqual(spans.first?.kind, .ssn)
    }

    func test_scan_detectsEmail() {
        let spans = analyzer.scan("Contact me at user@example.com for details.")
        XCTAssertEqual(spans.first?.kind, .email)
    }

    func test_scan_detectsPhone() {
        let spans = analyzer.scan("Call me at (555) 123-4567 tomorrow")
        XCTAssertEqual(spans.first?.kind, .phone)
    }

    func test_scan_detectsAccountNumber() {
        let spans = analyzer.scan("My account number is 1234567890 and it's acting up.")
        XCTAssertTrue(spans.contains(where: { $0.kind == .accountNumber }))
    }

    func test_scan_detectsCreditCard() {
        let spans = analyzer.scan("Card is 4532 1234 5678 9010")
        XCTAssertTrue(spans.contains(where: { $0.kind == .creditCard }))
    }

    func test_redact_replacesSSN() {
        let input = "SSN: 123-45-6789"
        let redacted = analyzer.redact(input)
        XCTAssertFalse(redacted.contains("123-45-6789"))
        XCTAssertTrue(redacted.contains("[REDACTED-ssn]"))
    }

    func test_scan_returnsEmptyForPlainText() {
        let spans = analyzer.scan("My router is slow, can you help?")
        XCTAssertTrue(spans.isEmpty)
    }
}
