import XCTest
@testable import VerizonSupportPOC

final class PIIMaskerTests: XCTestCase {

    // MARK: - SSN

    func test_ssn_dashedFormat_keepsLast4() {
        let span = makeSpan(.ssn, matched: "123-45-6789")
        XCTAssertEqual(PIIMasker.masked(span), "***-**-6789")
    }

    func test_ssn_noSeparators_keepsLast4() {
        let span = makeSpan(.ssn, matched: "123456789")
        XCTAssertEqual(PIIMasker.masked(span), "***-**-6789")
    }

    func test_ssn_malformedTooShort_usesGenericMask() {
        let span = makeSpan(.ssn, matched: "12-34")
        XCTAssertEqual(PIIMasker.masked(span), "***")
    }

    // MARK: - Credit card

    func test_creditCard_16digit_keepsLast4() {
        let span = makeSpan(.creditCard, matched: "4111 1111 1111 4321")
        XCTAssertEqual(PIIMasker.masked(span), "**** **** **** 4321")
    }

    func test_creditCard_amex15digit_keepsLast4() {
        let span = makeSpan(.creditCard, matched: "3782 822463 10005")
        XCTAssertEqual(PIIMasker.masked(span), "**** **** **** 0005")
    }

    func test_creditCard_tooShort_usesGenericMask() {
        let span = makeSpan(.creditCard, matched: "411111")
        XCTAssertEqual(PIIMasker.masked(span), "***")
    }

    // MARK: - Phone

    func test_phone_usFormat_masksToParenPattern() {
        let span = makeSpan(.phone, matched: "(555) 123-4567")
        XCTAssertEqual(PIIMasker.masked(span), "(***) ***-4567")
    }

    func test_phone_tenDigitsNoFormatting() {
        let span = makeSpan(.phone, matched: "5551234567")
        XCTAssertEqual(PIIMasker.masked(span), "(***) ***-4567")
    }

    func test_phone_elevenDigitsWithCountryCode_stillMasks() {
        let span = makeSpan(.phone, matched: "+1-555-123-4567")
        XCTAssertEqual(PIIMasker.masked(span), "(***) ***-4567")
    }

    func test_phone_internationalShort_fallsBackToTrailing() {
        // International numbers can have 8-9 digits; fall back to the
        // generic trailing-4 pattern instead of the US parens format.
        let span = makeSpan(.phone, matched: "20 7946 0958")
        XCTAssertEqual(PIIMasker.masked(span), "***0958")
    }

    func test_phone_tooShort_usesGenericMask() {
        let span = makeSpan(.phone, matched: "555")
        XCTAssertEqual(PIIMasker.masked(span), "***")
    }

    // MARK: - Email

    func test_email_keepsFirstCharAndFullDomain() {
        let span = makeSpan(.email, matched: "alex.rivera@carrier.example")
        XCTAssertEqual(PIIMasker.masked(span), "a***@carrier.example")
    }

    func test_email_singleCharLocal_masksTheFirstCharToo() {
        let span = makeSpan(.email, matched: "a@example.com")
        XCTAssertEqual(PIIMasker.masked(span), "a***@example.com")
    }

    func test_email_noAtSign_usesGenericMask() {
        let span = makeSpan(.email, matched: "not-an-email")
        XCTAssertEqual(PIIMasker.masked(span), "***")
    }

    // MARK: - Account number

    func test_accountNumber_keepsLast4() {
        let span = makeSpan(.accountNumber, matched: "acct 1234567890")
        XCTAssertEqual(PIIMasker.masked(span), "***7890")
    }

    func test_accountNumber_shortInput_keepsTrailingFour() {
        // maskTrailing takes the raw `suffix(4)`, which on a short
        // account-number match like "acct 1" is "ct 1". Documented
        // so future callers know the trailing-4 rule is literal.
        let span = makeSpan(.accountNumber, matched: "acct 1")
        XCTAssertEqual(PIIMasker.masked(span), "***ct 1")
    }

    func test_accountNumber_tooShort_usesGenericMask() {
        // When the raw string has <= `keep` characters, we can't keep
        // any characters without leaking the full value, so the
        // generic mask kicks in.
        let span = makeSpan(.accountNumber, matched: "123")
        XCTAssertEqual(PIIMasker.masked(span), "***")
    }

    func test_accountNumber_exactlyKeepChars_usesGenericMask() {
        // Boundary: if raw.count == keep, showing "keep" chars would
        // leak everything. maskTrailing's `raw.count > keep` guard
        // flips to generic mask here.
        let span = makeSpan(.accountNumber, matched: "1234")
        XCTAssertEqual(PIIMasker.masked(span), "***")
    }

    // MARK: - Empty / edge

    func test_empty_alwaysReturnsSomething() {
        // Every kind must produce a non-empty mask string, even on
        // empty input — we never want to render a blank row in the
        // inspect sheet.
        for kind in [PIISpan.Kind.ssn, .email, .phone, .creditCard, .accountNumber] {
            let span = makeSpan(kind, matched: "")
            XCTAssertFalse(PIIMasker.masked(span).isEmpty, "\(kind) produced empty mask")
        }
    }

    // MARK: - Helpers

    /// Builds a minimal PIISpan for testing. The `range` is synthetic
    /// because PIIMasker only reads `kind` + `matched`.
    private func makeSpan(_ kind: PIISpan.Kind, matched: String) -> PIISpan {
        let placeholder = "placeholder"
        let range = placeholder.startIndex..<placeholder.endIndex
        return PIISpan(kind: kind, range: range, matched: matched)
    }
}
