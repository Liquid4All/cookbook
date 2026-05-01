import XCTest
@testable import VerizonSupportPOC

final class SurfaceModeTests: XCTestCase {
    func test_allCases_coverBothAudiences() {
        let all = SurfaceMode.allCases
        XCTAssertEqual(all.count, 2)
        XCTAssertTrue(all.contains(.customer))
        XCTAssertTrue(all.contains(.operator_))
    }

    func test_customerMode_hasCustomerFacingLabels() {
        XCTAssertEqual(SurfaceMode.customer.displayName, "Customer")
        XCTAssertEqual(SurfaceMode.customer.longDisplayName, "Customer view")
    }

    func test_operatorMode_hasOperatorFacingLabels() {
        XCTAssertEqual(SurfaceMode.operator_.displayName, "Operator")
        XCTAssertEqual(SurfaceMode.operator_.longDisplayName, "Operator view")
    }

    func test_accessibilityHints_exist() {
        for mode in SurfaceMode.allCases {
            XCTAssertFalse(
                mode.accessibilityHint.isEmpty,
                "\(mode.rawValue) needs an accessibility hint"
            )
        }
    }

    func test_rawValueStability() {
        // The raw value is persisted — regressions here break sessions.
        XCTAssertEqual(SurfaceMode.customer.rawValue, "customer")
        XCTAssertEqual(SurfaceMode.operator_.rawValue, "operator")
    }
}
