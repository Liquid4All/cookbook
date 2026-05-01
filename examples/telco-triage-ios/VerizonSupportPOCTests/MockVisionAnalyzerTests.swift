import XCTest
import UIKit
@testable import VerizonSupportPOC

@MainActor
final class MockVisionAnalyzerTests: XCTestCase {
    private var packManager: SpecialistPackManager!
    private var analyzer: MockVisionAnalyzer!

    override func setUp() {
        super.setUp()
        let defaults = UserDefaults(suiteName: "MockVisionAnalyzerTests-\(UUID().uuidString)")!
        packManager = SpecialistPackManager(defaults: defaults)
        analyzer = MockVisionAnalyzer(packManager: packManager)
    }

    // MARK: - Keyword-based classification

    func test_analyze_routerKeyword_returnsRouterCategory() async throws {
        let img = Self.solidImage(color: .white, size: CGSize(width: 500, height: 400))
        let result = try await analyzer.analyze(image: img, prompt: "what do these router lights mean?")
        XCTAssertEqual(result.category, .router)
    }

    func test_analyze_billKeyword_returnsBillCategory() async throws {
        let img = Self.solidImage(color: .white, size: CGSize(width: 500, height: 400))
        let result = try await analyzer.analyze(image: img, prompt: "explain this charge on my bill")
        XCTAssertEqual(result.category, .bill)
    }

    func test_analyze_errorKeyword_returnsErrorScreenCategory() async throws {
        let img = Self.solidImage(color: .white, size: CGSize(width: 500, height: 400))
        let result = try await analyzer.analyze(image: img, prompt: "what does this error message mean?")
        XCTAssertEqual(result.category, .errorScreen)
    }

    // MARK: - Aspect-ratio / color fallbacks (no keyword)

    func test_analyze_portraitImageNoKeyword_returnsBill() async throws {
        let img = Self.solidImage(color: .gray, size: CGSize(width: 300, height: 700))
        let result = try await analyzer.analyze(image: img, prompt: "help")
        XCTAssertEqual(result.category, .bill)  // aspect < 0.7 → bill fallback
    }

    func test_analyze_whiteLandscapeImageNoKeyword_returnsRouter() async throws {
        let img = Self.solidImage(color: .white, size: CGSize(width: 800, height: 500))
        let result = try await analyzer.analyze(image: img, prompt: "help")
        XCTAssertEqual(result.category, .router)
    }

    func test_analyze_noPackInstalled_resultHasUsedPackFalse() async throws {
        let img = Self.solidImage(color: .white, size: CGSize(width: 500, height: 400))
        let result = try await analyzer.analyze(image: img, prompt: "router")
        XCTAssertFalse(result.usedPack)
        XCTAssertTrue(result.detail.lowercased().contains("pack"))
    }

    // MARK: - Helpers

    private static func solidImage(color: UIColor, size: CGSize) -> UIImage {
        UIGraphicsBeginImageContext(size)
        defer { UIGraphicsEndImageContext() }
        color.setFill()
        UIRectFill(CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? UIImage()
    }
}
