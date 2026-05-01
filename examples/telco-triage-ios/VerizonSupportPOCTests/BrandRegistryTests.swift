import XCTest
@testable import VerizonSupportPOC

@MainActor
final class BrandRegistryTests: XCTestCase {
    func test_defaultRegistry_startsOnTelcoTriage() {
        let defaults = UserDefaults(suiteName: "BrandRegistryTests-default-\(UUID().uuidString)")!
        let registry = BrandRegistry(defaults: defaults)
        XCTAssertEqual(registry.selected.id, "telco-triage")
    }

    func test_select_switchesBrand() {
        let defaults = UserDefaults(suiteName: "BrandRegistryTests-switch-\(UUID().uuidString)")!
        let registry = BrandRegistry(defaults: defaults)
        registry.select("liquid")
        XCTAssertEqual(registry.selected.id, "liquid")
    }

    func test_selectedBrand_persistsAcrossInstances() {
        let suiteName = "BrandRegistryTests-persist-\(UUID().uuidString)"
        let defaults1 = UserDefaults(suiteName: suiteName)!
        let registry1 = BrandRegistry(defaults: defaults1)
        registry1.select("liquid")

        let defaults2 = UserDefaults(suiteName: suiteName)!
        let registry2 = BrandRegistry(defaults: defaults2)
        XCTAssertEqual(registry2.selected.id, "liquid")
    }

    func test_unknownID_isIgnored() {
        let defaults = UserDefaults(suiteName: "BrandRegistryTests-unknown-\(UUID().uuidString)")!
        let registry = BrandRegistry(defaults: defaults)
        registry.select("does-not-exist")
        XCTAssertEqual(registry.selected.id, "telco-triage")
    }
}
