import XCTest
@testable import TelcoTriage

@MainActor
final class ToolRegistryTests: XCTestCase {
    private var registry: ToolRegistry!
    private var context: CustomerContext!

    override func setUp() {
        super.setUp()
        context = CustomerContext()
        registry = ToolRegistry.default(customerContext: context)
    }

    func test_default_registersEightTools() {
        // 8 tools — `set-downtime` intentionally excluded pending the
        // v3 tool-selector retrain documented in docs/FUTURE_SCOPE.md.
        XCTAssertEqual(registry.all.count, 8)
    }

    func test_lookupByIntent_returnsExpectedTool() {
        XCTAssertEqual(registry.tool(for: .restartRouter)?.id, "restart-router")
        XCTAssertEqual(registry.tool(for: .runSpeedTest)?.id, "run-speed-test")
        XCTAssertEqual(registry.tool(for: .checkConnection)?.id, "check-connection")
        XCTAssertEqual(registry.tool(for: .wpsPair)?.id, "enable-wps")
        XCTAssertEqual(registry.tool(for: .runDiagnostics)?.id, "run-diagnostics")
        XCTAssertEqual(registry.tool(for: .scheduleTechnician)?.id, "schedule-technician")
        XCTAssertEqual(registry.tool(for: .toggleParentalControls)?.id, "toggle-parental-controls")
        XCTAssertEqual(registry.tool(for: .rebootExtender)?.id, "reboot-extender")
    }

    func test_restartRouter_updatesEquipmentLastReboot() async throws {
        guard let tool = registry.tool(id: "restart-router") else {
            XCTFail("restart-router not registered")
            return
        }
        let routerSerial = context.profile.equipment.first(where: { $0.kind == .router })!.serial
        let before = context.profile.equipment.first(where: { $0.serial == routerSerial })!.lastReboot

        let result = try await tool.execute(arguments: .empty)

        XCTAssertEqual(result.status, .success)
        let after = context.profile.equipment.first(where: { $0.serial == routerSerial })!.lastReboot
        XCTAssertNotEqual(before, after)
    }

    func test_runSpeedTest_returnsStructuredResult() async throws {
        guard let tool = registry.tool(id: "run-speed-test") else {
            XCTFail("run-speed-test not registered")
            return
        }
        let result = try await tool.execute(arguments: .empty)
        XCTAssertEqual(result.status, .success)
        XCTAssertNotNil(result.structuredPayload["down_mbps"])
        XCTAssertNotNil(result.structuredPayload["up_mbps"])
    }

    func test_destructiveTools_requireConfirmation() {
        XCTAssertEqual(registry.tool(id: "restart-router")?.requiresConfirmation, true)
        XCTAssertEqual(registry.tool(id: "enable-wps")?.requiresConfirmation, true)
        XCTAssertEqual(registry.tool(id: "run-speed-test")?.requiresConfirmation, false)
        XCTAssertEqual(registry.tool(id: "check-connection")?.requiresConfirmation, false)
    }

    func test_toggleParentalControls_updatesManagedDeviceState() async throws {
        guard let tool = registry.tool(id: "toggle-parental-controls") else {
            XCTFail("toggle-parental-controls not registered")
            return
        }

        let result = try await tool.execute(arguments: ToolArguments([
            "action": "pause_internet",
            "target_device": "son's tablet",
        ]))

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(
            context.managedDevices.first(where: { $0.name == "Son's Tablet" })?.accessState,
            .paused
        )
    }

    func test_scheduleTechnician_createsAppointment() async throws {
        guard let tool = registry.tool(id: "schedule-technician") else {
            XCTFail("schedule-technician not registered")
            return
        }

        let result = try await tool.execute(arguments: ToolArguments([
            "preferred_date": "next week",
            "issue_summary": "packet loss on extender path",
        ]))

        XCTAssertEqual(result.status, .success)
        XCTAssertEqual(context.serviceAppointment?.windowLabel, "next week")
    }
}
