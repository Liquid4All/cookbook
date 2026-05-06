import XCTest
@testable import TelcoTriage

@MainActor
final class CustomerContextTests: XCTestCase {
    func test_demoProfile_hasExpectedEquipment() {
        let ctx = CustomerContext()
        XCTAssertEqual(ctx.profile.equipment.count, 3)
        XCTAssertTrue(ctx.profile.equipment.contains { $0.kind == .router })
        XCTAssertTrue(ctx.profile.equipment.contains { $0.kind == .extender })
        XCTAssertTrue(ctx.profile.equipment.contains { $0.kind == .setTopBox })
        XCTAssertEqual(ctx.managedDevices.count, 3)
    }

    func test_demoProfile_usesFakeName() {
        // Regression for S7 — never check in real personal data.
        let ctx = CustomerContext()
        XCTAssertEqual(ctx.profile.firstName, "Alex")
        XCTAssertNotEqual(ctx.profile.address.line1, "14305 Branham Lane")
    }

    func test_demoProfile_hasHomeNetworkFacts() {
        let ctx = CustomerContext()
        XCTAssertEqual(ctx.profile.homeNetwork.ssid, "Alex-Fiber-Home")
        XCTAssertEqual(ctx.profile.homeNetwork.guestSSID, "Alex-Fiber-Guest")
        XCTAssertEqual(ctx.profile.homeNetwork.securityMode, "WPA3/WPA2")
    }

    func test_customerProfileFactResolver_mapsSSIDPhrases() {
        let resolver = CustomerProfileFactResolver()
        XCTAssertEqual(resolver.resolve("Can you tell me what's my SSID?"), .homeSSID)
        XCTAssertEqual(resolver.resolve("What is my Wi-Fi name?"), .homeSSID)
        XCTAssertNil(resolver.resolve("Summarize my home network"))
    }

    func test_markRouterRebooted_setsStatusOnlineAndLastRebootRecent() {
        let ctx = CustomerContext()
        let router = ctx.profile.equipment.first { $0.kind == .router }!
        let priorReboot = router.lastReboot

        ctx.markRouterRebooted(serial: router.serial)
        let updated = ctx.profile.equipment.first { $0.serial == router.serial }!

        XCTAssertEqual(updated.status, .online)
        XCTAssertNotEqual(updated.lastReboot, priorReboot)
        if let newReboot = updated.lastReboot {
            XCTAssertLessThan(Date().timeIntervalSince(newReboot), 2.0)
        } else {
            XCTFail("lastReboot should be set")
        }
    }

    func test_markRouterRebooted_unknownSerial_noMutation() {
        let ctx = CustomerContext()
        let before = ctx.profile.equipment
        ctx.markRouterRebooted(serial: "DOES-NOT-EXIST")
        XCTAssertEqual(ctx.profile.equipment, before)
    }

    func test_markEquipmentStatus_updatesOnlyMatchingSerial() {
        let ctx = CustomerContext()
        let extender = ctx.profile.equipment.first { $0.kind == .extender }!
        XCTAssertEqual(extender.status, .unhealthy)

        ctx.markEquipmentStatus(serial: extender.serial, status: .online)
        let updated = ctx.profile.equipment.first { $0.serial == extender.serial }!
        XCTAssertEqual(updated.status, .online)

        // Router should be untouched
        let router = ctx.profile.equipment.first { $0.kind == .router }!
        XCTAssertEqual(router.status, .online)
    }

    func test_updateParentalControls_pausesManagedDevice() {
        let ctx = CustomerContext()
        let updated = ctx.updateParentalControls(deviceName: "son's tablet", action: "pause_internet")
        XCTAssertEqual(updated?.accessState, .paused)
        XCTAssertEqual(ctx.managedDevices.first?.accessState, .paused)
    }

    func test_scheduleTechnician_setsAppointment() {
        let ctx = CustomerContext()
        ctx.scheduleTechnician(windowLabel: "tomorrow afternoon", note: "packet loss")
        XCTAssertEqual(ctx.serviceAppointment?.windowLabel, "tomorrow afternoon")
    }

    func test_equipmentWith_copiesWithSelectiveOverride() {
        let original = CustomerProfile.Equipment(
            kind: .router,
            model: "G3100",
            serial: "S1",
            status: .offline,
            lastReboot: nil
        )
        let now = Date()
        let updated = original.with(status: .online, lastReboot: now)
        XCTAssertEqual(updated.kind, .router)
        XCTAssertEqual(updated.serial, "S1")
        XCTAssertEqual(updated.status, .online)
        XCTAssertEqual(updated.lastReboot, now)

        // Partial override — status only
        let partial = original.with(status: .unhealthy)
        XCTAssertEqual(partial.status, .unhealthy)
        XCTAssertEqual(partial.lastReboot, nil)
    }
}
