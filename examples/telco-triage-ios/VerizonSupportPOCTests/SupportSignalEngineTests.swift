import XCTest
@testable import VerizonSupportPOC

@MainActor
final class SupportSignalEngineTests: XCTestCase {

    // MARK: - Extender health

    func test_unhealthyExtender_producesAttentionSignal() {
        // The demo profile ships an unhealthy extender, so the default
        // context should surface the signal without any extra setup.
        let context = CustomerContext()
        let engine = SupportSignalEngine(context: context)

        let extenderSignals = engine.activeSignals.filter { $0.source == "equipment.extender.status" }
        XCTAssertEqual(extenderSignals.count, 1)
        XCTAssertEqual(extenderSignals.first?.severity, .attention)
        XCTAssertEqual(extenderSignals.first?.suggestedToolID, "reboot-extender")
    }

    // MARK: - Ranking

    func test_activeSignals_sortsByseverity() {
        let context = CustomerContext()
        let engine = SupportSignalEngine(context: context)

        let signals = engine.activeSignals
        XCTAssertGreaterThanOrEqual(signals.count, 2)
        // Each earlier signal must not be less severe than the next.
        for i in 0..<(signals.count - 1) {
            XCTAssertLessThanOrEqual(
                signals[i].severity.sortKey,
                signals[i + 1].severity.sortKey,
                "signal at \(i) should not be less severe than the one at \(i + 1)"
            )
        }
    }

    func test_activeSignals_capsAtThree() {
        let context = CustomerContext()
        let engine = SupportSignalEngine(context: context)
        XCTAssertLessThanOrEqual(engine.activeSignals.count, 3)
    }

    // MARK: - Suggested prompts preserve the NLU value prop

    func test_everySignal_providesSuggestedPrompt() {
        let context = CustomerContext()
        let engine = SupportSignalEngine(context: context)

        for signal in engine.activeSignals {
            XCTAssertFalse(
                signal.suggestedPrompt.isEmpty,
                "Signal \(signal.id) must carry a suggested natural-language prompt"
            )
        }
    }

    // MARK: - Refresh on mutation

    /// Mutating context state and re-running `refresh()` must update
    /// the signal set. This verifies the re-scoring logic directly;
    /// the Combine subscription that wires this automatically is a
    /// framework-level concern covered by the engine's ctor.
    func test_refresh_afterExtenderResolved_clearsSignal() {
        let context = CustomerContext()
        let engine = SupportSignalEngine(context: context)
        XCTAssertTrue(
            engine.activeSignals.contains { $0.source == "equipment.extender.status" }
        )

        context.markEquipmentStatus(serial: "EX22118B8QN", status: .online)
        engine.refresh()

        XCTAssertFalse(
            engine.activeSignals.contains { $0.source == "equipment.extender.status" }
        )
    }

    // MARK: - Injectable clock (W1)

    /// With a pristine context (no demo-profile competing signals),
    /// a paused device whose window has elapsed must produce the
    /// past-window signal when the injected clock is after `until`,
    /// and NOT produce it when the clock is before `until`.
    func test_pausedPastWindow_firesOnlyWhenClockPastWindow() {
        let baseline = Date(timeIntervalSince1970: 1_800_000_000) // fixed deterministic instant
        let windowEnd = baseline.addingTimeInterval(-3600)         // 1 h before baseline
        let context = Self.makePristineContext()
        context.applyDowntime(deviceName: "family-tablet", until: windowEnd)

        let pastClock = SupportSignalEngine(context: context, now: { baseline })
        XCTAssertTrue(
            pastClock.activeSignals.contains { $0.source == "managedDevices.downtimeUntil" },
            "signal should fire when current time is past window"
        )

        // Same state, clock set BEFORE the window ended: no signal.
        let beforeContext = Self.makePristineContext()
        beforeContext.applyDowntime(deviceName: "family-tablet", until: windowEnd)
        let preClock = SupportSignalEngine(
            context: beforeContext,
            now: { windowEnd.addingTimeInterval(-7200) } // 2 h before window ended
        )
        XCTAssertFalse(
            preClock.activeSignals.contains { $0.source == "managedDevices.downtimeUntil" }
        )
    }

    // MARK: - Pristine context builder

    /// Demo profile ships with an unhealthy extender + 3 speed-test
    /// failures + 42 outage minutes + heavy usage, which between
    /// them create up to 4 signals that compete with the one a
    /// given test wants to observe. This builder returns a
    /// "nothing wrong" context so a test can be sure any signal it
    /// sees came from the mutation it performed.
    private static func makePristineContext() -> CustomerContext {
        let profile = CustomerProfile(
            customerID: "PRISTINE",
            firstName: "T",
            lastName: "U",
            plan: .init(
                name: "Fiber Basic",
                downSpeedMbps: 100,
                upSpeedMbps: 100,
                monthlyPrice: 50
            ),
            address: .init(line1: "1 Test", city: "Anywhere", state: "CA", zip: "00000"),
            equipment: [
                .init(
                    kind: .router,
                    model: "Fiber Router G3100",
                    serial: "ROUTER",
                    status: .online,
                    lastReboot: nil
                ),
            ],
            recentIssues: [],
            usage: .init(
                periodDays: 30,
                downloadedGB: 10,
                uploadedGB: 1,
                connectedDeviceCount: 2,
                peakDeviceCount: 3,
                troubleshootCount: 0,
                avgDownMbps: 90,
                avgUpMbps: 50,
                speedTestFailures: 0,
                outageMinutes: 0,
                billCyclesAtOrOverCap: 0,
                activeBoltOns: []
            )
        )
        let devices: [CustomerContext.ManagedDevice] = [
            .init(
                id: "family-tablet",
                name: "family-tablet",
                kind: .tablet,
                location: "Kitchen",
                accessState: .unrestricted,
                detail: "demo device"
            ),
        ]
        return CustomerContext(
            profile: profile,
            managedDevices: devices,
            serviceAppointment: nil
        )
    }
}
