import XCTest
@testable import TelcoTriage

/// Pure-function tests for the ADR-015 5-lane router.
final class TelcoLaneRouterTests: XCTestCase {

    // MARK: - Lane: degraded (no ADR-015 heads)

    func test_degraded_when_adr015HeadsAbsent() {
        let v = TelcoDecisionVector(
            inputHash: 1,
            mode: .unavailable,
            kbEntry: .unavailable,
            tool: .unavailable,
            inferenceMode: .pairedAdapters,
            forwardPassMs: 0,
            headProjectionMs: 0
        )
        guard case .degraded = TelcoLaneRouter.route(v) else {
            return XCTFail("expected degraded when ADR-015 heads missing")
        }
    }

    // MARK: - Lane: localAnswer

    func test_localAnswer_for_deviceSetup_intent() {
        let v = adr015Vector(
            supportIntent: ("device_setup", 0.92),
            routingLane: ("local_answer", 0.85),
            cloudRequirements: [],
            requiredTool: ("no_tool", 0.7)
        )
        guard case .localAnswer(let intent, _) = TelcoLaneRouter.route(v) else {
            return XCTFail("expected localAnswer")
        }
        XCTAssertEqual(intent, .deviceSetup)
    }

    // MARK: - Lane: localTool

    func test_localTool_carries_required_tool() {
        let v = adr015Vector(
            supportIntent: ("troubleshooting", 0.95),
            routingLane: ("local_tool", 0.82),
            cloudRequirements: [],
            requiredTool: ("speed_test", 0.91)
        )
        guard case .localTool(let tool, let intent, _) = TelcoLaneRouter.route(v) else {
            return XCTFail("expected localTool")
        }
        XCTAssertEqual(tool, .speedTest)
        XCTAssertEqual(intent, .troubleshooting)
    }

    // MARK: - Lane: cloudAssist

    func test_cloudAssist_for_billing_intent() {
        let v = adr015Vector(
            supportIntent: ("billing", 0.88),
            routingLane: ("cloud_assist", 0.79),
            cloudRequirements: ["billing_record", "account_state"],
            requiredTool: ("cloud_only", 0.83),
            piiRisk: ("contains_account_data", 0.9),
            slotCompleteness: ["missing_account_auth"]
        )
        guard case .cloudAssist(let intent, let reqs, let slots, let pii, _) = TelcoLaneRouter.route(v) else {
            return XCTFail("expected cloudAssist")
        }
        XCTAssertEqual(intent, .billing)
        XCTAssertEqual(reqs, [.billingRecord, .accountState])
        XCTAssertEqual(slots, [.missingAccountAuth])
        XCTAssertEqual(pii, .containsAccountData)
    }

    // MARK: - Lane: humanEscalation

    func test_humanEscalation_when_supportIntent_is_agent_handoff() {
        let v = adr015Vector(
            supportIntent: ("agent_handoff", 0.98),
            routingLane: ("human_escalation", 0.95),
            cloudRequirements: [],
            requiredTool: ("no_tool", 0.7)
        )
        guard case .humanEscalation = TelcoLaneRouter.route(v) else {
            return XCTFail("expected humanEscalation")
        }
    }

    func test_humanEscalation_when_complaint_signal() {
        let v = adr015Vector(
            supportIntent: ("billing", 0.85),
            routingLane: ("cloud_assist", 0.55),
            cloudRequirements: ["billing_record"],
            requiredTool: ("cloud_only", 0.6),
            customerEscalationRisk: ("complaint", 0.78)
        )
        // Complaint should override the cloud_assist signal.
        guard case .humanEscalation = TelcoLaneRouter.route(v) else {
            return XCTFail("expected humanEscalation when escalation_risk=complaint")
        }
    }

    // MARK: - Lane: blocked (deterministic)

    func test_blocked_low_confidence_lane() {
        let v = adr015Vector(
            supportIntent: ("plan_account", 0.55),
            routingLane: ("blocked", 0.5),
            cloudRequirements: [],
            requiredTool: ("no_tool", 0.5)
        )
        guard case .blocked = TelcoLaneRouter.route(v) else {
            return XCTFail("expected blocked")
        }
    }

    // MARK: - Confidence floor + signal-derived lane

    func test_lowLaneConfidence_falls_back_to_cloud_when_requirements_present() {
        // Lane head's confidence below floor — derivation should pick
        // cloud_assist because cloud_requirements is non-empty.
        let v = adr015Vector(
            supportIntent: ("outage", 0.85),
            routingLane: ("local_answer", 0.30), // below floor
            cloudRequirements: ["live_network_status"],
            requiredTool: ("cloud_only", 0.7)
        )
        guard case .cloudAssist = TelcoLaneRouter.route(v) else {
            return XCTFail("expected cloudAssist via signal derivation")
        }
    }

    func test_lowLaneConfidence_falls_back_to_localAnswer_for_deviceSetup() {
        let v = adr015Vector(
            supportIntent: ("device_setup", 0.85),
            routingLane: ("cloud_assist", 0.20),
            cloudRequirements: [],
            requiredTool: ("no_tool", 0.6)
        )
        guard case .localAnswer = TelcoLaneRouter.route(v) else {
            return XCTFail("expected localAnswer via signal derivation")
        }
    }

    // MARK: - Helpers

    private func adr015Vector(
        supportIntent: (String, Double),
        routingLane: (String, Double),
        cloudRequirements: [String],
        requiredTool: (String, Double),
        customerEscalationRisk: (String, Double) = ("low", 0.9),
        piiRisk: (String, Double) = ("safe", 0.95),
        slotCompleteness: [String] = []
    ) -> TelcoDecisionVector {
        TelcoDecisionVector(
            inputHash: 42,
            mode: .unavailable,
            kbEntry: .unavailable,
            tool: .unavailable,
            supportIntent: TelcoHeadResult(label: supportIntent.0, confidence: supportIntent.1),
            issueComplexity: TelcoHeadResult(label: "guided", confidence: 0.7),
            routingLane: TelcoHeadResult(label: routingLane.0, confidence: routingLane.1),
            cloudRequirements: TelcoMultiLabelHeadResult(activeLabels: cloudRequirements),
            requiredTool: TelcoHeadResult(label: requiredTool.0, confidence: requiredTool.1),
            customerEscalationRisk: TelcoHeadResult(label: customerEscalationRisk.0, confidence: customerEscalationRisk.1),
            piiRisk: TelcoHeadResult(label: piiRisk.0, confidence: piiRisk.1),
            transcriptQuality: TelcoHeadResult(label: "clean", confidence: 0.95),
            slotCompleteness: TelcoMultiLabelHeadResult(activeLabels: slotCompleteness),
            inferenceMode: .sharedAdapter,
            forwardPassMs: 50,
            headProjectionMs: 1
        )
    }
}
