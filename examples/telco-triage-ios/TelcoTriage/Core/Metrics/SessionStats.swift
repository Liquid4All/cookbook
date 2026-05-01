import Foundation
import Combine

/// Session-level counters that complement `TokenLedger` — latency percentiles
/// and PII-catch counts. Published so SwiftUI views can subscribe.
@MainActor
public final class SessionStats: ObservableObject {
    @Published public private(set) var latenciesMS: [Int] = []
    @Published public private(set) var piiInstancesCaught: Int = 0
    @Published public private(set) var toolExecutions: Int = 0
    @Published public private(set) var truckRollRisksAvoided: Int = 0
    @Published public private(set) var appointmentsScheduled: Int = 0
    @Published public private(set) var privacyPreflightsApproved: Int = 0
    @Published public private(set) var privacyPreflightsCancelled: Int = 0

    public init() {}

    public var averageLatencyMS: Int {
        guard !latenciesMS.isEmpty else { return 0 }
        return latenciesMS.reduce(0, +) / latenciesMS.count
    }

    public var p95LatencyMS: Int {
        guard !latenciesMS.isEmpty else { return 0 }
        let sorted = latenciesMS.sorted()
        let index = min(sorted.count - 1, Int(Double(sorted.count) * 0.95))
        return sorted[index]
    }

    public func recordLatency(_ ms: Int) {
        latenciesMS.append(ms)
    }

    public func recordPII(_ count: Int) {
        piiInstancesCaught += count
    }

    public func recordToolExecution(toolID: String, status: ToolResult.Status) {
        guard status == .success else { return }
        toolExecutions += 1
        switch toolID {
        case "reboot-extender":
            truckRollRisksAvoided += 1
        case "schedule-technician":
            appointmentsScheduled += 1
        default:
            break
        }
    }

    public func recordPrivacyPreflight(approved: Bool) {
        if approved {
            privacyPreflightsApproved += 1
        } else {
            privacyPreflightsCancelled += 1
        }
    }

    public func reset() {
        latenciesMS.removeAll()
        piiInstancesCaught = 0
        toolExecutions = 0
        truckRollRisksAvoided = 0
        appointmentsScheduled = 0
        privacyPreflightsApproved = 0
        privacyPreflightsCancelled = 0
    }
}
