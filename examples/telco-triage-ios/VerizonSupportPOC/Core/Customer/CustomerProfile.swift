import Foundation

/// A mock customer profile. Real integration would pull this from the
/// carrier home internet app's session state. Used to personalize greetings and
/// ground tool responses ("restarted your G3100 router at 3:42 PM").
public struct CustomerProfile: Codable, Sendable, Equatable {
    public let customerID: String
    public let firstName: String
    public let lastName: String
    public let plan: Plan
    public let address: Address
    public let equipment: [Equipment]
    public let recentIssues: [PastIssue]
    public let usage: UsageSnapshot

    public struct Plan: Codable, Sendable, Equatable {
        public let name: String           // "Fiber Gigabit Connection"
        public let downSpeedMbps: Int     // 940
        public let upSpeedMbps: Int       // 880
        public let monthlyPrice: Double   // 89.99
    }

    public struct Address: Codable, Sendable, Equatable {
        public let line1: String
        public let city: String
        public let state: String
        public let zip: String
    }

    public struct Equipment: Codable, Sendable, Equatable, Identifiable {
        public var id: String { serial }
        public let kind: Kind
        public let model: String
        public let serial: String
        public let status: Status
        public let lastReboot: Date?

        public enum Kind: String, Codable, Sendable { case router, extender, setTopBox }
        public enum Status: String, Codable, Sendable { case online, unhealthy, offline }
    }

    /// Rolling 30-day usage/behavior snapshot. Feeds the Next-Best-Action
    /// engine — every NBA scores against this data. Real integration would
    /// stream this from the carrier telemetry pipeline; alpha seeds
    /// plausible values that drive realistic recommendations.
    public struct UsageSnapshot: Codable, Sendable, Equatable {
        public let periodDays: Int
        public let downloadedGB: Int
        public let uploadedGB: Int
        public let connectedDeviceCount: Int
        public let peakDeviceCount: Int
        public let troubleshootCount: Int
        public let avgDownMbps: Int
        public let avgUpMbps: Int
        public let speedTestFailures: Int            // tests below 70% of plan cap
        public let outageMinutes: Int
        public let billCyclesAtOrOverCap: Int         // 0..N past 3 cycles
        public let activeBoltOns: [String]            // e.g. ["TravelPass"]

        public init(
            periodDays: Int,
            downloadedGB: Int,
            uploadedGB: Int,
            connectedDeviceCount: Int,
            peakDeviceCount: Int,
            troubleshootCount: Int,
            avgDownMbps: Int,
            avgUpMbps: Int,
            speedTestFailures: Int,
            outageMinutes: Int,
            billCyclesAtOrOverCap: Int,
            activeBoltOns: [String]
        ) {
            self.periodDays = periodDays
            self.downloadedGB = downloadedGB
            self.uploadedGB = uploadedGB
            self.connectedDeviceCount = connectedDeviceCount
            self.peakDeviceCount = peakDeviceCount
            self.troubleshootCount = troubleshootCount
            self.avgDownMbps = avgDownMbps
            self.avgUpMbps = avgUpMbps
            self.speedTestFailures = speedTestFailures
            self.outageMinutes = outageMinutes
            self.billCyclesAtOrOverCap = billCyclesAtOrOverCap
            self.activeBoltOns = activeBoltOns
        }
    }

    public struct PastIssue: Codable, Sendable, Equatable, Identifiable {
        public var id: String { "\(timestamp.timeIntervalSince1970)-\(summary.prefix(16))" }
        public let timestamp: Date
        public let summary: String
        public let resolved: Bool
    }

    /// The demo profile we ship. Intentionally fake — any resemblance to
    /// real customers is coincidental. Never put real names, addresses, or
    /// account numbers here; this file ends up in screenshots and pitch
    /// decks.
    public static let demo: CustomerProfile = {
        let now = Date()
        return CustomerProfile(
            customerID: "TELCO-DEMO-0001",
            firstName: "Alex",
            lastName: "Rivera",
            plan: Plan(
                name: "Fiber Gigabit Connection",
                downSpeedMbps: 940,
                upSpeedMbps: 880,
                monthlyPrice: 89.99
            ),
            address: Address(
                line1: "100 Demo Street",
                city: "Anywhere",
                state: "CA",
                zip: "00000"
            ),
            equipment: [
                Equipment(
                    kind: .router,
                    model: "Fiber Router G3100",
                    serial: "CP18445A2FQ",
                    status: .online,
                    lastReboot: now.addingTimeInterval(-12 * 24 * 3600)
                ),
                Equipment(
                    kind: .extender,
                    model: "Mesh Extender E3200",
                    serial: "EX22118B8QN",
                    status: .unhealthy,
                    lastReboot: now.addingTimeInterval(-45 * 24 * 3600)
                ),
                Equipment(
                    kind: .setTopBox,
                    model: "Stream TV",
                    serial: "STVX9021PQ",
                    status: .online,
                    lastReboot: now.addingTimeInterval(-6 * 24 * 3600)
                ),
            ],
            recentIssues: [
                PastIssue(
                    timestamp: now.addingTimeInterval(-3 * 24 * 3600),
                    summary: "Speed test reported 430 Mbps (below plan)",
                    resolved: true
                ),
                PastIssue(
                    timestamp: now.addingTimeInterval(-9 * 24 * 3600),
                    summary: "Extender went offline briefly",
                    resolved: true
                ),
            ],
            // Usage tuned to trigger multiple NBAs: heavy data use + a few
            // speed-test failures → plan-optimize + retention-credit signal;
            // no travel bolt-on → TravelPass upsell; many connected devices
            // → mesh-upgrade suggestion; unhealthy extender → proactive
            // troubleshoot prompt. Keeps the demo story tight.
            usage: UsageSnapshot(
                periodDays: 30,
                downloadedGB: 412,
                uploadedGB: 58,
                connectedDeviceCount: 18,
                peakDeviceCount: 24,
                troubleshootCount: 3,
                avgDownMbps: 680,
                avgUpMbps: 520,
                speedTestFailures: 3,
                outageMinutes: 42,
                billCyclesAtOrOverCap: 0,
                activeBoltOns: []
            )
        )
    }()
}

public extension CustomerProfile {
    /// Copy-with-change helper so `CustomerContext` can update a single slice
    /// of profile state without repeating every field at each call site.
    func with(
        customerID: String? = nil,
        firstName: String? = nil,
        lastName: String? = nil,
        plan: Plan? = nil,
        address: Address? = nil,
        equipment: [Equipment]? = nil,
        recentIssues: [PastIssue]? = nil,
        usage: UsageSnapshot? = nil
    ) -> CustomerProfile {
        CustomerProfile(
            customerID: customerID ?? self.customerID,
            firstName: firstName ?? self.firstName,
            lastName: lastName ?? self.lastName,
            plan: plan ?? self.plan,
            address: address ?? self.address,
            equipment: equipment ?? self.equipment,
            recentIssues: recentIssues ?? self.recentIssues,
            usage: usage ?? self.usage
        )
    }
}
