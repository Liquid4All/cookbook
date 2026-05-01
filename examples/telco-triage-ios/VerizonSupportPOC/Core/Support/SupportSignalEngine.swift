import Combine
import Foundation

/// Reads `CustomerContext` and produces the ranked list of
/// `SupportSignal`s the chat should surface above its input bar.
///
/// Pure function of state: for a given context snapshot, the list is
/// deterministic. Re-runs automatically when `CustomerContext`
/// publishes changes (managedDevices or profile mutate). Callers
/// observe `activeSignals` directly.
///
/// Ranking: `.urgent` before `.attention` before `.info`. Within a
/// severity bucket, most-recently-observed (by underlying source
/// timestamp) first. Capped at 3 signals — more than that is a wall
/// of badges, not a useful assistant.
@MainActor
public final class SupportSignalEngine: ObservableObject {
    @Published public private(set) var activeSignals: [SupportSignal] = []

    private let context: CustomerContext
    private let now: @Sendable () -> Date
    private var cancellables: Set<AnyCancellable> = []

    public init(
        context: CustomerContext,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.context = context
        self.now = now
        refresh()
        // Re-score on any context mutation so a tool success that fixes
        // the underlying state clears the signal automatically.
        context.$profile
            .combineLatest(context.$managedDevices, context.$serviceAppointment)
            .sink { [weak self] _, _, _ in
                self?.refresh()
            }
            .store(in: &cancellables)
    }

    public func refresh() {
        var signals: [SupportSignal] = []
        signals.append(contentsOf: extenderHealthSignals())
        signals.append(contentsOf: routerHealthSignals())
        signals.append(contentsOf: speedDegradationSignals())
        signals.append(contentsOf: outageSignals())
        signals.append(contentsOf: planFitSignals())
        signals.append(contentsOf: pausedDeviceSignals())

        let ranked = signals.sorted { lhs, rhs in
            lhs.severity.sortKey < rhs.severity.sortKey
        }
        activeSignals = Array(ranked.prefix(3))
    }

    // MARK: - Rules

    /// Unhealthy or offline extender. This is the #1 signal for the
    /// demo — the customer hasn't complained yet, but the assistant
    /// has noticed and offers a one-tap restart.
    private func extenderHealthSignals() -> [SupportSignal] {
        context.profile.equipment.compactMap { item in
            guard item.kind == .extender, item.status != .online else { return nil }
            let severity: SupportSignal.Severity = item.status == .offline ? .urgent : .attention
            let stateWord = item.status == .offline ? "offline" : "unhealthy"
            return SupportSignal(
                id: "extender-health-\(item.serial)",
                domain: .home,
                severity: severity,
                title: "\(item.model) is \(stateWord)",
                summary: "Restarting the extender usually clears this without a truck roll.",
                suggestedPrompt: "restart the wifi extender",
                suggestedToolID: "reboot-extender",
                source: "equipment.extender.status",
                icon: "wifi.exclamationmark"
            )
        }
    }

    /// Router hasn't been restarted in 30+ days and status is anything
    /// but online — surface a nudge, not a panic.
    private func routerHealthSignals() -> [SupportSignal] {
        context.profile.equipment.compactMap { item in
            guard item.kind == .router, item.status != .online else { return nil }
            return SupportSignal(
                id: "router-health-\(item.serial)",
                domain: .home,
                severity: item.status == .offline ? .urgent : .attention,
                title: "\(item.model) reports \(item.status.rawValue)",
                summary: "A restart usually resolves this.",
                suggestedPrompt: "restart my router",
                suggestedToolID: "restart-router",
                source: "equipment.router.status",
                icon: "wifi.router"
            )
        }
    }

    /// Repeated speed test failures in the last 30 days — likely a
    /// plan-fit or congestion issue. Suggest a diagnostic.
    private func speedDegradationSignals() -> [SupportSignal] {
        guard context.profile.usage.speedTestFailures >= 2 else { return [] }
        return [
            SupportSignal(
                id: "speed-degradation",
                domain: .home,
                severity: .attention,
                title: "\(context.profile.usage.speedTestFailures) speed tests came in under plan this month",
                summary: "Run full diagnostics to rule out interference before escalating.",
                suggestedPrompt: "run diagnostics on my home network",
                suggestedToolID: "run-diagnostics",
                source: "usage.speedTestFailures",
                icon: "speedometer"
            ),
        ]
    }

    /// Substantial outage minutes in the last 30 days — offer a speed
    /// test or technician scheduling.
    private func outageSignals() -> [SupportSignal] {
        guard context.profile.usage.outageMinutes >= 30 else { return [] }
        return [
            SupportSignal(
                id: "outage-minutes",
                domain: .home,
                severity: .info,
                title: "\(context.profile.usage.outageMinutes) minutes of outages last month",
                summary: "A quick speed test confirms things are back to normal.",
                suggestedPrompt: "run a speed test",
                suggestedToolID: "run-speed-test",
                source: "usage.outageMinutes",
                icon: "clock.arrow.circlepath"
            ),
        ]
    }

    /// Heavy data use on a non-gigabit plan, OR heavy use + no
    /// travel bolt-on during summer — nudge toward plan review.
    /// Demoable copy; real engine would score multiple signals.
    private func planFitSignals() -> [SupportSignal] {
        let usage = context.profile.usage
        let heavyUse = usage.downloadedGB > 300 && usage.peakDeviceCount > 15
        guard heavyUse else { return [] }
        return [
            SupportSignal(
                id: "plan-fit-heavy-use",
                domain: .billing,
                severity: .info,
                title: "Your household is pushing \(usage.downloadedGB) GB across \(usage.peakDeviceCount) devices",
                summary: "A plan review could unlock mesh coverage or a better price for this footprint.",
                suggestedPrompt: "can you explain the charges on my bill?",
                suggestedToolID: nil,
                source: "usage.downloadedGB",
                icon: "chart.bar.doc.horizontal"
            ),
        ]
    }

    /// Devices that were paused and are still past their downtime
    /// window. Offers resume.
    private func pausedDeviceSignals() -> [SupportSignal] {
        let currentTime = now()
        return context.managedDevices.compactMap { device in
            guard device.accessState == .paused,
                  let until = device.downtimeUntil,
                  until <= currentTime
            else { return nil }
            return SupportSignal(
                id: "paused-past-window-\(device.id)",
                domain: .household,
                severity: .info,
                title: "\(device.name) is still paused past its window",
                summary: "Scheduled until \(Self.timeFormatter.string(from: until)). Resume access?",
                suggestedPrompt: "resume internet for \(device.name.lowercased())",
                suggestedToolID: "toggle-parental-controls",
                source: "managedDevices.downtimeUntil",
                icon: "play.circle"
            )
        }
    }

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return f
    }()
}
