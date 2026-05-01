import Foundation
import Combine

/// Session-level customer context the chat / tools can read from. Wraps the
/// immutable profile in an observable container so mutations (e.g.
/// last-reboot update after a restart-router tool call) propagate to any
/// view showing equipment status.
@MainActor
public final class CustomerContext: ObservableObject {
    @Published public private(set) var profile: CustomerProfile
    @Published public private(set) var managedDevices: [ManagedDevice]
    @Published public private(set) var serviceAppointment: ServiceAppointment?

    public init(
        profile: CustomerProfile = .demo,
        managedDevices: [ManagedDevice] = CustomerContext.demoManagedDevices,
        serviceAppointment: ServiceAppointment? = nil
    ) {
        self.profile = profile
        self.managedDevices = managedDevices
        self.serviceAppointment = serviceAppointment
    }

    public struct ManagedDevice: Identifiable, Equatable, Sendable {
        public enum Kind: String, Sendable { case tablet, laptop, tv, phone, console }
        public enum AccessState: String, Sendable { case unrestricted, paused }

        public let id: String
        public let name: String
        public let kind: Kind
        public let location: String
        public let accessState: AccessState
        public let detail: String
        /// If set, the device is paused until this timestamp; past it the
        /// device should be treated as unrestricted by callers consuming
        /// the managedDevices list. The engine doesn't clear this field
        /// automatically — we keep the end-of-window intent around for
        /// the signal engine to reason about ("tablet was paused until 7
        /// earlier tonight, suggest re-pause?").
        public let downtimeUntil: Date?

        public init(
            id: String,
            name: String,
            kind: Kind,
            location: String,
            accessState: AccessState,
            detail: String,
            downtimeUntil: Date? = nil
        ) {
            self.id = id
            self.name = name
            self.kind = kind
            self.location = location
            self.accessState = accessState
            self.detail = detail
            self.downtimeUntil = downtimeUntil
        }

        public func with(
            accessState: AccessState? = nil,
            detail: String? = nil,
            downtimeUntil: Date?? = nil
        ) -> ManagedDevice {
            ManagedDevice(
                id: self.id,
                name: self.name,
                kind: self.kind,
                location: self.location,
                accessState: accessState ?? self.accessState,
                detail: detail ?? self.detail,
                downtimeUntil: downtimeUntil ?? self.downtimeUntil
            )
        }
    }

    public struct ServiceAppointment: Identifiable, Equatable, Sendable {
        public enum Status: String, Sendable { case scheduled, confirmed }

        public let id: UUID
        public let title: String
        public let windowLabel: String
        public let note: String
        public let status: Status
        public let createdAt: Date

        public init(
            id: UUID = UUID(),
            title: String,
            windowLabel: String,
            note: String,
            status: Status,
            createdAt: Date = Date()
        ) {
            self.id = id
            self.title = title
            self.windowLabel = windowLabel
            self.note = note
            self.status = status
            self.createdAt = createdAt
        }
    }

    // MARK: - Equipment mutations driven by tool calls

    public func markRouterRebooted(serial: String, at time: Date = Date()) {
        updateEquipment(serial: serial) { item in
            item.with(status: .online, lastReboot: time)
        }
        appendRecentIssue("Router reboot started from on-device support assistant.")
    }

    public func markEquipmentStatus(serial: String, status: CustomerProfile.Equipment.Status) {
        updateEquipment(serial: serial) { item in
            item.with(status: status)
        }
    }

    @discardableResult
    public func markExtenderRebooted(
        requestedName: String?,
        at time: Date = Date()
    ) -> CustomerProfile.Equipment? {
        guard let extender = profile.equipment.first(where: { $0.kind == .extender }) else {
            return nil
        }
        updateEquipment(serial: extender.serial) { item in
            item.with(status: .online, lastReboot: time)
        }
        let resolvedName = requestedName ?? extender.model
        appendRecentIssue("Extender reboot initiated for \(resolvedName).")
        return extender
    }

    @discardableResult
    public func updateParentalControls(
        deviceName: String,
        action: String
    ) -> ManagedDevice? {
        guard let index = managedDevices.firstIndex(where: { matches($0.name, query: deviceName) }) else {
            return nil
        }

        let isPause = !action.lowercased().contains("resume")
        let updated = managedDevices[index].with(
            accessState: isPause ? .paused : .unrestricted,
            detail: isPause ? "Internet paused by on-device assistant" : "Internet access restored"
        )
        managedDevices[index] = updated
        appendRecentIssue(
            isPause
                ? "Parental controls paused internet for \(updated.name)."
                : "Parental controls restored internet for \(updated.name)."
        )
        return updated
    }

    /// Pause internet on the named device until `until`. Returns the
    /// updated device, or nil if no device matched the name hint. The
    /// caller is responsible for formatting the
    /// human-readable window label ("until 7 PM", "for 2 hours") — the
    /// context just stores the timestamp.
    @discardableResult
    public func applyDowntime(
        deviceName: String,
        until: Date
    ) -> ManagedDevice? {
        guard let index = managedDevices.firstIndex(where: { matches($0.name, query: deviceName) }) else {
            return nil
        }
        let updated = managedDevices[index].with(
            accessState: .paused,
            detail: "Internet paused until \(Self.downtimeFormatter.string(from: until))",
            downtimeUntil: until
        )
        managedDevices[index] = updated
        appendRecentIssue("Downtime scheduled for \(updated.name) until \(Self.downtimeFormatter.string(from: until)).")
        return updated
    }

    private static let downtimeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .none
        f.timeStyle = .short
        return f
    }()

    public func scheduleTechnician(windowLabel: String, note: String) {
        serviceAppointment = ServiceAppointment(
            title: "Home network technician visit",
            windowLabel: windowLabel,
            note: note,
            status: .scheduled
        )
        appendRecentIssue("Technician visit scheduled for \(windowLabel).")
    }

    public func appendRecentIssue(_ summary: String, resolved: Bool = true) {
        let updatedIssues = [
            CustomerProfile.PastIssue(timestamp: Date(), summary: summary, resolved: resolved),
        ] + Array(profile.recentIssues.prefix(5))
        profile = profile.with(recentIssues: Array(updatedIssues))
    }

    private func updateEquipment(
        serial: String,
        transform: (CustomerProfile.Equipment) -> CustomerProfile.Equipment
    ) {
        let updated = profile.equipment.map { item in
            item.serial == serial ? transform(item) : item
        }
        profile = profile.with(equipment: updated)
    }

    private func matches(_ value: String, query: String) -> Bool {
        let normalizedValue = value.lowercased()
        let normalizedQuery = query.lowercased()
        return normalizedValue.contains(normalizedQuery) || normalizedQuery.contains(normalizedValue)
    }
}

public extension CustomerContext {
    static let demoManagedDevices: [ManagedDevice] = [
        ManagedDevice(
            id: "sons-tablet",
            name: "Son's Tablet",
            kind: .tablet,
            location: "Upstairs bedroom",
            accessState: .unrestricted,
            detail: "Daily screen-time policy available"
        ),
        ManagedDevice(
            id: "work-laptop",
            name: "Work Laptop",
            kind: .laptop,
            location: "Home office",
            accessState: .unrestricted,
            detail: "Prioritized for video calls"
        ),
        ManagedDevice(
            id: "living-room-tv",
            name: "Living Room TV",
            kind: .tv,
            location: "Living room",
            accessState: .unrestricted,
            detail: "Streaming quality monitored"
        ),
    ]
}

public extension CustomerProfile.Equipment {
    /// Copy-with-change. Structs are immutable by design; this keeps call
    /// sites declarative.
    func with(
        status: CustomerProfile.Equipment.Status? = nil,
        lastReboot: Date? = nil
    ) -> CustomerProfile.Equipment {
        CustomerProfile.Equipment(
            kind: self.kind,
            model: self.model,
            serial: self.serial,
            status: status ?? self.status,
            lastReboot: lastReboot ?? self.lastReboot
        )
    }
}
