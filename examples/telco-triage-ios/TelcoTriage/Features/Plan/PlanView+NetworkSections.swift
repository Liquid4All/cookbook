import SwiftUI

/// Equipment, managed-device, service-visit, and recent-issues
/// sub-sections of the Household tab. Split out of `PlanView.swift`
/// so the main file stays under the 600-line CLAUDE.md guideline.
///
/// All types here are file-private helpers of the Household surface
/// and shouldn't leak outside the feature; `internal` access comes
/// from the default visibility — it's enough for same-module
/// cross-file references from `PlanView.swift` without exposing them
/// to the rest of the app.

struct EquipmentSection: View {
    let equipment: [CustomerProfile.Equipment]
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Equipment")
                .font(.caption).foregroundStyle(brand.textSecondary).textCase(.uppercase)
            ForEach(equipment) { item in
                EquipmentRow(item: item)
            }
        }
    }
}

struct ManagedDevicesSection: View {
    @ObservedObject var context: CustomerContext
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Family Devices")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)
            ForEach(context.managedDevices) { device in
                HStack(spacing: 12) {
                    Image(systemName: icon(for: device.kind))
                        .font(.title3)
                        .frame(width: 36, height: 36)
                        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 8))
                        .foregroundStyle(brand.textPrimary)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(device.name)
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        HStack(spacing: 6) {
                            DeviceStateChip(state: device.accessState)
                            Text(device.location)
                                .font(.caption)
                                .foregroundStyle(brand.textSecondary)
                        }
                        Text(device.detail)
                            .font(.caption2)
                            .foregroundStyle(brand.textSecondary)
                    }
                    Spacer()
                }
                .padding(12)
                .background(brand.surfaceElevated.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private func icon(for kind: CustomerContext.ManagedDevice.Kind) -> String {
        switch kind {
        case .tablet: return "ipad"
        case .laptop: return "laptopcomputer"
        case .tv: return "tv"
        case .phone: return "iphone"
        case .console: return "gamecontroller"
        }
    }
}

struct DeviceStateChip: View {
    let state: CustomerContext.ManagedDevice.AccessState
    @Environment(\.brand) private var brand

    var body: some View {
        Text(state == .paused ? "Paused" : "Open")
            .font(.caption2)
            .fontWeight(.bold)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(tint.opacity(0.14), in: Capsule())
            .foregroundStyle(tint)
    }

    private var tint: Color {
        switch state {
        case .unrestricted: return brand.success
        case .paused: return brand.warning
        }
    }
}

struct ServiceVisitSection: View {
    @ObservedObject var context: CustomerContext
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Service Desk")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .textCase(.uppercase)
            if let appointment = context.serviceAppointment {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(appointment.title)
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Spacer()
                        Text(appointment.status.rawValue.capitalized)
                            .font(.caption2)
                            .fontWeight(.bold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(brand.info.opacity(0.14), in: Capsule())
                            .foregroundStyle(brand.info)
                    }
                    Text(appointment.windowLabel)
                        .font(.headline)
                        .foregroundStyle(brand.textPrimary)
                    Text(appointment.note)
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
            } else {
                Text("No technician dispatch booked. If the assistant escalates to field support, it will appear here instantly.")
                    .font(.caption)
                    .foregroundStyle(brand.textSecondary)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(brand.surfaceElevated.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }
}

struct EquipmentRow: View {
    let item: CustomerProfile.Equipment
    @Environment(\.brand) private var brand

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .frame(width: 36, height: 36)
                .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: 8))
                .foregroundStyle(brand.textPrimary)
            VStack(alignment: .leading, spacing: 2) {
                Text(item.model).font(.subheadline).fontWeight(.semibold)
                HStack(spacing: 6) {
                    StatusDot(status: item.status)
                    Text(item.status.rawValue.capitalized)
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                    if let last = item.lastReboot {
                        Text("\u{2022} rebooted \(PlanFormatters.relative.localizedString(for: last, relativeTo: Date()))")
                            .font(.caption2)
                            .foregroundStyle(brand.textSecondary)
                    }
                }
            }
            Spacer()
        }
        .padding(12)
        .background(brand.surfaceElevated.opacity(0.4), in: RoundedRectangle(cornerRadius: 10))
    }

    private var icon: String {
        switch item.kind {
        case .router: return "wifi.router"
        case .extender: return "dot.radiowaves.up.forward"
        case .setTopBox: return "tv"
        }
    }
}

struct StatusDot: View {
    let status: CustomerProfile.Equipment.Status
    @Environment(\.brand) private var brand

    var body: some View {
        Circle()
            .fill(fill)
            .frame(width: 8, height: 8)
    }

    private var fill: Color {
        switch status {
        case .online: return brand.success
        case .unhealthy: return brand.warning
        case .offline: return brand.danger
        }
    }
}

struct RecentIssuesSection: View {
    let issues: [CustomerProfile.PastIssue]
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent Issues")
                .font(.caption).foregroundStyle(brand.textSecondary).textCase(.uppercase)
            if issues.isEmpty {
                Text("No recent issues on file.")
                    .font(.caption).foregroundStyle(brand.textSecondary)
            } else {
                ForEach(issues) { issue in
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: issue.resolved ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                            .foregroundStyle(issue.resolved ? brand.success : brand.warning)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(issue.summary).font(.caption).foregroundStyle(brand.textPrimary)
                            Text(PlanFormatters.relative.localizedString(for: issue.timestamp, relativeTo: Date()))
                                .font(.caption2).foregroundStyle(brand.textSecondary)
                        }
                        Spacer()
                    }
                    .padding(8)
                }
            }
        }
    }
}
