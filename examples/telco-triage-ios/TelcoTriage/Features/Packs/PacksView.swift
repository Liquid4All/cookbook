import SwiftUI

struct PacksView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.brand) private var brand

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    intro
                    ForEach(appState.packManager.packs) { pack in
                        PackCard(pack: pack, manager: appState.packManager)
                    }
                }
                .padding(16)
            }
            .background(brand.surfaceBackground.ignoresSafeArea())
            .navigationTitle("Intelligence Packs")
        }
    }

    private var intro: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Extend your assistant — on-device.")
                .font(brand.titleFont)
                .foregroundStyle(brand.textPrimary)
            Text("The base model (LFM2.5-350M) ships with the app. Download optional specialist packs to unlock voice or vision. Everything runs on your phone — no data leaves the device.")
                .font(.callout)
                .foregroundStyle(brand.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct PackCard: View {
    let pack: SpecialistPack
    @ObservedObject var manager: SpecialistPackManager
    @Environment(\.brand) private var brand

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Text(pack.summary)
                .font(.subheadline)
                .foregroundStyle(brand.textSecondary)
            valueProps
            stateRow
        }
        .padding(16)
        .background(brand.surfaceElevated, in: RoundedRectangle(cornerRadius: brand.cardCornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: brand.cardCornerRadius)
                .stroke(brand.border, lineWidth: 1)
        )
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: pack.icon)
                .font(.system(size: 24))
                .frame(width: 44, height: 44)
                .background(brand.primary.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(brand.primary)
            VStack(alignment: .leading, spacing: 2) {
                Text(pack.displayName)
                    .font(.headline)
                    .foregroundStyle(brand.textPrimary)
                HStack(spacing: 6) {
                    Text(pack.capabilityDisplayName)
                        .font(.caption2).fontWeight(.bold)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(brand.primary.opacity(0.12), in: Capsule())
                        .foregroundStyle(brand.primary)
                    Text(pack.formattedSize)
                        .font(.caption)
                        .foregroundStyle(brand.textSecondary)
                    Text("• \(pack.modelRepo)")
                        .font(.caption2)
                        .foregroundStyle(brand.textSecondary)
                        .lineLimit(1)
                }
            }
            Spacer()
        }
    }

    private var valueProps: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(pack.valueProps, id: \.self) { prop in
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "checkmark")
                        .font(.caption).foregroundStyle(brand.success)
                    Text(prop).font(.caption).foregroundStyle(brand.textPrimary)
                }
            }
        }
    }

    @ViewBuilder
    private var stateRow: some View {
        // Coming-soon packs preempt all state rendering. No install
        // button, no progress, no error — just the explainer so the
        // user knows what IS working today (Apple Speech for voice;
        // the vision capability is still wired to the mock analyzer).
        //
        // ViewBuilder doesn't support early `return`, so branch via
        // if/else rather than falling through to the switch.
        if case .comingSoon(let reason) = pack.availability {
            comingSoonRow(reason: reason)
        } else {
            installStateRow
        }
    }

    @ViewBuilder
    private var installStateRow: some View {
        switch manager.state(for: pack.id) {
        case .notInstalled:
            Button {
                manager.install(pack)
            } label: {
                HStack {
                    Image(systemName: "arrow.down.circle.fill")
                    Text("Download \(pack.formattedSize)")
                }
                .font(.subheadline).fontWeight(.semibold)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(brand.primary, in: RoundedRectangle(cornerRadius: 10))
                .foregroundStyle(brand.onPrimary)
            }

        case .downloading(let progress):
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Downloading… \(Int(progress * 100))%")
                        .font(.caption).foregroundStyle(brand.textSecondary)
                    Spacer()
                    Button("Cancel") { manager.cancelDownload(pack) }
                        .font(.caption)
                        .foregroundStyle(brand.danger)
                }
                ProgressView(value: progress)
                    .tint(brand.primary)
            }

        case .installed:
            HStack(spacing: 10) {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(brand.success)
                    VStack(alignment: .leading, spacing: 0) {
                        Text("Installed")
                            .font(.subheadline).fontWeight(.semibold)
                            .foregroundStyle(brand.textPrimary)
                        if manager.isCached(pack) {
                            Text("\(pack.formattedSize) on device")
                                .font(.caption2)
                                .foregroundStyle(brand.textSecondary)
                        }
                    }
                }
                Spacer()
                Button(role: .destructive) {
                    Task { await manager.uninstall(pack) }
                } label: {
                    Text("Remove")
                        .font(.caption).fontWeight(.medium)
                }
            }
            .padding(.vertical, 8)

        case .error(let message):
            VStack(alignment: .leading, spacing: 4) {
                Text("Install failed").font(.caption).foregroundStyle(brand.danger)
                Text(message).font(.caption2).foregroundStyle(brand.textSecondary)
                Button("Retry") { manager.install(pack) }
                    .font(.caption)
            }
        }
    }

    @ViewBuilder
    private func comingSoonRow(reason: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "hourglass")
                    .foregroundStyle(brand.textSecondary)
                Text("Coming Soon")
                    .font(.subheadline).fontWeight(.semibold)
                    .foregroundStyle(brand.textPrimary)
                Spacer()
            }
            Text(reason)
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            brand.surfaceBackground.opacity(0.5),
            in: RoundedRectangle(cornerRadius: 10)
        )
    }
}
