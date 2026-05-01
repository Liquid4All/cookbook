import SwiftUI

/// The Customer / Operator pivot. Sits at the top of the app, above the
/// tab bar. Tapping either half rebuilds the tab list underneath.
///
/// This is the demo's rhetorical device: the sales engineer runs a support
/// flow on the customer side, then taps "Operator" and the app reveals the
/// ROI + architecture layer. Keeping the pivot visible at all times (not a
/// hidden gesture) makes the distinction legible to execs who've never
/// seen the app before.
struct SurfaceModeBar: View {
    @Binding var mode: SurfaceMode
    let onSettings: () -> Void

    @Environment(\.brand) private var brand

    var body: some View {
        HStack(spacing: 10) {
            segmentedPicker
            settingsButton
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(
            Rectangle()
                .fill(brand.surfaceBackground)
                .shadow(color: .black.opacity(0.04), radius: 3, y: 1)
        )
    }

    private var segmentedPicker: some View {
        HStack(spacing: 0) {
            ForEach(SurfaceMode.allCases) { option in
                Button {
                    guard option != mode else { return }
                    withAnimation(.easeInOut(duration: 0.2)) {
                        mode = option
                    }
                } label: {
                    Text(option.displayName)
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(option == mode ? brand.onPrimary : brand.textSecondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 7)
                        .background(
                            option == mode
                                ? AnyView(Capsule().fill(brand.primary))
                                : AnyView(Color.clear)
                        )
                }
                .buttonStyle(.plain)
                .accessibilityLabel(option.longDisplayName)
                .accessibilityHint(option.accessibilityHint)
            }
        }
        .padding(3)
        .background(brand.surfaceElevated, in: Capsule())
    }

    private var settingsButton: some View {
        Button(action: onSettings) {
            Image(systemName: "gearshape")
                .font(.system(size: 18))
                .foregroundStyle(brand.textSecondary)
                .frame(width: 36, height: 36)
        }
        .accessibilityLabel("Settings")
    }
}
