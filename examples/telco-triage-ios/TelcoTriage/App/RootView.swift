import SwiftUI

/// App shell. For the generic telco pitch we show only the Customer surface —
/// the Operator (ROI / Architecture) tabs have been pulled out of the
/// root UI so a demoer can't accidentally land there. A gear icon in
/// each screen's toolbar opens Settings.
///
/// The Operator tabs still exist in code and can be re-enabled by
/// flipping `showsOperatorPivot` to `true` below.
struct RootView: View {
    @EnvironmentObject private var appState: AppState
    @State private var customerTab: CustomerTab = .chat

    /// Pitch-specific flag: keep the Operator pivot off by default.
    /// When re-enabled, RootView will restore the SurfaceModeBar and
    /// OperatorTab tree.
    private let showsOperatorPivot = false

    enum CustomerTab: Hashable { case chat, plan, packs }

    var body: some View {
        TabView(selection: $customerTab) {
            ChatView()
                .tabItem { Label("Support", systemImage: "bubble.left.and.bubble.right") }
                .tag(CustomerTab.chat)

            PlanView()
                .tabItem { Label("Household", systemImage: "person.crop.circle") }
                .badge(householdBadge)
                .tag(CustomerTab.plan)

            PacksView()
                .tabItem { Label("Packs", systemImage: "square.stack.3d.up") }
                .tag(CustomerTab.packs)
        }
        .tint(appState.brands.selected.primary)
    }

    private var householdBadge: String? {
        if appState.customerContext.serviceAppointment != nil {
            return "!"
        }
        let pausedCount = appState.customerContext.managedDevices
            .filter { $0.accessState == .paused }
            .count
        return pausedCount > 0 ? "\(pausedCount)" : nil
    }
}
