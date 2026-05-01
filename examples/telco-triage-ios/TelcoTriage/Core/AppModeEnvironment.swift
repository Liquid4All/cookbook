import SwiftUI

/// Environment key that propagates `AppMode` into the view hierarchy.
/// Views read `@Environment(\.appMode)` to decide between customer-clean
/// and engineering-instrumented rendering.
private struct AppModeKey: EnvironmentKey {
    static let defaultValue: AppMode = .customer
}

extension EnvironmentValues {
    var appMode: AppMode {
        get { self[AppModeKey.self] }
        set { self[AppModeKey.self] = newValue }
    }
}

extension View {
    func appMode(_ mode: AppMode) -> some View {
        environment(\.appMode, mode)
    }
}
