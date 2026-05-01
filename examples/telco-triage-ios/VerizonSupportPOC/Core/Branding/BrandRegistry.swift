import Foundation
import Combine

/// Plugin-style registry for brand themes. Adding a new carrier/brand is
/// (1) implement `BrandTheme`, (2) register here. Zero changes to views.
@MainActor
public final class BrandRegistry: ObservableObject {
    @Published public private(set) var selected: any BrandTheme
    public let available: [any BrandTheme]

    private let persistenceKey = "selectedBrandID"
    private let defaults: UserDefaults

    public init(
        available: [any BrandTheme] = [TelcoTriageTheme(), LiquidTheme()],
        defaults: UserDefaults = .standard
    ) {
        self.available = available
        self.defaults = defaults
        let savedID = defaults.string(forKey: persistenceKey) ?? "telco-triage"
        self.selected = available.first(where: { $0.id == savedID }) ?? TelcoTriageTheme()
    }

    public func select(_ id: String) {
        guard let match = available.first(where: { $0.id == id }) else { return }
        self.selected = match
        defaults.set(id, forKey: persistenceKey)
    }
}
