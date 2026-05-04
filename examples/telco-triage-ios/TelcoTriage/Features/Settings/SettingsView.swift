import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.brand) private var brand

    @State private var showResetConfirm = false

    var body: some View {
        NavigationStack {
            Form {
                modeSection
                kbSection
                modelsSection
                sessionSection
                aboutSection
            }
            .navigationTitle("Settings")
            .alert("Reset session?", isPresented: $showResetConfirm) {
                Button("Reset", role: .destructive, action: resetSession)
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Clears the token ledger, latency stats, and PII counts. Conversation history is not affected.")
            }
        }
    }

    private var modeSection: some View {
        Section(header: Text("Experience mode")) {
            Picker("Mode", selection: $appState.appMode) {
                Text("Customer").tag(AppMode.customer)
                Text("Engineering").tag(AppMode.engineering)
            }
            .pickerStyle(.segmented)
            Text(appState.appMode == .customer
                 ? "Clean chat experience. Traces and confidence scores are hidden."
                 : "Full instrumentation. Trace rows, tool cards, and latency counters visible.")
                .font(.caption).foregroundStyle(brand.textSecondary)
        }
    }

    private var kbSection: some View {
        Section(header: Text("Knowledge base")) {
            LabeledContent("Version", value: appState.knowledgeBase.version)
            LabeledContent("Entries", value: "\(appState.knowledgeBase.entries.count)")
            Text("Drop a newer `knowledge-base.json` into the app's Documents directory and relaunch to use it without rebuilding.")
                .font(.caption).foregroundStyle(brand.textSecondary)
        }
    }

    @ViewBuilder
    private var modelsSection: some View {
        if appState.appMode == .engineering {
            Section(header: Text("On-device models")) {
                LabeledContent("Base", value: TelcoModelBundle.baseModelName)
                LabeledContent("Decision heads", value: TelcoModelBundle.sharedClfAdapterPath() == nil ? "paired adapters" : TelcoModelBundle.sharedClfAdapterName)
                LabeledContent("Chat router", value: TelcoModelBundle.chatModeRouterAdapterName)
                LabeledContent("Tool selector", value: TelcoModelBundle.toolAdapterName)
                LabeledContent("KB extractor", value: TelcoModelBundle.kbExtractorAdapterName)
                Text("Simple support decisions run on-device. Complex requests can be handed to an existing cloud AI stack after privacy review.")
                    .font(.caption).foregroundStyle(brand.textSecondary)
            }
        }
    }

    @ViewBuilder
    private var sessionSection: some View {
        if appState.appMode == .engineering {
            Section(header: Text("Session")) {
                LabeledContent("Tokens kept on-device", value: "\(appState.tokenLedger.totalTokensSaved)")
                LabeledContent("On-device answers", value: "\(appState.tokenLedger.messagesOnDevice)")
                LabeledContent("Tool deflections", value: "\(appState.tokenLedger.messagesDeflected)")
                Button("Reset metrics", role: .destructive) { showResetConfirm = true }
            }
        }
    }

    private var aboutSection: some View {
        Section(header: Text("About")) {
            LabeledContent("Build", value: "\(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0") (\(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "0"))")
            LabeledContent("App", value: "\(brand.appName) \(brand.appSubtitle)")
            Text("Generic telco support assistant demo running LFM2.5-350M-Base with multi-head triage, local Q&A, voice-ready input, and cloud handoff for complex cases.")
                .font(.caption).foregroundStyle(brand.textSecondary)
        }
    }

    private func resetSession() {
        appState.tokenLedger.reset()
        appState.sessionStats.reset()
    }
}
