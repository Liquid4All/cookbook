import SwiftUI
import Combine

struct ChatView: View {
    @EnvironmentObject private var appState: AppState
    @StateObject private var holder = ChatViewModelHolder()
    @State private var showingPhotoPicker = false
    @State private var showingSettings = false
    @State private var toolConfirmationMessage: ChatMessage?
    @FocusState private var isInputFocused: Bool

    @Environment(\.brand) private var brand
    @Environment(\.appMode) private var appMode

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                messageList
                if shouldShowStarters {
                    starterChips
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                inputBar
            }
            .background(brand.surfaceBackground.ignoresSafeArea())
            .navigationTitle("\(brand.appName)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showingSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                            .foregroundStyle(brand.textPrimary)
                    }
                    .accessibilityLabel("Settings")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        holder.vm?.clear()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .foregroundStyle(brand.textPrimary)
                    }
                    .accessibilityLabel("Reset conversation")
                    .disabled(holder.vm == nil)
                }
                ToolbarItem(placement: .principal) {
                    LiquidAITitleLockup(
                        appName: brand.appName,
                        subtitle: appMode == .engineering ? "Engineering Mode" : brand.appSubtitle,
                        isEngineeringSubtitle: appMode == .engineering
                    )
                    // 2-second hold instead of the default 0.5s — an exec
                    // leaning on the title during the pitch won't flip the
                    // whole UI into engineering mode. Intentional demoer
                    // gesture stays discoverable; accidental trigger
                    // basically impossible.
                    .onLongPressGesture(minimumDuration: 2.0) {
                        let generator = UIImpactFeedbackGenerator(style: .medium)
                        generator.impactOccurred()
                        withAnimation(.easeInOut(duration: 0.2)) {
                            appState.appMode = appState.appMode == .customer ? .engineering : .customer
                        }
                    }
                    .accessibilityHint("Hold for 2 seconds to toggle engineering mode")
                }
            }
            .sheet(isPresented: $showingSettings) {
                NavigationStack { SettingsView() }
            }
            .sheet(item: Binding(
                get: { holder.vm?.privacyShieldQuery },
                set: { _ in holder.vm?.dismissPrivacyShield() }
            )) { state in
                PIIInspectSheet(state: state)
            }
            .sheet(item: Binding(
                get: { holder.vm?.readingArticle },
                set: { _ in holder.vm?.dismissKBArticle() }
            )) { entry in
                KBArticleView(entry: entry)
            }
            .sheet(isPresented: $showingPhotoPicker) {
                PhotoPicker(image: Binding(
                    get: { holder.vm?.attachedImage },
                    set: { new in
                        if let img = new { holder.vm?.attachImage(img) }
                    }
                ))
            }
            .sheet(item: $toolConfirmationMessage) { msg in
                if let decision = msg.toolDecision {
                    ToolConfirmationSheet(
                        decision: decision,
                        onConfirm: {
                            holder.vm?.confirmTool(messageID: msg.id)
                            toolConfirmationMessage = nil
                        },
                        onDecline: {
                            holder.vm?.declineTool(messageID: msg.id)
                            toolConfirmationMessage = nil
                        }
                    )
                    .presentationDetents([.medium])
                    .presentationDragIndicator(.visible)
                }
            }
            .onChange(of: appState.voice.state) { _, newValue in
                if case .finalized(let text) = newValue {
                    // Populate the text field and reset voice state so
                    // the user can review + edit + tap Send. Auto-send
                    // was too magical — with no visible confirmation the
                    // user couldn't tell if the mic had captured their
                    // words. They asked for "a way to stop recording
                    // and send the message"; this is it.
                    holder.vm?.inputText = text
                    appState.voice.reset()
                    isInputFocused = true
                }
            }
            .onChange(of: holder.vm?.messages.count ?? 0) { _, _ in
                // In customer mode, auto-present tool confirmation sheet
                // when a new message arrives with a tool decision.
                guard appMode == .customer,
                      let last = holder.vm?.messages.last,
                      last.role == .assistant,
                      last.toolDecision != nil,
                      toolConfirmationMessage == nil
                else { return }
                toolConfirmationMessage = last
            }
        }
        .onAppear {
            holder.bootstrap(from: appState)
        }
    }

    private var shouldShowStarters: Bool {
        guard let vm = holder.vm else { return false }
        let onlyWelcome = vm.messages.count == 1 && vm.messages.first?.role == .assistant
        return onlyWelcome && vm.inputText.isEmpty && !vm.isProcessing
    }

    /// Wrapped grid of starter prompts — each one maps to a distinct
    /// edge-AI primitive proven by the fine-tuned adapters. Tapping
    /// sends the prompt through the normal pipeline.
    private var starterChips: some View {
        VStack(spacing: 8) {
            Text("Try asking")
                .font(.caption)
                .foregroundStyle(brand.textSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 16)

            WrappingChipGrid(starters: appMode == .customer ? ConversationStarter.customer : ConversationStarter.all) { starter in
                guard let vm = holder.vm else { return }
                vm.inputText = starter.prompt
                vm.send()
            }
            .padding(.horizontal, 16)
        }
        .padding(.bottom, 8)
    }

    // MARK: - Subviews

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 16) {
                    ForEach(holder.vm?.messages ?? []) { message in
                        ChatMessageRow(
                            message: message,
                            onTapPII: { holder.vm?.openPrivacyShield(for: message) },
                            onExecuteVisionTool: { toolID, args in
                                holder.vm?.requestVisionProposedTool(toolID: toolID, arguments: args)
                            },
                            nbaForMessage: { id in holder.vm?.nba(for: id) },
                            onAcceptNBA: { id in holder.vm?.acceptNBA(id) },
                            onDeclineNBA: { id in holder.vm?.declineNBA(id) },
                            onConfirmTool: { id in holder.vm?.confirmTool(messageID: id) },
                            onDeclineTool: { id in holder.vm?.declineTool(messageID: id) },
                            onOpenArticle: { entry in holder.vm?.openKBArticle(entry) },
                            traceExpandedBinding: Binding(
                                get: { holder.vm?.isTraceExpanded(messageID: message.id) ?? true },
                                set: { _ in holder.vm?.toggleTraceExpanded(messageID: message.id) }
                            )
                        )
                        .id(message.id)
                    }
                    if holder.vm?.isProcessing == true {
                        if appMode == .customer, let stage = holder.vm?.routingStage {
                            RoutingStatusPill(stage: stage)
                        } else {
                            ProcessingRow()
                        }
                    }
                }
                .padding(.vertical, 16)
            }
            .scrollDismissesKeyboard(.interactively)
            .onTapGesture {
                isInputFocused = false
            }
            .onChange(of: holder.vm?.messages.count ?? 0) { _, _ in
                withAnimation { proxy.scrollTo(holder.vm?.messages.last?.id, anchor: .bottom) }
            }
        }
    }

    private var inputBar: some View {
        ChatInputBar(
            text: Binding(
                get: { holder.vm?.inputText ?? "" },
                set: { holder.vm?.inputText = $0 }
            ),
            isProcessing: holder.vm?.isProcessing ?? false,
            attachedImage: holder.vm?.attachedImage,
            isListening: appState.voice.isListening,
            listeningPartial: {
                if case .listening(let partial) = appState.voice.state { return partial }
                return ""
            }(),
            voiceError: {
                if case .error(let msg) = appState.voice.state { return msg }
                return nil
            }(),
            onSend: { holder.vm?.send() },
            onMicTap: {
                if appState.voice.isListening {
                    Task { await appState.voice.stop() }
                } else {
                    // Clear any prior error before re-starting so the
                    // user isn't greeted by a stale "permission denied"
                    // strip after they fix the setting.
                    appState.voice.reset()
                    appState.voice.start()
                }
            },
            onCameraTap: { showingPhotoPicker = true },
            onClearAttachment: { holder.vm?.clearAttachment() }
        )
    }
}

/// Defers ChatViewModel construction until AppState is in scope.
@MainActor
final class ChatViewModelHolder: ObservableObject {
    @Published var vm: ChatViewModel?
    private var vmCancellable: AnyCancellable?

    func bootstrap(from appState: AppState) {
        guard vm == nil else { return }
        let newVM = ChatViewModel(
            decisionEngine: appState.decisionEngine,
            chatModeRouter: appState.chatModeRouter,
            kbExtractor: appState.kbExtractor,
            provider: appState.modelProvider,
            piiAnalyzer: appState.piiAnalyzer,
            kb: appState.knowledgeBase,
            tokenLedger: appState.tokenLedger,
            sessionStats: appState.sessionStats,
            toolRegistry: appState.toolRegistry,
            visionAnalyzer: appState.visionAnalyzer,
            customerContext: appState.customerContext,
            nbaEngine: appState.nbaEngine,
            toolSelector: appState.toolSelector,
            toolExecutor: appState.toolExecutor,
            welcomeGreetingProvider: { [weak appState] in
                guard let appState else { return "" }
                let firstName = appState.customerContext.profile.firstName
                return appState.brands.selected.welcomeGreeting(firstName)
            }
        )
        // Forward every @Published change on the VM up through our own
        // ObservableObject so ChatView (which only observes `holder`)
        // re-renders on inputText/messages/isProcessing mutations.
        vmCancellable = newVM.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        vm = newVM
    }
}

struct ProcessingRow: View {
    @Environment(\.brand) private var brand

    var body: some View {
        HStack(spacing: 8) {
            TypingDots()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(brand.surfaceElevated, in: Capsule())
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
    }
}

/// Grid of suggestion chips that wraps naturally across lines. Tapping
/// sends through the NLU pipeline.
private struct WrappingChipGrid: View {
    let starters: [ConversationStarter]
    let onSelect: (ConversationStarter) -> Void

    @Environment(\.brand) private var brand

    private var rows: [[ConversationStarter]] {
        let mid = (starters.count + 1) / 2
        return [Array(starters.prefix(mid)), Array(starters.dropFirst(mid))]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                HStack(spacing: 8) {
                    ForEach(row) { starter in
                        Button { onSelect(starter) } label: {
                            Text(starter.prompt)
                                .font(.caption)
                                .foregroundStyle(brand.textSecondary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(brand.surfaceElevated, in: Capsule())
                                .overlay(Capsule().stroke(brand.border, lineWidth: 1))
                        }
                        .buttonStyle(.plain)
                        .accessibilityHint("Sends this as a message")
                    }
                }
            }
        }
    }
}

struct TypingDots: View {
    @State private var animating = false
    @Environment(\.brand) private var brand

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .frame(width: 6, height: 6)
                    .foregroundStyle(brand.textSecondary)
                    .opacity(animating ? 1.0 : 0.3)
                    .animation(
                        .easeInOut(duration: 0.6)
                            .repeatForever()
                            .delay(Double(index) * 0.15),
                        value: animating
                    )
            }
        }
        .onAppear { animating = true }
    }
}
