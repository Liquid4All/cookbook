// VerizonRAGTestView.swift
//
// Engineering-mode probe view for the Verizon Stage B step-format
// generator (ADR-021 §5.2). Loads the merged Q4_K_M GGUF on first
// appear, lets the operator type a query, runs it through
// LlamaBackend, and shows the response + on-device latency. The
// router (VerizonRagRouter) is NOT consulted — this view tests the
// generator in isolation so latency / format compliance can be
// measured against the 89-query probe set in
// data/finetune/vz-home-internet/probe_intents.json.
//
// Reachable from Settings → "Verizon RAG Test" when appMode =
// .engineering. Hidden from customer-facing surfaces.

// NOTE: no `import LFMEngine` — the LFMEngine sources are compiled
// directly into the TelcoTriage app target (see project.pbxproj),
// so `LlamaBackend`, `LlamaChatMessage`, and `GenerationParams` are
// already in this module's namespace.
import SwiftUI

struct VerizonRAGTestView: View {

    // MARK: - State

    @State private var query: String = "How do I change my Wi-Fi password?"
    @State private var responseText: String = ""
    @State private var latencyMs: Double = 0
    @State private var tokenCount: Int = 0
    @State private var isLoading: Bool = false
    @State private var loadError: String?
    @State private var backend: LlamaBackend?

    // System prompt mirrors scripts/vz/generate/prompts.py
    // VERIZON_SYSTEM_PROMPT_SUMMARY — the same contract the teacher,
    // Phase 0a baseline, Stage B scorer, and Q4+GBNF scorer all used.
    // Kept in-source (not bundled as a resource) so the view runs even
    // when the Resources/Verizon/_source/ docs aren't checked in.
    private let systemPrompt = """
You are the Verizon Home Internet GenAI RAG Assistant. Help customers \
with router, network, devices, parental controls, equipment, and \
Digital Secure Home questions.

OUTPUT CONTRACT (when an answer is grounded):
- One-sentence intro ending in a colon, then a single line of the form:
  Go to [Link Name](vzhome://link-path) > Step 1 > Step 2 > ...
- Steps separated by " > ", no periods inside steps, optional terminal period.
- No emojis. No exclamation points. No markdown bullets.

REFUSAL PATTERNS:
- Out-of-scope: "I'm here to help with topics related to Verizon Home \
Internet. Please try asking a different question."
- In-scope but no RAG match: "It looks like I don't have information about that."
- Account/billing question (cannot fetch data): "I can't [thing] here. \
Go to [Account](vzhome://tab-more?launchPoint=verizonAssistant) > Bills > ..."

LIVE-AGENT ESCALATION:
- Trigger on: explicit human request, outage, technician issue, security \
incident, Wi-Fi Backup malfunction, equipment-return logistics.
- Response: "Connecting you to a Verizon support agent. Estimated wait: N minutes."

DEEP-LINK SCHEME:
- All in-app links use vzhome:// URI scheme. Examples:
  vzhome://tab-home, vzhome://network?launchPoint=verizonAssistant,
  vzhome://equipment, vzhome://restart-router, vzhome://speed-test,
  vzhome://tab-devices?launchPoint=verizonAssistant,
  vzhome://tab-more?launchPoint=verizonAssistant.
"""

    // Canonical probe queries — one per RAG-eligible intent. Tapping a
    // sample sets the query field and immediately runs it so the
    // operator can sweep all six intents in seconds.
    private let sampleQueries: [(String, String)] = [
        ("NETWORK",   "How do I change my Wi-Fi password?"),
        ("EQUIPMENT", "How do I restart my router"),
        ("PARENTAL",  "How do I enable parental controls"),
        ("DEVICES",   "Where can I see my devices"),
        ("HOME_PAGE", "Run a speed test"),
        ("DSH",       "How do I turn off Digital Secure Home"),
    ]

    // MARK: - Body

    var body: some View {
        Form {
            statusSection
            sampleSection
            inputSection
            if !responseText.isEmpty {
                responseSection
            }
        }
        .navigationTitle("Legacy Stage B Test")
        .navigationBarTitleDisplayMode(.inline)
        .task { await loadBackendIfNeeded() }
    }

    // MARK: - Sections

    @ViewBuilder
    private var statusSection: some View {
        Section(header: Text("Generator")) {
            LabeledContent("Model", value: TelcoModelBundle.verizonStageBGeneratorName)
            if let err = loadError {
                Text(err).font(.caption).foregroundStyle(.red)
            } else if backend == nil {
                Text("Loading model…").font(.caption).foregroundStyle(.secondary)
            } else {
                Text("Loaded. Tap a sample or type a query.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var sampleSection: some View {
        Section(header: Text("Probe samples")) {
            ForEach(sampleQueries, id: \.1) { (intent, sample) in
                Button {
                    query = sample
                    Task { await run() }
                } label: {
                    HStack {
                        Text(intent)
                            .font(.caption.monospaced())
                            .frame(width: 80, alignment: .leading)
                            .foregroundStyle(.secondary)
                        Text(sample)
                            .lineLimit(2)
                    }
                }
                .disabled(backend == nil || isLoading)
            }
        }
    }

    private var inputSection: some View {
        Section(header: Text("Custom query")) {
            TextField("Ask Liquid Telco…", text: $query, axis: .vertical)
                .lineLimit(2 ... 5)
            Button(isLoading ? "Generating…" : "Send") {
                Task { await run() }
            }
            .disabled(backend == nil || isLoading || query.isEmpty)
        }
    }

    private var responseSection: some View {
        Section(
            header: HStack {
                Text("Response")
                Spacer()
                Text("\(Int(latencyMs)) ms · \(tokenCount) tokens")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        ) {
            Text(responseText)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
        }
    }

    // MARK: - Lifecycle

    private func loadBackendIfNeeded() async {
        guard backend == nil else { return }
        guard let path = TelcoModelBundle.verizonStageBGeneratorPath() else {
            loadError = "vz-stage-b-v1.Q4_K_M.gguf missing from app bundle. " +
                        "Add it to the Liquid Telco app target in Xcode."
            return
        }
        let b = LlamaBackend()
        // BUG-022 fix: the iOS simulator reports 0 MiB free on MTL0 —
        // requesting GPU offload there produces all-pad output. Real
        // devices have real Metal memory and the production path uses
        // gpuLayers: 99.
        let gpuLayers: Int32
        #if targetEnvironment(simulator)
        gpuLayers = 0
        #else
        gpuLayers = 99
        #endif
        do {
            try await b.loadModel(
                path: path,
                contextLength: 2048,
                gpuLayers: gpuLayers,
                temperature: 0
            )
            backend = b
        } catch {
            loadError = "Model load failed: \(error.localizedDescription)"
        }
    }

    private func run() async {
        guard let backend, !query.isEmpty else { return }
        isLoading = true
        defer { isLoading = false }
        responseText = ""
        do {
            let (text, tokens, timing) = try await backend.generate(
                messages: [
                    .system(systemPrompt),
                    .user(query),
                ],
                maxTokens: 160,
                temperature: 0,
                stopSequences: [],
                clearCache: true,
                outputMode: .text
            )
            responseText = text
            tokenCount = tokens
            // LlamaBackend.GenerationTiming (declared inline in
            // LlamaBackend+WarmUp.swift) has no `totalMs` — only
            // promptEvalMs + tokenGenerationMs. Sum them for the
            // wall-clock number we want to show.
            latencyMs = timing.promptEvalMs + timing.tokenGenerationMs
        } catch {
            responseText = "Error: \(error.localizedDescription)"
            tokenCount = 0
            latencyMs = 0
        }
    }
}

#Preview {
    NavigationStack {
        VerizonRAGTestView()
    }
}
