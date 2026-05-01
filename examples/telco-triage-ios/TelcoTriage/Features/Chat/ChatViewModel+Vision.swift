import Foundation
import UIKit

/// Vision pipeline for `ChatViewModel`. Split out so the main file
/// stays under the 600-line CLAUDE.md guideline. The methods here
/// call through the shared `VisionAnalyzer` protocol; the production
/// implementation is `MockVisionAnalyzer` (heuristic preview) with
/// `LFM2.5-VL-450M` as the Phase 3 swap target.
extension ChatViewModel {

    /// Heuristic confidence for vision-proposed tool decisions. The real
    /// LFM2.5-VL pack will supply its own confidence per diagnosis; this
    /// placeholder keeps the card honest until then.
    static let visionProposalConfidence: Double = 0.9

    /// Takes the attached photo, runs it through the vision analyzer,
    /// and renders a short bubble plus a `VisionDiagnosisCard`.
    /// Token accounting estimates the ~600 input tokens we'd pay at a
    /// cloud VLM, so the Savings dashboard stays honest.
    func processVisionQuery(query: String, image: UIImage) async {
        isProcessing = true
        // Vision path doesn't run the multi-head text classifier, so
        // the trace must not carry over from a prior text turn.
        lastTelcoVector = nil
        lastTelcoLane = nil
        defer { isProcessing = false }

        do {
            let diagnosis = try await visionAnalyzer.diagnose(
                image: image,
                prompt: query.isEmpty
                    ? "Describe what you see and any issue the user might be asking about."
                    : query
            )

            // Short text block — the diagnosis card handles the
            // detailed explanation. Keep the bubble copy tight so
            // the visual card is the hero.
            let text = "**\(diagnosis.headline)**"

            let visionEntry = ChatMessage(
                role: .assistant,
                text: text,
                routing: RoutingSummary(path: .answerWithRAG, toolIntent: nil, containsPII: false),
                latencyMS: diagnosis.latencyMS,
                trace: CallTrace(
                    surface: .visionPack,
                    inferenceMS: diagnosis.latencyMS
                ),
                visionDiagnosis: diagnosis
            )
            messages.append(visionEntry)

            tokenLedger.recordOnDevice(
                inputTokens: TokenEstimator.estimate(query) + 600, // image ≈ 600 tok on cloud VLMs
                outputTokens: TokenEstimator.estimate(text)
            )
            sessionStats.recordLatency(diagnosis.latencyMS)
        } catch {
            AppLog.vision.error("diagnose failed: \(error.localizedDescription, privacy: .public)")
            messages.append(ChatMessage(
                role: .assistant,
                text: "On-device vision analysis error: \(error.localizedDescription)",
                routing: RoutingSummary(path: .answerWithRAG, toolIntent: nil, containsPII: false)
            ))
        }
    }

    /// Show a tool decision card for a tool proposed by the vision
    /// diagnosis card. Surfaces the model's decision consistently with
    /// the text-based tool flow — the ToolDecisionCard's Confirm
    /// button will run the tool via ToolExecutor.
    func requestVisionProposedTool(toolID: String, arguments: [String: String] = [:]) {
        guard let tool = toolRegistry.tool(id: toolID) else { return }
        let decision = ToolDecision(
            intent: tool.intent,
            toolID: tool.id,
            displayName: tool.displayName,
            icon: tool.icon,
            description: tool.description,
            arguments: Self.formatArguments(ToolArguments(arguments)),
            confidence: Self.visionProposalConfidence,
            reasoning: "Proposed from image diagnosis.",
            requiresConfirmation: tool.requiresConfirmation,
            isDestructive: tool.isDestructive
        )
        messages.append(ChatMessage(
            role: .assistant,
            text: "Based on the image, I'd recommend **\(tool.displayName)**.",
            routing: RoutingSummary(path: .toolCall, toolIntent: tool.intent, containsPII: false),
            toolDecision: decision
        ))
    }
}
