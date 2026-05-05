import Foundation

/// Pure candidate router over telco head outputs.
///
/// No model calls live here. The function composes the ADR-015 head
/// outputs into a typed candidate action: answer from KB, propose a
/// tool, summarize local customer context, or decline locally.
///
/// Why this candidate is built from ADR-015 (`routing_lane` +
/// `required_tool`) instead of the legacy Phase-1 `mode`/`tool`/
/// `kbEntry` heads:
///
/// The Phase-1 heads were trained against per-task classification
/// adapters (`chat-mode-clf-v1`, `tool-selector-clf-v1`,
/// `kb-extract-clf-v1`). When `classifyAll()` runs against the shared
/// adapter, those heads see embeddings from a different distribution
/// than they were trained on — and silently misclassify ("how do I
/// restart my router" → kb_question instead of tool_action). The
/// ADR-015 heads (`routing_lane`, `required_tool`, `support_intent`,
/// …) WERE trained against the shared adapter, so their alignment is
/// correct and they're the right heads for cloud-assist, privacy, and
/// escalation policy.
///
/// The legacy heads remain in the vector for the engineering-mode
/// trace (so the demo can show them side-by-side), but they no
/// longer drive the actual dispatch. The chat surface then passes this
/// candidate through `TelcoModelModeArbiter`, which lets the dedicated
/// chat-mode LFM own the natural-language mode boundary.
public enum TelcoDecisionRouter {
    public static func route(
        _ vector: TelcoDecisionVector,
        kb: [KBEntry],
        availableTools: [Tool],
        extraction: ExtractionResult
    ) -> TelcoRoutingDecision {
        // Prefer the ADR-015 routing_lane when it's present and the
        // head is confident; fall back to the legacy mode head only
        // when the lane is unavailable (paired-adapter mode without
        // ADR-015 heads loaded).
        let mode: ChatMode = derivedChatMode(from: vector)
        let modePrediction = ChatModePrediction(
            mode: mode,
            confidence: routingConfidence(from: vector),
            reasoning: routingReasoning(for: vector),
            runtimeMS: Int(vector.totalMs.rounded())
        )

        switch mode {
        case .kbQuestion:
            // Citation is intentionally `noMatch` here — the chat
                // dispatch will run retrieval separately.
            // Returning a router-side citation from the misaligned
            // kb-entry head was a regression source ("restart my
            // router" → LED article). The trace UI surfaces both
            // signals so the engineering view stays informative.
            return .kbQuestion(
                modePrediction: modePrediction,
                citation: KBCitation.noMatch(runtimeMS: 0)
            )

        case .toolAction:
            return .toolAction(
                modePrediction: modePrediction,
                selection: toolSelection(
                    from: vector,
                    availableTools: availableTools,
                    extraction: extraction
                )
            )

        case .personalSummary:
            return .personalSummary(modePrediction: modePrediction)

        case .outOfScope:
            return .outOfScope(modePrediction: modePrediction)
        }
    }

    /// Map ADR-015 `routing_lane` to the legacy `ChatMode` enum the
    /// dispatch path consumes. When `routing_lane` is unavailable
    /// (paired-adapter fallback), use the legacy mode head as a
    /// last-resort signal.
    private static func derivedChatMode(from vector: TelcoDecisionVector) -> ChatMode {
        if vector.hasADR015Heads {
            switch vector.routingLane.label {
            case "local_answer":
                return .kbQuestion
            case "local_tool":
                return .toolAction
            case "cloud_assist":
                // Cloud assist + plan_account / billing intents land
                // on the personal-summary surface in this POC; other
                // intents read like KB questions to the customer.
                let intent = vector.supportIntent.label
                if intent == "plan_account" || intent == "billing" {
                    return .personalSummary
                }
                return .kbQuestion
            case "human_escalation", "blocked":
                return .outOfScope
            case "degraded":
                // Classifier couldn't process the input cleanly. We
                // refuse to refuse — the user asked a real question
                // and it's not their fault that confidence dropped.
                // Treat as a KB question so the retrieval path
                // still attempts retrieval; if RAG also misses, the
                // grounded-QA "no match" message is more honest than
                // an "off-topic" refusal.
                return .kbQuestion
            default:
                break
            }
        }
        return ChatMode(rawValue: vector.mode.label) ?? .outOfScope
    }

    private static func routingConfidence(from vector: TelcoDecisionVector) -> Double {
        if vector.hasADR015Heads {
            return vector.routingLane.confidence
        }
        return vector.mode.confidence
    }

    private static func routingReasoning(for vector: TelcoDecisionVector) -> String {
        if vector.hasADR015Heads {
            return "ADR-015 lane=\(vector.routingLane.label) intent=\(vector.supportIntent.label) tool=\(vector.requiredTool.label)"
        }
        return "legacy mode head (\(vector.inferenceMode.rawValue))"
    }

    private static func citation(
        from result: TelcoHeadResult,
        kb: [KBEntry]
    ) -> KBCitation {
        guard result.label != KBCitation.noMatchID,
              result.label != "unavailable",
              let entry = kb.first(where: { $0.id == result.label })
        else {
            return KBCitation(
                entryId: KBCitation.noMatchID,
                passage: "",
                confidence: result.confidence,
                runtimeMS: 0
            )
        }

        return KBCitation(
            entryId: entry.id,
            passage: firstSentence(from: entry.answer),
            confidence: result.confidence,
            runtimeMS: 0
        )
    }

    /// Resolve the tool from the ADR-015 `required_tool` head when
    /// available; fall back to the legacy `tool` head otherwise. The
    /// ADR-015 head's labels are kebab-case telco actions (e.g.
    /// `restart_gateway`, `speed_test`) which we translate to the
    /// app's `ToolIntent` taxonomy.
    private static func toolSelection(
        from vector: TelcoDecisionVector,
        availableTools: [Tool],
        extraction: ExtractionResult
    ) -> ToolSelection {
        let resolvedID: String
        let confidence: Double
        let reasoning: String
        if vector.hasADR015Heads, vector.requiredTool.label != "unavailable" {
            resolvedID = mapRequiredToolToToolID(vector.requiredTool.label)
            confidence = vector.requiredTool.confidence
            reasoning = "ADR-015 required_tool=\(vector.requiredTool.label)"
        } else {
            resolvedID = vector.tool.label
            confidence = vector.tool.confidence
            reasoning = "legacy tool head"
        }

        guard resolvedID != "none",
              resolvedID != "unavailable",
              resolvedID != "no_tool",
              resolvedID != "cloud_only",
              let intent = ToolIntent(toolID: resolvedID),
              availableTools.contains(where: { $0.intent == intent })
        else {
            return ToolSelection(
                intent: nil,
                confidence: confidence,
                reasoning: "\(reasoning): no matching tool",
                runtimeMS: 0
            )
        }

        return ToolSelection(
            intent: intent,
            confidence: confidence,
            arguments: arguments(for: intent, extraction: extraction),
            reasoning: reasoning,
            runtimeMS: 0
        )
    }

    /// Translate ADR-015 `required_tool` labels (snake_case telco
    /// vocabulary) into the iOS `ToolIntent` toolID strings
    /// (hyphen-case ids the registry expects).
    ///
    /// CRITICAL: `ToolIntent.init(toolID:)` parses HYPHENATED ids
    /// only — `"restart_router"` returns nil and the tool card
    /// silently never renders. This translation is the load-bearing
    /// bit between the classifier vocab and the iOS tool registry.
    private static func mapRequiredToolToToolID(_ requiredTool: String) -> String {
        switch requiredTool {
        case "restart_gateway":     return "restart-router"
        case "run_diagnostics":     return "run-diagnostics"
        case "speed_test":          return "run-speed-test"
        case "schedule_technician": return "schedule-technician"
        default:                    return requiredTool
        }
    }

    static func arguments(
        for intent: ToolIntent,
        extraction: ExtractionResult
    ) -> ToolArguments {
        var values: [String: String] = [:]

        switch intent {
        case .toggleParentalControls:
            values["action"] = parentalControlAction(from: extraction.requestedAction)
            if let target = extraction.targetDevice {
                values["target_device"] = target
            }

        case .rebootExtender:
            if let location = extraction.locationHint {
                values["extender_name"] = location
            }

        case .scheduleTechnician:
            values["preferred_date"] = extraction.requestedTime ?? "next_available"
            let issue = [
                extraction.errorCode,
                extraction.device,
                extraction.urgency == .high ? "high urgency" : nil,
            ]
                .compactMap { $0 }
                .joined(separator: " · ")
            if !issue.isEmpty {
                values["issue_summary"] = issue
            }

        case .restartRouter, .runSpeedTest, .checkConnection, .wpsPair, .runDiagnostics:
            break
        }

        return ToolArguments(values)
    }

    private static func parentalControlAction(from requestedAction: String?) -> String {
        switch requestedAction {
        case "resume_internet":
            return "disable"
        case "pause_internet":
            return "pause_internet"
        case "block":
            return "pause_internet"
        default:
            return "pause_internet"
        }
    }

    private static func firstSentence(from answer: String) -> String {
        let stripped = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        if let range = stripped.range(of: #"[.!?][\s\n]"#, options: .regularExpression) {
            return String(stripped[stripped.startIndex...range.lowerBound])
        }
        return String(stripped.prefix(200))
    }
}

public enum TelcoRoutingDecision: Sendable, Equatable {
    case kbQuestion(modePrediction: ChatModePrediction, citation: KBCitation)
    case toolAction(modePrediction: ChatModePrediction, selection: ToolSelection)
    case personalSummary(modePrediction: ChatModePrediction)
    case outOfScope(modePrediction: ChatModePrediction)
}
