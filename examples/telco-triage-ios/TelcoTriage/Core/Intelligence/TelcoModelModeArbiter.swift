import Foundation

/// Chooses the customer-visible chat branch from model outputs.
///
/// The ADR-015 shared classifier produces rich support signals for trace,
/// cloud assist, privacy, and escalation. The chat-mode router is the
/// adapter trained on the boundary that matters for the chat surface:
/// "answer this question" versus "execute this tool now".
///
/// No lexical question guards live here. The arbiter trusts the dedicated
/// chat-mode LFM when it returns a parseable prediction, while preserving
/// ADR-015 as the fail-closed candidate if that model call fails.
public enum TelcoModelModeArbiter {
    public static func arbitrate(
        candidate: TelcoRoutingDecision,
        modePrediction: ChatModePrediction
    ) -> TelcoRoutingDecision {
        guard modePrediction.confidence > 0 else {
            return candidate
        }

        switch modePrediction.mode {
        case .kbQuestion:
            return .kbQuestion(
                modePrediction: modePrediction,
                citation: KBCitation.noMatch(runtimeMS: 0)
            )

        case .toolAction:
            if case .toolAction(_, let selection) = candidate,
               selection.intent != nil {
                return .toolAction(
                    modePrediction: modePrediction,
                    selection: selection
                )
            }
            return .toolAction(
                modePrediction: modePrediction,
                selection: .none
            )

        case .personalSummary:
            return .personalSummary(modePrediction: modePrediction)

        case .outOfScope:
            return .outOfScope(modePrediction: modePrediction)
        }
    }
}
