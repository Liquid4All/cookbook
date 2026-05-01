import Foundation

/// Conversation-starter chips shown above the input bar when the chat
/// is fresh. Six chips, each anchored to a distinct edge-AI primitive
/// proven by the chat-mode-router-v2 + telco-tool-selector-v3 +
/// kb-extractor-v1 training sets and the scenario suite in
/// `scripts/test_telco_sidecar_scenarios.py`.
///
/// Every chip's pipeline terminates at an LFM-generated response —
/// there are no keyword fallbacks, no cloud mocks, no hardcoded
/// answer copy.
///
/// Chip → primitive table:
///
///  1. Grounded Q&A (RAG)             → retrieve KB, LFM synthesizes grounded answer
///  2. Tool call + device extraction  → toggle-parental-controls, pauses device in CustomerContext
///  3. Agentic diagnostics            → run-diagnostics tool → LFM summarizes telemetry
///  4. Location extraction            → reboot-extender with extender_name="upstairs"
///  5. Personalization                → LFM summarizes CustomerContext profile
///  6. Privacy boundary               → intent=unknown → LFM composes on-device-only decline
public struct ConversationStarter: Identifiable, Sendable {
    public let id: String
    public let primitive: String
    public let icon: String
    public let label: String
    public let prompt: String

    /// Customer mode: 6 chips that collectively exercise every route
    /// ChatModeRouter was trained on, every downstream specialist, and
    /// every capability a telco executive cares about. Ordering is the
    /// demo narrative, not alphabetical.
    ///
    /// Every prompt is word-for-word adjacent to a training example —
    /// changing a chip's wording will shift the routing distribution,
    /// so keep new phrasings inside the training manifold.
    ///
    /// Chip → route → specialist → exec takeaway:
    ///
    ///  1. `kb_question` → KBExtractor `internet-slow-troubleshoot`
    ///     (17 training examples, best-covered KB entry). #1 support
    ///     call reason, deflected on-device.
    ///
    ///  2. `kb_question` → KBExtractor `restart-router`. Tutorial on
    ///     demand — pairs with chip 3 to prove modality discrimination.
    ///
    ///  3. `tool_action` → ToolSelector `restart-router` (133 training
    ///     examples, destructive, shows ToolConfirmationSheet). Same
    ///     topic as chip 2, different modality — the model reads intent,
    ///     not keywords.
    ///
    ///  4. `tool_action` → ToolSelector `toggle-parental-controls` with
    ///     extracted args `{target_device: "tablet", action:
    ///     "pause_internet"}`. On-device argument extraction — no cloud
    ///     NLU ever sees the household.
    ///
    ///  5. `personal_summary` → base model over CustomerContext. Shows
    ///     the model knows THIS customer's household without the data
    ///     leaving the phone.
    ///
    ///  6. `out_of_scope` → graceful refusal. Model stays in its lane,
    ///     doesn't hallucinate weather — trust boundary for production.
    ///
    /// IDs are prefixed with `c-` to distinguish from the engineering
    /// `.all` array.
    public static let customer: [ConversationStarter] = [
        ConversationStarter(
            id: "c-wifi-slow-kb",
            primitive: "Grounded Q&A",
            icon: "wifi.exclamationmark",
            label: "Why is my WiFi slow",
            prompt: "why is my wifi slow"
        ),
        ConversationStarter(
            id: "c-restart-router-kb",
            primitive: "Grounded Q&A",
            icon: "wifi.router",
            label: "How to restart my router",
            prompt: "how do I restart my router"
        ),
        ConversationStarter(
            id: "c-restart-router-tool",
            primitive: "Tool call",
            icon: "arrow.clockwise.circle",
            label: "Restart my router",
            prompt: "restart my router"
        ),
        ConversationStarter(
            id: "c-parental-pause",
            primitive: "Tool call + arg extraction",
            icon: "hand.raised",
            label: "Pause my son's tablet",
            prompt: "pause internet for my son's tablet"
        ),
        ConversationStarter(
            id: "c-connected-devices",
            primitive: "Personal summary",
            icon: "house.and.flag",
            label: "Show my connected devices",
            prompt: "show my connected devices"
        ),
        ConversationStarter(
            id: "c-out-of-scope",
            primitive: "Out-of-scope refusal",
            icon: "lock.shield",
            label: "Ask something off-topic",
            prompt: "what's the weather today"
        ),
    ]

    /// Engineering mode: all 6 chips, each showcasing a distinct primitive.
    public static let all: [ConversationStarter] = [
        // 1. Grounded Q&A — maps to intent `restart_equipment`,
        //    KB hit `restart-router`, RAG answer mode.
        ConversationStarter(
            id: "rag-restart-router",
            primitive: "Grounded Q&A",
            icon: "text.book.closed",
            label: "How to restart my router",
            prompt: "how do I restart my router"
        ),

        // 2. Tool call + device extraction — maps to intent
        //    `parental_controls`, tool `toggle-parental-controls` with
        //    action=pause_internet + target_device extracted from free
        //    text. Uses "pause" as the action verb because that phrasing
        //    is dense in the tool-selector training distribution; the
        //    "block …" phrasing (earlier copy) caused the LoRA to
        //    hallucinate a non-existent tool id `block-parental-
        //    controls` — verified on the local harness.
        ConversationStarter(
            id: "tool-parental-pause",
            primitive: "Tool call + device extraction",
            icon: "hand.raised.square",
            label: "Pause my son's tablet",
            prompt: "pause internet for my son's tablet"
        ),

        // 3. Agentic diagnostics — maps to intent
        //    `troubleshoot_connectivity`, tool `run-diagnostics`.
        //    The tool returns structured telemetry and the LFM
        //    summarizes it in plain English; the demo profile's
        //    unhealthy extender gives the summary substance.
        ConversationStarter(
            id: "agentic-diagnostics",
            primitive: "Agentic diagnostics",
            icon: "waveform.path.ecg.rectangle",
            label: "Run diagnostics",
            prompt: "run diagnostics on my home network"
        ),

        // 4. Location extraction — maps to intent `restart_equipment`,
        //    tool `reboot-extender` with extender_name="upstairs".
        //    Scenario `restart_extender` is an explicit v2 regression
        //    fix in the eval suite.
        ConversationStarter(
            id: "tool-reboot-extender-upstairs",
            primitive: "Location extraction",
            icon: "dot.radiowaves.up.forward",
            label: "Restart the upstairs extender",
            prompt: "restart the wifi extender upstairs"
        ),

        // 5. Personalization — LFMChatModeRouter classifies this
        //    as `.personalSummary`, which routes directly to
        //    LFMChatProvider.Mode.profileSummary over the
        //    CustomerContext snapshot.
        ConversationStarter(
            id: "personalized-summary",
            primitive: "Personalization over private context",
            icon: "house.and.flag",
            label: "Summarize my home network",
            prompt: "summarize my home network"
        ),

        // 6. Privacy boundary — the intent classifier is trained with
        //    ~80 out-of-domain examples that produce `intent=unknown`
        //    with confidence 0.15–0.40. The LFM composes an on-device
        //    decline; the trace row shows 0-byte egress.
        ConversationStarter(
            id: "privacy-out-of-scope",
            primitive: "Privacy boundary",
            icon: "lock.shield",
            label: "Ask something off-topic",
            prompt: "what is the weather in new york"
        ),
    ]
}
