import Foundation

/// Chooses which tool (if any) to invoke for a given query. This is the
/// production **Tool Selector** — backed by the `telco-tool-selector`
/// LoRA adapter over the bundled LFM2.5-350M base. Emits the exact
/// prompt format the fine-tune was trained against (see
/// `scripts/generate_telco_tool_selector.py`).
///
/// There is no keyword fallback: for a pitch demo we require the
/// fine-tuned model to run. If the GGUFs are missing, `AppState.init`
/// fails fast rather than silently degrading.
public protocol ToolSelector: Sendable {
    func select(
        query: String,
        extraction: ExtractionResult,
        availableTools: [Tool]
    ) async -> ToolSelection
}

public struct ToolSelection: Sendable, Equatable {
    public let intent: ToolIntent?
    public let confidence: Double
    public let arguments: ToolArguments
    public let reasoning: String
    public let runtimeMS: Int

    public init(
        intent: ToolIntent?,
        confidence: Double,
        arguments: ToolArguments = .empty,
        reasoning: String,
        runtimeMS: Int
    ) {
        self.intent = intent
        self.confidence = confidence
        self.arguments = arguments
        self.reasoning = reasoning
        self.runtimeMS = runtimeMS
    }

    public static let none = ToolSelection(
        intent: nil,
        confidence: 0,
        reasoning: "no matching tool",
        runtimeMS: 0
    )
}

// MARK: - LFM selector (production path)

/// Production tool selector backed by the `telco-tool-selector-v3` LoRA
/// adapter. Falls back to `ToolSelection.none` on any inference or parse
/// failure so one flaky generation never bricks the chat flow.
public struct LFMToolSelector: ToolSelector {
    private let backend: AdapterInferenceBackend
    private let adapterPath: String
    private let maxTokens: Int

    public init(
        backend: AdapterInferenceBackend,
        adapterPath: String,
        maxTokens: Int = 256
    ) {
        // 256 is sized for the widest observed tool-selection JSON (the
        // `reasoning` field often runs 25–40 tokens after fine-tune).
        self.backend = backend
        self.adapterPath = adapterPath
        self.maxTokens = maxTokens
    }

    public func select(
        query: String,
        extraction: ExtractionResult,
        availableTools: [Tool]
    ) async -> ToolSelection {
        let start = Date()
        let userPrompt = Self.buildPrompt(query: query)

        // This LoRA was trained via leap-finetune with ChatML wrapping,
        // so we MUST dispatch through the chat-template path. See
        // AdapterInferenceBackend.swift.
        let raw: String
        do {
            raw = try await backend.generate(
                messages: [.user(userPrompt)],
                adapterPath: adapterPath,
                maxTokens: maxTokens
            )
        } catch {
            AppLog.intelligence.error("tool selector backend failed: \(error.localizedDescription, privacy: .public)")
            return ToolSelection(
                intent: nil,
                confidence: 0,
                reasoning: "",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        guard let parsed = Self.parseAssistantJSON(raw) else {
            AppLog.intelligence.error("tool selector produced unparseable output (len=\(raw.count, privacy: .public))")
            return ToolSelection(
                intent: nil,
                confidence: 0,
                reasoning: "",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        // Model emits "none" when no tool matches — collapse to the
        // shared sentinel rather than inventing a .none enum case.
        if parsed.toolID == "none" {
            return ToolSelection(
                intent: nil,
                confidence: parsed.confidence,
                reasoning: parsed.reasoning,
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        guard let intent = ToolIntent(toolID: parsed.toolID) else {
            // Model emitted a tool_id we don't know (e.g. "set-downtime"
            // if a future adapter version ships ahead of the Swift
            // enum). Log + fall through — the chat flow handles
            // `intent: nil` as "no tool", not as an error.
            AppLog.intelligence.warning("tool selector emitted unknown tool_id: \(parsed.toolID, privacy: .public)")
            return ToolSelection(
                intent: nil,
                confidence: 0,
                reasoning: "",
                runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
            )
        }

        return ToolSelection(
            intent: intent,
            confidence: parsed.confidence,
            arguments: ToolArguments(parsed.arguments),
            reasoning: parsed.reasoning,
            runtimeMS: Int(Date().timeIntervalSince(start) * 1000)
        )
    }

    // MARK: - Prompt

    /// Mirrors the user-message template in
    /// `scripts/generate_telco_tool_selector.py::build_production_prompt`.
    /// Critical: the LoRA adapter was trained against this exact string —
    /// every whitespace change costs accuracy. Keep in sync with the
    /// generator's `PRODUCTION_PROMPT_TEMPLATE`.
    static func buildPrompt(query: String) -> String {
        return """
        Select the correct tool and fill parameters for this telco home internet customer query.
        The intent router has already determined this query requires a tool execution.

        Query: "\(query)"

        Available tools:
        \(Self.toolCatalogJSON)

        Select the best tool from the catalog. Fill any required parameters based on the query context.
        If no tool matches the query, return tool_id "none".

        Return JSON: {"tool_id": "restart-router", "arguments": {"target": "router"}, "reasoning": "Customer asked to reboot their router", "requires_confirmation": true, "confidence": 0.95}

        JSON:
        """
    }

    /// Static tool catalog, JSON-encoded. MUST match the catalog in
    /// `scripts/generate_telco_tool_selector.py::TELCO_TOOLS` exactly —
    /// including omitting `set-downtime`, which the adapter was never
    /// trained on (see docs/FUTURE_SCOPE.md for the retrain plan).
    private static let toolCatalogJSON: String = """
    [
      {
        "id": "restart-router",
        "display_name": "Restart Router",
        "description": "Reboots the primary router. All connected devices will briefly lose internet.",
        "requires_confirmation": true,
        "is_destructive": true,
        "parameters": {}
      },
      {
        "id": "run-speed-test",
        "display_name": "Run Speed Test",
        "description": "Measures download and upload speeds at the router.",
        "requires_confirmation": false,
        "is_destructive": false,
        "parameters": {}
      },
      {
        "id": "check-connection",
        "display_name": "Check Connection",
        "description": "Checks connection status of all devices and equipment on the network.",
        "requires_confirmation": false,
        "is_destructive": false,
        "parameters": {}
      },
      {
        "id": "enable-wps",
        "display_name": "Enable WPS",
        "description": "Starts WPS pairing mode so a device can connect without entering the WiFi password.",
        "requires_confirmation": true,
        "is_destructive": true,
        "parameters": {}
      },
      {
        "id": "run-diagnostics",
        "display_name": "Run Network Diagnostics",
        "description": "Runs a comprehensive network diagnostic: latency, packet loss, DNS resolution, gateway reachability.",
        "requires_confirmation": false,
        "is_destructive": false,
        "parameters": {}
      },
      {
        "id": "schedule-technician",
        "display_name": "Schedule Technician",
        "description": "Schedules an in-home technician visit for issues that cannot be resolved remotely.",
        "requires_confirmation": true,
        "is_destructive": true,
        "parameters": {
          "issue_summary": {
            "type": "string",
            "description": "Brief description of the issue for the technician"
          },
          "preferred_date": {
            "type": "string",
            "description": "Preferred date (YYYY-MM-DD) or 'next_available'",
            "default": "next_available"
          }
        }
      },
      {
        "id": "toggle-parental-controls",
        "display_name": "Toggle Parental Controls",
        "description": "Enables or disables parental controls for a specific device or profile.",
        "requires_confirmation": true,
        "is_destructive": true,
        "parameters": {
          "action": {
            "type": "string",
            "description": "Action to take",
            "enum": [
              "enable",
              "disable",
              "pause_internet"
            ]
          },
          "target_device": {
            "type": "string",
            "description": "Device name or 'all'",
            "default": "all"
          }
        }
      },
      {
        "id": "reboot-extender",
        "display_name": "Reboot Extender",
        "description": "Reboots a specific WiFi extender or mesh node. Devices connected through it will briefly disconnect.",
        "requires_confirmation": true,
        "is_destructive": true,
        "parameters": {
          "extender_name": {
            "type": "string",
            "description": "Name or location of the extender to reboot",
            "default": "primary_extender"
          }
        }
      }
    ]
    """

    // MARK: - Parsing

    static func parseAssistantJSON(_ raw: String) -> (
        toolID: String,
        arguments: [String: String],
        reasoning: String,
        confidence: Double
    )? {
        let trimmed = JSONExtract.stripFences(raw).trimmingCharacters(in: .whitespacesAndNewlines)

        guard let jsonSlice = JSONExtract.firstJSONObject(in: trimmed) else { return nil }
        guard let data = jsonSlice.data(using: .utf8) else { return nil }
        guard let any = try? JSONSerialization.jsonObject(with: data) else { return nil }
        guard let dict = any as? [String: Any] else { return nil }

        guard let toolID = dict["tool_id"] as? String else { return nil }

        // Coerce argument values to Strings. Order matters: Bool must
        // be checked before NSNumber because Swift/Foundation bridging
        // makes __NSCFBoolean match both — checking Bool first avoids
        // stringifying "true" as "1".
        var arguments: [String: String] = [:]
        if let raw = dict["arguments"] as? [String: Any] {
            for (k, v) in raw {
                if let s = v as? String {
                    arguments[k] = s
                } else if let b = v as? Bool {
                    arguments[k] = b ? "true" : "false"
                } else if let n = v as? NSNumber {
                    arguments[k] = n.stringValue
                }
            }
        }

        let reasoning = (dict["reasoning"] as? String) ?? ""
        let confidence = (dict["confidence"] as? Double)
            ?? (dict["confidence"] as? NSNumber)?.doubleValue
            ?? 0.0

        return (toolID, arguments, reasoning, confidence)
    }
}
