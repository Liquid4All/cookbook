import Foundation

extension LlamaBackend {

    func shouldStop(for text: String, mode: GenerationParams.OutputMode) -> Bool {
        switch mode {
        case .text:
            return false
        case .jsonObject:
            return hasCompleteJSONObject(text)
        }
    }

    /// Bail early if the model is producing prose instead of JSON.
    /// After 16 tokens with no opening brace, the model is off-track —
    /// stop generating to avoid wasting ~2s on garbage output.
    func shouldBailNoJSON(for text: String, tokenCount: Int, mode: GenerationParams.OutputMode) -> Bool {
        guard mode == .jsonObject else { return false }
        guard tokenCount >= 16 else { return false }
        return !text.contains("{")
    }

    /// Check whether the generated text contains a complete top-level JSON object.
    ///
    /// Tolerates preamble text before the JSON (e.g., reasoning or markdown that
    /// SVD-extracted LoRA adapters sometimes emit). Scans past any non-`{` characters
    /// to find the first opening brace, then tracks brace depth with proper string
    /// literal handling.
    ///
    /// This matches the approach in `ResponseSynthesizer.extractJSON()` which also
    /// scans for `firstIndex(of: "{")` in potentially messy model output.
    private func hasCompleteJSONObject(_ text: String) -> Bool {
        var depth = 0
        var inString = false
        var isEscaped = false

        for character in text {
            // Skip all characters until we find the first '{'
            if depth == 0 && !inString {
                if character == "{" {
                    depth = 1
                }
                continue
            }

            if inString {
                if isEscaped {
                    isEscaped = false
                    continue
                }
                if character == "\\" {
                    isEscaped = true
                } else if character == "\"" {
                    inString = false
                }
                continue
            }

            switch character {
            case "\"":
                inString = true
            case "{":
                depth += 1
            case "}":
                depth -= 1
                if depth == 0 {
                    return true
                }
            default:
                continue
            }
        }

        return false
    }
}
