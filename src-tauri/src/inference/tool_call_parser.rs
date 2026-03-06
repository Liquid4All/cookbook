//! Tool call parsing — normalizes model output to `ToolCall` structs.
//!
//! Supports three formats (per `_models/config.yaml` `tool_call_format`):
//!
//! 1. **native_json** — Standard OpenAI JSON tool calls (Qwen, GPT).
//!    The model returns `tool_calls` in the response delta with function name
//!    and JSON-encoded arguments.
//!
//! 2. **pythonic** — LFM2.5-style text-based calls.
//!    The model emits lines like:
//!    ```text
//!    Tool: filesystem.list_dir
//!    Arguments: {"path": "/Users/chintan"}
//!    ```
//!    These must be parsed and converted to the standard `ToolCall` struct.
//!
//! 3. **bracket** — LFM2-24B-A2B bracket format.
//!    The model emits tool calls in text content using brackets:
//!    ```text
//!    [filesystem.list_dir(path="/tmp")]
//!    ```
//!    Or with special tokens:
//!    ```text
//!    <|tool_call_start|>[filesystem.list_dir(path="/tmp")]<|tool_call_end|>
//!    ```

use uuid::Uuid;

use super::config::ToolCallFormat;
use super::errors::InferenceError;
use super::types::ToolCall;

// ─── Native JSON Parsing ─────────────────────────────────────────────────────

/// Parse a tool call from the accumulated streaming deltas (native_json format).
///
/// `name` and `arguments_json` are the concatenated values from all chunks
/// for a single tool call index.
pub fn parse_native_json_tool_call(
    id: Option<&str>,
    name: &str,
    arguments_json: &str,
) -> Result<ToolCall, InferenceError> {
    let call_id = id
        .map(String::from)
        .unwrap_or_else(|| format!("call_{}", Uuid::new_v4()));

    if name.is_empty() {
        return Err(InferenceError::ToolCallParseError {
            raw_response: arguments_json.to_string(),
            reason: "empty tool name".into(),
        });
    }

    let arguments: serde_json::Value =
        serde_json::from_str(arguments_json).map_err(|e| InferenceError::ToolCallParseError {
            raw_response: arguments_json.to_string(),
            reason: format!("invalid JSON arguments: {e}"),
        })?;

    Ok(ToolCall {
        id: call_id,
        name: name.to_string(),
        arguments,
    })
}

// ─── Pythonic Format Parsing ─────────────────────────────────────────────────

/// Extract tool calls from text content that uses the Pythonic format.
///
/// Looks for patterns like:
/// ```text
/// Tool: server.tool_name
/// Arguments: {"key": "value"}
/// ```
///
/// Returns all tool calls found in the text.
pub fn parse_pythonic_tool_calls(text: &str) -> Result<Vec<ToolCall>, InferenceError> {
    let mut calls = Vec::new();
    let lines: Vec<&str> = text.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();

        // Look for "Tool: <name>" lines
        if let Some(name) = line.strip_prefix("Tool:").or_else(|| line.strip_prefix("tool:")) {
            let tool_name = name.trim().to_string();

            if tool_name.is_empty() {
                i += 1;
                continue;
            }

            // The next line should be "Arguments: <json>"
            let arguments = if i + 1 < lines.len() {
                let next_line = lines[i + 1].trim();
                if let Some(args_str) = next_line
                    .strip_prefix("Arguments:")
                    .or_else(|| next_line.strip_prefix("arguments:"))
                {
                    let args_str = args_str.trim();
                    i += 1; // consume the arguments line

                    serde_json::from_str(args_str).map_err(|e| {
                        InferenceError::ToolCallParseError {
                            raw_response: args_str.to_string(),
                            reason: format!("invalid Pythonic arguments JSON: {e}"),
                        }
                    })?
                } else {
                    // No arguments line — use empty object
                    serde_json::Value::Object(serde_json::Map::new())
                }
            } else {
                serde_json::Value::Object(serde_json::Map::new())
            };

            calls.push(ToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name: tool_name,
                arguments,
            });
        }

        i += 1;
    }

    Ok(calls)
}

// ─── Bracket Format Parsing ─────────────────────────────────────────────────

/// Extract tool calls from text that uses the bracket format (LFM2-24B-A2B).
///
/// Supports multiple modes (tried in order, first match wins):
///
/// 1. **Special tokens**: `<|tool_call_start|>[server.tool(args)]<|tool_call_end|>`
/// 2. **Bare bracket**: `[server.tool_name(key="value")]`
/// 3. **Backtick mention**: `` `server.tool_name` `` (no arguments)
///
/// Arguments inside parens are parsed as Python-style kwargs and converted to JSON.
pub fn parse_bracket_tool_calls(text: &str) -> Result<Vec<ToolCall>, InferenceError> {
    // Mode 1: Special token markers
    let mut calls = parse_bracket_special_tokens(text)?;
    if !calls.is_empty() {
        return Ok(calls);
    }

    // Mode 2: Bare bracket [server.tool_name(args)]
    calls = parse_bracket_bare(text)?;
    if !calls.is_empty() {
        return Ok(calls);
    }

    // Mode 3: Backtick or bare mention of tool names (no arguments)
    calls = parse_bracket_mention(text)?;

    Ok(calls)
}

/// Mode 1: Parse `<|tool_call_start|>…<|tool_call_end|>` blocks.
fn parse_bracket_special_tokens(text: &str) -> Result<Vec<ToolCall>, InferenceError> {
    const START_TAG: &str = "<|tool_call_start|>";
    const END_TAG: &str = "<|tool_call_end|>";

    let mut calls = Vec::new();
    let mut search_from = 0;

    while let Some(start_offset) = text[search_from..].find(START_TAG) {
        let abs_start = search_from + start_offset + START_TAG.len();
        if let Some(end_offset) = text[abs_start..].find(END_TAG) {
            let block = text[abs_start..abs_start + end_offset].trim();
            search_from = abs_start + end_offset + END_TAG.len();

            // Strip optional outer brackets
            let inner = if block.starts_with('[') && block.ends_with(']') {
                &block[1..block.len() - 1]
            } else {
                block
            };

            if inner.is_empty() {
                continue;
            }

            if let Some(call) = parse_bracket_expression(inner)? {
                calls.push(call);
            }
        } else {
            break;
        }
    }

    Ok(calls)
}

/// Mode 2: Parse bare `[server.tool_name(args)]` patterns.
fn parse_bracket_bare(text: &str) -> Result<Vec<ToolCall>, InferenceError> {
    let mut calls = Vec::new();

    // Match [word.word_name(anything)] — the tool name must be dotted
    // Use a manual scan to handle nested parens correctly
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // Find the matching close bracket
            if let Some(close) = find_matching_bracket(text, i) {
                let inner = &text[i + 1..close];
                // Check if it matches tool_name(args) pattern
                if let Some(paren) = inner.find('(') {
                    let name = inner[..paren].trim();
                    if is_dotted_tool_name(name) {
                        let args_str = if inner.ends_with(')') {
                            &inner[paren + 1..inner.len() - 1]
                        } else {
                            &inner[paren + 1..]
                        };
                        let arguments = parse_bracket_args(args_str);
                        calls.push(ToolCall {
                            id: format!("call_{}", Uuid::new_v4()),
                            name: name.to_string(),
                            arguments,
                        });
                    }
                }
                i = close + 1;
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    Ok(calls)
}

/// Mode 3: Parse backtick mentions `` `server.tool` `` or bare `server.tool` references.
fn parse_bracket_mention(text: &str) -> Result<Vec<ToolCall>, InferenceError> {
    let mut calls = Vec::new();

    // Try backtick first: `server.tool_name`
    let mut search_from = 0;
    while let Some(start) = text[search_from..].find('`') {
        let abs_start = search_from + start + 1;
        if let Some(end_offset) = text[abs_start..].find('`') {
            let name = &text[abs_start..abs_start + end_offset];
            if is_dotted_tool_name(name) {
                calls.push(ToolCall {
                    id: format!("call_{}", Uuid::new_v4()),
                    name: name.to_string(),
                    arguments: serde_json::Value::Object(serde_json::Map::new()),
                });
                // Only take the first backtick mention
                return Ok(calls);
            }
            search_from = abs_start + end_offset + 1;
        } else {
            break;
        }
    }

    Ok(calls)
}

/// Common file extensions — used to reject filenames that structurally
/// resemble dotted tool names (e.g., `original_me.png`, `report.txt`).
///
/// This is a closed, stable set (file extensions rarely change) unlike
/// server prefixes which evolve as new MCP servers are added. Actual
/// tool name validation against the runtime registry happens downstream
/// in `ToolRegistry::resolve()`.
const FILE_EXTENSIONS: &[&str] = &[
    // Images
    "png", "jpg", "jpeg", "gif", "bmp", "svg", "webp", "ico", "tiff",
    // Documents
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt",
    // Text / data
    "txt", "md", "csv", "json", "xml", "yaml", "yml", "toml",
    // Web / code
    "html", "htm", "css", "js", "ts", "py", "rs", "go", "rb",
    // Archives
    "zip", "tar", "gz", "bz", "rar", "dmg", "iso",
    // Media
    "mp3", "mp4", "wav", "avi", "mov", "mkv", "flac",
    // Misc
    "log", "bak", "tmp", "swp",
];

/// Check if a string looks structurally like a dotted tool name
/// (e.g., `filesystem.list_dir`).
///
/// This is a **syntactic pre-filter** for the bracket parser — it does not
/// validate against the runtime tool registry. False positives that slip
/// through are caught downstream by `ToolRegistry::resolve()`.
///
/// Rules:
/// 1. Exactly one dot separating two parts
/// 2. Both parts are non-empty, lowercase ASCII letters + underscores only
/// 3. The suffix (part after the dot) must not be a known file extension
fn is_dotted_tool_name(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() != 2 {
        return false;
    }
    let server = parts[0];
    let tool = parts[1];

    if server.is_empty() || tool.is_empty() {
        return false;
    }

    let valid_char = |c: char| c.is_ascii_lowercase() || c == '_';
    if !server.chars().all(valid_char) || !tool.chars().all(valid_char) {
        return false;
    }

    // Reject if the suffix is a known file extension
    !FILE_EXTENSIONS.contains(&tool)
}

/// Find the matching `]` for a `[` at position `start`.
///
/// Brackets inside quoted strings are ignored to handle cases like:
/// `[task.create_task(assignments="[team]")]` where the inner `[` is part
/// of a string value, not a nested bracket.
fn find_matching_bracket(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0;
    let mut in_string = false;
    let mut string_char = 0u8;
    let mut i = start;

    while i < bytes.len() {
        let b = bytes[i];

        // Track string boundaries (skip brackets inside quoted strings)
        if !in_string && (b == b'"' || b == b'\'') {
            in_string = true;
            string_char = b;
            i += 1;
            continue;
        }
        if in_string {
            if b == string_char && (i == 0 || bytes[i - 1] != b'\\') {
                in_string = false;
            }
            i += 1;
            continue;
        }

        // Only count brackets outside strings
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Parse a single bracket expression like `server.tool(key="value", key2=123)`.
fn parse_bracket_expression(expr: &str) -> Result<Option<ToolCall>, InferenceError> {
    let paren_idx = match expr.find('(') {
        Some(idx) => idx,
        None => {
            // No parens — just a tool name
            let name = expr.trim();
            if name.is_empty() {
                return Ok(None);
            }
            return Ok(Some(ToolCall {
                id: format!("call_{}", Uuid::new_v4()),
                name: name.to_string(),
                arguments: serde_json::Value::Object(serde_json::Map::new()),
            }));
        }
    };

    let tool_name = expr[..paren_idx].trim();
    if tool_name.is_empty() {
        return Ok(None);
    }

    // Extract args between ( and final )
    let args_str = if expr.ends_with(')') {
        &expr[paren_idx + 1..expr.len() - 1]
    } else {
        &expr[paren_idx + 1..]
    };

    let arguments = parse_bracket_args(args_str);

    Ok(Some(ToolCall {
        id: format!("call_{}", Uuid::new_v4()),
        name: tool_name.to_string(),
        arguments,
    }))
}

/// Parse Python-style kwargs like `key="value", key2=123` into a JSON object.
///
/// Handles: string values (single/double quoted), numeric values, booleans,
/// and raw JSON. Falls back to empty object if kwargs parsing finds nothing.
fn parse_bracket_args(raw: &str) -> serde_json::Value {
    let raw = raw.trim();
    if raw.is_empty() {
        return serde_json::Value::Object(serde_json::Map::new());
    }

    // Try parsing as raw JSON first (some models emit JSON in brackets)
    if raw.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
            return v;
        }
    }

    // Parse Python-style kwargs: key="value", key2=123
    let mut map = serde_json::Map::new();
    let mut remaining = raw;

    while !remaining.is_empty() {
        remaining = remaining.trim_start_matches([',', ' '].as_ref()).trim();
        if remaining.is_empty() {
            break;
        }

        // Find key=value
        let eq_idx = match remaining.find('=') {
            Some(idx) => idx,
            None => break,
        };

        let key = remaining[..eq_idx].trim().trim_matches('"').trim_matches('\'');
        remaining = &remaining[eq_idx + 1..];

        // Parse the value
        let (value, rest) = parse_bracket_value(remaining);
        map.insert(key.to_string(), value);
        remaining = rest;
    }

    if map.is_empty() {
        // Fallback: wrap as empty object
        serde_json::Value::Object(serde_json::Map::new())
    } else {
        serde_json::Value::Object(map)
    }
}

/// Parse a single value from a kwargs expression. Returns `(value, remaining_str)`.
fn parse_bracket_value(input: &str) -> (serde_json::Value, &str) {
    let input = input.trim();

    // Quoted string (double or single)
    if input.starts_with('"') || input.starts_with('\'') {
        let quote = input.as_bytes()[0] as char;
        let mut end = 1;
        let mut escaped = false;
        for ch in input[1..].chars() {
            if escaped {
                escaped = false;
                end += ch.len_utf8();
                continue;
            }
            if ch == '\\' {
                escaped = true;
                end += 1;
                continue;
            }
            if ch == quote {
                let val = &input[1..end];
                let rest = &input[end + 1..];
                return (serde_json::Value::String(val.to_string()), rest);
            }
            end += ch.len_utf8();
        }
        // Unterminated string — take everything
        return (serde_json::Value::String(input[1..].to_string()), "");
    }

    // Find the next comma or end
    let end_idx = input.find(',').unwrap_or(input.len());
    let val_str = input[..end_idx].trim().trim_end_matches(')');

    // Try numeric
    if let Ok(n) = val_str.parse::<i64>() {
        return (serde_json::Value::Number(n.into()), &input[end_idx..]);
    }
    if let Ok(n) = val_str.parse::<f64>() {
        if let Some(num) = serde_json::Number::from_f64(n) {
            return (serde_json::Value::Number(num), &input[end_idx..]);
        }
    }

    // Boolean / null
    match val_str.to_lowercase().as_str() {
        "true" => return (serde_json::Value::Bool(true), &input[end_idx..]),
        "false" => return (serde_json::Value::Bool(false), &input[end_idx..]),
        "none" | "null" => return (serde_json::Value::Null, &input[end_idx..]),
        _ => {}
    }

    // Fallback: treat as string
    (serde_json::Value::String(val_str.to_string()), &input[end_idx..])
}

// ─── Malformed JSON Repair ──────────────────────────────────────────────────

/// Extract tool name and raw arguments JSON from an Ollama HTTP 500 error body.
///
/// Ollama's Harmony parser returns errors like:
/// ```json
/// {"error":{"message":"error parsing tool call: raw='{...}', err=..."}}
/// ```
///
/// Returns `Some((tool_name, raw_arguments))` if the error matches, `None` otherwise.
pub fn extract_tool_call_from_error(error_body: &str) -> Option<(String, String)> {
    // Parse the error JSON to extract the message
    let parsed: serde_json::Value = serde_json::from_str(error_body).ok()?;
    let message = parsed
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())?;

    // Must be a tool call parse error
    if !message.contains("error parsing tool call") {
        return None;
    }

    // Extract raw='{...}' — find the boundaries
    let raw_start = message.find("raw='")?;
    let raw_content_start = raw_start + 5; // skip "raw='"
    let raw_end = message[raw_content_start..].rfind("', err=")?;
    let raw_json = &message[raw_content_start..raw_content_start + raw_end];

    // Try to extract tool name from the raw JSON keys or from a
    // best-effort parse. The raw JSON is the arguments object, so the tool
    // name comes from Ollama's Harmony format. We look for known tool
    // argument patterns to infer the tool.
    // For now, we return an empty tool name — the caller must resolve it
    // from context (e.g., the last tool call the model was attempting).
    //
    // In practice, Ollama's error doesn't include the tool name in the raw
    // field. The tool name is available in the Harmony channel marker
    // (`to=functions.<name>`) which is in the full response but not the
    // error message. We return empty and let the caller handle it.
    Some((String::new(), raw_json.to_string()))
}

/// Attempt to repair malformed JSON arguments from a model tool call.
///
/// Common malformations observed in production:
/// 1. Double quotes: `"key":"value"` (extra quote before value)
/// 2. Trailing commas: `{"a":1,}`
/// 3. Missing closing brace (unbalanced)
/// 4. Unescaped control characters in string values
///
/// Returns `Some(value)` if repair succeeds, `None` if irreparable.
pub fn repair_malformed_tool_call_json(raw: &str) -> Option<serde_json::Value> {
    // First, try parsing as-is (maybe it's already valid)
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) {
        return Some(v);
    }

    let mut repaired = raw.to_string();

    // Repair 1: Fix double-quote patterns like `":"` → `":"`
    // The model sometimes generates `"key":"value"` instead of `"key":"value"`
    // Detect `":"` (quote-colon-quote-quote) and collapse to `":"`
    repaired = repair_double_quotes(&repaired);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        return Some(v);
    }

    // Repair 2: Remove trailing commas before closing braces/brackets
    repaired = repair_trailing_commas(&repaired);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        return Some(v);
    }

    // Repair 3: Balance braces — append missing closing braces
    repaired = repair_unbalanced_braces(&repaired);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        return Some(v);
    }

    // Repair 4: Strip control characters (except \n, \r, \t which are valid in JSON strings)
    repaired = repair_control_characters(&repaired);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        return Some(v);
    }

    None
}

/// Fix double-quote patterns: `":"` → `":"`
///
/// Scans character-by-character to find `":"` sequences and collapses the
/// extra quote. This handles the exact failure observed in production:
/// `"destination":""/Users/...` → `"destination":"/Users/...`
fn repair_double_quotes(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut result = Vec::with_capacity(bytes.len());
    let mut i = 0;

    while i < bytes.len() {
        // Look for pattern: `":"` (colon followed by two quotes)
        if i + 2 < bytes.len()
            && bytes[i] == b':'
            && bytes[i + 1] == b'"'
            && bytes[i + 2] == b'"'
        {
            // Check this isn't a legitimate empty string `:""`
            // Empty string would be followed by `,` or `}` or end
            if i + 3 < bytes.len() && bytes[i + 3] != b',' && bytes[i + 3] != b'}' {
                // Extra quote — skip it: emit `:"`  and skip the second `"`
                result.push(b':');
                result.push(b'"');
                i += 3; // skip `:""`
                continue;
            }
        }
        result.push(bytes[i]);
        i += 1;
    }

    String::from_utf8(result).unwrap_or_else(|_| input.to_string())
}

/// Remove trailing commas before `}` or `]`.
fn repair_trailing_commas(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == ',' {
            // Look ahead past whitespace for `}` or `]`
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                // Skip the trailing comma
                i += 1;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    result
}

/// Append closing braces to balance unmatched opening braces.
fn repair_unbalanced_braces(input: &str) -> String {
    let mut brace_depth: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for ch in input.chars() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if ch == '{' {
                brace_depth += 1;
            } else if ch == '}' {
                brace_depth -= 1;
            }
        }
    }

    if brace_depth > 0 {
        let mut result = input.to_string();
        for _ in 0..brace_depth {
            result.push('}');
        }
        result
    } else {
        input.to_string()
    }
}

/// Remove non-printable control characters that break JSON parsing.
/// Preserves `\n`, `\r`, `\t` which are valid in JSON strings.
fn repair_control_characters(input: &str) -> String {
    input
        .chars()
        .filter(|&c| !c.is_control() || c == '\n' || c == '\r' || c == '\t')
        .collect()
}

/// Parse tool calls from accumulated content, using the configured format.
pub fn parse_tool_calls(
    format: ToolCallFormat,
    content: &str,
    native_calls: &[(Option<String>, String, String)],
) -> Result<Vec<ToolCall>, InferenceError> {
    match format {
        ToolCallFormat::NativeJson => {
            let mut calls = Vec::new();
            for (id, name, args) in native_calls {
                calls.push(parse_native_json_tool_call(
                    id.as_deref(),
                    name,
                    args,
                )?);
            }
            Ok(calls)
        }
        ToolCallFormat::Pythonic => parse_pythonic_tool_calls(content),
        ToolCallFormat::Bracket => parse_bracket_tool_calls(content),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_native_json_valid() {
        let result = parse_native_json_tool_call(
            Some("call_123"),
            "filesystem.list_dir",
            r#"{"path": "/tmp"}"#,
        )
        .unwrap();

        assert_eq!(result.id, "call_123");
        assert_eq!(result.name, "filesystem.list_dir");
        assert_eq!(result.arguments["path"], "/tmp");
    }

    #[test]
    fn test_parse_native_json_generates_id() {
        let result = parse_native_json_tool_call(
            None,
            "filesystem.read_file",
            r#"{"path": "/etc/hosts"}"#,
        )
        .unwrap();

        assert!(result.id.starts_with("call_"));
        assert_eq!(result.name, "filesystem.read_file");
    }

    #[test]
    fn test_parse_native_json_empty_name() {
        let result = parse_native_json_tool_call(None, "", r#"{}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_native_json_invalid_json() {
        let result = parse_native_json_tool_call(None, "test.tool", "not json");
        assert!(result.is_err());
        if let Err(InferenceError::ToolCallParseError { reason, .. }) = result {
            assert!(reason.contains("invalid JSON"));
        }
    }

    #[test]
    fn test_parse_pythonic_single_call() {
        let text = "Tool: filesystem.list_dir\nArguments: {\"path\": \"/tmp\"}";
        let calls = parse_pythonic_tool_calls(text).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(calls[0].arguments["path"], "/tmp");
    }

    #[test]
    fn test_parse_pythonic_multiple_calls() {
        let text = "\
I'll list the directory and then read a file.

Tool: filesystem.list_dir
Arguments: {\"path\": \"/tmp\"}

Tool: filesystem.read_file
Arguments: {\"path\": \"/tmp/test.txt\"}";

        let calls = parse_pythonic_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(calls[1].name, "filesystem.read_file");
    }

    #[test]
    fn test_parse_pythonic_no_arguments() {
        let text = "Tool: system.get_info\nSome other text";
        let calls = parse_pythonic_tool_calls(text).unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "system.get_info");
        assert!(calls[0].arguments.is_object());
    }

    #[test]
    fn test_parse_pythonic_no_tool_calls() {
        let text = "Just a regular response with no tool calls.";
        let calls = parse_pythonic_tool_calls(text).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_tool_calls_native() {
        let native = vec![(
            Some("id1".to_string()),
            "test.tool".to_string(),
            r#"{"key": "val"}"#.to_string(),
        )];

        let calls = parse_tool_calls(ToolCallFormat::NativeJson, "", &native).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "test.tool");
    }

    #[test]
    fn test_parse_tool_calls_pythonic() {
        let content = "Tool: test.tool\nArguments: {\"key\": \"val\"}";
        let calls = parse_tool_calls(ToolCallFormat::Pythonic, content, &[]).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "test.tool");
    }

    // ─── JSON Repair Tests ──────────────────────────────────────────────

    #[test]
    fn test_repair_double_quote() {
        // The exact failure from agent.log: `"destination":""/Users/...`
        let raw = r#"{"create_dirs":true,"destination":""/Users/chintan/Desktop/file.png","source":"/Users/chintan/Desktop/file.png"}"#;
        let result = repair_malformed_tool_call_json(raw);
        assert!(result.is_some(), "should repair double-quote pattern");
        let v = result.unwrap();
        assert_eq!(v["destination"], "/Users/chintan/Desktop/file.png");
        assert_eq!(v["source"], "/Users/chintan/Desktop/file.png");
        assert_eq!(v["create_dirs"], true);
    }

    #[test]
    fn test_repair_trailing_comma() {
        let raw = r#"{"path": "/tmp", "recursive": true,}"#;
        let result = repair_malformed_tool_call_json(raw);
        assert!(result.is_some(), "should repair trailing comma");
        assert_eq!(result.unwrap()["path"], "/tmp");
    }

    #[test]
    fn test_repair_missing_closing_brace() {
        let raw = r#"{"path": "/tmp", "recursive": true"#;
        let result = repair_malformed_tool_call_json(raw);
        assert!(result.is_some(), "should repair missing closing brace");
        assert_eq!(result.unwrap()["path"], "/tmp");
    }

    #[test]
    fn test_repair_already_valid() {
        let raw = r#"{"path": "/tmp"}"#;
        let result = repair_malformed_tool_call_json(raw);
        assert!(result.is_some());
        assert_eq!(result.unwrap()["path"], "/tmp");
    }

    #[test]
    fn test_repair_irreparable() {
        let raw = "this is not json at all and cannot be repaired";
        let result = repair_malformed_tool_call_json(raw);
        assert!(result.is_none(), "should return None for irreparable input");
    }

    #[test]
    fn test_extract_tool_call_from_error_valid() {
        let body = r#"{"error":{"message":"error parsing tool call: raw='{\"path\":\"\"/tmp\"}', err=invalid character '/' after object key:value pair"}}"#;
        let result = extract_tool_call_from_error(body);
        assert!(result.is_some());
        let (name, raw) = result.unwrap();
        assert!(name.is_empty(), "tool name not available in Ollama error");
        assert!(raw.contains("path"));
    }

    #[test]
    fn test_extract_tool_call_from_error_non_matching() {
        let body = r#"{"error":{"message":"model not found"}}"#;
        let result = extract_tool_call_from_error(body);
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_tool_call_from_error_invalid_json() {
        let result = extract_tool_call_from_error("not json");
        assert!(result.is_none());
    }

    // ─── Bracket Format Tests ─────────────────────────────────────────────

    #[test]
    fn test_bracket_special_tokens() {
        let text = r#"<|tool_call_start|>[filesystem.list_dir(path="/tmp")]<|tool_call_end|>"#;
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(calls[0].arguments["path"], "/tmp");
    }

    #[test]
    fn test_bracket_special_tokens_multiple() {
        let text = r#"I'll help you.
<|tool_call_start|>[filesystem.list_dir(path="/tmp")]<|tool_call_end|>
<|tool_call_start|>[filesystem.read_file(path="/tmp/test.txt")]<|tool_call_end|>"#;
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(calls[1].name, "filesystem.read_file");
    }

    #[test]
    fn test_bracket_bare() {
        let text = r#"I'll search for the file. [filesystem.search_files(pattern="*.pdf", path="/home")]"#;
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.search_files");
        assert_eq!(calls[0].arguments["pattern"], "*.pdf");
        assert_eq!(calls[0].arguments["path"], "/home");
    }

    #[test]
    fn test_bracket_no_args() {
        let text = "<|tool_call_start|>system.get_system_info<|tool_call_end|>";
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "system.get_system_info");
        assert!(calls[0].arguments.is_object());
    }

    #[test]
    fn test_bracket_backtick_mention() {
        let text = "You should use `filesystem.list_dir` to browse the directory.";
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
    }

    #[test]
    fn test_bracket_numeric_and_bool_args() {
        let text = r#"[data.query_sqlite(query="SELECT *", limit=50, verbose=true)]"#;
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "data.query_sqlite");
        assert_eq!(calls[0].arguments["query"], "SELECT *");
        assert_eq!(calls[0].arguments["limit"], 50);
        assert_eq!(calls[0].arguments["verbose"], true);
    }

    #[test]
    fn test_bracket_no_tool_calls() {
        let text = "Just a regular response with no tool calls at all.";
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_bracket_json_args() {
        let text = r#"<|tool_call_start|>[filesystem.list_dir({"path": "/tmp"})]<|tool_call_end|>"#;
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
        assert_eq!(calls[0].arguments["path"], "/tmp");
    }

    #[test]
    fn test_parse_tool_calls_bracket() {
        let content = r#"[filesystem.list_dir(path="/tmp")]"#;
        let calls = parse_tool_calls(ToolCallFormat::Bracket, content, &[]).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "filesystem.list_dir");
    }

    #[test]
    fn test_is_dotted_tool_name() {
        // Valid tool names — all pass structural check
        assert!(is_dotted_tool_name("filesystem.list_dir"));
        assert!(is_dotted_tool_name("ocr.extract_text_from_image"));
        assert!(is_dotted_tool_name("document.extract_text"));
        assert!(is_dotted_tool_name("knowledge.search"));
        assert!(is_dotted_tool_name("meeting.transcribe"));
        assert!(is_dotted_tool_name("security.scan_for_pii"));
        assert!(is_dotted_tool_name("calendar.list_events"));
        assert!(is_dotted_tool_name("email.send"));
        assert!(is_dotted_tool_name("task.create"));
        assert!(is_dotted_tool_name("data.query_sqlite"));
        assert!(is_dotted_tool_name("audit.list_entries"));
        assert!(is_dotted_tool_name("clipboard.read"));
        assert!(is_dotted_tool_name("system.get_system_info"));

        // Valid structurally — unknown prefixes pass the syntactic check.
        // Downstream ToolRegistry::resolve() handles semantic validation.
        assert!(is_dotted_tool_name("unknown.tool_name"));
        assert!(is_dotted_tool_name("custom.something"));

        // Invalid — not dotted at all
        assert!(!is_dotted_tool_name("not_dotted"));

        // Invalid — structural issues
        assert!(!is_dotted_tool_name(".starts_with_dot"));
        assert!(!is_dotted_tool_name("ends_with_dot."));
        assert!(!is_dotted_tool_name("has.two.dots"));
        assert!(!is_dotted_tool_name("has.UPPER"));

        // Invalid — filenames (suffix is a file extension)
        assert!(!is_dotted_tool_name("original_me.png"));
        assert!(!is_dotted_tool_name("screenshot_2026.pdf"));
        assert!(!is_dotted_tool_name("report_final.txt"));
        assert!(!is_dotted_tool_name("my_photo.jpg"));
        assert!(!is_dotted_tool_name("archive.zip"));
        assert!(!is_dotted_tool_name("audio_clip.mp3"));
        assert!(!is_dotted_tool_name("config.yaml"));
        assert!(!is_dotted_tool_name("debug.log"));
    }

    // ─── Fix F8: Brackets inside quoted strings ─────────────────────────

    #[test]
    fn find_matching_bracket_simple() {
        let text = "[filesystem.list_dir(path=\"/tmp\")]";
        assert_eq!(find_matching_bracket(text, 0), Some(text.len() - 1));
    }

    #[test]
    fn find_matching_bracket_with_inner_bracket_in_string() {
        // The "[" inside the quoted string should NOT be counted as a nested bracket
        let text = r#"[task.create_task(assignments="[team]")]"#;
        assert_eq!(find_matching_bracket(text, 0), Some(text.len() - 1));
    }

    #[test]
    fn find_matching_bracket_with_unmatched_bracket_in_value() {
        // This was the Phase 2c Session 2 failure: assignments="["
        let text = r#"[task.create_task(title="Review Q4", assignments="[")]"#;
        assert_eq!(find_matching_bracket(text, 0), Some(text.len() - 1));
    }

    #[test]
    fn parse_bracket_with_brackets_in_args() {
        let text = r#"[task.create_task(title="Review Q4", assignments="[team]")]"#;
        let calls = parse_bracket_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "task.create_task");
        assert_eq!(
            calls[0].arguments.get("title").and_then(|v| v.as_str()),
            Some("Review Q4")
        );
    }
}
