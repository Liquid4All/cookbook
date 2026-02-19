//! Plan output parsers for the orchestrator planner model.
//!
//! Supports two formats:
//! - **Bracket format** (primary): `[plan.add_step(step=1, server="fs", description="...")]`
//!   This matches LFM2-24B-A2B's native bracket tool-call syntax.
//! - **JSON format** (fallback): `{"needs_tools":true,"steps":[...]}`
//!   For models that can produce structured JSON.
//!
//! The bracket parser is tried first because LFM2-24B-A2B had a 94% JSON parse
//! failure rate in orchestrator benchmarks — the model naturally produces bracket
//! syntax, not raw JSON.

use crate::agent_core::orchestrator::{PlanStep, StepPlan};

// ─── Bracket-Format Parser ──────────────────────────────────────────────────

/// Parse bracket-format plan output from LFM2-24B-A2B.
///
/// Extracts calls in the form:
///   `[plan.add_step(step=1, server="filesystem", description="...")]`
///   `[plan.respond(message="direct answer")]`
///   `[plan.done()]`
pub fn parse_bracket_plan(text: &str) -> Option<StepPlan> {
    let mut steps: Vec<PlanStep> = Vec::new();
    let mut direct_response: Option<String> = None;
    let mut found_any_call = false;

    for line in text.lines() {
        let line = line.trim();

        // Match [plan.add_step(...)]
        if let Some(inner) = extract_bracket_call(line, "plan.add_step") {
            found_any_call = true;
            if let Some(step) = parse_add_step_args(inner) {
                steps.push(step);
            }
        }

        // Match [plan.respond(message="...")]
        if let Some(inner) = extract_bracket_call(line, "plan.respond") {
            found_any_call = true;
            if let Some(msg) = extract_named_string_arg(inner, "message") {
                direct_response = Some(msg);
            }
        }

        // Match [plan.done()] — signals end of plan
        if extract_bracket_call(line, "plan.done").is_some() {
            found_any_call = true;
        }
    }

    if !found_any_call {
        return None;
    }

    if let Some(ref response) = direct_response {
        Some(StepPlan {
            needs_tools: false,
            direct_response: Some(response.clone()),
            steps: Vec::new(),
        })
    } else if !steps.is_empty() {
        Some(StepPlan {
            needs_tools: true,
            direct_response: None,
            steps,
        })
    } else {
        None
    }
}

// ─── JSON-Format Parser ─────────────────────────────────────────────────────

/// Parse JSON-format plan output (fallback for models that produce JSON).
pub fn parse_json_plan(text: &str) -> Result<StepPlan, String> {
    let json_str = extract_json(text);
    serde_json::from_str::<StepPlan>(json_str)
        .map_err(|e| format!("failed to parse plan JSON: {e}"))
}

/// Extract JSON from text that may be wrapped in markdown code fences.
fn extract_json(text: &str) -> &str {
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            return &text[start..=end];
        }
    }
    text
}

// ─── Bracket Argument Extractors ─────────────────────────────────────────────

/// Extract the inner arguments from a bracket call like `[fn_name(args)]`.
///
/// Returns the argument string between parentheses, or `None` if the line
/// doesn't match the expected function name.
fn extract_bracket_call<'a>(line: &'a str, fn_name: &str) -> Option<&'a str> {
    let pattern = format!("[{fn_name}(");
    let start = line.find(&pattern)?;
    let args_start = start + pattern.len();

    // Find matching closing )]
    let rest = &line[args_start..];
    let close = rest.rfind(")]")?;

    Some(&rest[..close])
}

/// Parse arguments from a `plan.add_step(step=1, server="fs", description="...")` call.
fn parse_add_step_args(args: &str) -> Option<PlanStep> {
    let step_num = extract_named_int_arg(args, "step").unwrap_or(1);
    let server = extract_named_string_arg(args, "server");
    let description = extract_named_string_arg(args, "description")?;

    Some(PlanStep {
        step_number: step_num,
        description,
        expected_server: server,
        hint_params: None,
    })
}

/// Extract a named string argument like `key="value"` from a comma-separated arg list.
///
/// Handles escaped quotes within the value.
pub fn extract_named_string_arg(args: &str, key: &str) -> Option<String> {
    let pattern = format!("{key}=\"");
    let start = args.find(&pattern)?;
    let value_start = start + pattern.len();
    let rest = &args[value_start..];

    // Find the closing quote, handling escaped quotes
    let mut end = 0;
    let bytes = rest.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' && (end == 0 || bytes[end - 1] != b'\\') {
            break;
        }
        end += 1;
    }

    if end >= bytes.len() {
        return None;
    }

    let value = &rest[..end];
    Some(value.replace("\\\"", "\""))
}

/// Extract a named integer argument like `step=1` from a comma-separated arg list.
pub fn extract_named_int_arg(args: &str, key: &str) -> Option<u32> {
    let pattern = format!("{key}=");
    let start = args.find(&pattern)?;
    let value_start = start + pattern.len();
    let rest = &args[value_start..];

    // Read digits until non-digit
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Bracket-format plan parsing ─────────────────────────────────────

    #[test]
    fn parse_bracket_single_step() {
        let text = r#"[plan.add_step(step=1, server="filesystem", description="List files in /tmp")]
[plan.done()]"#;
        let plan = parse_bracket_plan(text).unwrap();
        assert!(plan.needs_tools);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].step_number, 1);
        assert_eq!(plan.steps[0].expected_server.as_deref(), Some("filesystem"));
        assert_eq!(plan.steps[0].description, "List files in /tmp");
    }

    #[test]
    fn parse_bracket_multi_step() {
        let text = r#"[plan.add_step(step=1, server="filesystem", description="List all PDF files in /Users/chintan/Downloads")]
[plan.add_step(step=2, server="document", description="Using the result from step 1, extract text from the first PDF")]
[plan.add_step(step=3, server="knowledge", description="Index the extracted text for semantic search")]
[plan.done()]"#;
        let plan = parse_bracket_plan(text).unwrap();
        assert!(plan.needs_tools);
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0].step_number, 1);
        assert_eq!(plan.steps[1].step_number, 2);
        assert_eq!(plan.steps[2].step_number, 3);
        assert_eq!(plan.steps[2].expected_server.as_deref(), Some("knowledge"));
        assert!(plan.steps[1].description.contains("result from step 1"));
    }

    #[test]
    fn parse_bracket_direct_response() {
        let text = r#"[plan.respond(message="The capital of France is Paris.")]"#;
        let plan = parse_bracket_plan(text).unwrap();
        assert!(!plan.needs_tools);
        assert_eq!(
            plan.direct_response.as_deref(),
            Some("The capital of France is Paris.")
        );
        assert!(plan.steps.is_empty());
    }

    #[test]
    fn parse_bracket_with_escaped_quotes() {
        let text =
            r#"[plan.add_step(step=1, server="filesystem", description="Search for files named \"report.pdf\" in Downloads")]
[plan.done()]"#;
        let plan = parse_bracket_plan(text).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(
            plan.steps[0].description,
            r#"Search for files named "report.pdf" in Downloads"#
        );
    }

    #[test]
    fn parse_bracket_no_server_hint() {
        let text =
            r#"[plan.add_step(step=1, description="Find all receipts from last month")]
[plan.done()]"#;
        let plan = parse_bracket_plan(text).unwrap();
        assert_eq!(plan.steps.len(), 1);
        assert!(plan.steps[0].expected_server.is_none());
    }

    #[test]
    fn parse_bracket_with_surrounding_text() {
        let text = r#"Here is the plan:
[plan.add_step(step=1, server="filesystem", description="List files in /tmp")]
[plan.done()]"#;
        let plan = parse_bracket_plan(text).unwrap();
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn parse_bracket_returns_none_for_plain_text() {
        let text = "I can help you with that! Let me list the files.";
        assert!(parse_bracket_plan(text).is_none());
    }

    #[test]
    fn parse_bracket_returns_none_for_empty() {
        assert!(parse_bracket_plan("").is_none());
    }

    // ─── Extract bracket call internals ──────────────────────────────────

    #[test]
    fn extract_bracket_call_basic() {
        let line = r#"[plan.add_step(step=1, server="fs", description="test")]"#;
        let inner = extract_bracket_call(line, "plan.add_step").unwrap();
        assert_eq!(inner, r#"step=1, server="fs", description="test""#);
    }

    #[test]
    fn extract_bracket_call_no_args() {
        let line = "[plan.done()]";
        let inner = extract_bracket_call(line, "plan.done").unwrap();
        assert_eq!(inner, "");
    }

    #[test]
    fn extract_bracket_call_wrong_name() {
        let line = "[plan.add_step(step=1)]";
        assert!(extract_bracket_call(line, "plan.respond").is_none());
    }

    // ─── Named argument extraction ───────────────────────────────────────

    #[test]
    fn extract_string_arg_basic() {
        let args = r#"step=1, server="filesystem", description="List files""#;
        assert_eq!(
            extract_named_string_arg(args, "server"),
            Some("filesystem".to_string())
        );
        assert_eq!(
            extract_named_string_arg(args, "description"),
            Some("List files".to_string())
        );
    }

    #[test]
    fn extract_string_arg_missing() {
        let args = r#"step=1, description="test""#;
        assert!(extract_named_string_arg(args, "server").is_none());
    }

    #[test]
    fn extract_int_arg_basic() {
        let args = r#"step=3, server="filesystem""#;
        assert_eq!(extract_named_int_arg(args, "step"), Some(3));
    }

    #[test]
    fn extract_int_arg_missing() {
        let args = r#"server="filesystem""#;
        assert!(extract_named_int_arg(args, "step").is_none());
    }

    // ─── JSON fallback parsing ───────────────────────────────────────────

    #[test]
    fn parse_json_step_plan() {
        let json = r#"{"needs_tools":true,"steps":[{"step_number":1,"description":"List files in /tmp","expected_server":"filesystem"}]}"#;
        let plan = parse_json_plan(json).unwrap();
        assert!(plan.needs_tools);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].expected_server.as_deref(), Some("filesystem"));
    }

    #[test]
    fn parse_json_no_tools_plan() {
        let json = r#"{"needs_tools":false,"direct_response":"The answer is 42."}"#;
        let plan = parse_json_plan(json).unwrap();
        assert!(!plan.needs_tools);
        assert_eq!(plan.direct_response.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn extract_json_from_markdown() {
        let wrapped = "```json\n{\"needs_tools\":true,\"steps\":[]}\n```";
        let result = extract_json(wrapped);
        assert_eq!(result, "{\"needs_tools\":true,\"steps\":[]}");
    }

    #[test]
    fn extract_json_bare() {
        let bare = "{\"needs_tools\":false}";
        let result = extract_json(bare);
        assert_eq!(result, bare);
    }
}
