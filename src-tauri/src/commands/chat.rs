//! Tauri IPC commands for the chat interface.
//!
//! These commands are called from the React frontend via `invoke()`.
//! They bridge the frontend to the agent core (ConversationManager,
//! ToolRouter, and InferenceClient).

use std::sync::Mutex;

use futures::StreamExt;
use serde::Serialize;
use uuid::Uuid;

use crate::agent_core::permissions::{PermissionScope, PermissionStatus, PermissionStore};
// NOTE: response_analysis functions (is_incomplete_response, is_deflection_response)
// remain in the codebase and are tested, but are no longer called from the agent loop.
// They are available for the Orchestrator (ADR-009) or re-enablement via config.
// Tests below still exercise them for regression coverage.
use crate::agent_core::tokens::truncate_utf8;
use crate::agent_core::tool_router::{generate_preview, is_destructive_action};
use crate::agent_core::{AuditStatus, ConfirmationRequest, ConfirmationResponse};
use crate::agent_core::ConversationManager;
use crate::inference::config::{find_config_path, load_models_config};
use crate::inference::types::{SamplingOverrides, ToolDefinition};
use crate::inference::InferenceClient;
use crate::mcp_client::{CategoryRegistry, McpClient, ToolResolution};
use crate::SamplingConfig;
use crate::{PendingConfirmation, TokioMutex};

// ─── Two-Pass Tool Selection ────────────────────────────────────────────────

/// Tracks the two-pass tool selection state within the agent loop.
///
/// On `Categories` phase, the model sees ~15 category meta-tools (~1,500 tokens).
/// On `Expanded`, the model sees real tools from selected categories.
/// On `Flat` (legacy), all tools are sent every turn (~8,670 tokens).
#[derive(Debug, Clone)]
enum ToolSelectionPhase {
    /// First turn: model sees category meta-tools.
    Categories {
        /// The category registry used for expansion.
        cat_registry: CategoryRegistry,
    },
    /// Subsequent turns: model sees real tools from selected categories.
    Expanded {
        /// Category names that were selected (retained for diagnostics).
        _selected_categories: Vec<String>,
    },
    /// Legacy flat mode: all tools every turn.
    Flat,
}

/// Minimum number of registered tools to activate two-pass mode.
/// Below this threshold, flat mode is used regardless of config.
/// Set to 30 because category meta-tools confuse LFM2-24B-A2B at ≤21 tools
/// (model responds with text instead of calling tools). Two-pass is only
/// worthwhile at 67+ tools where it saves ~7k tokens/turn.
const TWO_PASS_MIN_TOOLS: usize = 30;

// ─── Response Types ─────────────────────────────────────────────────────────

/// Session start response.
#[derive(Debug, Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    /// Whether this is a newly created session or a resumed one.
    pub resumed: bool,
}

// ─── System prompt ──────────────────────────────────────────────────────────

/// Identity and intro — static portion of the system prompt.
///
/// Kept short: research shows small LLMs perform better with concise identity
/// statements. The capabilities section (dynamic) is inserted after this.
const SYSTEM_PROMPT_INTRO: &str = "\
You are LocalCowork, a private on-device AI assistant. You call tools to help the user.";

/// Behavioral rules and few-shot examples — dynamic portion of the system prompt.
///
/// Optimized for small LLMs (24B MoE) based on research:
/// - XML section tags for clear structure (models parse sections, not paragraphs)
/// - Pre-computed relative dates (model doesn't need to reason about "today")
/// - ≤6 rules (small models lose track beyond 5-7 instructions)
/// - Calendar/date example in position 1 (primacy effect)
/// - Concrete dates in examples (no indirection)
fn system_prompt_rules(today: &str, tomorrow: &str, week_start: &str, week_end: &str) -> String {
    let home = dirs::home_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| {
            if cfg!(target_os = "windows") {
                r"C:\Users\user".to_string()
            } else if cfg!(target_os = "macos") {
                "/Users/user".to_string()
            } else {
                "/home/user".to_string()
            }
        });

    format!("\
<rules>\n\
1. Use fully-qualified tool names: filesystem.list_dir, NOT list_dir.\n\
2. Use absolute paths: {home}/Documents/file.txt, NOT ~/Documents/file.txt. \
If a WORKING FOLDER is set, use ONLY the exact paths listed there.\n\
3. READ tools: call immediately. WRITE tools: call directly (system shows confirmation).\n\
4. After a scan returns results, present findings and STOP. Do NOT auto-chain to \
mutable tools (encrypt, delete, move) without the user asking.\n\
5. Never call the same tool with the same arguments twice. Use the result you already have.\n\
6. Be concise. Respond after 1-3 tool calls unless the user asked for exhaustive processing.\n\
</rules>\n\n\
<examples>\n\
Example 1 — calendar query (use pre-computed dates, never ask the user):\n\
  User: \"What's on my calendar today?\"\n\
  You call: calendar.list_events({{\"start_date\": \"{today}\", \"end_date\": \"{today}\"}})\n\
  User: \"Any meetings tomorrow?\"\n\
  You call: calendar.list_events({{\"start_date\": \"{tomorrow}\", \"end_date\": \"{tomorrow}\"}})\n\
  User: \"What do I have this week?\"\n\
  You call: calendar.list_events({{\"start_date\": \"{week_start}\", \"end_date\": \"{week_end}\"}})\n\
  WRONG: Asking the user what today's date is. You already know it.\n\n\
Example 2 — file listing:\n\
  User: \"List my Documents folder.\"\n\
  You call: filesystem.list_dir({{\"path\": \"{home}/Documents\"}})\n\n\
Example 3 — security scan:\n\
  User: \"Scan for secrets.\"\n\
  You call: security.scan_for_secrets({{\"path\": \"{home}/Projects\"}})\n\
  Then: present findings and STOP.\n\
</examples>")
}

/// Build the system prompt with dynamic tool capabilities from the MCP registry.
///
/// Structure (optimized for small LLMs):
/// ```text
/// Identity (1 line)
/// <datetime> block with pre-computed dates </datetime>
/// <capabilities> from MCP registry </capabilities>
/// <rules> consolidated behavioral rules </rules>
/// <examples> few-shot examples (calendar first) </examples>
/// ```
///
/// Key optimizations for 24B MoE models:
/// - XML section tags (research: small models parse structured prompts better)
/// - Pre-computed relative dates (today, tomorrow, week range) — no reasoning needed
/// - Date block at position 2 (high primacy) and repeated in rules reminder
/// - ≤6 rules instead of 12 (small models lose track beyond 5-7)
fn build_system_prompt(
    registry: &crate::mcp_client::registry::ToolRegistry,
    two_pass_active: bool,
) -> String {
    use chrono::{Datelike, Duration};

    let capabilities = registry.capability_summary();

    // Pre-compute all relative dates so the model never needs to reason about them.
    // Research: small LLMs fail at date arithmetic; pre-computing eliminates the problem.
    let now = chrono::Local::now();
    let today = now.format("%Y-%m-%d").to_string();
    let day_of_week = now.format("%A").to_string(); // e.g. "Monday"
    let tomorrow = (now + Duration::days(1)).format("%Y-%m-%d").to_string();

    // Compute Monday (start) and Sunday (end) of the current week
    let weekday_num = now.weekday().num_days_from_monday(); // Mon=0, Sun=6
    let week_start = (now - Duration::days(weekday_num as i64))
        .format("%Y-%m-%d")
        .to_string();
    let week_end = (now + Duration::days((6 - weekday_num) as i64))
        .format("%Y-%m-%d")
        .to_string();

    let time_str = now.format("%H:%M").to_string();

    // Date block — prominent, structured, with pre-computed values.
    // Placed immediately after identity for maximum primacy.
    let date_block = format!(
        "<datetime>\n\
         today = {today} ({day_of_week})\n\
         tomorrow = {tomorrow}\n\
         this_week = {week_start} to {week_end}\n\
         current_time = {time_str}\n\
         Use these exact values when the user says \"today\", \"tomorrow\", \"this week\".\n\
         NEVER ask the user for a date.\n\
         </datetime>"
    );

    let rules = system_prompt_rules(&today, &tomorrow, &week_start, &week_end);

    if two_pass_active {
        let two_pass_instruction = "\n\nIMPORTANT: You will first see category-level tools \
            (like file_browse, image_ocr, data_analysis, etc.). Call 1-3 categories that match \
            the user's request. You will then receive the specific tools within those categories. \
            Always select the categories FIRST before trying to use specific tools. \
            After selecting categories and receiving the expanded tools, call the minimum \
            tools needed to answer the user's question, then provide your response.";
        format!(
            "{SYSTEM_PROMPT_INTRO}\n\n\
             {date_block}\n\n\
             <capabilities>\n{capabilities}\n</capabilities>\
             {two_pass_instruction}\n\n\
             {rules}"
        )
    } else {
        format!(
            "{SYSTEM_PROMPT_INTRO}\n\n\
             {date_block}\n\n\
             <capabilities>\n{capabilities}\n</capabilities>\n\n\
             {rules}"
        )
    }
}

/// Maximum number of tool-call round-trips per user message.
///
/// Each round allows one model response + one set of tool executions.
/// Complex tasks (e.g., OCR on 10 files) may use many rounds.
/// The model gets one call per tool per round (it can batch multiple
/// tool calls in a single response, but typically does one at a time).
const MAX_TOOL_ROUNDS: usize = 10;

/// Maximum consecutive empty responses before forcing a summary.
///
/// If the model returns 0 text AND 0 tool calls this many times in a row,
/// it's stuck (likely due to context confusion or timeout). We inject a
/// summary prompt to force text output.
const MAX_EMPTY_RETRIES: usize = 2;

/// Maximum consecutive rounds with ALL tool calls failing before injecting
/// a corrective hint.
///
/// When the model repeatedly calls the same non-existent tool (e.g.,
/// `filesystem.rename_file` instead of `filesystem.move_file`), this
/// prevents burning all 20 rounds on the same error. After this many
/// consecutive all-error rounds, we inject a hint telling the model
/// which tools actually exist.
const MAX_CONSECUTIVE_ERROR_ROUNDS: usize = 2;

/// Maximum times a single tool can fail before it's removed from the tool
/// definitions and the model is told to stop retrying.
///
/// This catches the case where the model alternates between a succeeding tool
/// and a failing one — the per-round counter (`consecutive_error_rounds`) resets
/// on every success, so this per-tool counter is the only thing that can break
/// that loop.
const MAX_SAME_TOOL_FAILURES: usize = 3;

/// Maximum consecutive duplicate tool calls (same tool name with identical
/// arguments) before the agent loop breaks.
///
/// When the model gets stuck calling the same tool repeatedly with identical
/// params (e.g., `list_directory("~/Downloads")` 3× in a row), the loop
/// should detect this and exit.
///
/// Note: `consecutive_duplicate_count()` returns 1 for the first occurrence,
/// so a threshold of 2 means "one genuine duplicate" (the tool was called
/// twice with identical args). Before reaching this hard break, the soft
/// interception in the tool execution loop will skip the redundant call and
/// inject a "you already have these results" nudge, giving the model a
/// chance to produce text.
const MAX_DUPLICATE_TOOL_CALLS: usize = 2;

/// Minimum remaining token budget to start a new agent loop round.
///
/// If the context window has fewer than this many tokens remaining, the
/// agent loop exits early rather than risk context overflow and degraded
/// model quality. Set to accommodate a model response (~500 tokens) plus
/// a tool result (~1000 tokens).
const MIN_ROUND_TOKEN_BUDGET: u32 = 1500;

/// Configuration for tool result compression.
/// These values can be made configurable via SamplingConfig in the future.
const COMPRESSION_THRESHOLD_CHARS: usize = 3_000;
const MAX_TOOL_RESULT_CHARS: usize = 6_000;

impl SamplingConfig {
    /// Get compression threshold or use default.
    pub fn compression_threshold(&self) -> usize {
        self.compression_threshold_chars.unwrap_or(COMPRESSION_THRESHOLD_CHARS)
    }

    /// Get max tool result chars or use default.
    pub fn max_tool_result_chars(&self) -> usize {
        self.max_tool_result_chars.unwrap_or(MAX_TOOL_RESULT_CHARS)
    }
}

/// Truncate a tool result if it exceeds `MAX_TOOL_RESULT_CHARS`.
///
/// Preserves the beginning of the result (which usually contains the most
/// useful information) and appends a truncation notice.
fn truncate_tool_result(result: &str, tool_name: &str) -> String {
    if result.len() <= MAX_TOOL_RESULT_CHARS {
        return result.to_string();
    }

    // Try smart compression for known tool types
    if let Some(summary) = compress_tool_result(result, tool_name) {
        if summary.len() <= MAX_TOOL_RESULT_CHARS {
            tracing::info!(
                tool = %tool_name,
                original_len = result.len(),
                compressed_len = summary.len(),
                "tool result compressed via smart summary"
            );
            return summary;
        }
    }

    // Fall back to simple truncation
    let truncated = &result[..MAX_TOOL_RESULT_CHARS];
    tracing::warn!(
        tool = %tool_name,
        original_len = result.len(),
        truncated_to = MAX_TOOL_RESULT_CHARS,
        "tool result truncated — exceeded MAX_TOOL_RESULT_CHARS"
    );
    format!(
        "{truncated}\n\n[... truncated: showing first {MAX_TOOL_RESULT_CHARS} of {} chars]",
        result.len()
    )
}

/// Compress tool results using smart extraction for known data patterns.
///
/// For directory listings, extracts just filenames and counts.
/// For search results, extracts matches and count.
/// For JSON/structured data, extracts key summaries.
///
/// Returns None if compression isn't beneficial for this tool type.
fn compress_tool_result(result: &str, tool_name: &str) -> Option<String> {
    // Only compress for read/search type operations
    let compressible_tools = [
        "list_dir",
        "search_files",
        "scan_for_secrets",
        "scan_for_pii",
        "query_knowledge",
        "list_events",
        "list_tasks",
        "search_emails",
    ];

    let is_compressible = compressible_tools
        .iter()
        .any(|t| tool_name.contains(t));

    if !is_compressible || result.len() < COMPRESSION_THRESHOLD_CHARS {
        return None;
    }

    // For directory listings: extract just file/dir names and summary
    if tool_name.contains("list_dir") {
        return compress_directory_listing(result);
    }

    // For search results: extract match counts and key matches
    if tool_name.contains("search") || tool_name.contains("scan") {
        return compress_search_results(result);
    }

    // For JSON-like results: extract key fields
    if result.starts_with('{') || result.starts_with('[') {
        return compress_json_result(result);
    }

    None
}

/// Compress a directory listing to just names and counts.
fn compress_directory_listing(result: &str) -> Option<String> {
    let lines: Vec<&str> = result.lines().collect();
    if lines.is_empty() {
        return None;
    }

    let mut files = Vec::new();
    let mut dirs = Vec::new();

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with("📁") {
            if let Some(name) = trimmed.strip_prefix("📁").map(|s| s.trim()) {
                dirs.push(name.to_string());
            }
        } else if trimmed.starts_with("📄") {
            if let Some(name) = trimmed.strip_prefix("📄").map(|s| s.trim()) {
                files.push(name);
            }
        } else {
            // Try to extract name from parenthetical format: "name (123 KB)"
            if let Some(paren_idx) = trimmed.find('(') {
                let name = trimmed[..paren_idx].trim();
                if !name.is_empty() {
                    files.push(name);
                }
            }
        }
    }

    let total = files.len() + dirs.len();
    let mut summary = format!("Total: {} items ({} files, {} directories)\n\n", total, files.len(), dirs.len());

    if !dirs.is_empty() {
        summary.push_str("Directories:\n");
        for d in dirs.iter().take(20) {
            summary.push_str(&format!("  📁 {}\n", d));
        }
        if dirs.len() > 20 {
            summary.push_str(&format!("  ... and {} more\n", dirs.len() - 20));
        }
        summary.push('\n');
    }

    if !files.is_empty() {
        summary.push_str("Files:\n");
        for f in files.iter().take(30) {
            summary.push_str(&format!("  📄 {}\n", f));
        }
        if files.len() > 30 {
            summary.push_str(&format!("  ... and {} more\n", files.len() - 30));
        }
    }

    Some(summary)
}

/// Compress search/scan results to match counts and key findings.
fn compress_search_results(result: &str) -> Option<String> {
    let lower = result.to_lowercase();

    // Try to extract match count
    let count_patterns = [
        ("found ", " matches"),
        ("matches: ", ""),
        ("results: ", ""),
        ("total: ", " items"),
    ];

    for (prefix, suffix) in &count_patterns {
        if let Some(idx) = lower.find(prefix) {
            let after_prefix = &result[idx + prefix.len()..];
            let end_idx = after_prefix
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(after_prefix.len());
            let count = &after_prefix[..end_idx];
            if !count.is_empty() && count.len() <= 6 {
                // Found a count, now get first few matches
                let matches: Vec<&str> = result
                    .lines()
                    .filter(|l| {
                        let l_lower = l.to_lowercase();
                        !l_lower.contains("found")
                            && !l_lower.contains("total")
                            && !l_lower.contains("scan")
                            && !l_lower.contains("error")
                            && l.trim().len() > 3
                    })
                    .take(10)
                    .collect();

                let mut summary = format!("{}{}{}\n\n", prefix, count, suffix);
                if !matches.is_empty() {
                    summary.push_str("Key findings:\n");
                    for m in matches {
                        let trimmed = m.trim();
                        if trimmed.len() > 100 {
                            summary.push_str(&format!("  {}\n", &trimmed[..100]));
                        } else {
                            summary.push_str(&format!("  {}\n", trimmed));
                        }
                    }
                }
                return Some(summary);
            }
        }
    }

    // Fallback: just take first 15 lines
    let lines: Vec<&str> = result.lines().take(15).collect();
    if lines.is_empty() {
        return None;
    }

    let summary = format!(
        "[... {} lines total ...]\n\n{}",
        result.lines().count(),
        lines.join("\n")
    );
    Some(summary)
}

/// Compress JSON results by extracting key fields.
fn compress_json_result(result: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(result).ok()?;

    // For arrays, extract count and first few items
    if let Some(arr) = parsed.as_array() {
        if arr.is_empty() {
            return Some("[] (empty)".to_string());
        }

        let count = arr.len();
        let mut summary = format!("[{} items]\n\n", count);

        for (i, item) in arr.iter().take(5).enumerate() {
            summary.push_str(&format!("{}. ", i + 1));
            if let Some(obj) = item.as_object() {
                // Extract common "name" or "text" fields
                if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
                    summary.push_str(&format!("name: {}", name));
                } else if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                    let preview = if text.len() > 50 {
                        format!("{}...", &text[..50])
                    } else {
                        text.to_string()
                    };
                    summary.push_str(&preview);
                } else if let Some(path) = obj.get("path").and_then(|v| v.as_str()) {
                    summary.push_str(&format!("path: {}", path));
                } else {
                    // Just stringify the object
                    let s = serde_json::to_string(item).ok()?;
                    let preview = if s.len() > 80 {
                        format!("{}...", &s[..80])
                    } else {
                        s
                    };
                    summary.push_str(&preview);
                }
            } else if let Some(s) = item.as_str() {
                let preview = if s.len() > 80 {
                    format!("{}...", &s[..80])
                } else {
                    s.to_string()
                };
                summary.push_str(&preview);
            }
            summary.push('\n');
        }

        if count > 5 {
            summary.push_str(&format!("... and {} more\n", count - 5));
        }

        return Some(summary);
    }

    // For objects, extract key fields
    if let Some(obj) = parsed.as_object() {
        let key_fields = ["text", "content", "name", "path", "message", "result", "total", "count"];
        let mut summary = String::new();

        for key in &key_fields {
            if let Some(val) = obj.get(*key) {
                if let Some(s) = val.as_str() {
                    summary.push_str(&format!("{}: {}\n", key, s));
                } else if let Some(n) = val.as_u64() {
                    summary.push_str(&format!("{}: {}\n", key, n));
                }
            }
        }

        // If we extracted nothing useful, return original
        if summary.is_empty() {
            // Just return first 500 chars
            let preview = if result.len() > 500 {
                format!("{}...", &result[..500])
            } else {
                result.to_string()
            };
            return Some(preview);
        }

        return Some(summary);
    }

    None
}

// ─── Tool Definitions ──────────────────────────────────────────────────────

/// Built-in tool definitions (filesystem operations handled in-process).
fn builtin_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            r#type: "function".to_string(),
            function: crate::inference::types::FunctionDefinition {
                name: "list_directory".to_string(),
                description: "List files and directories at the given path. \
                    Returns name, type (file/dir), size, and modification date \
                    for each entry. Use ~/path for home-relative paths."
                    .to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list, e.g. ~/Desktop"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
        ToolDefinition {
            r#type: "function".to_string(),
            function: crate::inference::types::FunctionDefinition {
                name: "read_file".to_string(),
                description: "Read the text contents of a file at the given path. \
                    Returns the file content as a string. Only works for text files."
                    .to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read, e.g. ~/Desktop/notes.txt"
                        }
                    },
                    "required": ["path"]
                }),
            },
        },
    ]
}

/// Build merged tool definitions: built-in + MCP tools from the registry.
///
/// Built-in tools (`list_directory`, `read_file`) are suppressed when the MCP
/// registry already contains their equivalents (`filesystem.list_dir`,
/// `filesystem.read_file`). This avoids confusing the model with near-duplicate
/// tools, which causes it to pick the wrong one or get stuck in loops.
fn build_all_tool_definitions(mcp_client: &McpClient) -> Vec<ToolDefinition> {
    // Map of built-in tool name → MCP equivalent that supersedes it
    let builtin_mcp_equivalents: &[(&str, &str)] = &[
        ("list_directory", "filesystem.list_dir"),
        ("read_file", "filesystem.read_file"),
    ];

    // Only include built-ins whose MCP equivalent is NOT in the registry
    let mut tools: Vec<ToolDefinition> = builtin_tool_definitions()
        .into_iter()
        .filter(|tool| {
            let name = &tool.function.name;
            !builtin_mcp_equivalents.iter().any(|(builtin, mcp)| {
                name == builtin && mcp_client.registry.get_tool(mcp).is_some()
            })
        })
        .collect();

    // Append MCP tool definitions from the registry
    let mcp_tools = mcp_client.registry.to_openai_tools();
    for mcp_tool_json in mcp_tools {
        if let Ok(tool_def) = serde_json::from_value::<ToolDefinition>(mcp_tool_json) {
            tools.push(tool_def);
        }
    }

    tools
}

/// Build tool definitions from category meta-tools (two-pass mode).
///
/// Each category becomes a synthetic OpenAI function with a single `"intent"`
/// parameter. The model calls these to signal which capability areas it needs.
/// Built-in tools (`list_directory`, `read_file`) are always included.
fn build_category_tool_definitions(cat_registry: &CategoryRegistry) -> Vec<ToolDefinition> {
    let mut tools = builtin_tool_definitions();

    let cat_tools = cat_registry.to_openai_tools();
    for cat_json in cat_tools {
        if let Ok(tool_def) = serde_json::from_value::<ToolDefinition>(cat_json) {
            tools.push(tool_def);
        }
    }

    tools
}

// ─── Tool Execution ──────────────────────────────────────────────────────

/// Execute a built-in tool call and return the result as a string.
fn execute_builtin_tool(name: &str, arguments: &serde_json::Value) -> String {
    match name {
        "list_directory" => {
            let path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or(".");
            match super::filesystem::list_directory(path.to_string()) {
                Ok(entries) => {
                    if entries.is_empty() {
                        "Directory is empty.".to_string()
                    } else {
                        let mut lines = Vec::new();
                        for e in &entries {
                            let type_icon = if e.entry_type == "dir" {
                                "📁"
                            } else {
                                "📄"
                            };
                            let size_str = if e.entry_type == "dir" {
                                String::new()
                            } else {
                                format_file_size(e.size)
                            };
                            lines.push(format!(
                                "{} {} {}",
                                type_icon, e.name, size_str
                            ));
                        }
                        lines.join("\n")
                    }
                }
                Err(e) => format!("Error: {e}"),
            }
        }
        "read_file" => {
            let path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let resolved = if path.starts_with('~') {
                if let Some(home) = dirs::home_dir() {
                    home.join(path.strip_prefix("~/").unwrap_or(path))
                } else {
                    std::path::PathBuf::from(path)
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read_to_string(&resolved) {
                Ok(content) => {
                    if content.len() > 8000 {
                        format!(
                            "{}\n\n[... truncated, showing first ~8000 chars of {} total]",
                            truncate_utf8(&content, 8000),
                            content.len()
                        )
                    } else {
                        content
                    }
                }
                Err(e) => format!("Error reading file: {e}"),
            }
        }
        _ => format!("Unknown built-in tool: {name}"),
    }
}

// ─── Tool Execution Outcome ─────────────────────────────────────────────────

/// Typed result from executing a single tool call in the agent loop.
///
/// Preserves the success/failure distinction through types instead of string
/// matching. The agent loop uses this to:
/// - Feed the right text back to the model (via `model_text()`)
/// - Track error patterns for loop detection (via `is_error()`)
/// - Build correction hints from `ToolResolution` suggestions
#[derive(Debug)]
#[allow(dead_code)] // tool_name fields used for Debug output and future ToolRouter integration
enum ToolExecutionOutcome {
    /// Tool executed successfully.
    Success { tool_name: String, text: String },

    /// Tool exists but returned an application-level error
    /// (e.g., "file not found", "permission denied").
    ToolError { tool_name: String, text: String },

    /// Tool name not found in the registry. `resolution` carries the
    /// registry's analysis (suggestions, nearest matches, etc.).
    UnknownTool {
        tool_name: String,
        resolution: ToolResolution,
        text: String,
    },

    /// Infrastructure error: timeout, server crash, transport failure.
    InfraError { tool_name: String, text: String },
}

impl ToolExecutionOutcome {
    /// The text to feed back to the model as the tool result message.
    fn model_text(&self) -> &str {
        match self {
            Self::Success { text, .. }
            | Self::ToolError { text, .. }
            | Self::UnknownTool { text, .. }
            | Self::InfraError { text, .. } => text,
        }
    }

    /// Whether this outcome represents an error (any variant except Success).
    fn is_error(&self) -> bool {
        !matches!(self, Self::Success { .. })
    }
}

/// Minimum similarity score (0.0–1.0) for auto-correcting tool names.
///
/// Below this threshold, the registry returns `NotFound` instead of
/// `Corrected`. Set conservatively to avoid correcting to the wrong tool.
const TOOL_RESOLUTION_THRESHOLD: f64 = 0.5;

/// Execute a tool call: built-in tools run in-process, MCP tools route
/// through the McpClient.
///
/// Tool names are resolved via `ToolRegistry::resolve()` which handles:
/// - Exact matches (tool exists as-is)
/// - Unprefixed names (model dropped the `server.` prefix)
/// - Fuzzy correction (model hallucinated a similar name)
///
/// Results are capped at `MAX_TOOL_RESULT_CHARS` to prevent a single large
/// result from consuming the entire context window budget.
async fn execute_tool(
    name: &str,
    arguments: &serde_json::Value,
    mcp_client: &mut McpClient,
) -> ToolExecutionOutcome {
    // Built-in tools (handled in-process for speed)
    if name == "list_directory" || name == "read_file" {
        let text = truncate_tool_result(&execute_builtin_tool(name, arguments), name);
        return ToolExecutionOutcome::Success {
            tool_name: name.to_string(),
            text,
        };
    }

    // Resolve tool name via the registry (exact → unprefixed → fuzzy)
    let resolution = mcp_client.registry.resolve(name, TOOL_RESOLUTION_THRESHOLD);

    let resolved_name = match &resolution {
        ToolResolution::Exact(n) => n.clone(),
        ToolResolution::Unprefixed { resolved, original } => {
            tracing::info!(
                original = %original,
                resolved = %resolved,
                "resolved unprefixed tool name"
            );
            resolved.clone()
        }
        ToolResolution::Corrected {
            resolved,
            original,
            score,
        } => {
            tracing::info!(
                original = %original,
                resolved = %resolved,
                score = score,
                "auto-corrected tool name via edit distance"
            );
            resolved.clone()
        }
        ToolResolution::NotFound {
            original,
            suggestions,
        } => {
            let text = if suggestions.is_empty() {
                format!(
                    "Unknown tool: '{original}'. Use fully-qualified names \
                     (e.g., filesystem.list_dir, security.scan_for_secrets)."
                )
            } else {
                format!(
                    "Unknown tool: '{original}'. Did you mean: {}?",
                    suggestions.join(", ")
                )
            };
            return ToolExecutionOutcome::UnknownTool {
                tool_name: original.clone(),
                resolution,
                text,
            };
        }
    };

    // Track whether we auto-corrected the name so we can annotate errors.
    let correction_context: Option<String> = match &resolution {
        ToolResolution::Corrected {
            original, resolved, ..
        } => Some(format!(
            "NOTE: '{original}' does not exist. Auto-corrected to '{resolved}'. "
        )),
        _ => None,
    };

    // Expand `~` prefixes in string arguments before MCP dispatch.
    // Built-in tools handle tilde themselves; MCP servers expect absolute paths.
    let expanded_arguments = expand_tilde_in_arguments(arguments);

    // Execute via MCP
    match mcp_client
        .call_tool(&resolved_name, expanded_arguments)
        .await
    {
        Ok(result) => {
            let raw_text = if result.success {
                extract_mcp_result_text(&result.result)
            } else {
                result
                    .error
                    .unwrap_or_else(|| "Tool execution failed".to_string())
            };
            let text = truncate_tool_result(&raw_text, &resolved_name);
            if result.success {
                ToolExecutionOutcome::Success {
                    tool_name: resolved_name,
                    text,
                }
            } else {
                // Prepend correction context so the model understands the
                // mis-dispatch: e.g. "rename_file does not exist, corrected
                // to move_file. <actual error>".
                let annotated = if let Some(ctx) = &correction_context {
                    format!("{ctx}{text}")
                } else {
                    text
                };
                ToolExecutionOutcome::ToolError {
                    tool_name: resolved_name,
                    text: annotated,
                }
            }
        }
        Err(e) => {
            let base = format!("MCP error for '{resolved_name}': {e}");
            let text = if let Some(ctx) = &correction_context {
                format!("{ctx}{base}")
            } else {
                base
            };
            ToolExecutionOutcome::InfraError {
                tool_name: resolved_name,
                text,
            }
        }
    }
}

// `is_incomplete_response` and `is_deflection_response` are now in
// `agent_core::response_analysis` — no longer called from the agent loop,
// but still tested for regression coverage and available for the Orchestrator.

/// Detect when a model's final text claims task completion but tool history
/// disagrees — i.e., the model confabulated a summary.
///
/// This catches the pattern where the model says "I've successfully renamed
/// all 9 files" but `move_file` never appeared in `tool_call_history`.
///
/// Returns `true` when the response looks like a confabulated completion.
///
/// NOTE: Currently only used by tests. The agent loop no longer calls this
/// (continuation heuristics were removed in favour of trusting the model).
/// Retained for the Orchestrator (ADR-009) and regression test coverage.
#[cfg(test)]
fn has_unverified_completion(text: &str, tool_call_history: &[String]) -> bool {
    let lower = text.to_lowercase();

    // Only trigger on text that claims the task is done.
    let claims_done = [
        "successfully",
        "completed",
        "all files",
        "renamed",
        "processed all",
        "all done",
        "task complete",
        "finished processing",
    ];
    let claims_completion = claims_done.iter().any(|s| lower.contains(s));
    if !claims_completion {
        return false;
    }

    // Mutable operations the model might claim to have done.
    // If the model claims completion but never called any of these, it confabulated.
    // This list covers all mutable tools across all 13 MCP servers.
    let mutable_tools = [
        // Filesystem
        "move_file",
        "write_file",
        "copy_file",
        "create_dir",
        "move_to_trash",
        "rename_file",
        // Task management
        "create_task",
        "update_task",
        "delete_task",
        "complete_task",
        // Calendar
        "create_event",
        "update_event",
        "delete_event",
        // Email
        "send_email",
        "draft_email",
        // Security
        "encrypt_file",
        "decrypt_file",
        "propose_cleanup",
        // Knowledge
        "index_document",
        "delete_index",
        // Document
        "convert_document",
        "merge_documents",
    ];

    let called_any_mutable = tool_call_history
        .iter()
        .any(|t| mutable_tools.iter().any(|m| t.contains(m)));

    // If model claims done AND actually called mutable tools → not confabulated.
    if called_any_mutable {
        return false;
    }

    // If the model never called any mutable tool but claims completion, it
    // MAY be confabulated. However, we need to distinguish two cases:
    //
    // 1. Read-only task genuinely complete: "What files are in Downloads?" →
    //    model calls list_dir, says "all done" → NOT confabulation.
    //
    // 2. Mutable task not executed: "Rename all screenshots" → model calls
    //    list_dir + OCR but says "all files renamed" → IS confabulation.
    //
    // Heuristic: check if the completion text specifically claims a mutable
    // action (rename, create, move, delete, write, send, encrypt, etc.).
    // Generic "all done" / "completed" without mutable verbs is likely a
    // legitimate read-only task completion.
    // Edge case: ZERO tool calls but model claims completion — always confabulated.
    // The model literally did nothing but claims to have finished.
    if tool_call_history.is_empty() {
        return true;
    }

    // The model called tools but none were mutable. Check if the completion
    // text specifically claims a mutable action (rename, create, move, etc.).
    // Generic "all done" / "completed" without mutable verbs is likely a
    // legitimate read-only task completion.
    let mutable_action_claims = [
        "renamed",
        "moved",
        "deleted",
        "created",
        "written",
        "sent",
        "encrypted",
        "decrypted",
        "copied",
        "converted",
        "merged",
        "updated",
        "modified",
        "saved",
    ];

    let claims_mutable_action = mutable_action_claims.iter().any(|v| lower.contains(v));

    // Only confabulation if model claims a mutable action it never performed.
    // "All done" after read-only work → not confabulation (let it exit).
    // "Successfully renamed all files" after only reading → confabulation.
    claims_mutable_action
}

/// Detect if the model is stuck calling the same tool with the same arguments.
///
/// Returns the number of consecutive times the last tool call signature has
/// repeated. The caller compares this against `MAX_DUPLICATE_TOOL_CALLS`.
///
/// A "signature" is `"tool_name|arguments_json"` — if the model calls
/// `list_directory(path="~/Downloads")` three rounds in a row, this returns 3.
fn consecutive_duplicate_count(history: &[(String, String)]) -> usize {
    if history.is_empty() {
        return 0;
    }
    let last = &history[history.len() - 1];
    let mut count = 1;
    for entry in history.iter().rev().skip(1) {
        if entry.0 == last.0 && entry.1 == last.1 {
            count += 1;
        } else {
            break;
        }
    }
    count
}

/// Format a correction hint from the `ToolResolution` data collected during
/// a round where all tool calls failed.
///
/// Uses the suggestions already computed by `ToolRegistry::resolve()` — no
/// extra registry queries needed.
fn format_correction_hint(unknown_tools: &[(String, ToolResolution)]) -> String {
    if unknown_tools.is_empty() {
        return "TOOL ERROR: All tool calls in this round failed. \
                Check your tool names and try again."
            .to_string();
    }

    let mut parts = Vec::new();
    for (name, resolution) in unknown_tools {
        match resolution {
            ToolResolution::NotFound { suggestions, .. } if !suggestions.is_empty() => {
                parts.push(format!(
                    "'{name}' does not exist. Did you mean: {}?",
                    suggestions.join(", ")
                ));
            }
            _ => {
                parts.push(format!("'{name}' does not exist."));
            }
        }
    }

    format!(
        "TOOL ERROR: {}. Use ONLY tools listed in your available tools.",
        parts.join(" ")
    )
}

/// Expand `~` or `~/` prefixes to the user's home directory in any string
/// argument value that looks like a file path.
///
/// MCP servers expect absolute paths. The LLM frequently generates `~/...`
/// despite system-prompt rules. Rather than relying on each MCP server to
/// handle tildes, we expand them centrally before dispatch.
///
/// Also fixes cross-platform path hallucination:
/// - `/home/<user>/...` on macOS → `/Users/<user>/...`
/// - `/Users/{user}/...` (placeholder) → real home dir
/// - `/Users/<wrong_user>/...` → real home dir
///
/// Only replaces `~` or `~/...` at the start of a string value. Values like
/// `~other_user/` or `~suffix` are left untouched (we can't resolve those).
fn expand_tilde_in_arguments(args: &serde_json::Value) -> serde_json::Value {
    match args {
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k.clone(), expand_tilde_in_arguments(v));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::String(s) => {
            if let Some(fixed) = fix_path_string(s) {
                serde_json::Value::String(fixed)
            } else {
                serde_json::Value::String(s.clone())
            }
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(expand_tilde_in_arguments).collect())
        }
        other => other.clone(),
    }
}

/// Fix a single path string: tilde expansion + cross-platform path correction.
///
/// Returns `Some(fixed)` if the path was modified, `None` if no fix was needed.
///
/// The model hallucinates paths in several forms:
///   - `~/Documents`         → tilde shorthand
///   - `Projects`            → bare relative dir name
///   - `/home/user/...`      → wrong OS prefix (Linux on macOS/Windows)
///   - `/Users/{user}/...`   → template placeholders
///   - `C:\Users\{user}\...` → template placeholders (Windows)
///
/// All corrections use `std::path::Path::join` so separators are always
/// correct for the target platform.
fn fix_path_string(s: &str) -> Option<String> {
    use std::path::MAIN_SEPARATOR;

    let home = dirs::home_dir()?;
    let home_str = home.to_string_lossy();

    // ── 1. Tilde expansion: ~/... → <home>/... ──────────────────────────────
    if s.starts_with("~/") || s.starts_with("~\\") {
        let rest = &s[2..];
        return Some(home.join(rest).to_string_lossy().into_owned());
    }
    if s == "~" {
        return Some(home_str.into_owned());
    }

    // ── 2. Bare relative path that matches a well-known home subdirectory ───
    //    Model outputs "Projects" or "Downloads" instead of an absolute path.
    //    Guard: skip strings that look like absolute paths or URLs.
    let looks_absolute = s.starts_with('/')
        || s.starts_with('\\')
        || (s.len() >= 3 && s.as_bytes()[1] == b':'); // C:\ or D:\
    if !looks_absolute && !s.contains("://") {
        let first_segment = s.split(&['/', '\\'][..]).next().unwrap_or(s);
        let well_known = [
            "Desktop",
            "Documents",
            "Downloads",
            "Projects",
            "Pictures",
            "Music",
            "Videos",   // Windows
            "Movies",   // macOS
            "Library",  // macOS
        ];
        if well_known.iter().any(|d| d.eq_ignore_ascii_case(first_segment)) {
            return Some(home.join(s).to_string_lossy().into_owned());
        }
    }

    // ── 3. Foreign OS home prefix → real home dir ─────────────────────────
    //    LLMs hallucinate Linux-style /home/... on macOS/Windows and
    //    macOS-style /Users/... on Linux/Windows.  A foreign prefix means
    //    the entire path is hallucinated — rewrite any username unconditionally.
    let foreign_prefixes: &[&str] = if cfg!(target_os = "macos") {
        &["/home/"] // /Users/ is native on macOS — handled separately below
    } else if cfg!(target_os = "linux") {
        &["/Users/"] // /home/ is native on Linux — handled separately below
    } else {
        &["/home/", "/Users/"] // both are foreign on Windows
    };

    for prefix in foreign_prefixes {
        if let Some(after_prefix) = s.strip_prefix(prefix) {
            if let Some(slash_idx) = after_prefix.find('/') {
                let rest = &after_prefix[slash_idx + 1..];
                return Some(home.join(rest).to_string_lossy().into_owned());
            }
        }
    }

    // ── 4. Native OS home prefix with template placeholder ──────────────
    //    /Users/{user}/... on macOS, /home/{user}/... on Linux.
    //    Only rewrite if the "username" is a known template placeholder —
    //    never silently replace a real username on a multi-user system.
    let native_prefix: &str = if cfg!(target_os = "macos") {
        "/Users/"
    } else if cfg!(target_os = "linux") {
        "/home/"
    } else {
        "" // Windows native prefix handled in section 5
    };

    if !native_prefix.is_empty() && s.starts_with(native_prefix) {
        // Already matches our home dir — nothing to fix
        if s.starts_with(&*home_str) {
            return None;
        }

        let after_prefix = &s[native_prefix.len()..];
        if let Some(slash_idx) = after_prefix.find('/') {
            let placeholder = &after_prefix[..slash_idx];
            let rest = &after_prefix[slash_idx + 1..];

            let is_template =
                (placeholder.starts_with('{') && placeholder.ends_with('}'))
                    || (placeholder.starts_with('<') && placeholder.ends_with('>'))
                    || (placeholder.starts_with('[') && placeholder.ends_with(']'));

            if is_template {
                return Some(home.join(rest).to_string_lossy().into_owned());
            }

            // Common LLM placeholder words (not real usernames)
            let placeholder_lower = placeholder.to_ascii_lowercase();
            let known_placeholders = ["user", "username", "your_name", "me"];
            if known_placeholders.contains(&placeholder_lower.as_str()) {
                return Some(home.join(rest).to_string_lossy().into_owned());
            }
        }
    }

    // ── 5. Windows C:\Users\{placeholder}\... ───────────────────────────────
    let win_prefix = "C:\\Users\\";
    let win_prefix_fwd = "C:/Users/"; // model may use forward slashes on Windows too
    for prefix in &[win_prefix, win_prefix_fwd] {
        if let Some(after_prefix) = s.strip_prefix(prefix) {
            // Already matches our home dir — nothing to fix
            if s.starts_with(&*home_str) {
                return None;
            }

            let sep_idx = after_prefix.find(&['/', '\\'][..]);

            if let Some(idx) = sep_idx {
                let placeholder = &after_prefix[..idx];
                let rest = &after_prefix[idx + 1..];

                let is_template =
                    (placeholder.starts_with('{') && placeholder.ends_with('}'))
                        || (placeholder.starts_with('<') && placeholder.ends_with('>'))
                        || (placeholder.starts_with('[') && placeholder.ends_with(']'));

                if is_template {
                    return Some(home.join(rest).to_string_lossy().into_owned());
                }

                let placeholder_lower = placeholder.to_ascii_lowercase();
                let known_placeholders = ["user", "username", "your_name", "me"];
                if known_placeholders.contains(&placeholder_lower.as_str()) {
                    return Some(home.join(rest).to_string_lossy().into_owned());
                }
            }
        }
    }

    // Suppress unused-variable warning on platforms where MAIN_SEPARATOR is `/`
    let _ = MAIN_SEPARATOR;

    None
}

/// Extract readable text from an MCP tool result.
///
/// MCP results follow the format: `{ "content": [{ "type": "text", "text": "..." }] }`
/// The `text` field may itself be a JSON-serialized result object (e.g. from Python
/// pydantic `.model_dump()` + `json.dumps()`), so we attempt to extract a human-readable
/// summary from known fields like "text", "content", "message", or "result".
fn extract_mcp_result_text(result: &Option<serde_json::Value>) -> String {
    let Some(value) = result else {
        return "No result returned.".to_string();
    };

    // Try standard MCP content format
    if let Some(content_arr) = value.get("content").and_then(|c| c.as_array()) {
        let texts: Vec<&str> = content_arr
            .iter()
            .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
            .collect();
        if !texts.is_empty() {
            let raw = texts.join("\n");
            // The text might be a JSON-serialized tool result (e.g. from json.dumps).
            // Try to parse it and extract human-readable content.
            return unwrap_tool_result_json(&raw);
        }
    }

    // Fallback: stringify the entire result
    match serde_json::to_string_pretty(value) {
        Ok(s) => s,
        Err(_) => format!("{value:?}"),
    }
}

/// If `raw` is a JSON object with known text fields, extract and format them
/// for human readability. Otherwise return the original string unchanged.
///
/// This handles the case where Python MCP servers serialize their result model
/// via `json.dumps(result.model_dump())`, producing strings like:
/// `{"text": "extracted text...", "confidence": 0.9, "engine": "lfm_vision"}`
fn unwrap_tool_result_json(raw: &str) -> String {
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(raw) else {
        return raw.to_string(); // Not JSON, return as-is
    };

    let obj = match parsed.as_object() {
        Some(o) => o,
        None => return raw.to_string(), // JSON but not an object
    };

    // Look for a primary text field in priority order
    for key in &["text", "content", "message", "result", "output"] {
        if let Some(val) = obj.get(*key).and_then(|v| v.as_str()) {
            if !val.is_empty() {
                // Build a summary with the primary text and any useful metadata
                let mut parts = vec![val.to_string()];
                for meta_key in &["engine", "confidence", "language", "page_count"] {
                    if let Some(meta_val) = obj.get(*meta_key) {
                        let display = match meta_val {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            other => other.to_string(),
                        };
                        parts.push(format!("[{meta_key}: {display}]"));
                    }
                }
                return parts.join("\n");
            }
        }
    }

    // JSON object but no recognized text field — return the formatted JSON
    raw.to_string()
}

/// Format bytes into human-readable size.
fn format_file_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("({bytes} B)")
    } else if bytes < 1024 * 1024 {
        format!("({:.1} KB)", bytes as f64 / 1024.0)
    } else {
        format!("({:.1} MB)", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// Emit context budget to the frontend.
fn emit_context_budget(
    app_handle: &tauri::AppHandle,
    mgr: &ConversationManager,
    session_id: &str,
) {
    use tauri::Emitter;
    if let Ok(budget) = mgr.get_budget(session_id) {
        let _ = app_handle.emit(
            "context-budget",
            serde_json::json!({
                "total": budget.total,
                "systemPrompt": budget.system_prompt,
                "toolDefinitions": budget.tool_definitions,
                "conversationHistory": budget.conversation_history,
                "outputReservation": budget.output_reservation,
                "remaining": budget.remaining,
            }),
        );
    }
}

// ─── Commands ───────────────────────────────────────────────────────────────

/// Start or resume a chat session.
///
/// On first launch, creates a new session. On subsequent app opens,
/// returns the most recent session that has user messages.
/// If explicitly called with `force_new = true`, always creates a new session.
#[tauri::command]
pub async fn start_session(
    force_new: Option<bool>,
    state: tauri::State<'_, Mutex<ConversationManager>>,
    mcp_state: tauri::State<'_, TokioMutex<McpClient>>,
) -> Result<SessionInfo, String> {
    // Phase 1: Check for resumable sessions (lock ConversationManager, then drop).
    // std::sync::MutexGuard is !Send, so it MUST be dropped before any .await.
    {
        let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;

        if force_new != Some(true) {
            if let Ok(sessions) = mgr.db().list_sessions() {
                for session in &sessions {
                    if let Ok(count) = mgr.db().message_count(&session.id) {
                        if count > 1 {
                            tracing::info!(
                                session_id = %session.id,
                                message_count = count,
                                "resuming existing session"
                            );
                            return Ok(SessionInfo {
                                session_id: session.id.clone(),
                                resumed: true,
                            });
                        }
                    }
                }
            }
        }
    } // mgr lock dropped here — safe to .await below

    // Phase 2: Build dynamic system prompt from MCP registry (async lock).
    //          Check if two-pass mode should be noted in the system prompt.
    let system_prompt = {
        let mcp = mcp_state.lock().await;
        let cwd = std::env::current_dir().unwrap_or_default();
        let two_pass_active = if let Ok(cfg_path) = find_config_path(&cwd) {
            load_models_config(&cfg_path)
                .ok()
                .and_then(|cfg| cfg.two_pass_tool_selection)
                .unwrap_or(false)
                && mcp.registry.len() > TWO_PASS_MIN_TOOLS
        } else {
            false
        };
        build_system_prompt(&mcp.registry, two_pass_active)
    };

    // Phase 3: Create the new session (re-acquire ConversationManager).
    let session_id = Uuid::new_v4().to_string();

    {
        let mut mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;

        mgr.new_session(&session_id, &system_prompt)
            .map_err(|e| format!("Failed to create session: {e}"))?;

        // Set accurate system prompt budget from the actual dynamic prompt
        let actual_prompt_tokens =
            crate::agent_core::tokens::estimate_system_prompt_tokens(&system_prompt);
        mgr.set_system_prompt_budget(actual_prompt_tokens);

        tracing::info!(
            session_id = %session_id,
            prompt_tokens = actual_prompt_tokens,
            "new chat session created with dynamic system prompt"
        );
    }

    Ok(SessionInfo {
        session_id,
        resumed: false,
    })
}

/// Send a user message and get an assistant response.
///
/// Implements the agent loop:
/// 1. Persist user message, build history
/// 2. Call LLM with tool definitions (built-in + MCP)
/// 3. If model returns tool calls → execute them → feed results back → repeat
/// 4. When model returns text → stream it to frontend
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn send_message(
    session_id: String,
    content: String,
    working_directory: Option<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, Mutex<ConversationManager>>,
    mcp_state: tauri::State<'_, TokioMutex<McpClient>>,
    permission_state: tauri::State<'_, TokioMutex<PermissionStore>>,
    pending_confirm: tauri::State<'_, PendingConfirmation>,
    sampling_state: tauri::State<'_, TokioMutex<crate::commands::settings::SamplingConfig>>,
    in_flight: tauri::State<'_, crate::InFlightRequests>,
) -> Result<(), String> {
    use tauri::Emitter;

    // Request deduplication: check if there's already a request in flight for this session
    {
        let mut in_flight_guard = in_flight.lock().await;
        if in_flight_guard.get(&session_id) == Some(&true) {
            tracing::warn!(session_id = %session_id, "duplicate request ignored");
            return Ok(()); // Silently ignore duplicate request
        }
        in_flight_guard.insert(session_id.clone(), true);
    }

    // Generate trace ID for this request (for correlation across logs)
    let trace_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    
    tracing::info!(trace_id = %trace_id, session_id = %session_id, content_len = content.len(), "starting message processing");

    // Read sampling config once at the start of this request.
    let sampling_cfg = sampling_state.lock().await.clone();
    let tool_turn_sampling = SamplingOverrides {
        temperature: Some(sampling_cfg.tool_temperature),
        top_p: Some(sampling_cfg.tool_top_p),
    };
    let conversational_sampling = SamplingOverrides {
        temperature: Some(sampling_cfg.conversational_temperature),
        top_p: Some(sampling_cfg.conversational_top_p),
    };

    // 1. Persist user message and build conversation history
    let mut messages = {
        let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;

        mgr.add_user_message(&session_id, &content)
            .map_err(|e| format!("Failed to save user message: {e}"))?;

        let evicted = mgr
            .evict_if_needed(&session_id)
            .map_err(|e| format!("Eviction error: {e}"))?;
        if evicted > 0 {
            tracing::info!(evicted_tokens = evicted, "evicted old messages");
        }

        mgr.build_chat_messages(&session_id)
            .map_err(|e| format!("Failed to build messages: {e}"))?
    };

    // 1b. Inject date context directly into the user message when temporal words
    //     are detected. Small LLMs (24B) have strong training priors for 2023/2024
    //     dates and will ignore system prompt dates. Putting the date IN the user
    //     message forces the model to see it as part of the query itself.
    {
        use chrono::{Datelike, Duration};

        let content_lower = content.to_lowercase();
        let has_temporal = content_lower.contains("today")
            || content_lower.contains("tomorrow")
            || content_lower.contains("this week")
            || content_lower.contains("next week")
            || content_lower.contains("yesterday")
            || content_lower.contains("calendar")
            || content_lower.contains("schedule")
            || content_lower.contains("meeting");

        if has_temporal {
            let now = chrono::Local::now();
            let today_str = now.format("%Y-%m-%d").to_string();
            let tomorrow_str = (now + Duration::days(1)).format("%Y-%m-%d").to_string();
            let weekday_num = now.weekday().num_days_from_monday();
            let week_start = (now - Duration::days(weekday_num as i64))
                .format("%Y-%m-%d")
                .to_string();
            let week_end = (now + Duration::days((6 - weekday_num) as i64))
                .format("%Y-%m-%d")
                .to_string();

            let date_prefix = format!(
                "[Today is {today_str}. Tomorrow is {tomorrow_str}. \
                 This week is {week_start} to {week_end}.]\n"
            );

            // Find the last user message and prepend the date context
            if let Some(last_user_msg) = messages
                .iter_mut()
                .rev()
                .find(|m| m.role == crate::inference::types::Role::User)
            {
                if let Some(ref mut msg_content) = last_user_msg.content {
                    let original = msg_content.clone();
                    msg_content.clear();
                    msg_content.push_str(&date_prefix);
                    msg_content.push_str(&original);
                    tracing::info!(
                        date_injected = %today_str,
                        "injected date context into user message"
                    );
                }
            }
        }
    }

    // 1b2. Inject working folder PATH (not file listing) into the user message.
    //      Same strategy as date injection: small LLMs ignore system prompt paths
    //      and hallucinate /path/to/... from training data. Putting the folder
    //      path IN the user message makes it impossible to ignore.
    //
    //      IMPORTANT: Only the path goes here, NOT the file listing. If we put
    //      files in the user message, the model skips tool calls (it already has
    //      the answer) and the user never sees the tool trace UI. The full file
    //      listing stays in the system prompt to guide tool argument selection.
    if let Some(ref dir) = working_directory {
        if let Some(last_user_msg) = messages
            .iter_mut()
            .rev()
            .find(|m| m.role == crate::inference::types::Role::User)
        {
            if let Some(ref mut msg_content) = last_user_msg.content {
                let folder_prefix = format!(
                    "[Working folder: {dir}. Use tools on files in this folder.]\n"
                );

                // Prepend — but AFTER any date prefix that may already be there
                let original = msg_content.clone();
                msg_content.clear();
                if original.starts_with("[Today is") {
                    // Date prefix exists — insert folder after it
                    if let Some(newline_pos) = original.find("]\n") {
                        let after_date = newline_pos + 2; // skip "]\n"
                        msg_content.push_str(&original[..after_date]);
                        msg_content.push_str(&folder_prefix);
                        msg_content.push_str(&original[after_date..]);
                    } else {
                        msg_content.push_str(&folder_prefix);
                        msg_content.push_str(&original);
                    }
                } else {
                    msg_content.push_str(&folder_prefix);
                    msg_content.push_str(&original);
                }

                tracing::info!(
                    working_directory = %dir,
                    "injected working folder into user message"
                );
            }
        }
    }

    // 1c. Inject working directory context + file listing into the system message.
    //     This is a per-request overlay — not persisted in the DB — so it
    //     automatically reflects the user's current folder selection.
    //     Including the actual file listing is a product-level optimization:
    //     same pattern as Cowork's project indexing — the model sees concrete
    //     file names without needing to call list_dir first.
    const MAX_FOLDER_ENTRIES: usize = 50;

    if let Some(ref dir) = working_directory {
        let mut file_count: usize = 0;
        if let Some(system_msg) = messages.first_mut() {
            if system_msg.role == crate::inference::types::Role::System {
                if let Some(ref mut content) = system_msg.content {
                    // Build the working folder context block with XML tags
                    let mut folder_ctx = format!(
                        "<working_folder path=\"{dir}\">\n\
                         Use ONLY the file paths listed below. \
                         Do NOT invent or guess paths."
                    );

                    // List directory contents (skip hidden files, cap at 50)
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        let mut files: Vec<String> = entries
                            .filter_map(|e| e.ok())
                            .filter(|e| {
                                !e.file_name()
                                    .to_string_lossy()
                                    .starts_with('.')
                            })
                            .map(|e| {
                                let full_path =
                                    e.path().to_string_lossy().into_owned();
                                if e.path().is_dir() {
                                    format!("  {full_path}/")
                                } else {
                                    format!("  {full_path}")
                                }
                            })
                            .collect();
                        files.sort();

                        let total = files.len();
                        file_count = total;
                        if total > MAX_FOLDER_ENTRIES {
                            files.truncate(MAX_FOLDER_ENTRIES);
                            files.push(format!(
                                "  (and {} more files...)",
                                total - MAX_FOLDER_ENTRIES
                            ));
                        }
                        if !files.is_empty() {
                            folder_ctx.push_str("\nFiles:\n");
                            folder_ctx.push_str(&files.join("\n"));
                        }
                    }

                    folder_ctx.push_str("\n</working_folder>\n");

                    // RECENCY REMINDER: shorter repetition block for the end.
                    // Research shows repeating key instructions at the end improves
                    // accuracy for smaller models (primacy + recency positions).
                    let mut folder_reminder = format!(
                        "\n\n<reminder>\n\
                         working_folder = {dir}\n\
                         Files:"
                    );
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        let mut files: Vec<String> = entries
                            .filter_map(|e| e.ok())
                            .filter(|e| {
                                !e.file_name()
                                    .to_string_lossy()
                                    .starts_with('.')
                            })
                            .map(|e| {
                                let full_path =
                                    e.path().to_string_lossy().into_owned();
                                format!("  {full_path}")
                            })
                            .collect();
                        files.sort();
                        if files.len() > MAX_FOLDER_ENTRIES {
                            files.truncate(MAX_FOLDER_ENTRIES);
                        }
                        if !files.is_empty() {
                            folder_reminder.push('\n');
                            folder_reminder.push_str(&files.join("\n"));
                        }
                    }
                    folder_reminder.push_str(
                        "\nUse ONLY these paths. Do NOT invent paths.\n\
                         </reminder>"
                    );

                    // SANDWICH PATTERN: insert working folder at TOP and BOTTOM
                    // of system prompt. The model sees the file paths at the
                    // strongest positions (primacy + recency).
                    let original = content.clone();
                    content.clear();

                    // TOP: Insert after the first paragraph (identity intro)
                    if let Some(pos) = original.find("\n\n") {
                        content.push_str(&original[..pos]);
                        content.push_str("\n\n");
                        content.push_str(&folder_ctx);
                        content.push_str(&original[pos..]);
                    } else {
                        content.push_str(&folder_ctx);
                        content.push_str("\n\n");
                        content.push_str(&original);
                    }

                    // BOTTOM: Append reminder at the very end
                    content.push_str(&folder_reminder);
                }
            }
        }
        tracing::info!(
            working_directory = %dir,
            file_count,
            "injected working folder into system prompt"
        );
    }

    // 2. Create inference client and build merged tool list
    let cwd = std::env::current_dir().unwrap_or_default();
    let config_path =
        find_config_path(&cwd).map_err(|e| format!("Config error: {e}"))?;
    let config =
        load_models_config(&config_path).map_err(|e| format!("Config error: {e}"))?;
    let mut client = InferenceClient::from_config(config.clone())
        .map_err(|e| format!("Inference client error: {e}"))?;

    // 2a. Build tool definitions — either flat (all tools) or category meta-tools.
    //     Two-pass mode sends ~15 categories on the first turn (~1,500 tokens)
    //     instead of all ~67 tools (~8,670 tokens). Selected categories are
    //     expanded to real tools on subsequent turns.
    let (mut tool_phase, mut tools) = {
        let mcp = mcp_state.lock().await;
        let use_two_pass = config.two_pass_tool_selection.unwrap_or(false)
            && mcp.registry.len() > TWO_PASS_MIN_TOOLS;

        if use_two_pass {
            let cat_registry = CategoryRegistry::build(&mcp.registry);
            let cat_tools = build_category_tool_definitions(&cat_registry);
            tracing::info!(
                category_count = cat_registry.len(),
                tool_count_saved = mcp.registry.len(),
                "two-pass mode: sending category meta-tools instead of all tools"
            );
            (
                ToolSelectionPhase::Categories { cat_registry },
                cat_tools,
            )
        } else {
            let all_tools = build_all_tool_definitions(&mcp);
            (ToolSelectionPhase::Flat, all_tools)
        }
    };

    // Measure actual tool definition tokens and update the budget.
    // The default TOOL_DEFINITIONS_BUDGET (2000) was calibrated for stub schemas.
    // With real JSON Schema from zod-to-json-schema, 15 tools consume 5000-8000+
    // tokens. Using the measured value ensures accurate eviction timing.
    {
        let tools_json: Vec<serde_json::Value> = tools
            .iter()
            .filter_map(|t| serde_json::to_value(t).ok())
            .collect();
        let actual_tool_tokens =
            crate::agent_core::tokens::estimate_tool_definitions_tokens(&tools_json);

        tracing::info!(
            tool_count = tools.len(),
            tool_tokens = actual_tool_tokens,
            two_pass = matches!(tool_phase, ToolSelectionPhase::Categories { .. }),
            "measured actual tool definition tokens"
        );

        let mut mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;
        mgr.set_tool_definitions_budget(actual_tool_tokens);
    }

    // Response text — set by either the orchestrator or the agent loop.
    let mut full_response = String::new();
    // Set to true when the orchestrator already persisted the response to DB.
    let mut already_persisted = false;

    // 2b. Dual-model orchestrator (ADR-009) — if enabled, try the planner+router
    //     pipeline before falling into the single-model agent loop.
    if let Some(ref orch_config) = config.orchestrator {
        if orch_config.enabled {
            tracing::info!("orchestrator enabled — attempting dual-model pipeline");
            match crate::agent_core::orchestrator::orchestrate_dual_model(
                &session_id,
                &content,
                &messages,
                &config,
                orch_config,
                &app_handle,
                &state,
                &mcp_state,
            )
            .await
            {
                Ok(result) if !result.fell_back => {
                    // Fix F3: Check if orchestrator "succeeded" but no tools were
                    // actually called. This happens when the router fails to produce
                    // bracket-format tool calls for every step.
                    let any_tool_called = result
                        .step_results
                        .iter()
                        .any(|r| r.tool_called.is_some());

                    if !result.all_steps_succeeded && !any_tool_called {
                        tracing::warn!(
                            session_id = %session_id,
                            failed_steps = result.step_results.len(),
                            "orchestrator: no tools called — falling back to single-model"
                        );
                        // Fall through to single-model agent loop
                    } else {
                        tracing::info!(
                            steps = result.step_results.len(),
                            all_succeeded = result.all_steps_succeeded,
                            tools_called = any_tool_called,
                            "orchestrator completed — skipping single-model loop"
                        );
                        // Set the response so the normal completion path (step 5)
                        // emits the properly-formatted stream-complete event.
                        // The orchestrator already persisted the message to the DB.
                        full_response = result.synthesis;
                        already_persisted = true;
                    }
                }
                Ok(_) => {
                    tracing::warn!(
                        "orchestrator fell back — continuing to single-model agent loop"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "orchestrator error — continuing to single-model agent loop"
                    );
                }
            }
        }
    }

    // 3. Agent loop: call model → execute tools → repeat
    // Variables used by both the agent loop and the force-summary path.
    let mut empty_response_count: usize = 0;
    let mut tool_call_history: Vec<String> = Vec::new();

    // ── Turn-level tool call accumulator ──────────────────────────────
    // The bracket format emits one tool call per inference round, so a
    // multi-tool response spans multiple rounds:
    //   assistant(toolCalls:[A]) → tool(resultA) → assistant(toolCalls:[B]) → ...
    //
    // To present this as a single "2 tools executed" block in the UI, we
    // accumulate all tool calls under a stable message ID and re-emit the
    // growing list on each round. The frontend upserts by ID.
    let turn_message_id = chrono::Utc::now().timestamp_millis();
    let mut turn_tool_calls: Vec<serde_json::Value> = Vec::new();

    // Skip entirely if the orchestrator already produced a response.
    if full_response.is_empty() {

    // Track (tool_name, arguments) pairs to detect duplicate calls
    let mut tool_call_signatures: Vec<(String, String)> = Vec::new();
    let mut consecutive_error_rounds: usize = 0;
    let mut tool_failure_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for round in 0..MAX_TOOL_ROUNDS {
        // ── Token budget gate ──────────────────────────────────────────
        // Before each LLM call, check that we have enough remaining
        // tokens for a productive round. If not, break early to avoid
        // context overflow and degraded model quality.
        {
            let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;
            let budget = mgr
                .get_budget(&session_id)
                .map_err(|e| format!("Budget error: {e}"))?;
            if budget.remaining < MIN_ROUND_TOKEN_BUDGET {
                tracing::warn!(
                    round = round,
                    remaining = budget.remaining,
                    threshold = MIN_ROUND_TOKEN_BUDGET,
                    "token budget exhausted — ending agent loop"
                );
                break;
            }
        }

        tracing::info!(
            session_id = %session_id,
            round = round,
            message_count = messages.len(),
            total_content_bytes = messages.iter()
                .map(|m| m.content.as_deref().unwrap_or("").len())
                .sum::<usize>(),
            "=== AGENT LOOP ROUND START ==="
        );

        let mut round_text = String::new();
        let mut tool_calls_detected: Vec<crate::inference::types::ToolCall> = Vec::new();

        // Measure model inference time (from request to full response parsed).
        let inference_start = std::time::Instant::now();

        match client
            .chat_completion_stream(messages.clone(), Some(tools.clone()), Some(tool_turn_sampling))
            .await
        {
            Ok(stream) => {
                futures::pin_mut!(stream);

                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(chunk) => {
                            if let Some(token) = &chunk.token {
                                round_text.push_str(token);
                                if tool_calls_detected.is_empty() {
                                    let _ = app_handle.emit(
                                        "stream-token",
                                        token.clone(),
                                    );
                                }
                            }
                            if let Some(ref calls) = chunk.tool_calls {
                                for tc in calls {
                                    if !tool_calls_detected
                                        .iter()
                                        .any(|existing| existing.id == tc.id)
                                    {
                                        tool_calls_detected.push(tc.clone());
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                round = round,
                                error = %e,
                                "stream error in agent loop"
                            );
                            // Don't abort the whole loop — treat as empty
                            // response and let retry logic handle it
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                let fallback =
                    crate::inference::client::static_fallback_response();
                if let Some(token) = &fallback.token {
                    full_response = token.clone();
                    let _ = app_handle.emit("stream-token", token.clone());
                }
                tracing::warn!(error = %e, "all models unavailable, using static fallback");
                break;
            }
        }

        let inference_time_ms = inference_start.elapsed().as_millis() as u64;

        tracing::info!(
            session_id = %session_id,
            round = round,
            round_text_len = round_text.len(),
            tool_calls_count = tool_calls_detected.len(),
            tool_names = ?tool_calls_detected.iter().map(|tc| tc.name.as_str()).collect::<Vec<_>>(),
            inference_time_ms = inference_time_ms,
            "=== MODEL RESPONSE ==="
        );

        // ── Handle empty response (0 text AND 0 tool calls) ────────
        // This is abnormal — typically caused by timeout, context overflow,
        // or model confusion. Retry a limited number of times, then force
        // a summary.
        if tool_calls_detected.is_empty() && round_text.trim().is_empty() {
            empty_response_count += 1;
            tracing::warn!(
                round = round,
                empty_count = empty_response_count,
                max_retries = MAX_EMPTY_RETRIES,
                "model returned empty response (0 text, 0 tools)"
            );

            if empty_response_count >= MAX_EMPTY_RETRIES {
                tracing::warn!("max empty retries reached — forcing summary");
                break;
            }

            // Inject a nudge prompt instead of retrying with identical messages.
            // Retrying unchanged context causes the same stall. A new user message
            // gives the model fresh input to work from.
            let nudge = if tool_call_history.is_empty() {
                "You returned an empty response. Please answer the user's question \
                 or call the appropriate tool now."
                    .to_string()
            } else {
                format!(
                    "You returned an empty response after processing {} tool call(s). \
                     If there are more files to process, call the next tool now. \
                     If the task is complete, provide a final summary of what was done.",
                    tool_call_history.len()
                )
            };

            messages.push(crate::inference::types::ChatMessage {
                role: crate::inference::types::Role::User,
                content: Some(nudge),
                tool_call_id: None,
                tool_calls: None,
            });

            tracing::info!(
                round = round,
                tools_completed = tool_call_history.len(),
                "injected nudge prompt after empty response"
            );
            continue;
        }

        // Reset empty counter on any successful response
        empty_response_count = 0;

        // ── Text response (0 tool calls) — accept and exit ─────────
        // When the model returns text without tool calls, it has decided
        // the task is complete. Trust the model's judgment and exit.
        //
        // This is the same pattern as Claude Code: model produces text →
        // loop ends. If the user wants more, they say "continue."
        //
        // Previously, heuristic detectors (is_incomplete_response,
        // has_unverified_completion, is_deflection_response) would
        // second-guess the model and inject continuation prompts. These
        // caused more harm than good — a valid 324-char system info
        // summary would trigger "FM-3 deflection" because it contained
        // "let me know", causing the model to spiral into unnecessary
        // tool calls and produce a worse answer.
        //
        // Multi-step tasks that need continuation belong in the
        // Orchestrator (ADR-009), not in heuristic string-matching.
        if tool_calls_detected.is_empty() {
            full_response.push_str(&round_text);
            break;
        }

        // ── Two-pass category expansion ─────────────────────────────
        // If we're in Categories phase and the model called category meta-tools,
        // expand them to real tools for subsequent rounds. Category "tool calls"
        // are NOT executed — they just tell us which capability areas are needed.
        if let ToolSelectionPhase::Categories { ref cat_registry } = tool_phase {
            let mut selected_categories: Vec<String> = Vec::new();
            let mut direct_tool_calls: Vec<crate::inference::types::ToolCall> = Vec::new();

            for tc in &tool_calls_detected {
                if cat_registry.is_category(&tc.name) {
                    selected_categories.push(tc.name.clone());
                } else {
                    // Model called a real tool directly — handle gracefully
                    direct_tool_calls.push(tc.clone());
                }
            }

            if !selected_categories.is_empty() {
                // Expand categories to real tool names
                let expanded_names = cat_registry.expand_categories(&selected_categories);

                // Build expanded tool definitions from the live registry
                let expanded_defs = {
                    let mcp = mcp_state.lock().await;
                    let mut defs = builtin_tool_definitions();
                    let mcp_tools = mcp.registry.to_openai_tools_filtered(&expanded_names);
                    for tool_json in mcp_tools {
                        if let Ok(td) =
                            serde_json::from_value::<ToolDefinition>(tool_json)
                        {
                            defs.push(td);
                        }
                    }
                    defs
                };

                tracing::info!(
                    session_id = %session_id,
                    round = round,
                    categories = ?selected_categories,
                    expanded_tool_count = expanded_defs.len(),
                    "two-pass: expanded categories to real tools"
                );

                // Update token budget for the expanded (smaller) tool set
                {
                    let tools_json: Vec<serde_json::Value> = expanded_defs
                        .iter()
                        .filter_map(|t| serde_json::to_value(t).ok())
                        .collect();
                    let expanded_tokens =
                        crate::agent_core::tokens::estimate_tool_definitions_tokens(
                            &tools_json,
                        );
                    let mut mgr =
                        state.lock().map_err(|e| format!("Lock error: {e}"))?;
                    mgr.set_tool_definitions_budget(expanded_tokens);
                    tracing::info!(
                        expanded_tool_tokens = expanded_tokens,
                        "updated token budget for expanded tools"
                    );
                }

                // Transition phase and update tools
                tool_phase = ToolSelectionPhase::Expanded {
                    _selected_categories: selected_categories.clone(),
                };
                tools = expanded_defs;

                // Inject an assistant message noting the category selection
                // (in-memory only — not persisted, same pattern as continuation prompts)
                let cat_text = format!(
                    "Selected capability areas: {}. Now proceeding with specific tools.",
                    selected_categories.join(", ")
                );
                messages.push(crate::inference::types::ChatMessage {
                    role: crate::inference::types::Role::Assistant,
                    content: Some(cat_text),
                    tool_call_id: None,
                    tool_calls: None,
                });

                // If the model also called real tools directly, process them
                if !direct_tool_calls.is_empty() {
                    tracing::info!(
                        direct_tool_count = direct_tool_calls.len(),
                        "two-pass: model also called real tools directly — \
                         processing as fallback"
                    );
                    tool_calls_detected = direct_tool_calls;
                    // Fall through to normal tool execution below
                } else {
                    // Re-prompt with the expanded real tools — no tool execution
                    // this round. The model will now see the specific tools.
                    continue;
                }
            }
            // If no categories were selected (model called only real tools),
            // fall through to normal execution — graceful degradation.
        }

        // ── Tool execution round ──────────────────────────────────────

        if !round_text.is_empty() {
            let _ = app_handle.emit("stream-clear", ());
        }

        tracing::info!(
            round = round,
            tool_count = tool_calls_detected.len(),
            "executing tool calls"
        );

        // Persist the assistant's tool-call message
        {
            let mgr =
                state.lock().map_err(|e| format!("Lock error: {e}"))?;
            mgr.add_tool_call_message(&session_id, &tool_calls_detected)
                .map_err(|e| format!("Failed to save tool call: {e}"))?;
        }

        // ── Accumulate tool calls for the turn ─────────────────────────
        // Push this round's calls into the turn-level accumulator, then
        // emit ALL accumulated calls under the same stable message ID.
        // The frontend upserts by ID, so the ToolTrace grows in-place
        // rather than spawning a new block each round.
        for tc in &tool_calls_detected {
            turn_tool_calls.push(serde_json::json!({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }));
        }

        let _ = app_handle.emit(
            "tool-call",
            serde_json::json!({
                "id": turn_message_id,
                "sessionId": session_id,
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "role": "assistant",
                "toolCalls": turn_tool_calls,
                "tokenCount": 10,
            }),
        );

        // Execute each tool and collect typed outcomes.
        let mut round_error_count: usize = 0;
        let round_call_count = tool_calls_detected.len();
        let mut round_unknown_tools: Vec<(String, ToolResolution)> = Vec::new();

        for tc in &tool_calls_detected {
            // Auto-inject session_id into audit tool arguments so the model
            // doesn't need to guess it. Audit tools expect a session_id param
            // that matches the agent_core audit log's session column.
            // Always override — the model often hallucinates placeholder values
            // like "SESSION_ID_FROM_CURRENT_CONTEXT" or tool_call_ids.
            let mut effective_arguments = if tc.name.starts_with("audit.") {
                let mut args = tc.arguments.clone();
                if let Some(obj) = args.as_object_mut() {
                    obj.insert(
                        "session_id".to_string(),
                        serde_json::Value::String(session_id.clone()),
                    );
                }
                args
            } else {
                tc.arguments.clone()
            };

            // ── HITL confirmation check ──────────────────────────────
            // Built-in tools (list_directory, read_file) are always read-only.
            // MCP tools check the registry's confirmation_required metadata.
            // If the user has previously granted permission, skip the dialog.
            let is_builtin = tc.name == "list_directory" || tc.name == "read_file";
            let needs_confirmation = !is_builtin && {
                let mcp = mcp_state.lock().await;
                mcp.registry.requires_confirmation(&tc.name)
            };

            let mut user_confirmed = !needs_confirmation;

            if needs_confirmation {
                // Check if permission was previously granted
                let already_allowed = {
                    let perms = permission_state.lock().await;
                    perms.check(&tc.name) == PermissionStatus::Allowed
                };

                if already_allowed {
                    user_confirmed = true;
                    tracing::debug!(
                        tool = %tc.name,
                        "skipping confirmation — permission granted"
                    );
                } else {
                    // Build and emit a confirmation request
                    let supports_undo = {
                        let mcp = mcp_state.lock().await;
                        mcp.registry.supports_undo(&tc.name)
                    };
                    let preview = generate_preview(&tc.name, &effective_arguments);
                    let is_destructive = is_destructive_action(&tc.name);

                    let request = ConfirmationRequest {
                        request_id: Uuid::new_v4().to_string(),
                        tool_name: tc.name.clone(),
                        arguments: effective_arguments.clone(),
                        preview,
                        confirmation_required: true,
                        undo_supported: supports_undo,
                        is_destructive,
                    };

                    tracing::info!(
                        tool = %tc.name,
                        request_id = %request.request_id,
                        is_destructive,
                        "awaiting user confirmation"
                    );

                    // Create a oneshot channel for this confirmation
                    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                    {
                        let mut pending = pending_confirm.lock().await;
                        *pending = Some(resp_tx);
                    }

                    // Emit confirmation-request event to frontend
                    let _ = app_handle.emit("confirmation-request", &request);

                    // Wait for user response (blocks the agent loop)
                    match resp_rx.await {
                        Ok(ConfirmationResponse::Rejected) => {
                            tracing::info!(
                                tool = %tc.name,
                                "tool call rejected by user"
                            );
                            // Write rejection to audit log
                            {
                                let mgr = state
                                    .lock()
                                    .map_err(|e| format!("Lock error: {e}"))?;
                                let _ = mgr.db().insert_audit_entry(
                                    &session_id,
                                    &tc.name,
                                    &effective_arguments,
                                    None,
                                    AuditStatus::RejectedByUser,
                                    false,
                                    0,
                                );
                            }

                            let rejection_text =
                                format!("Tool '{}' was rejected by the user.", tc.name);

                            // Emit rejection result to frontend
                            let _ = app_handle.emit(
                                "tool-result",
                                serde_json::json!({
                                    "id": chrono::Utc::now().timestamp_millis(),
                                    "sessionId": session_id,
                                    "timestamp": chrono::Utc::now().to_rfc3339(),
                                    "role": "tool",
                                    "content": rejection_text,
                                    "toolCallId": tc.id,
                                    "toolResult": {
                                        "success": false,
                                        "result": rejection_text,
                                        "toolCallId": tc.id,
                                        "toolName": tc.name,
                                    },
                                    "tokenCount": rejection_text.len() / 4,
                                }),
                            );

                            // Persist rejection so the model knows
                            {
                                let mgr = state
                                    .lock()
                                    .map_err(|e| format!("Lock error: {e}"))?;
                                let result_json =
                                    serde_json::Value::String(rejection_text);
                                mgr.add_tool_result_message(
                                    &session_id,
                                    &tc.id,
                                    &result_json,
                                )
                                .map_err(|e| {
                                    format!("Failed to save tool result: {e}")
                                })?;
                            }

                            // Add to conversation history for the LLM
                            messages.push(crate::inference::types::ChatMessage {
                                role: crate::inference::types::Role::Tool,
                                content: Some(format!(
                                    "Tool '{}' was rejected by the user.",
                                    tc.name
                                )),
                                tool_call_id: Some(tc.id.clone()),
                                tool_calls: None,
                            });

                            round_error_count += 1;
                            tool_call_history.push(tc.name.clone());
                            tool_call_signatures.push((
                                tc.name.clone(),
                                tc.arguments.to_string(),
                            ));
                            continue;
                        }
                        Ok(ConfirmationResponse::ConfirmedForSession) => {
                            let mut perms = permission_state.lock().await;
                            perms.grant(&tc.name, PermissionScope::Session);
                            user_confirmed = true;
                        }
                        Ok(ConfirmationResponse::ConfirmedAlways) => {
                            let mut perms = permission_state.lock().await;
                            perms.grant(&tc.name, PermissionScope::Always);
                            user_confirmed = true;
                        }
                        Ok(ConfirmationResponse::Confirmed) => {
                            user_confirmed = true;
                        }
                        Ok(ConfirmationResponse::EditedAndConfirmed {
                            new_arguments,
                        }) => {
                            effective_arguments = new_arguments;
                            user_confirmed = true;
                        }
                        Err(_) => {
                            tracing::warn!(
                                tool = %tc.name,
                                "confirmation channel closed — skipping tool"
                            );
                            continue;
                        }
                    }
                }
            }

            // ── Duplicate call interception ──────────────────────────
            // If the model is requesting the exact same tool+args as a
            // previous call in this conversation, skip execution entirely.
            // Instead, feed back a short "you already have this" nudge so
            // the model transitions to summarising the results it already
            // has. This is cheaper and more robust than executing the
            // duplicate and relying on post-hoc detection to break the loop.
            let call_sig = (tc.name.clone(), tc.arguments.to_string());
            let is_duplicate = tool_call_signatures.contains(&call_sig);

            if is_duplicate {
                tracing::info!(
                    session_id = %session_id,
                    round = round,
                    tool = %tc.name,
                    "skipping duplicate tool call — returning cached nudge"
                );

                let nudge = format!(
                    "You already called {} with these exact arguments. \
                     The results are in the conversation above. \
                     Summarize those results for the user now.",
                    tc.name
                );

                // Record the signature so the hard-break counter still works
                tool_call_history.push(tc.name.clone());
                tool_call_signatures.push(call_sig);

                // Push the nudge as the tool result so the model sees it
                messages.push(crate::inference::types::ChatMessage {
                    role: crate::inference::types::Role::Tool,
                    content: Some(nudge.clone()),
                    tool_call_id: Some(tc.id.clone()),
                    tool_calls: None,
                });

                // Persist so windowed rebuild includes it
                {
                    let mgr = state
                        .lock()
                        .map_err(|e| format!("Lock error: {e}"))?;
                    let nudge_json = serde_json::Value::String(nudge);
                    mgr.add_tool_result_message(&session_id, &tc.id, &nudge_json)
                        .map_err(|e| format!("Failed to save nudge: {e}"))?;
                }

                continue; // skip to next tool call (or next round)
            }

            // ── Execute tool with Error Boundary ───────────────────────
            let tool_start = std::time::Instant::now();
            let outcome: ToolExecutionOutcome = {
                let mut mcp = mcp_state.lock().await;
                // Error boundary: wrap in a timeout to prevent hung tool executions
                // The try_read_with_timeout helper handles both success and error cases
                match tokio::time::timeout(
                    std::time::Duration::from_secs(120), // 2 min timeout per tool
                    execute_tool(&tc.name, &effective_arguments, &mut mcp),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_elapsed) => {
                        tracing::error!(
                            tool = %tc.name,
                            timeout_secs = 120,
                            "tool execution timed out — caught by error boundary"
                        );
                        ToolExecutionOutcome::InfraError {
                            tool_name: tc.name.clone(),
                            text: format!(
                                "Tool '{}' timed out after 120 seconds. Please try again or use a different tool.",
                                tc.name
                            ),
                        }
                    }
                }
            };
            let execution_time_ms = tool_start.elapsed().as_millis() as u64;

            let is_error = outcome.is_error();
            let result_text = outcome.model_text().to_string();

            // ── Audit log write ──────────────────────────────────────
            // Record every tool execution in the audit_log table so
            // audit.get_tool_log / audit.generate_audit_report can read them.
            {
                let mgr = state
                    .lock()
                    .map_err(|e| format!("Lock error: {e}"))?;
                let audit_status = if is_error {
                    AuditStatus::Error
                } else {
                    AuditStatus::Success
                };
                let result_val = serde_json::Value::String(result_text.clone());
                if let Err(e) = mgr.db().insert_audit_entry(
                    &session_id,
                    &tc.name,
                    &effective_arguments,
                    Some(&result_val),
                    audit_status,
                    user_confirmed,
                    execution_time_ms,
                ) {
                    tracing::warn!(
                        session_id = %session_id,
                        tool = %tc.name,
                        error = %e,
                        "failed to write audit log entry"
                    );
                }
            }

            if is_error {
                round_error_count += 1;
                *tool_failure_counts.entry(tc.name.clone()).or_default() += 1;
            }

            // Collect UnknownTool resolutions for correction hints
            if let ToolExecutionOutcome::UnknownTool {
                ref tool_name,
                ref resolution,
                ..
            } = outcome
            {
                round_unknown_tools.push((tool_name.clone(), resolution.clone()));
            }

            tool_call_history.push(tc.name.clone());
            tool_call_signatures.push((
                tc.name.clone(),
                tc.arguments.to_string(),
            ));

            if is_error {
                tracing::warn!(
                    session_id = %session_id,
                    tool = %tc.name,
                    tool_call_id = %tc.id,
                    result_len = result_text.len(),
                    result_preview = %truncate_utf8(&result_text, 200),
                    execution_time_ms = execution_time_ms,
                    tools_completed = tool_call_history.len(),
                    "tool call FAILED"
                );
            } else {
                tracing::info!(
                    session_id = %session_id,
                    tool = %tc.name,
                    tool_call_id = %tc.id,
                    result_len = result_text.len(),
                    execution_time_ms = execution_time_ms,
                    tools_completed = tool_call_history.len(),
                    user_confirmed,
                    "tool execution complete"
                );
            }

            let _ = app_handle.emit(
                "tool-result",
                serde_json::json!({
                    "id": chrono::Utc::now().timestamp_millis(),
                    "sessionId": session_id,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "role": "tool",
                    "content": result_text,
                    "toolCallId": tc.id,
                    "toolResult": {
                        "success": !is_error,
                        "result": result_text,
                        "toolCallId": tc.id,
                        "toolName": tc.name,
                        "executionTimeMs": execution_time_ms,
                        "inferenceTimeMs": inference_time_ms,
                    },
                    "tokenCount": result_text.len() / 4,
                }),
            );

            // Persist tool result in conversation
            {
                let mgr = state
                    .lock()
                    .map_err(|e| format!("Lock error: {e}"))?;
                let result_json = serde_json::Value::String(result_text);
                mgr.add_tool_result_message(
                    &session_id,
                    &tc.id,
                    &result_json,
                )
                .map_err(|e| format!("Failed to save tool result: {e}"))?;
            }
        }

        // ── Consecutive error round tracking ─────────────────────────
        // If ALL tool calls in this round errored, the model may be stuck
        // in a loop calling a non-existent tool (e.g., filesystem.rename_file).
        // After MAX_CONSECUTIVE_ERROR_ROUNDS, inject a corrective hint using
        // the suggestions already computed by ToolRegistry::resolve().
        if round_error_count > 0 && round_error_count == round_call_count {
            consecutive_error_rounds += 1;
            tracing::warn!(
                session_id = %session_id,
                round = round,
                consecutive_error_rounds = consecutive_error_rounds,
                failed_tools = ?tool_calls_detected.iter().map(|tc| tc.name.as_str()).collect::<Vec<_>>(),
                "all tool calls in round failed"
            );

            if consecutive_error_rounds >= MAX_CONSECUTIVE_ERROR_ROUNDS {
                let hint = format_correction_hint(&round_unknown_tools);

                tracing::info!(
                    round = round,
                    hint_len = hint.len(),
                    "injecting tool correction hint after repeated failures"
                );

                // Persist the corrective hint as a user message
                {
                    let mgr = state
                        .lock()
                        .map_err(|e| format!("Lock error: {e}"))?;
                    mgr.add_user_message(&session_id, &hint)
                        .map_err(|e| format!("Failed to save hint: {e}"))?;
                }

                // Reset counter so the model gets another chance
                consecutive_error_rounds = 0;
            }
        } else {
            // At least one tool succeeded — reset the counter
            consecutive_error_rounds = 0;
        }

        // ── Per-tool failure circuit breaker ──────────────────────────
        // Even when the per-round counter resets (because the model alternates
        // between a succeeding tool and a failing one), the per-tool counter
        // keeps accumulating. Once a tool hits MAX_SAME_TOOL_FAILURES, remove
        // it from the definitions and inject a hard stop hint.
        let stuck_tools: Vec<String> = tool_failure_counts
            .iter()
            .filter(|(_, &count)| count >= MAX_SAME_TOOL_FAILURES)
            .map(|(name, _)| name.clone())
            .collect();

        if !stuck_tools.is_empty() {
            let hint = format!(
                "STOP: The following tools have each failed {} or more times and have been \
                 removed: {}. Do NOT attempt to call them again. Respond to the user with \
                 what you know so far, or try a completely different approach.",
                MAX_SAME_TOOL_FAILURES,
                stuck_tools.join(", ")
            );

            tracing::warn!(
                session_id = %session_id,
                round = round,
                stuck_tools = ?stuck_tools,
                "per-tool failure limit reached — removing stuck tools from definitions"
            );

            // Remove stuck tools from the active tool definitions
            tools.retain(|t| !stuck_tools.contains(&t.function.name));

            // Clear the counters for removed tools so we don't re-trigger
            for name in &stuck_tools {
                tool_failure_counts.remove(name);
            }

            // Inject the hint as a user message
            {
                let mgr = state
                    .lock()
                    .map_err(|e| format!("Lock error: {e}"))?;
                mgr.add_user_message(&session_id, &hint)
                    .map_err(|e| format!("Failed to save stuck-tool hint: {e}"))?;
            }
        }

        // ── Duplicate tool call detection ─────────────────────────────
        // If the model is calling the same tool with the same arguments
        // repeatedly (e.g., list_directory("~/Downloads") 3× in a row),
        // the results won't change. Break to prevent wasting rounds.
        let dup_count = consecutive_duplicate_count(&tool_call_signatures);
        if dup_count >= MAX_DUPLICATE_TOOL_CALLS {
            tracing::warn!(
                session_id = %session_id,
                round = round,
                duplicate_count = dup_count,
                tool = %tool_call_signatures.last().map(|(n, _)| n.as_str()).unwrap_or("?"),
                "duplicate tool call detected — model is stuck, breaking loop"
            );
            break;
        }

        // ── Mid-loop eviction ───────────────────────────────────────
        // After persisting tool results, check if context window needs
        // eviction before the next round. This prevents unbounded growth
        // during long multi-step workflows.
        {
            let mgr =
                state.lock().map_err(|e| format!("Lock error: {e}"))?;
            let evicted = mgr
                .evict_if_needed(&session_id)
                .map_err(|e| format!("Eviction error: {e}"))?;
            if evicted > 0 {
                tracing::info!(
                    round = round,
                    evicted_tokens = evicted,
                    "mid-loop eviction"
                );
            }
        }

        // Rebuild messages (windowed — compress old tool results to save tokens)
        messages = {
            let mgr =
                state.lock().map_err(|e| format!("Lock error: {e}"))?;
            mgr.build_windowed_chat_messages(&session_id, 4)
                .map_err(|e| format!("Failed to build messages: {e}"))?
        };
    }

    } // end if full_response.is_empty() (skip agent loop when orchestrator succeeded)

    // 4. If the agent loop finished without generating text, force a
    //    summary. This can happen when:
    //    - All rounds were used on tool calls (normal for large batches)
    //    - Model returned empty responses (timeout / context overflow)
    //    - Streaming errors caused early exit
    //
    //    Strategy: inject a short, explicit "summarize now" user message
    //    and call the model WITHOUT tools, so it MUST produce text.
    if full_response.is_empty() {
        tracing::info!(
            session_id = %session_id,
            rounds_used = empty_response_count,
            tool_calls_total = tool_call_history.len(),
            "forcing summary — injecting summarize prompt"
        );

        // Inject a constrained summary instruction that prevents confabulation.
        // The model MUST only report results it actually received from tools.
        let summary_instruction = crate::inference::types::ChatMessage {
            role: crate::inference::types::Role::User,
            content: Some(
                "Based on the tool results above, provide a concise summary.\n\
                 CRITICAL RULES:\n\
                 - ONLY report results you actually received from tool calls above.\n\
                 - If a file was not processed, say 'not processed' — do NOT guess or invent results.\n\
                 - If no tool results are visible, say 'I was unable to complete the task.'\n\
                 Do NOT call any more tools."
                    .to_string(),
            ),
            tool_call_id: None,
            tool_calls: None,
        };
        messages.push(summary_instruction);

        match client
            .chat_completion_stream(messages, None, Some(conversational_sampling)) // No tools → model MUST produce text
            .await
        {
            Ok(stream) => {
                futures::pin_mut!(stream);
                while let Some(chunk_result) = stream.next().await {
                    if let Ok(chunk) = chunk_result {
                        if let Some(token) = &chunk.token {
                            full_response.push_str(token);
                            let _ = app_handle.emit("stream-token", token.clone());
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "summary call failed");
            }
        }

        // If even the summary call returned nothing, use a static fallback
        if full_response.is_empty() {
            tracing::warn!("summary call also returned empty — using static fallback text");
            full_response = "I processed the requested files using the tools above. \
                You can see the individual results in the tool trace. \
                Please ask a follow-up question if you'd like me to continue."
                .to_string();
            let _ = app_handle.emit("stream-token", full_response.clone());
        }
    }

    // 5. Persist final assistant text response
    //    (skip if the orchestrator already persisted it)
    {
        let mgr = state.lock().map_err(|e| format!("Lock error: {e}"))?;

        if !full_response.is_empty() && !already_persisted {
            mgr.add_assistant_message(&session_id, &full_response)
                .map_err(|e| format!("Failed to save assistant message: {e}"))?;
        }

        emit_context_budget(&app_handle, &mgr, &session_id);
    }

    // 5. Emit the complete message
    let message = serde_json::json!({
        "id": chrono::Utc::now().timestamp_millis(),
        "sessionId": session_id,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "role": "assistant",
        "content": full_response,
        "tokenCount": full_response.len() / 4,
    });

    let _ = app_handle.emit("stream-complete", message);

    // Release in-flight lock
    {
        let mut in_flight_guard = in_flight.lock().await;
        in_flight_guard.insert(session_id.clone(), false);
    }

    Ok(())
}

/// Respond to a confirmation request from the agent loop.
///
/// The frontend calls this when the user clicks Confirm/Cancel on a
/// confirmation dialog. The response is forwarded to the agent loop
/// via the pending oneshot channel.
#[tauri::command]
pub async fn respond_to_confirmation(
    request_id: String,
    response: serde_json::Value,
    pending: tauri::State<'_, PendingConfirmation>,
) -> Result<(), String> {
    tracing::info!(
        request_id = %request_id,
        response = %response,
        "confirmation response received"
    );

    let parsed: ConfirmationResponse = serde_json::from_value(response)
        .map_err(|e| format!("Invalid confirmation response: {e}"))?;

    let mut lock = pending.lock().await;
    if let Some(tx) = lock.take() {
        // oneshot::Sender::send returns Err if receiver was dropped
        tx.send(parsed).map_err(|_| {
            "Confirmation channel closed — agent loop may have timed out".to_string()
        })?;
    } else {
        tracing::warn!(
            request_id = %request_id,
            "no pending confirmation — response ignored"
        );
    }

    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_core::response_analysis::is_incomplete_response;

    #[test]
    fn test_unwrap_tool_result_json_extracts_text() {
        // Simulates what Python MCP servers send: json.dumps(result.model_dump())
        let raw = r#"{"text": "LocalCowork OCR Test\nInvoice #12345", "confidence": 0.9, "engine": "lfm_vision"}"#;
        let result = unwrap_tool_result_json(raw);
        assert!(result.starts_with("LocalCowork OCR Test"));
        assert!(result.contains("[engine: lfm_vision]"));
        assert!(result.contains("[confidence: 0.9]"));
    }

    #[test]
    fn test_unwrap_tool_result_json_plain_text() {
        let raw = "Just a plain text result";
        let result = unwrap_tool_result_json(raw);
        assert_eq!(result, "Just a plain text result");
    }

    #[test]
    fn test_unwrap_tool_result_json_no_text_field() {
        let raw = r#"{"headers": ["col1", "col2"], "rows": [["a", "b"]]}"#;
        let result = unwrap_tool_result_json(raw);
        // No recognized text field, should return raw JSON
        assert_eq!(result, raw);
    }

    #[test]
    fn test_extract_mcp_result_text_with_content_array() {
        let value = serde_json::json!({
            "content": [{"type": "text", "text": "{\"text\": \"hello\", \"engine\": \"tesseract\"}"}]
        });
        let result = extract_mcp_result_text(&Some(value));
        assert!(result.starts_with("hello"));
        assert!(result.contains("[engine: tesseract]"));
    }

    #[test]
    fn test_extract_mcp_result_text_none() {
        let result = extract_mcp_result_text(&None);
        assert_eq!(result, "No result returned.");
    }

    #[test]
    fn test_truncate_tool_result_short() {
        let result = truncate_tool_result("short result", "test_tool");
        assert_eq!(result, "short result");
    }

    #[test]
    fn test_truncate_tool_result_long() {
        let long = "x".repeat(10_000);
        let result = truncate_tool_result(&long, "test_tool");
        assert!(result.len() < long.len());
        assert!(result.contains("[... truncated: showing first 6000 of 10000 chars]"));
    }

    #[test]
    fn test_is_incomplete_response_remaining() {
        assert!(is_incomplete_response(
            "I've processed 3 files. There are 4 remaining screenshots to rename."
        ));
    }

    #[test]
    fn test_is_incomplete_response_next_file() {
        assert!(is_incomplete_response(
            "Renamed screenshot 1. Moving on to the next file."
        ));
    }

    #[test]
    fn test_is_incomplete_response_complete() {
        assert!(!is_incomplete_response(
            "All screenshots have been renamed successfully."
        ));
    }

    #[test]
    fn test_is_incomplete_response_no_signals() {
        // No incomplete or complete signals — defaults to false (task done)
        assert!(!is_incomplete_response(
            "Here is the result of your request."
        ));
    }

    /// Helper to create an McpClient with registered tools for testing.
    fn mcp_client_with_tools(tools: Vec<(&str, &str)>) -> McpClient {
        use crate::mcp_client::types::{McpServersConfig, McpToolDefinition};

        let config = McpServersConfig {
            servers: std::collections::HashMap::new(),
        };
        let mut client = McpClient::new(config, None);

        // Group tools by server name and register them
        let mut server_tools: std::collections::HashMap<&str, Vec<McpToolDefinition>> =
            std::collections::HashMap::new();
        for (server, tool) in tools {
            server_tools
                .entry(server)
                .or_default()
                .push(McpToolDefinition {
                    name: tool.to_string(),
                    description: format!("Test tool: {tool}"),
                    params_schema: serde_json::json!({"type": "object", "properties": {}}),
                    returns_schema: serde_json::json!({}),
                    confirmation_required: false,
                    undo_supported: false,
                });
        }
        for (server, defs) in server_tools {
            client.registry.register_server_tools(server, defs);
        }

        client
    }

    #[test]
    fn test_resolve_exact_match() {
        let client = mcp_client_with_tools(vec![("filesystem", "move_file")]);
        let resolution = client.registry.resolve("filesystem.move_file", 0.5);
        assert!(matches!(resolution, ToolResolution::Exact(_)));
        assert_eq!(resolution.resolved_name(), Some("filesystem.move_file"));
    }

    #[test]
    fn test_resolve_unprefixed() {
        let client = mcp_client_with_tools(vec![
            ("filesystem", "move_file"),
            ("filesystem", "copy_file"),
            ("ocr", "extract_text_from_image"),
        ]);
        let resolution = client.registry.resolve("move_file", 0.5);
        assert!(matches!(resolution, ToolResolution::Unprefixed { .. }));
        assert_eq!(resolution.resolved_name(), Some("filesystem.move_file"));
    }

    #[test]
    fn test_resolve_unknown_unprefixed() {
        let client = mcp_client_with_tools(vec![("filesystem", "move_file")]);
        let resolution = client.registry.resolve("nonexistent_tool", 0.5);
        assert!(matches!(resolution, ToolResolution::NotFound { .. }));
        assert_eq!(resolution.resolved_name(), None);
    }

    #[test]
    fn test_resolve_wrong_server_prefix() {
        let client = mcp_client_with_tools(vec![("filesystem", "move_file")]);
        // "wrong_server" doesn't exist — no same-server tools to match against
        let resolution = client.registry.resolve("wrong_server.move_file", 0.5);
        assert!(matches!(resolution, ToolResolution::NotFound { .. }));
    }

    #[test]
    fn test_resolve_ambiguous_unprefixed() {
        let client = mcp_client_with_tools(vec![
            ("ocr", "process"),
            ("document", "process"),
        ]);
        // Ambiguous — two servers have "process"
        let resolution = client.registry.resolve("process", 0.5);
        assert!(matches!(resolution, ToolResolution::NotFound { .. }));
    }

    #[test]
    fn test_build_system_prompt_includes_server_names() {
        use crate::mcp_client::registry::ToolRegistry;
        use crate::mcp_client::types::McpToolDefinition;

        let mut registry = ToolRegistry::new();
        registry.register_server_tools(
            "filesystem",
            vec![McpToolDefinition {
                name: "list_dir".to_string(),
                description: "List directory".to_string(),
                params_schema: serde_json::json!({"type": "object"}),
                returns_schema: serde_json::json!({}),
                confirmation_required: false,
                undo_supported: false,
            }],
        );
        registry.register_server_tools(
            "email",
            vec![McpToolDefinition {
                name: "send_draft".to_string(),
                description: "Send draft".to_string(),
                params_schema: serde_json::json!({"type": "object"}),
                returns_schema: serde_json::json!({}),
                confirmation_required: true,
                undo_supported: false,
            }],
        );

        let prompt = build_system_prompt(&registry, false);
        assert!(prompt.contains("filesystem (1)"));
        assert!(prompt.contains("email (1)"));
        assert!(prompt.contains("2 tools across 2 servers"));
        assert!(prompt.contains("LocalCowork"));
        // Should include XML-tagged rules section
        assert!(prompt.contains("<rules>"));
        assert!(prompt.contains("fully-qualified tool names"));
    }

    #[test]
    fn test_build_system_prompt_empty_registry() {
        use crate::mcp_client::registry::ToolRegistry;

        let registry = ToolRegistry::new();
        let prompt = build_system_prompt(&registry, false);
        assert!(prompt.contains("No MCP tools currently available"));
        // Should still include the rules and examples sections
        assert!(prompt.contains("<rules>"));
        assert!(prompt.contains("<examples>"));
        assert!(prompt.contains("filesystem.list_dir"));
    }

    #[test]
    fn test_build_system_prompt_with_two_pass() {
        use crate::mcp_client::registry::ToolRegistry;

        let registry = ToolRegistry::new();
        let prompt = build_system_prompt(&registry, true);
        assert!(prompt.contains("category-level tools"));
        assert!(prompt.contains("file_browse"));
    }

    #[test]
    fn test_build_system_prompt_has_precomputed_dates() {
        use crate::mcp_client::registry::ToolRegistry;

        let registry = ToolRegistry::new();
        let prompt = build_system_prompt(&registry, false);
        // Must contain the <datetime> block with pre-computed dates
        assert!(prompt.contains("<datetime>"));
        assert!(prompt.contains("</datetime>"));
        assert!(prompt.contains("today ="));
        assert!(prompt.contains("tomorrow ="));
        assert!(prompt.contains("this_week ="));
        assert!(prompt.contains("NEVER ask the user for a date"));
    }

    // ── has_unverified_completion tests ──────────────────────────────

    #[test]
    fn test_unverified_completion_claims_done_no_mutable_calls() {
        // Model says "all files renamed" but history has no move_file
        let history = vec![
            "filesystem.list_dir".to_string(),
            "ocr.extract_text_from_image".to_string(),
            "ocr.extract_text_from_image".to_string(),
        ];
        assert!(has_unverified_completion(
            "I've successfully renamed all 9 files.",
            &history,
        ));
    }

    #[test]
    fn test_unverified_completion_claims_done_with_mutable_calls() {
        // Model says "all files renamed" AND move_file is in history — genuine
        let history = vec![
            "filesystem.list_dir".to_string(),
            "ocr.extract_text_from_image".to_string(),
            "filesystem.move_file".to_string(),
        ];
        assert!(!has_unverified_completion(
            "I've successfully renamed all 9 files.",
            &history,
        ));
    }

    #[test]
    fn test_unverified_completion_no_completion_claim() {
        // Model doesn't claim completion — no confabulation check needed
        let history = vec!["filesystem.list_dir".to_string()];
        assert!(!has_unverified_completion(
            "Here are the files I found on your desktop.",
            &history,
        ));
    }

    #[test]
    fn test_unverified_completion_empty_history() {
        // Empty tool history + completion claim = confabulation
        assert!(has_unverified_completion(
            "All done! Finished processing everything.",
            &[],
        ));
    }

    #[test]
    fn test_unverified_completion_write_file_counts_as_mutable() {
        // write_file is a mutable operation — should count
        let history = vec!["filesystem.write_file".to_string()];
        assert!(!has_unverified_completion(
            "Task complete. All files processed.",
            &history,
        ));
    }

    #[test]
    fn test_unverified_completion_create_task_counts_as_mutable() {
        // create_task should now be recognized as mutable
        let history = vec![
            "filesystem.read_file".to_string(),
            "task.create_task".to_string(),
        ];
        assert!(!has_unverified_completion(
            "Successfully created the task.",
            &history,
        ));
    }

    #[test]
    fn test_unverified_completion_read_only_generic_done() {
        // Read-only task (list files) saying "all done" — NOT confabulation.
        // The model legitimately completed a read-only request.
        let history = vec![
            "filesystem.list_dir".to_string(),
        ];
        assert!(!has_unverified_completion(
            "All done! Here are the files in your Downloads folder.",
            &history,
        ));
    }

    #[test]
    fn test_unverified_completion_read_only_claims_rename() {
        // Read-only tools but claims "renamed" → confabulation
        let history = vec![
            "filesystem.list_dir".to_string(),
            "ocr.extract_text_from_image".to_string(),
        ];
        assert!(has_unverified_completion(
            "I've successfully renamed all 9 files.",
            &history,
        ));
    }

    #[test]
    fn test_unverified_completion_scan_then_complete() {
        // Security scan (read-only) followed by "completed" → not confabulation
        // (it's a genuinely complete read-only scan task)
        let history = vec![
            "security.scan_for_pii".to_string(),
            "security.scan_for_secrets".to_string(),
        ];
        assert!(!has_unverified_completion(
            "All done! Here's what I found in the scan.",
            &history,
        ));
    }

    // ── consecutive_duplicate_count tests ────────────────────────────

    #[test]
    fn test_duplicate_count_empty() {
        let history: Vec<(String, String)> = vec![];
        assert_eq!(consecutive_duplicate_count(&history), 0);
    }

    #[test]
    fn test_duplicate_count_single() {
        let history = vec![("list_dir".into(), r#"{"path":"~/Downloads"}"#.into())];
        assert_eq!(consecutive_duplicate_count(&history), 1);
    }

    #[test]
    fn test_duplicate_count_three_identical() {
        let history = vec![
            ("list_dir".into(), r#"{"path":"~/Downloads"}"#.into()),
            ("list_dir".into(), r#"{"path":"~/Downloads"}"#.into()),
            ("list_dir".into(), r#"{"path":"~/Downloads"}"#.into()),
        ];
        assert_eq!(consecutive_duplicate_count(&history), 3);
    }

    #[test]
    fn test_duplicate_count_different_args() {
        let history = vec![
            ("list_dir".into(), r#"{"path":"~/Downloads"}"#.into()),
            ("list_dir".into(), r#"{"path":"~/Documents"}"#.into()),
        ];
        assert_eq!(consecutive_duplicate_count(&history), 1);
    }

    #[test]
    fn test_duplicate_count_interrupted_by_different_tool() {
        let history = vec![
            ("list_dir".into(), r#"{"path":"~/Downloads"}"#.into()),
            ("read_file".into(), r#"{"path":"file.txt"}"#.into()),
            ("list_dir".into(), r#"{"path":"~/Downloads"}"#.into()),
        ];
        // Only the last consecutive run counts (just 1)
        assert_eq!(consecutive_duplicate_count(&history), 1);
    }

    // ── expand_tilde_in_arguments tests ──────────────────────────────

    #[test]
    fn test_expand_tilde_simple_path() {
        let args = serde_json::json!({"path": "~/Documents/file.txt"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert!(!path.starts_with('~'), "tilde should be expanded: {path}");
        assert!(path.ends_with("/Documents/file.txt"));
    }

    #[test]
    fn test_expand_tilde_bare() {
        let args = serde_json::json!({"path": "~"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert!(!path.starts_with('~'));
        assert!(!path.is_empty());
    }

    #[test]
    fn test_expand_tilde_leaves_absolute_paths() {
        let args = serde_json::json!({"path": "/Users/chintan/Documents/file.txt"});
        let expanded = expand_tilde_in_arguments(&args);
        assert_eq!(
            expanded["path"].as_str().unwrap(),
            "/Users/chintan/Documents/file.txt"
        );
    }

    #[test]
    fn test_expand_tilde_leaves_other_user() {
        // ~other_user/... should NOT be expanded
        let args = serde_json::json!({"path": "~other_user/file.txt"});
        let expanded = expand_tilde_in_arguments(&args);
        assert_eq!(expanded["path"].as_str().unwrap(), "~other_user/file.txt");
    }

    #[test]
    fn test_expand_tilde_nested_object() {
        let args = serde_json::json!({
            "source": "~/Desktop/a.png",
            "destination": "/tmp/b.png",
            "options": {"backup": "~/backup/"}
        });
        let expanded = expand_tilde_in_arguments(&args);
        assert!(!expanded["source"].as_str().unwrap().starts_with('~'));
        assert_eq!(expanded["destination"].as_str().unwrap(), "/tmp/b.png");
        assert!(!expanded["options"]["backup"].as_str().unwrap().starts_with('~'));
    }

    #[test]
    fn test_expand_tilde_non_string_values() {
        let args = serde_json::json!({"count": 42, "flag": true, "path": "~/file"});
        let expanded = expand_tilde_in_arguments(&args);
        assert_eq!(expanded["count"], 42);
        assert_eq!(expanded["flag"], true);
        assert!(!expanded["path"].as_str().unwrap().starts_with('~'));
    }

    #[test]
    fn test_expand_tilde_array_values() {
        let args = serde_json::json!({"paths": ["~/a.txt", "/b.txt", "~/c.txt"]});
        let expanded = expand_tilde_in_arguments(&args);
        let paths = expanded["paths"].as_array().unwrap();
        assert!(!paths[0].as_str().unwrap().starts_with('~'));
        assert_eq!(paths[1].as_str().unwrap(), "/b.txt");
        assert!(!paths[2].as_str().unwrap().starts_with('~'));
    }

    // ── fix_path_string: cross-platform path correction tests ───────

    /// Helper: build the expected path using Path::join (platform-correct).
    fn expected_home_join(suffix: &str) -> String {
        dirs::home_dir()
            .unwrap()
            .join(suffix)
            .to_string_lossy()
            .into_owned()
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_fix_foreign_os_prefix() {
        // On macOS, /home/ is foreign — any username is hallucinated
        let args = serde_json::json!({"path": "/home/chintan/Downloads"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert_eq!(path, expected_home_join("Downloads"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_native_prefix_real_username_not_rewritten() {
        // On macOS, /Users/<other_real_user>/... should NOT be rewritten
        // (could be a legitimate multi-user path)
        let args = serde_json::json!({"path": "/Users/admin/shared/notes.txt"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert_eq!(path, "/Users/admin/shared/notes.txt", "Real username should not be rewritten");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_native_prefix_template_user() {
        // /Users/{user}/Downloads on macOS — template on native prefix
        let args = serde_json::json!({"path": "/Users/{user}/Downloads"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert!(
            !path.contains("{user}"),
            "Placeholder should be replaced: {path}"
        );
        assert_eq!(path, expected_home_join("Downloads"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_native_prefix_template_username() {
        // /Users/{username}/Documents on macOS
        let args = serde_json::json!({"path": "/Users/{username}/Documents"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert!(
            !path.contains("{username}"),
            "Placeholder should be replaced: {path}"
        );
        assert_eq!(path, expected_home_join("Documents"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_native_prefix_angle_bracket() {
        // /Users/<username>/Downloads on macOS
        let args = serde_json::json!({"path": "/Users/<username>/Downloads"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert!(
            !path.contains("<username>"),
            "Angle-bracket placeholder should be replaced: {path}"
        );
        assert_eq!(path, expected_home_join("Downloads"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_native_prefix_square_bracket() {
        // /Users/[USER]/Documents/Projects on macOS
        let args = serde_json::json!({"path": "/Users/[USER]/Documents/Projects"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert!(
            !path.contains("[USER]"),
            "Square-bracket placeholder should be replaced: {path}"
        );
        assert_eq!(path, expected_home_join("Documents/Projects"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_native_prefix_known_placeholder_word() {
        // /Users/user/Documents on macOS — "user" is a known placeholder
        let args = serde_json::json!({"path": "/Users/user/Documents"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert_eq!(path, expected_home_join("Documents"));
    }

    #[test]
    fn test_fix_bare_relative_path() {
        // Model generates just "Projects" instead of an absolute path
        let args = serde_json::json!({"path": "Projects"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert_eq!(path, expected_home_join("Projects"));
    }

    #[test]
    fn test_fix_bare_downloads_relative_path() {
        // Model generates "Downloads"
        let args = serde_json::json!({"path": "Downloads"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert_eq!(path, expected_home_join("Downloads"));
    }

    #[test]
    fn test_tilde_expansion() {
        let args = serde_json::json!({"path": "~/Documents/file.txt"});
        let expanded = expand_tilde_in_arguments(&args);
        let path = expanded["path"].as_str().unwrap();
        assert_eq!(path, expected_home_join("Documents/file.txt"));
    }

    #[test]
    fn test_no_fix_for_correct_path() {
        // Already-correct absolute path should not be modified
        let home = dirs::home_dir().unwrap();
        let correct = home.join("Documents").join("test.txt");
        let correct_str = correct.to_string_lossy().into_owned();
        let args = serde_json::json!({"path": correct_str});
        let expanded = expand_tilde_in_arguments(&args);
        assert_eq!(expanded["path"].as_str().unwrap(), correct_str);
    }

    #[test]
    fn test_no_fix_for_urls() {
        // URL-like strings should not be modified
        let args = serde_json::json!({"url": "https://example.com/Documents/file"});
        let expanded = expand_tilde_in_arguments(&args);
        assert_eq!(
            expanded["url"].as_str().unwrap(),
            "https://example.com/Documents/file"
        );
    }

    // ─── Tool Result Compression Tests (PR #59) ─────────────────────────────

    #[test]
    fn test_truncate_tool_result_small() {
        let result = "short result";
        let truncated = truncate_tool_result(result, "test_tool");
        assert_eq!(truncated, result);
    }

    #[test]
    fn test_truncate_tool_result_large_no_compression() {
        // Large result that doesn't match compression patterns
        let result = "x".repeat(10000);
        let truncated = truncate_tool_result(&result, "unknown_tool");
        assert!(truncated.contains("truncated"));
        assert!(truncated.len() < result.len());
    }

    #[test]
    fn test_compress_directory_listing() {
        let listing = r#"📁 src/
📁 tests/
📄 file1.txt (1 KB)
📄 file2.txt (2 KB)
📁 subdir/
📄 long_file_name.txt (100 KB)"#;
        let compressed = compress_directory_listing(listing);
        assert!(compressed.is_some());
        let summary = compressed.unwrap();
        assert!(summary.contains("Total:"));
        assert!(summary.contains("Directories:"));
        assert!(summary.contains("Files:"));
    }

    #[test]
    fn test_compress_directory_listing_empty() {
        let compressed = compress_directory_listing("");
        assert!(compressed.is_none());
    }

    #[test]
    fn test_compress_search_results_with_count() {
        let result = "Scanning files...\nFound 42 matches\n\n/path/to/file1.txt: line 10\n/path/to/file2.txt: line 20";
        let compressed = compress_search_results(result);
        assert!(compressed.is_some());
        let summary = compressed.unwrap();
        assert!(summary.contains("42"));
        assert!(summary.contains("Key findings"));
    }

    #[test]
    fn test_compress_search_results_no_count() {
        // Search result without clear count pattern
        let result = "File A\nFile B\nFile C\nFile D\nFile E";
        let compressed = compress_search_results(result);
        assert!(compressed.is_some());
    }

    #[test]
    fn test_compress_json_result_array() {
        let json = r#"[
            {"name": "item1", "size": 100},
            {"name": "item2", "size": 200},
            {"name": "item3", "size": 300}
        ]"#;
        let compressed = compress_json_result(json);
        assert!(compressed.is_some());
        let summary = compressed.unwrap();
        assert!(summary.contains("3 items"));
        assert!(summary.contains("item1"));
    }

    #[test]
    fn test_compress_json_result_object() {
        let json = r#"{"name": "test", "content": "hello world", "count": 42}"#;
        let compressed = compress_json_result(json);
        assert!(compressed.is_some());
        let summary = compressed.unwrap();
        assert!(summary.contains("name:"));
        assert!(summary.contains("test"));
    }

    #[test]
    fn test_compress_json_result_empty_array() {
        let json = "[]";
        let compressed = compress_json_result(json);
        assert!(compressed.is_some());
        assert!(compressed.unwrap().contains("empty"));
    }

    #[test]
    fn test_compress_tool_result_skips_small() {
        // Small results should not be compressed
        let result = "short";
        let compressed = compress_tool_result(result, "list_dir");
        assert!(compressed.is_none());
    }

    #[test]
    fn test_compress_tool_result_non_compressible_tool() {
        // Non-compressible tools should return None even for large results
        let result = "x".repeat(10000);
        let compressed = compress_tool_result(&result, "write_file");
        assert!(compressed.is_none());
    }

    #[test]
    fn test_compress_tool_result_compressible_tool() {
        // Compressible tools should attempt compression (need 3000+ chars)
        let result = "📁 folder1\n📁 folder2\n📄 file1.txt (1 KB)\n📄 file2.txt (2 KB)\n".repeat(500);
        let compressed = compress_tool_result(&result, "list_dir");
        assert!(compressed.is_some());
    }

    #[test]
    fn test_builtin_tool_definitions() {
        let tools = builtin_tool_definitions();
        assert!(!tools.is_empty());
        
        let names: Vec<String> = tools.iter()
            .map(|t| t.function.name.clone())
            .collect();
        
        assert!(names.contains(&"list_directory".to_string()));
        assert!(names.contains(&"read_file".to_string()));
    }

    #[test]
    fn test_builtin_tool_definitions_have_descriptions() {
        let tools = builtin_tool_definitions();
        for tool in &tools {
            assert!(!tool.function.description.is_empty());
            assert!(!tool.function.parameters.is_null());
        }
    }
}
