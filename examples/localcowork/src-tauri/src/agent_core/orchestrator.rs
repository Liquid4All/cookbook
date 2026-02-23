//! Dual-model orchestrator: Planner (LFM2-24B-A2B) + Router (LFM2.5-1.2B-Router-FT).
//!
//! Architecture (ADR-009):
//! 1. **Plan** — planner model (MoE, ~2B active) decomposes user request into steps
//! 2. **Execute** — router model selects and calls one tool per step (RAG pre-filtered K=15)
//! 3. **Synthesize** — planner model generates user-facing summary from step results
//!
//! Each step is a clean single-turn interaction with the router model — no
//! conversation history, no context accumulation. This preserves the 78%
//! single-step accuracy that degrades to 8% in multi-turn context.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

use tokio::sync::Mutex as TokioMutex;

use crate::agent_core::plan_parser::{parse_bracket_plan, parse_json_plan};
use crate::agent_core::tokens::truncate_utf8;
use crate::agent_core::tool_prefilter::ToolEmbeddingIndex;
use crate::inference::client::InferenceClient;
use crate::inference::config::{ModelsConfig, OrchestratorConfig};
use crate::inference::types::{
    ChatMessage, Role, SamplingOverrides, ToolCall,
};
use crate::mcp_client::client::McpClient;

// ─── Types ──────────────────────────────────────────────────────────────────

/// A single step in the execution plan (from the planner model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_number: u32,
    /// Self-contained instruction for the router model.
    pub description: String,
    /// Hint: which MCP server is likely needed (e.g., "filesystem").
    #[serde(default)]
    pub expected_server: Option<String>,
    /// Hint: key parameter values from the user's request.
    #[serde(default)]
    pub hint_params: Option<serde_json::Value>,
}

/// Structured plan output from the planner model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepPlan {
    /// Whether the request needs tool calls at all.
    pub needs_tools: bool,
    /// Direct response when no tools needed.
    #[serde(default)]
    pub direct_response: Option<String>,
    /// Ordered sequence of tool steps.
    #[serde(default)]
    pub steps: Vec<PlanStep>,
}

/// Result of executing a single plan step.
#[derive(Debug, Clone, Serialize)]
pub struct StepExecutionResult {
    pub step_number: u32,
    pub description: String,
    pub tool_called: Option<String>,
    pub tool_arguments: Option<serde_json::Value>,
    pub tool_result: Option<String>,
    pub success: bool,
    pub error: Option<String>,
}

/// Full orchestration result.
#[derive(Debug, Clone)]
pub struct OrchestrationResult {
    pub step_results: Vec<StepExecutionResult>,
    pub synthesis: String,
    pub all_steps_succeeded: bool,
    /// True if the orchestrator aborted and the caller should fall back.
    pub fell_back: bool,
}

// ─── Orchestrator Entry Point ───────────────────────────────────────────────

/// Execute the dual-model orchestration pipeline.
///
/// Returns `Ok(result)` on success. If `result.fell_back` is true, the caller
/// should fall through to the single-model agent loop.
#[allow(clippy::too_many_arguments)]
pub async fn orchestrate_dual_model(
    session_id: &str,
    user_message: &str,
    conversation_history: &[ChatMessage],
    models_config: &ModelsConfig,
    orch_config: &OrchestratorConfig,
    app_handle: &tauri::AppHandle,
    conv_state: &Mutex<crate::agent_core::ConversationManager>,
    mcp_state: &TokioMutex<McpClient>,
) -> Result<OrchestrationResult, String> {
    // Create separate clients for planner and router
    let mut planner = InferenceClient::from_config_with_model(
        models_config.clone(),
        &orch_config.planner_model,
    )
    .map_err(|e| format!("planner client error: {e}"))?;

    let mut router = InferenceClient::from_config_with_model(
        models_config.clone(),
        &orch_config.router_model,
    )
    .map_err(|e| format!("router client error: {e}"))?;

    tracing::info!(
        planner = planner.current_model_name(),
        router = router.current_model_name(),
        "orchestrator: starting dual-model pipeline"
    );

    // ── Phase 0: Template match (M1) ────────────────────────────────────
    // Check if the user's message matches a known use case pattern.
    // If matched, skip the planner entirely — use the pre-built plan.
    let mut plan_is_template = false;

    // ── Phase 1: Plan ───────────────────────────────────────────────────
    let mut plan = if let Some(template_plan) =
        crate::agent_core::plan_templates::try_template_match(user_message)
    {
        tracing::info!(
            steps = template_plan.steps.len(),
            "orchestrator: using template-matched plan (skipping planner)"
        );
        plan_is_template = true;
        template_plan
    } else {
        match plan_steps(&mut planner, user_message, conversation_history).await {
            Ok(plan) => plan,
            Err(e) => {
                tracing::warn!(error = %e, "orchestrator: plan failed — falling back");
                return Ok(OrchestrationResult {
                    step_results: Vec::new(),
                    synthesis: String::new(),
                    all_steps_succeeded: false,
                    fell_back: true,
                });
            }
        }
    };

    // Post-plan decomposition check (Fix F11): only for model-generated plans.
    // Template plans are pre-decomposed and don't need this check.
    if !plan_is_template && plan_needs_decomposition(&plan, user_message) {
        tracing::info!(
            original_steps = plan.steps.len(),
            "orchestrator: plan under-decomposed — re-planning with stronger prompt"
        );
        let retry_message = format!(
            "{}\n\n\
             CRITICAL: This request requires MULTIPLE steps across DIFFERENT servers. \
             You MUST break it into separate steps. Each step calls ONE tool from ONE server. \
             Do NOT combine scanning, reading, and task creation into a single step. \
             Look for these signals in the request: \"and\", \"then\", \"create a task\", \
             \"scan for X and Y\" — each signals a separate step.",
            user_message
        );
        if let Ok(retry_plan) =
            plan_steps(&mut planner, &retry_message, conversation_history).await
        {
            if retry_plan.needs_tools && retry_plan.steps.len() > plan.steps.len() {
                tracing::info!(
                    new_steps = retry_plan.steps.len(),
                    "orchestrator: re-plan produced more steps — using new plan"
                );
                plan = retry_plan;
            }
        }
    }

    // If no tools needed, stream the direct response.
    // Note: stream-complete is NOT emitted here — the caller (chat.rs)
    // handles persistence and the properly-formatted ChatMessage emission.
    if !plan.needs_tools {
        let response = plan.direct_response.unwrap_or_default();
        let _ = tauri::Emitter::emit(app_handle, "stream-token", &response);
        return Ok(OrchestrationResult {
            step_results: Vec::new(),
            synthesis: response,
            all_steps_succeeded: true,
            fell_back: false,
        });
    }

    let _ = tauri::Emitter::emit(app_handle, "plan-created", &plan.steps);

    tracing::info!(step_count = plan.steps.len(), "orchestrator: plan created");

    // ── Build tool embedding index ──────────────────────────────────────
    let tool_pairs: Vec<(String, String)> = {
        let mcp = mcp_state.lock().await;
        mcp.registry.tool_name_description_pairs()
    };

    let tool_index = match ToolEmbeddingIndex::build(
        router.current_base_url(),
        &tool_pairs,
    )
    .await
    {
        Ok(index) => index,
        Err(e) => {
            tracing::warn!(error = %e, "orchestrator: tool index build failed — falling back");
            return Ok(OrchestrationResult {
                step_results: Vec::new(),
                synthesis: String::new(),
                all_steps_succeeded: false,
                fell_back: true,
            });
        }
    };

    tracing::info!(tool_count = tool_index.len(), "orchestrator: tool index built");

    // ── Plan validation gate (Improvement I4) ─────────────────────────
    {
        let mcp = mcp_state.lock().await;
        for step in &plan.steps {
            if let Some(ref server) = step.expected_server {
                let prefix = format!("{}.", server);
                let has_tools = mcp
                    .registry
                    .tool_name_description_pairs()
                    .iter()
                    .any(|(name, _)| name.starts_with(&prefix));
                if !has_tools {
                    tracing::warn!(
                        step = step.step_number,
                        server = %server,
                        "orchestrator: plan references unknown server"
                    );
                }
            }
        }
    }

    // ── Phase 2: Execute each step ──────────────────────────────────────
    let mut step_results: Vec<StepExecutionResult> = Vec::new();
    let mut any_critical_failure = false;
    let total_steps = plan.steps.len();

    for step in &plan.steps {
        // Richer step progress events (Improvement I3)
        let _ = tauri::Emitter::emit(
            app_handle,
            "step-executing",
            &serde_json::json!({
                "step_number": step.step_number,
                "total_steps": total_steps,
                "description": step.description,
                "server": step.expected_server,
            }),
        );

        let result = execute_step(
            step,
            &step_results,
            &mut router,
            &tool_index,
            orch_config,
            mcp_state,
        )
        .await;

        let _ = tauri::Emitter::emit(
            app_handle,
            "step-completed",
            &serde_json::json!({
                "step_number": step.step_number,
                "total_steps": total_steps,
                "success": result.success,
                "tool_called": result.tool_called,
                "result_preview": result.tool_result.as_deref()
                    .map(|r| truncate_utf8(r, 200)),
            }),
        );

        if !result.success {
            tracing::warn!(
                step = step.step_number,
                error = result.error.as_deref().unwrap_or("unknown"),
                "orchestrator: step failed"
            );
            // Check if subsequent steps reference this step's result
            let step_ref = format!("step {}", step.step_number);
            let is_critical = plan.steps.iter().any(|s| {
                s.step_number > step.step_number
                    && s.description.to_lowercase().contains(&step_ref)
            });

            if is_critical {
                any_critical_failure = true;
                step_results.push(result);
                break;
            }
        }

        step_results.push(result);
    }

    // If a critical step failed, fall back to single-model mode
    if any_critical_failure {
        tracing::warn!("orchestrator: critical step failed — falling back");
        return Ok(OrchestrationResult {
            step_results,
            synthesis: String::new(),
            all_steps_succeeded: false,
            fell_back: true,
        });
    }

    // ── Phase 3: Synthesize ─────────────────────────────────────────────
    let synthesis = synthesize_response(
        &mut planner,
        user_message,
        &step_results,
        app_handle,
    )
    .await
    .unwrap_or_else(|e| {
        tracing::warn!(error = %e, "orchestrator: synthesis failed");
        // Build a basic summary from step results
        step_results
            .iter()
            .filter(|r| r.success)
            .map(|r| {
                format!(
                    "- {}: {}",
                    r.description,
                    r.tool_result.as_deref().unwrap_or("done")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    });

    // Persist the synthesized response
    {
        let mgr = conv_state.lock().map_err(|e| format!("Lock error: {e}"))?;
        let _ = mgr.add_assistant_message(session_id, &synthesis);
    }

    // Note: stream-complete is NOT emitted here — the caller (chat.rs)
    // handles the properly-formatted ChatMessage emission to avoid
    // duplicating the message format in two places.

    let all_succeeded = step_results.iter().all(|r| r.success);

    Ok(OrchestrationResult {
        step_results,
        synthesis,
        all_steps_succeeded: all_succeeded,
        fell_back: false,
    })
}

// ─── Phase 1: Plan ──────────────────────────────────────────────────────────

/// System prompt for the planner model — uses bracket-format calls that LFM2-24B-A2B
/// can reliably produce (JSON output had 94% parse failure rate in benchmarks).
const PLANNER_SYSTEM_PROMPT: &str = r#"You are a task planner for LocalCowork. Given a user request, decompose it into a sequence of tool-calling steps. You do NOT call tools yourself. Output your plan using bracket function calls.

Available capability areas (servers):
- filesystem: list, read, write, move, copy, delete, search files
- document: extract text from PDF/DOCX, convert formats, diff, create PDF/DOCX
- ocr: extract text from images/screenshots, extract structured data
- data: CSV/SQLite operations, deduplication, anomaly detection
- knowledge: semantic search across indexed documents, RAG Q&A
- security: PII/secrets scanning, file encryption, duplicate finding
- task: create/update/list tasks, daily briefing
- calendar: list events, create events, find free slots
- email: draft/send emails, search, summarize threads
- meeting: transcribe audio, extract action items, generate minutes
- audit: tool usage logs, session summaries
- clipboard: read/write system clipboard
- system: system info, open apps, take screenshots

Rules:
1. Use bracket function calls to build the plan. No prose before or after.
2. If the request does NOT require tools, call: [plan.respond(message="your direct answer")]
3. Each step description must be COMPLETE and self-contained.
4. Include file paths, search terms, and specifics from the user message in each step.
5. For steps needing a prior result, write: "Using the result from step N, ..."
6. Maximum 10 steps.
7. End with [plan.done()]

DECOMPOSITION RULES (critical):
- Each step calls EXACTLY ONE tool from ONE server. Never combine multiple operations.
- If the user says "scan for SSNs and API keys", that is TWO steps: one for PII scanning, one for secrets scanning.
- If the user says "do X and then create a task", that is at least TWO steps: the action + task creation.
- If scanning multiple files, create one step to list/discover them, then steps to scan each file type.
- Keywords that signal separate steps: "and", "then", "also", "follow up", "create a task".
- NEVER collapse a multi-server workflow into one step. When in doubt, create MORE steps.

Examples:

Simple single-step:
[plan.add_step(step=1, server="filesystem", description="List all files in /Users/chintan/Downloads")]
[plan.done()]

Two-server chain (filesystem + task):
[plan.add_step(step=1, server="filesystem", description="Read the file /Users/chintan/Projects/localCoWork/tests/fixtures/uc4/downloads/quarterly_report.txt")]
[plan.add_step(step=2, server="task", description="Using the content from step 1, create a task titled 'Review Q4 numbers' with due date Friday, including key findings from the quarterly report in the description")]
[plan.done()]

Multi-server workflow (filesystem + security + task):
[plan.add_step(step=1, server="filesystem", description="List all files in /Users/chintan/Projects/localCoWork/tests/fixtures/uc3/sample_files/")]
[plan.add_step(step=2, server="security", description="Using the result from step 1, scan each file found for PII (SSNs, phone numbers, addresses)")]
[plan.add_step(step=3, server="security", description="Using the result from step 1, scan each file found for secrets (API keys, passwords, tokens)")]
[plan.add_step(step=4, server="task", description="Using the results from steps 2 and 3, create a follow-up task to remediate any sensitive files found, including the file paths and findings in the description")]
[plan.done()]

Document analysis with knowledge search and email (filesystem + document + knowledge + email):
[plan.add_step(step=1, server="filesystem", description="List all PDF and DOCX files in /Users/chintan/Documents/Contracts/")]
[plan.add_step(step=2, server="document", description="Using the result from step 1, extract text from the contract file found")]
[plan.add_step(step=3, server="knowledge", description="Using the extracted text from step 2, search the knowledge base for similar clauses or related documents")]
[plan.add_step(step=4, server="email", description="Using the findings from steps 2 and 3, draft an email summarizing the key contract points and any related precedents found")]
[plan.done()]

File scan with OCR, PII detection, and remediation (filesystem + ocr + security + task + email):
[plan.add_step(step=1, server="filesystem", description="List all files in /Users/chintan/Downloads/ including PDFs and images")]
[plan.add_step(step=2, server="ocr", description="Using the result from step 1, extract text from any image files found (PNG, JPG, screenshots)")]
[plan.add_step(step=3, server="security", description="Using the results from steps 1 and 2, scan all extracted content for PII (SSNs, credit card numbers, phone numbers)")]
[plan.add_step(step=4, server="task", description="Using the results from step 3, create a remediation task listing each file with PII findings and recommended actions")]
[plan.add_step(step=5, server="email", description="Using the results from steps 3 and 4, draft a notification email summarizing the PII scan findings and the remediation task created")]
[plan.done()]

Meeting processing with tasks, calendar, and follow-up (meeting + task + calendar + knowledge + email):
[plan.add_step(step=1, server="meeting", description="Transcribe the audio file /Users/chintan/Recordings/standup-2026-02-19.m4a")]
[plan.add_step(step=2, server="meeting", description="Using the transcript from step 1, extract action items and commitments from the meeting")]
[plan.add_step(step=3, server="task", description="Using the action items from step 2, create a task for each commitment with the assigned person and due date")]
[plan.add_step(step=4, server="calendar", description="Using the tasks from step 3, find free time slots this week to schedule focused work blocks for the high-priority tasks")]
[plan.add_step(step=5, server="knowledge", description="Using the transcript from step 1, index the meeting notes in the knowledge base for future search")]
[plan.add_step(step=6, server="email", description="Using the action items from step 2 and tasks from step 3, draft a meeting summary email to attendees with the action items and deadlines")]
[plan.done()]

For non-tool requests:
[plan.respond(message="The answer to your question is...")]"#;

/// Call the planner model to decompose the request into steps.
async fn plan_steps(
    planner: &mut InferenceClient,
    user_message: &str,
    conversation_history: &[ChatMessage],
) -> Result<StepPlan, String> {
    let mut messages = vec![ChatMessage {
        role: Role::System,
        content: Some(PLANNER_SYSTEM_PROMPT.to_string()),
        tool_call_id: None,
        tool_calls: None,
    }];

    // Include recent conversation history for context (last 6 turns max)
    let history_window = conversation_history
        .iter()
        .filter(|m| m.role != Role::System)
        .rev()
        .take(6)
        .collect::<Vec<_>>();
    for msg in history_window.into_iter().rev() {
        messages.push(msg.clone());
    }

    messages.push(ChatMessage {
        role: Role::User,
        content: Some(user_message.to_string()),
        tool_call_id: None,
        tool_calls: None,
    });

    let sampling = SamplingOverrides {
        temperature: Some(0.1),
        top_p: Some(0.2),
    };

    let result = planner
        .chat_completion(messages, None, Some(sampling))
        .await
        .map_err(|e| format!("planner inference error: {e}"))?;

    let text = result.token.unwrap_or_default();
    let trimmed = text.trim();

    // Try bracket-format parsing first (primary for LFM2-24B-A2B)
    if let Some(plan) = parse_bracket_plan(trimmed) {
        tracing::info!("orchestrator: parsed bracket-format plan");
        return Ok(plan);
    }

    // Fall back to JSON parsing (for models that support it)
    match parse_json_plan(trimmed) {
        Ok(plan) => {
            tracing::info!("orchestrator: parsed JSON-format plan (fallback)");
            Ok(plan)
        }
        Err(json_err) => Err(format!(
            "failed to parse plan (bracket and JSON both failed)\n\
             {json_err}\n\
             Raw output: {trimmed}"
        )),
    }
}

/// Check if a plan likely under-decomposed a compound request.
///
/// If the user message contains signals of a multi-step workflow (e.g., "scan AND
/// create a task", "read file THEN create task") but the planner only produced one
/// step, we should re-plan with a stronger decomposition prompt.
fn plan_needs_decomposition(plan: &StepPlan, user_message: &str) -> bool {
    // Already multi-step or no-tool — no re-plan needed
    if plan.steps.len() > 1 || !plan.needs_tools {
        return false;
    }

    let lower = user_message.to_lowercase();

    // Explicit compound keywords (user says "do X and then do Y")
    let compound_signals = [
        " and then ",
        " and create ",
        " then create ",
        " then tell ",
        " then make ",
        " also ",
        " follow up ",
        " and scan ",
        " and a task",
        ", create a task",
        ", then ",
    ];
    for signal in &compound_signals {
        if lower.contains(signal) {
            return true;
        }
    }

    // Multi-operation pairs (user mentions two distinct action types)
    let multi_op_pairs = [
        ("scan", "create"),
        ("read", "create"),
        ("list", "create"),
        ("scan", "task"),
        ("search", "task"),
        ("extract", "task"),
        ("read", "task"),
        // Security + task combinations
        ("ssn", "task"),
        ("pii", "task"),
        ("secret", "task"),
        ("api key", "task"),
        // Scan for multiple things
        ("ssn", "api key"),
        ("pii", "secret"),
    ];
    for (a, b) in &multi_op_pairs {
        if lower.contains(a) && lower.contains(b) {
            return true;
        }
    }

    false
}

// ─── Phase 2: Execute ───────────────────────────────────────────────────────

/// Build a system prompt for the router that matches the fine-tuning training format.
///
/// The fine-tuned router was trained with tools as a numbered text list in the system
/// prompt (`generate_training_data_v2.py` lines 281-290). Sending tools via the OpenAI
/// `tools` JSON parameter causes llama-server to reformat them via its chat template,
/// which the 1.2B model has never seen — causing 0% tool call rate.
fn build_router_system_prompt(
    filtered_names: &[String],
    mcp: &McpClient,
) -> String {
    let mut tool_lines = Vec::new();
    for (i, name) in filtered_names.iter().enumerate() {
        let desc = mcp
            .registry
            .get_tool(name)
            .map(|d| d.description.clone())
            .unwrap_or_default();
        tool_lines.push(format!("{}. {} — {}", i + 1, name, desc));
    }

    format!(
        "You are LocalCowork, a desktop AI assistant that runs entirely on-device. \
         You have access to the following tools. ALWAYS call exactly one tool using \
         bracket syntax: [server.tool(param=\"value\")]. NEVER ask questions. \
         NEVER say you cannot help. ALWAYS select the most appropriate tool.\n\n\
         Available tools:\n{}",
        tool_lines.join("\n")
    )
}

/// Server-aware adaptive tool selection (Improvement I1).
///
/// Guarantees all tools from the planner's hinted server are included in the
/// candidate set. Remaining slots are filled by RAG similarity.
async fn adaptive_filter(
    tool_index: &ToolEmbeddingIndex,
    step: &PlanStep,
    router_base_url: &str,
    description: &str,
    base_k: usize,
    mcp: &McpClient,
) -> Vec<String> {
    // Start with all tools from the hinted server (if available)
    let mut selected: Vec<String> = Vec::new();
    if let Some(ref server) = step.expected_server {
        let prefix = format!("{}.", server);
        let server_tools: Vec<String> = mcp
            .registry
            .tool_name_description_pairs()
            .iter()
            .filter(|(name, _)| name.starts_with(&prefix))
            .map(|(name, _)| name.clone())
            .collect();
        selected.extend(server_tools);
    }

    // Fill remaining slots with RAG-filtered tools (excluding already-selected)
    let remaining_k = base_k.saturating_sub(selected.len());
    if remaining_k > 0 {
        // Fetch more than needed so we can deduplicate
        let fetch_k = base_k * 2;
        if let Ok((rag_names, _)) =
            tool_index.filter(router_base_url, description, fetch_k).await
        {
            for name in rag_names {
                if selected.len() >= base_k {
                    break;
                }
                if !selected.contains(&name) {
                    selected.push(name);
                }
            }
        }
    }

    selected
}

/// Last-resort extraction: if the router produced text mentioning exactly one
/// tool name but without proper bracket syntax, construct the call (Improvement I2).
fn extract_fallback_tool_call(
    response_text: &str,
    filtered_names: &[String],
) -> Option<ToolCall> {
    let mentioned: Vec<&str> = filtered_names
        .iter()
        .filter(|name| response_text.contains(name.as_str()))
        .map(|s| s.as_str())
        .collect();

    if mentioned.len() == 1 {
        let name = mentioned[0];
        // Try to extract simple path= or query= arguments from the response text
        let args = extract_inline_args(response_text);
        tracing::info!(
            tool = name,
            args_keys = ?args.as_object().map(|o| o.keys().collect::<Vec<_>>()),
            "fallback: extracted tool call from router text"
        );
        Some(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4()),
            name: name.to_string(),
            arguments: args,
        })
    } else {
        None
    }
}

/// Extract inline `key="value"` arguments from free-form text.
fn extract_inline_args(text: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();

    // Scan for key="value" patterns (common in router output)
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        // Look for '="' which signals a key="value" pair
        if i + 1 < bytes.len() && bytes[i] == b'=' && bytes[i + 1] == b'"' {
            // Walk backwards to find the key (alphanumeric + underscore)
            let eq_pos = i;
            let mut key_start = eq_pos;
            while key_start > 0
                && (bytes[key_start - 1].is_ascii_alphanumeric() || bytes[key_start - 1] == b'_')
            {
                key_start -= 1;
            }
            let key = &text[key_start..eq_pos];

            // Walk forward to find the closing quote
            let val_start = i + 2;
            if let Some(val_end_offset) = text[val_start..].find('"') {
                let val = &text[val_start..val_start + val_end_offset];
                if !key.is_empty() && !val.is_empty() {
                    map.insert(
                        key.to_string(),
                        serde_json::Value::String(val.to_string()),
                    );
                }
                i = val_start + val_end_offset + 1;
            } else {
                i += 2;
            }
        } else {
            i += 1;
        }
    }

    serde_json::Value::Object(map)
}

/// Construct tool arguments from the step description context instead of relying
/// on the router's arguments (which are often hallucinated from training data).
///
/// The 1.2B router is excellent at tool selection (100% in Phase 2c) but poor at
/// argument construction — it regurgitates memorized example paths like
/// `~/Documents/example.txt` instead of extracting the actual path from the user's
/// message. This function extracts arguments from the step description, which
/// contains the user's actual intent and specific paths/values.
fn construct_args_from_context(
    tool_name: &str,
    step_description: &str,
    mcp: &McpClient,
) -> serde_json::Value {
    let mut args = serde_json::Map::new();

    // Get the tool's parameter schema from MCP registry
    let schema = mcp.registry.get_tool(tool_name).map(|t| &t.params_schema);

    if let Some(schema) = schema {
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            for (key, _prop_schema) in props {
                if let Some(value) = extract_param_value(key, step_description) {
                    args.insert(key.clone(), value);
                }
            }
        }
    }

    serde_json::Value::Object(args)
}

/// Extract a parameter value from the step description based on the parameter name.
fn extract_param_value(
    param_name: &str,
    description: &str,
) -> Option<serde_json::Value> {
    match param_name {
        // Path-like parameters: extract file/directory paths from description
        "path" | "file_path" | "dir_path" | "directory" | "source" | "destination" => {
            extract_path_from_text(description).map(serde_json::Value::String)
        }
        // Title-like parameters: extract from quoted text or keywords
        "title" | "name" => extract_title_from_text(description).map(serde_json::Value::String),
        // Due date parameters
        "due" | "due_date" => extract_date_from_text(description).map(serde_json::Value::String),
        // Description/details parameters — use the step description itself
        "description" | "details" | "body" | "content" | "text" => {
            // Don't auto-fill content fields from step description — let the router handle it
            // or let the tool use defaults. Step description is meta-info, not content.
            None
        }
        _ => None,
    }
}

/// Extract a file/directory path from natural language text.
///
/// Priority order:
/// 1. Explicit paths (starting with `/` or `~/`)
/// 2. Backtick-quoted paths
/// 3. Well-known directory references ("Downloads folder")
pub(crate) fn extract_path_from_text(text: &str) -> Option<String> {
    // Priority 1: Backtick-quoted paths (most explicit)
    let mut search_from = 0;
    while let Some(start) = text[search_from..].find('`') {
        let abs_start = search_from + start + 1;
        if let Some(end) = text[abs_start..].find('`') {
            let content = &text[abs_start..abs_start + end];
            if content.contains('/') {
                return Some(content.to_string());
            }
            search_from = abs_start + end + 1;
        } else {
            break;
        }
    }

    // Priority 2: Absolute/home-relative paths in the text
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| {
            c == '`' || c == '\'' || c == '"' || c == ',' || c == ')'
        });
        if (clean.starts_with('/') || clean.starts_with("~/")) && clean.len() > 2 {
            return Some(clean.to_string());
        }
    }

    // Priority 3: Well-known directory references
    let lower = text.to_lowercase();
    if lower.contains("downloads folder")
        || lower.contains("downloads directory")
        || lower.contains("my downloads")
        || (lower.contains("downloads") && lower.contains("folder"))
    {
        return Some("~/Downloads".to_string());
    }
    if lower.contains("documents folder") || lower.contains("documents directory") {
        return Some("~/Documents".to_string());
    }
    if lower.contains("desktop folder")
        || lower.contains("desktop directory")
        || lower.contains("my desktop")
    {
        return Some("~/Desktop".to_string());
    }
    if lower.contains("home folder") || lower.contains("home directory") {
        return Some("~".to_string());
    }

    None
}

/// Extract a title from text — looks for quoted strings or "titled X" patterns.
fn extract_title_from_text(text: &str) -> Option<String> {
    // Look for quoted strings (single or double)
    for quote in ['"', '\''] {
        let mut search_from = 0;
        while let Some(start) = text[search_from..].find(quote) {
            let abs_start = search_from + start + 1;
            if let Some(end) = text[abs_start..].find(quote) {
                let content = &text[abs_start..abs_start + end];
                // Skip very short or very long content
                if content.len() >= 3 && content.len() <= 200 {
                    return Some(content.to_string());
                }
                search_from = abs_start + end + 1;
            } else {
                break;
            }
        }
    }

    // Look for "titled X" or "called X" pattern
    for prefix in ["titled ", "called ", "named "] {
        if let Some(idx) = text.to_lowercase().find(prefix) {
            let after = &text[idx + prefix.len()..];
            let end = after
                .find([',', '.', '\n'])
                .unwrap_or(after.len());
            let title = after[..end].trim().trim_matches('\'').trim_matches('"');
            if !title.is_empty() && title.len() <= 200 {
                return Some(title.to_string());
            }
        }
    }

    None
}

/// Extract a date reference from text.
fn extract_date_from_text(text: &str) -> Option<String> {
    let lower = text.to_lowercase();

    // Day names
    for day in [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ] {
        if lower.contains(day) {
            // Capitalize first letter for the output
            let capitalized = format!("{}{}", &day[..1].to_uppercase(), &day[1..]);
            return Some(capitalized);
        }
    }

    // Relative dates
    if lower.contains("tomorrow") {
        return Some("tomorrow".to_string());
    }
    if lower.contains("today") {
        return Some("today".to_string());
    }
    if lower.contains("next week") {
        return Some("next week".to_string());
    }
    if lower.contains("end of week") || lower.contains("end of the week") {
        return Some("Friday".to_string());
    }

    // ISO date patterns (YYYY-MM-DD)
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '-');
        if clean.len() == 10
            && clean.as_bytes().get(4) == Some(&b'-')
            && clean.as_bytes().get(7) == Some(&b'-')
        {
            return Some(clean.to_string());
        }
    }

    None
}

/// Check if a value looks like a placeholder from training data rather than a
/// real user-supplied argument. The 1.2B router often regurgitates example
/// values from its fine-tuning data instead of extracting from the user message.
fn is_placeholder_value(value: &serde_json::Value) -> bool {
    if let Some(s) = value.as_str() {
        let lower = s.to_lowercase();
        // Generic example paths from training data
        lower == "~/documents/example.txt"
            || lower == "~/documents/"
            || lower == "~/documents"
            || lower.contains("/example.txt")
            || lower.contains("/example/")
            // Literal placeholder strings
            || lower == "value"
            || lower == "content"
            || lower == "text"
            || lower == "description"
            || lower == "placeholder"
            // Schema-literal patterns (router copies from schema description)
            || lower.contains("iso 8601")
            || lower == "due date"
            || lower == "search query"
            || lower == "file path"
            || lower == "directory path"
            // Empty-ish
            || s.trim().is_empty()
    } else {
        false
    }
}

/// Merge two argument objects with smart override logic.
///
/// - **Path-like keys** (`path`, `file_path`, etc.) always prefer `primary` because
///   the router almost always hallucinates training-data paths.
/// - **Non-path keys** (`title`, `name`, etc.) only override the router's value
///   if the router produced a placeholder (detected by `is_placeholder_value`).
/// - Keys absent in `secondary` are filled from `primary`.
/// - Keys absent in `primary` are kept from `secondary`.
fn merge_args(primary: &serde_json::Value, secondary: &serde_json::Value) -> serde_json::Value {
    let mut merged = serde_json::Map::new();

    // Start with secondary (router's args — may be wrong but fills gaps)
    if let Some(obj) = secondary.as_object() {
        for (k, v) in obj {
            merged.insert(k.clone(), v.clone());
        }
    }

    // Selectively override with primary (context-extracted args)
    if let Some(obj) = primary.as_object() {
        for (k, v) in obj {
            // Only consider non-empty extracted values
            let is_meaningful = match v {
                serde_json::Value::String(s) => !s.is_empty(),
                serde_json::Value::Null => false,
                _ => true,
            };
            if !is_meaningful {
                continue;
            }

            let is_path_key = matches!(
                k.as_str(),
                "path"
                    | "file_path"
                    | "dir_path"
                    | "directory"
                    | "source"
                    | "destination"
                    | "folder"
            );

            if is_path_key {
                // Path keys: ALWAYS override — router paths are almost always wrong
                merged.insert(k.clone(), v.clone());
            } else if !merged.contains_key(k) {
                // Key not in router args: fill from context extraction
                merged.insert(k.clone(), v.clone());
            } else {
                // Key exists in router args: only override if router value is a placeholder
                let router_is_placeholder = merged
                    .get(k)
                    .map(is_placeholder_value)
                    .unwrap_or(true);
                if router_is_placeholder {
                    merged.insert(k.clone(), v.clone());
                }
                // Otherwise keep the router's value (it's probably correct)
            }
        }
    }

    serde_json::Value::Object(merged)
}

/// Execute a single plan step using the router model.
///
/// Tools are presented in the system prompt as a numbered text list matching the
/// fine-tuning training format. The router's bracket-format output is parsed by
/// `parse_non_streaming_response` → `parse_bracket_tool_calls`.
async fn execute_step(
    step: &PlanStep,
    prior_results: &[StepExecutionResult],
    router: &mut InferenceClient,
    tool_index: &ToolEmbeddingIndex,
    config: &OrchestratorConfig,
    mcp_state: &TokioMutex<McpClient>,
) -> StepExecutionResult {
    let description =
        interpolate_prior_results(step.step_number, &step.description, prior_results);

    // Adaptive tool selection: server hint → RAG fill (Improvement I1)
    let filtered_names = {
        let mcp = mcp_state.lock().await;
        adaptive_filter(
            tool_index,
            step,
            router.current_base_url(),
            &description,
            config.router_top_k as usize,
            &mcp,
        )
        .await
    };

    if filtered_names.is_empty() {
        return StepExecutionResult {
            step_number: step.step_number,
            description,
            tool_called: None,
            tool_arguments: None,
            tool_result: None,
            success: false,
            error: Some("no tools available after filtering".to_string()),
        };
    }

    // Build the system prompt matching training format (Fix F1 — the critical fix)
    let router_system = {
        let mcp = mcp_state.lock().await;
        build_router_system_prompt(&filtered_names, &mcp)
    };

    tracing::info!(
        step = step.step_number,
        filtered_tool_count = filtered_names.len(),
        filtered_tools = ?filtered_names.iter().take(5).collect::<Vec<_>>(),
        "router: step tools selected"
    );

    let sampling = SamplingOverrides {
        temperature: Some(0.1),
        top_p: Some(0.1),
    };

    let mut last_response_text = String::new();

    // Try up to step_retries times
    for attempt in 0..config.step_retries {
        let prompt = if attempt == 0 {
            description.clone()
        } else {
            // Enhanced retry with bracket format example (Fix F2)
            format!(
                "{}\n\nIMPORTANT: You MUST respond with exactly one tool call in bracket \
                 format. Example: [filesystem.list_dir(path=\"/Users/chintan/Downloads\")]\n\
                 Choose from: {}",
                description,
                filtered_names
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: Some(router_system.clone()),
                tool_call_id: None,
                tool_calls: None,
            },
            ChatMessage {
                role: Role::User,
                content: Some(prompt),
                tool_call_id: None,
                tool_calls: None,
            },
        ];

        // Pass tools: None — tools are in the system prompt, not the API parameter.
        // parse_non_streaming_response() handles bracket-format extraction from text.
        let result = match router.chat_completion(messages, None, Some(sampling)).await {
            Ok(chunk) => chunk,
            Err(e) => {
                tracing::warn!(
                    step = step.step_number,
                    attempt = attempt,
                    error = %e,
                    "router inference error"
                );
                continue;
            }
        };

        // Diagnostic logging (Fix F5)
        let response_text = result.token.as_deref().unwrap_or("<empty>");
        let has_native_tool_calls = result.tool_calls.is_some();
        tracing::info!(
            step = step.step_number,
            attempt = attempt,
            response_text_len = response_text.len(),
            response_text_preview = %truncate_utf8(response_text, 200),
            has_native_tool_calls,
            "router raw response"
        );
        last_response_text = response_text.to_string();

        // Check for tool calls in the response (bracket-parsed or native)
        if let Some(ref tool_calls) = result.tool_calls {
            if let Some(tc) = tool_calls.first() {
                // Override router's args with context-extracted args (Fix F6).
                // The 1.2B router is great at tool selection but hallucinates
                // arguments from training data. Extract real args from the
                // step description which contains the user's actual paths/values.
                let overridden_args = {
                    let mcp_ref = mcp_state.lock().await;
                    let context_args =
                        construct_args_from_context(&tc.name, &description, &mcp_ref);
                    merge_args(&context_args, &tc.arguments)
                };

                tracing::info!(
                    step = step.step_number,
                    tool = %tc.name,
                    router_args = %tc.arguments,
                    final_args = %overridden_args,
                    "router selected tool"
                );

                // Execute the tool via MCP with overridden args
                let mut mcp = mcp_state.lock().await;
                let tool_result =
                    match mcp.call_tool(&tc.name, overridden_args.clone()).await {
                        Ok(res) if res.success => {
                            let text = res
                                .result
                                .and_then(|v| {
                                    v.get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|s| s.to_string())
                                        .or_else(|| serde_json::to_string(&v).ok())
                                })
                                .unwrap_or_else(|| "ok".to_string());
                            tracing::info!(
                                step = step.step_number,
                                tool = %tc.name,
                                result_len = text.len(),
                                result_preview = %truncate_utf8(&text, 200),
                                "step tool execution succeeded"
                            );
                            text
                        }
                        Ok(res) => {
                            let err =
                                res.error.unwrap_or_else(|| "tool failed".to_string());
                            tracing::warn!(
                                step = step.step_number,
                                tool = %tc.name,
                                error = %err,
                                "step tool execution failed"
                            );
                            err
                        }
                        Err(e) => {
                            tracing::warn!(
                                step = step.step_number,
                                tool = %tc.name,
                                error = %e,
                                "step tool MCP error"
                            );
                            format!("MCP error: {e}")
                        }
                    };

                return StepExecutionResult {
                    step_number: step.step_number,
                    description: description.clone(),
                    tool_called: Some(tc.name.clone()),
                    tool_arguments: Some(overridden_args),
                    tool_result: Some(tool_result),
                    success: true,
                    error: None,
                };
            }
        }

        tracing::info!(
            step = step.step_number,
            attempt = attempt,
            "router returned no tool call — retrying"
        );
    }

    // All retries exhausted — try fallback extraction (Improvement I2)
    if !last_response_text.is_empty() {
        if let Some(tc) = extract_fallback_tool_call(&last_response_text, &filtered_names) {
            // Apply same argument override as normal path (Fix F6)
            let overridden_args = {
                let mcp_ref = mcp_state.lock().await;
                let context_args =
                    construct_args_from_context(&tc.name, &description, &mcp_ref);
                merge_args(&context_args, &tc.arguments)
            };

            tracing::info!(
                step = step.step_number,
                tool = %tc.name,
                final_args = %overridden_args,
                "router fallback: extracted tool from text"
            );
            let mut mcp = mcp_state.lock().await;
            let tool_result = match mcp.call_tool(&tc.name, overridden_args.clone()).await {
                Ok(res) if res.success => {
                    let text = res
                        .result
                        .and_then(|v| {
                            v.get("text")
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string())
                                .or_else(|| serde_json::to_string(&v).ok())
                        })
                        .unwrap_or_else(|| "ok".to_string());
                    tracing::info!(
                        step = step.step_number,
                        tool = %tc.name,
                        result_len = text.len(),
                        result_preview = %truncate_utf8(&text, 200),
                        "fallback tool execution succeeded"
                    );
                    text
                }
                Ok(res) => res.error.unwrap_or_else(|| "tool failed".to_string()),
                Err(e) => format!("MCP error: {e}"),
            };

            return StepExecutionResult {
                step_number: step.step_number,
                description: description.clone(),
                tool_called: Some(tc.name.clone()),
                tool_arguments: Some(overridden_args),
                tool_result: Some(tool_result),
                success: true,
                error: None,
            };
        }
    }

    StepExecutionResult {
        step_number: step.step_number,
        description,
        tool_called: None,
        tool_arguments: None,
        tool_result: None,
        success: false,
        error: Some(format!(
            "router failed to produce a tool call after {} attempts",
            config.step_retries
        )),
    }
}

/// Condense a step execution result into a 1-2 line summary.
///
/// Extracts the key information (tool name, outcome, key data) rather than
/// dumping the full result text. Keeps the router's context clean and focused.
fn condense_step_result(result: &StepExecutionResult) -> String {
    let tool = result.tool_called.as_deref().unwrap_or("unknown");

    match &result.tool_result {
        Some(text) if result.success => {
            let summary = if text.len() <= 200 {
                text.clone()
            } else {
                format!("{}... ({} chars total)", truncate_utf8(text, 150), text.len())
            };
            format!("Step {} ({}) succeeded: {}", result.step_number, tool, summary)
        }
        _ if !result.success => {
            let err = result.error.as_deref().unwrap_or("unknown error");
            format!("Step {} ({}) failed: {}", result.step_number, tool, err)
        }
        _ => {
            format!("Step {} ({}) succeeded", result.step_number, tool)
        }
    }
}

/// Enhance the step description with prior step results (M3).
///
/// Three forwarding mechanisms:
/// 1. **Immediate predecessor**: Step N always gets step N-1's condensed result,
///    regardless of whether the description references it explicitly.
/// 2. **Explicit references**: If the description mentions "step M", that step's
///    condensed result is also included (preserving existing behavior).
/// 3. **Deduplication**: Each step's result appears at most once in the context block.
fn interpolate_prior_results(
    step_number: u32,
    description: &str,
    prior_results: &[StepExecutionResult],
) -> String {
    if prior_results.is_empty() {
        return description.to_string();
    }

    let mut context_lines: Vec<String> = Vec::new();
    let mut included_steps: Vec<u32> = Vec::new();

    // 1. Always include the immediately preceding step's result
    if let Some(prev) = prior_results.iter().rfind(|r| r.step_number == step_number - 1) {
        if prev.success {
            context_lines.push(condense_step_result(prev));
            included_steps.push(prev.step_number);
        }
    }

    // 2. Include any explicitly referenced steps (e.g., "step 2" in step 5's description)
    let lower_desc = description.to_lowercase();
    for prior in prior_results {
        if !prior.success || included_steps.contains(&prior.step_number) {
            continue;
        }
        let step_ref = format!("step {}", prior.step_number);
        if lower_desc.contains(&step_ref) {
            context_lines.push(condense_step_result(prior));
            included_steps.push(prior.step_number);
        }
    }

    // 3. Build enhanced description with a clean [Prior step context] block
    if context_lines.is_empty() {
        description.to_string()
    } else {
        format!(
            "{}\n\n[Prior step context]:\n{}",
            description,
            context_lines.join("\n")
        )
    }
}

// ─── Phase 3: Synthesize ────────────────────────────────────────────────────

/// Generate a user-facing summary from accumulated step results.
async fn synthesize_response(
    planner: &mut InferenceClient,
    user_message: &str,
    step_results: &[StepExecutionResult],
    app_handle: &tauri::AppHandle,
) -> Result<String, String> {
    let results_summary: String = step_results
        .iter()
        .map(|r| {
            if r.success {
                format!(
                    "Step {}: {} → {} → {}",
                    r.step_number,
                    r.description,
                    r.tool_called.as_deref().unwrap_or("none"),
                    r.tool_result
                        .as_deref()
                        .map(|s| truncate_utf8(s, 500))
                        .unwrap_or("ok")
                )
            } else {
                format!(
                    "Step {}: {} → FAILED: {}",
                    r.step_number,
                    r.description,
                    r.error.as_deref().unwrap_or("unknown")
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let synthesis_prompt = format!(
        "The user asked: \"{user_message}\"\n\n\
         The following tool actions were executed:\n{results_summary}\n\n\
         Provide a clear, helpful summary of what was done and the results. \
         Be concise. Summarize ONLY the results that succeeded. For failed steps, \
         honestly report that the action could not be completed. NEVER fabricate results."
    );

    let messages = vec![
        ChatMessage {
            role: Role::System,
            content: Some(
                "You are LocalCowork, an on-device AI assistant. Summarize the tool \
                 results for the user. Be clear and concise. Only report what actually \
                 happened."
                    .to_string(),
            ),
            tool_call_id: None,
            tool_calls: None,
        },
        ChatMessage {
            role: Role::User,
            content: Some(synthesis_prompt),
            tool_call_id: None,
            tool_calls: None,
        },
    ];

    let sampling = SamplingOverrides {
        temperature: Some(0.7),
        top_p: Some(0.9),
    };

    // Stream the synthesis to the frontend
    use futures::StreamExt;

    let stream = planner
        .chat_completion_stream(messages, None, Some(sampling))
        .await
        .map_err(|e| format!("synthesis streaming error: {e}"))?;

    futures::pin_mut!(stream);

    let mut full_text = String::new();
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                if let Some(ref token) = chunk.token {
                    full_text.push_str(token);
                    let _ = tauri::Emitter::emit(app_handle, "stream-token", token);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "synthesis stream error");
                break;
            }
        }
    }

    if full_text.is_empty() {
        return Err("synthesis produced empty response".to_string());
    }

    Ok(full_text)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_no_prior_results() {
        let desc = "List files in /tmp";
        let result = interpolate_prior_results(1, desc, &[]);
        assert_eq!(result, desc);
    }

    #[test]
    fn interpolate_with_explicit_step_reference() {
        let desc = "Using the result from step 1, extract text from image";
        let prior = vec![StepExecutionResult {
            step_number: 1,
            description: "List files".into(),
            tool_called: Some("filesystem.list_dir".into()),
            tool_arguments: None,
            tool_result: Some("[\"file1.png\", \"file2.png\"]".into()),
            success: true,
            error: None,
        }];
        let result = interpolate_prior_results(2, desc, &prior);
        assert!(result.contains("[Prior step context]"));
        assert!(result.contains("file1.png"));
    }

    #[test]
    fn interpolate_skips_failed_steps() {
        let desc = "Using the result from step 1, continue";
        let prior = vec![StepExecutionResult {
            step_number: 1,
            description: "Failed step".into(),
            tool_called: None,
            tool_arguments: None,
            tool_result: None,
            success: false,
            error: Some("timeout".into()),
        }];
        let result = interpolate_prior_results(2, desc, &prior);
        assert!(!result.contains("[Prior step context]"));
    }

    #[test]
    fn interpolate_always_forwards_predecessor() {
        // Step 2 should get step 1's result even without explicit "step 1" reference
        let desc = "Extract text from the document";
        let prior = vec![StepExecutionResult {
            step_number: 1,
            description: "List files".into(),
            tool_called: Some("filesystem.list_dir".into()),
            tool_arguments: None,
            tool_result: Some("[\"report.pdf\", \"notes.txt\"]".into()),
            success: true,
            error: None,
        }];
        let result = interpolate_prior_results(2, desc, &prior);
        assert!(result.contains("[Prior step context]"), "should include context block");
        assert!(result.contains("report.pdf"), "should include predecessor result");
    }

    #[test]
    fn interpolate_deduplicates_predecessor_and_explicit() {
        // Step 2 references "step 1" explicitly, and step 1 is also the predecessor.
        // The result should appear only ONCE.
        let desc = "Using the result from step 1, extract text";
        let prior = vec![StepExecutionResult {
            step_number: 1,
            description: "List files".into(),
            tool_called: Some("filesystem.list_dir".into()),
            tool_arguments: None,
            tool_result: Some("[\"a.txt\"]".into()),
            success: true,
            error: None,
        }];
        let result = interpolate_prior_results(2, desc, &prior);
        let context_count = result.matches("filesystem.list_dir").count();
        assert_eq!(context_count, 1, "predecessor + explicit should not duplicate");
    }

    #[test]
    fn interpolate_includes_explicit_and_predecessor() {
        // Step 3 references "step 1" explicitly; step 2 is the predecessor.
        // Both should appear.
        let desc = "Using the results from step 1, create a task";
        let prior = vec![
            StepExecutionResult {
                step_number: 1,
                description: "Scan files".into(),
                tool_called: Some("security.scan_for_pii".into()),
                tool_arguments: None,
                tool_result: Some("Found 3 files with SSNs".into()),
                success: true,
                error: None,
            },
            StepExecutionResult {
                step_number: 2,
                description: "Scan secrets".into(),
                tool_called: Some("security.scan_for_secrets".into()),
                tool_arguments: None,
                tool_result: Some("Found 1 API key".into()),
                success: true,
                error: None,
            },
        ];
        let result = interpolate_prior_results(3, desc, &prior);
        assert!(result.contains("Found 1 API key"), "should include step 2 (predecessor)");
        assert!(result.contains("Found 3 files"), "should include step 1 (referenced)");
    }

    #[test]
    fn interpolate_skips_failed_predecessor() {
        let desc = "Continue processing";
        let prior = vec![StepExecutionResult {
            step_number: 1,
            description: "Failed step".into(),
            tool_called: None,
            tool_arguments: None,
            tool_result: None,
            success: false,
            error: Some("timeout".into()),
        }];
        let result = interpolate_prior_results(2, desc, &prior);
        assert!(!result.contains("[Prior step context]"));
    }

    #[test]
    fn condense_short_result_includes_full_text() {
        let step = StepExecutionResult {
            step_number: 1,
            description: "List files".into(),
            tool_called: Some("filesystem.list_dir".into()),
            tool_arguments: None,
            tool_result: Some("[\"a.txt\", \"b.pdf\"]".into()),
            success: true,
            error: None,
        };
        let condensed = condense_step_result(&step);
        assert!(condensed.contains("filesystem.list_dir"));
        assert!(condensed.contains("succeeded"));
        assert!(condensed.contains("a.txt"));
    }

    #[test]
    fn condense_long_result_truncates() {
        let long_text = "x".repeat(500);
        let step = StepExecutionResult {
            step_number: 2,
            description: "Extract text".into(),
            tool_called: Some("document.extract_text".into()),
            tool_arguments: None,
            tool_result: Some(long_text),
            success: true,
            error: None,
        };
        let condensed = condense_step_result(&step);
        assert!(condensed.len() < 300, "condensed should be much shorter than 500");
        assert!(condensed.contains("500 chars total"));
    }

    #[test]
    fn condense_failed_result() {
        let step = StepExecutionResult {
            step_number: 3,
            description: "Scan PII".into(),
            tool_called: Some("security.scan_for_pii".into()),
            tool_arguments: None,
            tool_result: None,
            success: false,
            error: Some("file not found".into()),
        };
        let condensed = condense_step_result(&step);
        assert!(condensed.contains("failed"));
        assert!(condensed.contains("file not found"));
    }

    #[test]
    fn extract_inline_args_basic() {
        let text = r#"I'll call filesystem.list_dir with path="/Users/chintan/Downloads""#;
        let args = extract_inline_args(text);
        assert_eq!(
            args.get("path").and_then(|v| v.as_str()),
            Some("/Users/chintan/Downloads")
        );
    }

    #[test]
    fn extract_inline_args_multiple() {
        let text = r#"query="find documents" path="/tmp/data""#;
        let args = extract_inline_args(text);
        assert_eq!(
            args.get("query").and_then(|v| v.as_str()),
            Some("find documents")
        );
        assert_eq!(
            args.get("path").and_then(|v| v.as_str()),
            Some("/tmp/data")
        );
    }

    #[test]
    fn extract_inline_args_empty_when_no_patterns() {
        let text = "Just some plain text without any arguments";
        let args = extract_inline_args(text);
        assert!(args.as_object().unwrap().is_empty());
    }

    #[test]
    fn extract_fallback_single_mention() {
        let names = vec![
            "filesystem.list_dir".to_string(),
            "filesystem.read_file".to_string(),
            "document.extract_text".to_string(),
        ];
        let text = "I would use filesystem.list_dir to list the Downloads folder";
        let result = extract_fallback_tool_call(text, &names);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "filesystem.list_dir");
    }

    #[test]
    fn extract_fallback_multiple_mentions_returns_none() {
        let names = vec![
            "filesystem.list_dir".to_string(),
            "filesystem.read_file".to_string(),
        ];
        let text = "I could use filesystem.list_dir or filesystem.read_file";
        let result = extract_fallback_tool_call(text, &names);
        assert!(result.is_none());
    }

    #[test]
    fn extract_fallback_no_mention_returns_none() {
        let names = vec!["filesystem.list_dir".to_string()];
        let text = "I don't know which tool to use";
        let result = extract_fallback_tool_call(text, &names);
        assert!(result.is_none());
    }

    // ─── Fix F6: Argument extraction tests ──────────────────────────────

    #[test]
    fn extract_path_absolute() {
        let text = "List all files in /Users/chintan/Downloads";
        assert_eq!(
            extract_path_from_text(text),
            Some("/Users/chintan/Downloads".to_string())
        );
    }

    #[test]
    fn extract_path_tilde() {
        let text = "Read ~/Projects/localCoWork/README.md";
        assert_eq!(
            extract_path_from_text(text),
            Some("~/Projects/localCoWork/README.md".to_string())
        );
    }

    #[test]
    fn extract_path_backtick() {
        let text = "Scan files in `tests/fixtures/uc3/sample_files/` for PII";
        assert_eq!(
            extract_path_from_text(text),
            Some("tests/fixtures/uc3/sample_files/".to_string())
        );
    }

    #[test]
    fn extract_path_downloads_folder() {
        let text = "What files are in my Downloads folder?";
        assert_eq!(
            extract_path_from_text(text),
            Some("~/Downloads".to_string())
        );
    }

    #[test]
    fn extract_path_desktop() {
        let text = "Show me what's on my Desktop";
        assert_eq!(
            extract_path_from_text(text),
            Some("~/Desktop".to_string())
        );
    }

    #[test]
    fn extract_path_none_when_no_path() {
        let text = "Create a new task to review the report";
        assert_eq!(extract_path_from_text(text), None);
    }

    #[test]
    fn extract_title_quoted() {
        let text = "Create a task titled 'Review Q4 numbers' with due date Friday";
        assert_eq!(
            extract_title_from_text(text),
            Some("Review Q4 numbers".to_string())
        );
    }

    #[test]
    fn extract_title_double_quoted() {
        let text = r#"Create a task titled "Fix the login bug" by tomorrow"#;
        assert_eq!(
            extract_title_from_text(text),
            Some("Fix the login bug".to_string())
        );
    }

    #[test]
    fn extract_date_friday() {
        let text = "Review Q4 numbers by Friday";
        assert_eq!(
            extract_date_from_text(text),
            Some("Friday".to_string())
        );
    }

    #[test]
    fn extract_date_tomorrow() {
        let text = "Follow up on this tomorrow";
        assert_eq!(
            extract_date_from_text(text),
            Some("tomorrow".to_string())
        );
    }

    #[test]
    fn extract_date_iso() {
        let text = "Schedule the meeting for 2026-03-15";
        assert_eq!(
            extract_date_from_text(text),
            Some("2026-03-15".to_string())
        );
    }

    #[test]
    fn merge_args_path_always_overrides() {
        // Path keys always prefer primary (context-extracted) over secondary (router)
        let primary = serde_json::json!({"path": "~/Downloads"});
        let secondary = serde_json::json!({"path": "~/Documents/example.txt", "recursive": true});
        let merged = merge_args(&primary, &secondary);
        assert_eq!(merged.get("path").unwrap(), "~/Downloads");
        assert_eq!(merged.get("recursive").unwrap(), true);
    }

    #[test]
    fn merge_args_empty_primary_keeps_secondary() {
        let primary = serde_json::json!({});
        let secondary = serde_json::json!({"path": "/tmp/file.txt"});
        let merged = merge_args(&primary, &secondary);
        assert_eq!(merged.get("path").unwrap(), "/tmp/file.txt");
    }

    #[test]
    fn merge_args_null_primary_keeps_secondary() {
        let primary = serde_json::json!({"path": null});
        let secondary = serde_json::json!({"path": "/tmp/file.txt"});
        let merged = merge_args(&primary, &secondary);
        assert_eq!(merged.get("path").unwrap(), "/tmp/file.txt");
    }

    #[test]
    fn merge_args_preserves_good_router_title() {
        // Router produced a real title (not a placeholder) — keep it
        let primary = serde_json::json!({"title": "Some extracted title"});
        let secondary = serde_json::json!({"title": "Review Q4 numbers"});
        let merged = merge_args(&primary, &secondary);
        // Router's title "Review Q4 numbers" is not a placeholder, so it should be kept
        assert_eq!(merged.get("title").unwrap(), "Review Q4 numbers");
    }

    #[test]
    fn merge_args_overrides_placeholder_title() {
        // Router produced a placeholder title — override with context-extracted
        let primary = serde_json::json!({"title": "Follow up on sensitive files"});
        let secondary = serde_json::json!({"title": "content"});
        let merged = merge_args(&primary, &secondary);
        assert_eq!(merged.get("title").unwrap(), "Follow up on sensitive files");
    }

    #[test]
    fn merge_args_overrides_placeholder_due_date() {
        // Router produced a schema-literal due date — override
        let primary = serde_json::json!({"due_date": "Friday"});
        let secondary = serde_json::json!({"due_date": "Due date (ISO 8601)"});
        let merged = merge_args(&primary, &secondary);
        assert_eq!(merged.get("due_date").unwrap(), "Friday");
    }

    #[test]
    fn merge_args_fills_missing_router_keys() {
        // Router didn't produce a due_date — fill from primary
        let primary = serde_json::json!({"due_date": "Friday"});
        let secondary = serde_json::json!({"title": "Review Q4 numbers"});
        let merged = merge_args(&primary, &secondary);
        assert_eq!(merged.get("title").unwrap(), "Review Q4 numbers");
        assert_eq!(merged.get("due_date").unwrap(), "Friday");
    }

    // ─── Fix F10: Placeholder detection tests ───────────────────────────

    #[test]
    fn placeholder_detects_example_path() {
        assert!(is_placeholder_value(&serde_json::json!(
            "~/Documents/example.txt"
        )));
    }

    #[test]
    fn placeholder_detects_schema_literal() {
        assert!(is_placeholder_value(&serde_json::json!(
            "Due date (ISO 8601)"
        )));
    }

    #[test]
    fn placeholder_detects_generic_words() {
        assert!(is_placeholder_value(&serde_json::json!("content")));
        assert!(is_placeholder_value(&serde_json::json!("value")));
        assert!(is_placeholder_value(&serde_json::json!("description")));
    }

    #[test]
    fn placeholder_rejects_real_values() {
        assert!(!is_placeholder_value(&serde_json::json!(
            "Review Q4 numbers"
        )));
        assert!(!is_placeholder_value(&serde_json::json!("Friday")));
        assert!(!is_placeholder_value(&serde_json::json!(
            "/Users/chintan/Downloads"
        )));
    }

    #[test]
    fn placeholder_detects_empty_string() {
        assert!(is_placeholder_value(&serde_json::json!("")));
        assert!(is_placeholder_value(&serde_json::json!("  ")));
    }

    // ─── Fix F11: Plan decomposition check tests ────────────────────────

    fn single_step_plan() -> StepPlan {
        StepPlan {
            needs_tools: true,
            direct_response: None,
            steps: vec![PlanStep {
                step_number: 1,
                description: "Do something".into(),
                expected_server: Some("security".into()),
                hint_params: None,
            }],
        }
    }

    fn multi_step_plan() -> StepPlan {
        StepPlan {
            needs_tools: true,
            direct_response: None,
            steps: vec![
                PlanStep {
                    step_number: 1,
                    description: "Scan files".into(),
                    expected_server: Some("security".into()),
                    hint_params: None,
                },
                PlanStep {
                    step_number: 2,
                    description: "Create task".into(),
                    expected_server: Some("task".into()),
                    hint_params: None,
                },
            ],
        }
    }

    #[test]
    fn decomp_triggers_on_scan_and_task() {
        let plan = single_step_plan();
        let msg = "Scan files for SSNs and API keys, then create a task to follow up";
        assert!(plan_needs_decomposition(&plan, msg));
    }

    #[test]
    fn decomp_triggers_on_read_and_create() {
        let plan = single_step_plan();
        let msg = "Read the quarterly report and create a task to review the numbers";
        assert!(plan_needs_decomposition(&plan, msg));
    }

    #[test]
    fn decomp_triggers_on_ssn_and_api_key() {
        let plan = single_step_plan();
        let msg = "Scan for SSN and API key issues in my documents";
        assert!(plan_needs_decomposition(&plan, msg));
    }

    #[test]
    fn decomp_skips_already_multi_step() {
        let plan = multi_step_plan();
        let msg = "Scan files and create a task";
        assert!(!plan_needs_decomposition(&plan, msg));
    }

    #[test]
    fn decomp_skips_simple_request() {
        let plan = single_step_plan();
        let msg = "What files are in my Downloads folder?";
        assert!(!plan_needs_decomposition(&plan, msg));
    }

    #[test]
    fn decomp_skips_no_tools_plan() {
        let plan = StepPlan {
            needs_tools: false,
            direct_response: Some("Hello!".into()),
            steps: vec![],
        };
        let msg = "Hello, how are you?";
        assert!(!plan_needs_decomposition(&plan, msg));
    }
}
