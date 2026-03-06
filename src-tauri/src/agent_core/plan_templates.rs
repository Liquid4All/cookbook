//! Template-based plan decomposition for known use case patterns.
//!
//! Before calling the planner model, the orchestrator checks if the user's
//! message matches a known use case pattern (from PRD UC-1 through UC-10).
//! If matched, a pre-built `StepPlan` is returned directly, bypassing the
//! planner model call entirely. This guarantees correct multi-step decomposition
//! for common workflows and saves ~2-3s of planner latency.

use crate::agent_core::orchestrator::{PlanStep, StepPlan};

/// Attempt to match the user's message against known use case templates.
///
/// Returns `Some(StepPlan)` if a high-confidence match is found, `None` otherwise.
/// The caller should fall through to the planner model when `None` is returned.
///
/// Match confidence requires at least 3 signal keyword groups for multi-step
/// templates to avoid false positives on simple requests.
pub fn try_template_match(user_message: &str) -> Option<StepPlan> {
    let lower = user_message.to_lowercase();

    // UC-4: Download triage (5 steps) — checked before UC-1 because UC-4's
    // "download" keyword is more specific; without this ordering, generic
    // file-management messages can false-positive into UC-1.
    let uc4_score = keyword_score(&lower, &[
        &["download"],
        &["organize", "classify", "sort", "clean up", "triage"],
        &["move", "file", "rename"],
        &["pii", "sensitive", "scan", "security"],
        &["task", "follow up", "remediat"],
    ]);
    if uc4_score >= 3 {
        tracing::info!(score = uc4_score, "template match: UC-4 download triage");
        return Some(build_uc4_download_triage_template(user_message));
    }

    // UC-1: Receipt reconciliation (4 steps)
    let uc1_score = keyword_score(&lower, &[
        &["receipt", "invoice", "expense"],
        &["folder", "directory", "files in"],
        &["organize", "reconcil", "spreadsheet", "csv", "categoriz"],
        &["scan", "extract", "ocr"],
    ]);
    if uc1_score >= 3 {
        tracing::info!(score = uc1_score, "template match: UC-1 receipt reconciliation");
        return Some(build_uc1_receipt_template(user_message));
    }

    // UC-7: Contract copilot (3 steps)
    let uc7_score = keyword_score(&lower, &[
        &["contract", "nda", "agreement", "legal"],
        &["compare", "diff", "review", "analyz"],
        &["email", "draft", "send", "counsel"],
    ]);
    if uc7_score >= 3 {
        tracing::info!(score = uc7_score, "template match: UC-7 contract copilot");
        return Some(build_uc7_contract_copilot_template(user_message));
    }

    None
}

/// Score a message against keyword groups.
///
/// Each group is a set of synonymous terms. A group is "matched" if ANY term
/// in it appears in the message. Returns the count of matched groups.
fn keyword_score(lower_message: &str, groups: &[&[&str]]) -> usize {
    groups
        .iter()
        .filter(|group| group.iter().any(|kw| lower_message.contains(kw)))
        .count()
}

/// Extract a file/directory path hint from the user message.
///
/// Delegates to the orchestrator's existing path extraction logic.
fn extract_path_hint(user_message: &str) -> Option<String> {
    crate::agent_core::orchestrator::extract_path_from_text(user_message)
}

// ─── Template Builders ──────────────────────────────────────────────────────

/// UC-1: Receipt Reconciliation — list → extract → write CSV → create task.
fn build_uc1_receipt_template(user_message: &str) -> StepPlan {
    let path = extract_path_hint(user_message).unwrap_or_else(|| "~/Downloads".to_string());
    StepPlan {
        needs_tools: true,
        direct_response: None,
        steps: vec![
            PlanStep {
                step_number: 1,
                description: format!(
                    "List all files in {path} to find receipts, invoices, and expense documents"
                ),
                expected_server: Some("filesystem".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 2,
                description: "Using the result from step 1, extract text from each receipt \
                    or invoice file (OCR for images, text extraction for PDFs)"
                    .to_string(),
                expected_server: Some("document".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 3,
                description: "Using the extracted text from step 2, write the structured \
                    receipt data (vendor, date, amount, category) to a CSV spreadsheet"
                    .to_string(),
                expected_server: Some("data".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 4,
                description: "Using the results from step 3, create a follow-up task to \
                    review the reconciled receipts and flag any anomalies"
                    .to_string(),
                expected_server: Some("task".to_string()),
                hint_params: None,
            },
        ],
    }
}

/// UC-4: Download Triage — list → extract → scan PII → move → create task.
fn build_uc4_download_triage_template(user_message: &str) -> StepPlan {
    let path = extract_path_hint(user_message).unwrap_or_else(|| "~/Downloads".to_string());
    StepPlan {
        needs_tools: true,
        direct_response: None,
        steps: vec![
            PlanStep {
                step_number: 1,
                description: format!(
                    "List all files in {path} to identify what needs to be triaged"
                ),
                expected_server: Some("filesystem".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 2,
                description: "Using the result from step 1, extract text from document files \
                    (PDFs, DOCX) to understand their content for classification"
                    .to_string(),
                expected_server: Some("document".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 3,
                description: "Using the result from step 1, scan all files for PII \
                    (SSNs, credit card numbers) and secrets (API keys, passwords)"
                    .to_string(),
                expected_server: Some("security".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 4,
                description: format!(
                    "Using the results from steps 2 and 3, move files from {path} \
                     to appropriate categorized folders"
                ),
                expected_server: Some("filesystem".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 5,
                description: "Using the results from steps 3 and 4, create a remediation \
                    task for any files with PII or security findings"
                    .to_string(),
                expected_server: Some("task".to_string()),
                hint_params: None,
            },
        ],
    }
}

/// UC-7: Contract Copilot — extract text → search knowledge → draft email.
fn build_uc7_contract_copilot_template(user_message: &str) -> StepPlan {
    let _path = extract_path_hint(user_message);
    StepPlan {
        needs_tools: true,
        direct_response: None,
        steps: vec![
            PlanStep {
                step_number: 1,
                description: "Extract text from the contract or NDA document provided"
                    .to_string(),
                expected_server: Some("document".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 2,
                description: "Using the extracted text from step 1, search the knowledge \
                    base for similar clauses or related contract precedents"
                    .to_string(),
                expected_server: Some("knowledge".to_string()),
                hint_params: None,
            },
            PlanStep {
                step_number: 3,
                description: "Using the analysis from steps 1 and 2, draft an email \
                    summarizing the key findings, risk flags, and recommendations"
                    .to_string(),
                expected_server: Some("email".to_string()),
                hint_params: None,
            },
        ],
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uc1_receipt_match() {
        let msg = "Scan and organize the receipts in my ~/Documents/Expenses folder \
                   and extract the data into a CSV spreadsheet";
        let result = try_template_match(msg);
        assert!(result.is_some(), "should match UC-1");
        let plan = result.unwrap();
        assert_eq!(plan.steps.len(), 4);
        assert_eq!(plan.steps[0].expected_server.as_deref(), Some("filesystem"));
        assert_eq!(plan.steps[1].expected_server.as_deref(), Some("document"));
        assert_eq!(plan.steps[2].expected_server.as_deref(), Some("data"));
        assert_eq!(plan.steps[3].expected_server.as_deref(), Some("task"));
    }

    #[test]
    fn uc4_download_triage_match() {
        let msg = "Triage my downloads folder — organize files, scan for sensitive PII, \
                   move them to the right place, and create a task for anything flagged";
        let result = try_template_match(msg);
        assert!(result.is_some(), "should match UC-4");
        let plan = result.unwrap();
        assert_eq!(plan.steps.len(), 5);
        assert_eq!(plan.steps[0].expected_server.as_deref(), Some("filesystem"));
        assert_eq!(plan.steps[2].expected_server.as_deref(), Some("security"));
        assert_eq!(plan.steps[4].expected_server.as_deref(), Some("task"));
    }

    #[test]
    fn uc7_contract_copilot_match() {
        let msg = "Review the NDA contract, compare it against our legal knowledge base, \
                   and draft an email to counsel with your analysis";
        let result = try_template_match(msg);
        assert!(result.is_some(), "should match UC-7");
        let plan = result.unwrap();
        assert_eq!(plan.steps.len(), 3);
        assert_eq!(plan.steps[0].expected_server.as_deref(), Some("document"));
        assert_eq!(plan.steps[1].expected_server.as_deref(), Some("knowledge"));
        assert_eq!(plan.steps[2].expected_server.as_deref(), Some("email"));
    }

    #[test]
    fn no_match_simple_request() {
        let msg = "List my Downloads folder";
        assert!(try_template_match(msg).is_none(), "simple request should not match");
    }

    #[test]
    fn no_match_partial_keywords() {
        let msg = "Organize my files";
        assert!(try_template_match(msg).is_none(), "only 1-2 keyword groups should not match");
    }

    #[test]
    fn uc1_extracts_path() {
        let msg = "Scan the receipts in /Users/chintan/Expenses and extract to CSV";
        let result = try_template_match(msg);
        assert!(result.is_some());
        let plan = result.unwrap();
        assert!(
            plan.steps[0].description.contains("/Users/chintan/Expenses"),
            "should extract path from message"
        );
    }

    #[test]
    fn keyword_score_all_groups_match() {
        let score = keyword_score(
            "scan receipts in folder and organize into csv",
            &[
                &["receipt", "invoice"],
                &["folder", "directory"],
                &["organize", "csv"],
            ],
        );
        assert_eq!(score, 3);
    }

    #[test]
    fn keyword_score_partial_match() {
        let score = keyword_score(
            "organize my files",
            &[
                &["receipt", "invoice"],
                &["folder", "directory"],
                &["organize", "csv"],
            ],
        );
        assert_eq!(score, 1);
    }
}
