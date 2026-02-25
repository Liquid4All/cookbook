//! Response analysis for the agent loop.
//!
//! Detects incomplete tasks, conversational deflection (FM-3), and completion
//! summaries in model text responses. Used by the agent loop in `commands/chat.rs`
//! to decide whether to inject continuation prompts or exit.

/// Detect if a model's text response indicates an incomplete task.
///
/// A local model often "fatigues" mid-task and produces a partial summary
/// after processing 3-4 items out of 7+. This function looks for signals that
/// the model stopped before finishing: mentions of remaining files, "next" steps,
/// partial progress reports, etc.
///
/// Returns `true` if the response suggests the task is unfinished and the model
/// should be prompted to continue.
pub fn is_incomplete_response(text: &str) -> bool {
    let lower = text.to_lowercase();

    // Patterns that indicate the model is reporting incomplete progress
    let incomplete_signals = [
        "remaining",
        "left to process",
        "still need to",
        "continue with",
        "next file",
        "next screenshot",
        "more files",
        "not yet processed",
        "will process",
        "haven't processed",
        "need to rename",
        "files left",
        "let me continue",
        "i'll continue",
        "proceeding to",
        "moving on to",
    ];

    if is_completion_summary(&lower) {
        return false;
    }

    for signal in &incomplete_signals {
        if lower.contains(signal) {
            return true;
        }
    }

    false
}

/// Detect if a model's text response is a conversational deflection (FM-3).
///
/// Called AFTER [`is_incomplete_response`] returns `false`. Identifies cases
/// where the model received tool results but responds with a question or
/// narration instead of calling the next tool.
///
/// Three detection layers:
/// - **Result presentation guard**: text presenting tool results without
///   a question is NOT deflection (e.g., "Here are the files in your folder").
/// - **Layer A**: Explicit deflection phrases — model asks user instead of acting.
/// - **Layer B**: Short-question heuristic for novel formulations.
///
/// Gated on `round > 0 AND tool_call_count > 0` so that round-0 text
/// responses and direct answers are never flagged.
///
/// # Arguments
/// * `text` — The model's text response.
/// * `round` — Current agent loop round (0-based).
/// * `tool_call_count` — Total tool calls executed so far in this loop.
pub fn is_deflection_response(text: &str, round: usize, tool_call_count: usize) -> bool {
    // Gate: only check after at least one tool has been executed
    if round == 0 || tool_call_count == 0 {
        return false;
    }

    // Trust gate: after 3+ tool calls the model has done meaningful work.
    // Its text response is a summary or answer, not a deflection.
    if tool_call_count >= 3 {
        return false;
    }

    let lower = text.to_lowercase();

    // If the response is a genuine completion summary, don't flag it
    if is_completion_summary(&lower) {
        return false;
    }

    // Result presentation guard: if the model is presenting tool results
    // (file listings, scan findings, data summaries) WITHOUT asking a
    // question or deferring to the user, this is expected behavior.
    if is_presenting_results(&lower) && !text.contains('?') && !has_deferral(&lower) {
        return false;
    }

    // Layer A: Explicit deflection patterns — model asks user instead of acting
    let deflection_patterns = [
        "what would you like",
        "how would you like",
        "how should i",
        "what should i",
        "would you like me to",
        "shall i",
        "do you want me to",
        "let me know",
        "please let me know",
        "i can help you",
        "i can assist",
        "what do you think",
        "which one",
        "which files",
        "here are some options",
    ];

    for pattern in &deflection_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }

    // Layer A extension: Pure narration with no substance — model describes
    // what it sees without presenting actionable results. Patterns that
    // present results ("i found the following", "here are the files") are
    // handled by the result-presentation guard above, not here.
    let narration_patterns = [
        "i see the files",
        "i see the following",
        "i notice",
    ];

    for pattern in &narration_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }

    // Layer B: Short-question heuristic — catches novel deflection formulations.
    // If the response is short (<300 chars), contains a question mark, and we've
    // already executed tools, the model is likely asking instead of acting.
    if text.len() < 300 && text.contains('?') {
        return true;
    }

    false
}

/// Check if text contains deferral phrases that hand control back to the user.
///
/// These phrases indicate the model is waiting for instructions rather than
/// proceeding autonomously. Used alongside the result-presentation guard
/// to catch "I found X files. Let me know which ones you want." patterns.
fn has_deferral(lower: &str) -> bool {
    let deferral_phrases = [
        "let me know",
        "please let me know",
        "would you like",
        "shall i",
        "do you want",
        "which one",
        "which files",
        "which of",
    ];

    for phrase in &deferral_phrases {
        if lower.contains(phrase) {
            return true;
        }
    }

    false
}

/// Check if a model's text is presenting tool results to the user.
///
/// This distinguishes "Here are the files in your Downloads folder: ..."
/// (presenting results — correct behavior) from "I see files on your Desktop.
/// What would you like me to do?" (deflection — incorrect).
///
/// Called by [`is_deflection_response`] to guard against false positives when
/// the model is doing exactly what the user asked: calling a tool and
/// reporting the results.
fn is_presenting_results(lower: &str) -> bool {
    let result_patterns = [
        "here is",          // "Here is what I found"
        "here are",         // "Here are the files"
        "here's what",      // "Here's what the scan found"
        "i found",          // "I found 5 files"
        "the scan found",   // "The scan found 2 secrets"
        "the scan shows",   // "The scan shows no issues"
        "the results",      // "The results show"
        "contains",         // "The folder contains"
        "total of",         // "A total of 12 files"
        "no files",         // "No files found"
        "no results",       // "No results"
        "the folder is",    // "The folder is empty"
        "the directory",    // "The directory contains"
        "found in",         // "Found in the folder"
        "listed below",     // "Files listed below"
    ];

    for p in &result_patterns {
        if lower.contains(p) {
            return true;
        }
    }

    false
}

/// Check if text matches known task-completion signals.
///
/// Used by both [`is_incomplete_response`] and [`is_deflection_response`]
/// to avoid false positives on legitimate summaries.
fn is_completion_summary(lower: &str) -> bool {
    let complete_signals = [
        "all files have been",
        "all screenshots have been",
        "completed all",
        "finished processing",
        "all done",
        "successfully renamed all",
        "processed all",
        "no more files",
        "task complete",
        "that's all",
        "here is a summary",
        "here's what i did",
        "here's what was done",
        "i've completed",
        "i have completed",
        "summary of changes",
        "all files processed",
    ];

    for signal in &complete_signals {
        if lower.contains(signal) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_deflection_response tests ──────────────────────────────────

    #[test]
    fn deflection_question_after_tool() {
        // Model asks the user what to do — clear deflection
        let text = "I see the files on your Desktop. How would you like me to process them?";
        assert!(is_deflection_response(text, 1, 1));
    }

    #[test]
    fn deflection_narration_with_question() {
        // Narration + question = deflection (the "?" makes it past result guard)
        let text = "I found the following files in your Downloads folder. \
                    Let me know which ones you'd like to rename.";
        assert!(is_deflection_response(text, 1, 1));
    }

    #[test]
    fn deflection_shall_i_pattern() {
        let text = "There are 7 screenshots. Shall I extract text from each one?";
        assert!(is_deflection_response(text, 1, 1));
    }

    #[test]
    fn deflection_short_question_heuristic() {
        let text = "I see 7 files. Which ones should I process?";
        assert!(is_deflection_response(text, 1, 1));
    }

    #[test]
    fn no_deflection_on_round_zero() {
        let text = "What would you like me to do?";
        assert!(!is_deflection_response(text, 0, 0));
    }

    #[test]
    fn no_deflection_on_zero_tools() {
        let text = "How would you like me to process them?";
        assert!(!is_deflection_response(text, 1, 0));
    }

    #[test]
    fn no_deflection_on_completion_summary() {
        let text = "All files have been renamed successfully. Here's what I did: \
                    renamed 3 screenshots based on their OCR content.";
        assert!(!is_deflection_response(text, 5, 7));
    }

    #[test]
    fn no_deflection_on_legitimate_answer() {
        let text = "The meeting notes contain the following action items: \
                    1. Prepare the quarterly report by Friday. \
                    2. Schedule a follow-up with the design team. \
                    3. Review the updated budget spreadsheet and send comments.";
        assert!(!is_deflection_response(text, 1, 1));
    }

    // ── Result presentation — NOT deflection ─────────────────────────

    #[test]
    fn no_deflection_when_presenting_file_listing() {
        // This is exactly the Test 1 scenario: model lists files after list_dir
        let text = "Here are the files in your Downloads folder: \
                    DEMO CARD styles.png, benchmark_results.csv, \
                    Liquid AI Notes.pdf, and 12 others.";
        assert!(!is_deflection_response(text, 1, 1));
    }

    #[test]
    fn no_deflection_when_reporting_scan_results() {
        // Model reports scan findings without asking a question
        let text = "The scan found 2 files containing secrets: \
                    .env (AWS key), config.yaml (API token).";
        assert!(!is_deflection_response(text, 1, 1));
    }

    #[test]
    fn no_deflection_when_folder_empty() {
        let text = "No files were found matching your criteria in the folder.";
        assert!(!is_deflection_response(text, 1, 1));
    }

    #[test]
    fn no_deflection_here_is_what_i_found() {
        let text = "Here is what I found in your Documents folder: 3 PDF files, \
                    2 spreadsheets, and 1 text file containing notes.";
        assert!(!is_deflection_response(text, 1, 1));
    }

    #[test]
    fn deflection_result_plus_question() {
        // Presenting results BUT also asking a question — still deflection
        let text = "I found 5 files. Which ones should I process?";
        assert!(is_deflection_response(text, 1, 1));
    }

    // ── Trust gate tests ─────────────────────────────────────────────

    #[test]
    fn no_deflection_after_three_tool_calls() {
        // After 3+ tool calls, model has done real work — trust it
        let text = "I found the following text in your screenshots: ...";
        assert!(!is_deflection_response(text, 4, 3));
        assert!(!is_deflection_response(text, 8, 5));
        assert!(!is_deflection_response(text, 14, 13));
    }

    #[test]
    fn deflection_still_fires_with_few_tool_calls() {
        // With 1-2 tool calls, deflection detection still active for questions
        let text = "I found the following files. Which ones should I process?";
        assert!(is_deflection_response(text, 1, 1));
        assert!(is_deflection_response(text, 2, 2));
    }

    // ── is_incomplete_response tests ──────────────────────────────────

    #[test]
    fn incomplete_remaining_files() {
        assert!(is_incomplete_response(
            "I've processed 3 files. There are 4 remaining."
        ));
    }

    #[test]
    fn incomplete_next_file() {
        assert!(is_incomplete_response("Moving on to the next file now."));
    }

    #[test]
    fn complete_all_done() {
        assert!(!is_incomplete_response(
            "All done! I renamed all 7 screenshots."
        ));
    }

    #[test]
    fn complete_finished() {
        assert!(!is_incomplete_response("Finished processing all files."));
    }

    #[test]
    fn neutral_text_not_incomplete() {
        assert!(!is_incomplete_response(
            "The file contains a quarterly revenue report."
        ));
    }

    // ── is_completion_summary tests ───────────────────────────────────

    #[test]
    fn summary_detected() {
        assert!(is_completion_summary(
            "all files have been renamed successfully"
        ));
    }

    #[test]
    fn summary_not_detected() {
        assert!(!is_completion_summary("i see the files on your desktop"));
    }

    // ── is_presenting_results tests ──────────────────────────────────

    #[test]
    fn presenting_file_list() {
        assert!(is_presenting_results(
            "here are the files in your downloads folder"
        ));
    }

    #[test]
    fn presenting_scan_findings() {
        assert!(is_presenting_results("the scan found 2 secrets"));
    }

    #[test]
    fn presenting_empty_results() {
        assert!(is_presenting_results("no files matching your criteria"));
    }

    #[test]
    fn not_presenting_pure_question() {
        assert!(!is_presenting_results(
            "what would you like me to do with these"
        ));
    }

}
