/**
 * Type definitions for the LocalCowork Model Behavior Test Suite.
 *
 * These interfaces define the shape of test cases that verify
 * the LLM's tool-calling accuracy across all 13 MCP servers.
 */

/** A single tool-selection test: one prompt maps to one or more expected tools. */
export interface ToolSelectionTest {
  readonly id: string;
  readonly category: string;
  readonly prompt: string;
  readonly context?: readonly string[];
  readonly expectedTools: readonly string[];
  readonly expectedParamKeys?: Readonly<Record<string, readonly string[]>>;
  readonly difficulty: 'easy' | 'medium' | 'hard';
}

/** A step within a multi-step chain test. */
export interface MultiStepEntry {
  readonly description: string;
  readonly prompt: string;
  readonly expectedTools: readonly string[];
}

/** A multi-step scenario requiring a chain of 3+ tool calls. */
export interface MultiStepTest {
  readonly id: string;
  readonly category: string;
  readonly scenario: string;
  readonly steps: readonly MultiStepEntry[];
  readonly difficulty: 'easy' | 'medium' | 'hard';
}

/** An edge-case test verifying the model handles unusual inputs correctly. */
export interface EdgeCaseTest {
  readonly id: string;
  readonly category: string;
  readonly prompt: string;
  readonly expectedBehavior: 'clarify' | 'fallback' | 'refuse' | 'suggest_alternative';
  readonly expectedTools?: readonly string[];
  readonly reason: string;
}

/** Aggregated results from a test run. */
export interface TestRunResults {
  readonly runId: string;
  readonly timestamp: string;
  readonly modelEndpoint: string | null;
  readonly total: number;
  readonly passed: number;
  readonly failed: number;
  readonly skipped: number;
  readonly accuracyPercent: number;
  readonly categories: Readonly<Record<string, CategoryResult>>;
  readonly durationMs: number;
}

/** Per-category breakdown of test results. */
export interface CategoryResult {
  readonly total: number;
  readonly passed: number;
  readonly failed: number;
  readonly accuracyPercent: number;
}

/** Result of an individual test case. */
export interface IndividualTestResult {
  readonly testId: string;
  readonly status: 'passed' | 'failed' | 'skipped';
  readonly expectedTools: readonly string[];
  readonly actualTools?: readonly string[];
  readonly error?: string;
  readonly durationMs: number;
}

/** The complete set of valid tool names across all 13 MCP servers. */
export const VALID_TOOL_NAMES: readonly string[] = [
  // filesystem (9 tools)
  'filesystem.list_dir',
  'filesystem.read_file',
  'filesystem.write_file',
  'filesystem.move_file',
  'filesystem.copy_file',
  'filesystem.delete_file',
  'filesystem.search_files',
  'filesystem.get_metadata',
  'filesystem.watch_folder',
  // document (8 tools)
  'document.extract_text',
  'document.convert_format',
  'document.diff_documents',
  'document.create_pdf',
  'document.fill_pdf_form',
  'document.merge_pdfs',
  'document.create_docx',
  'document.read_spreadsheet',
  // ocr (4 tools)
  'ocr.extract_text_from_image',
  'ocr.extract_text_from_pdf',
  'ocr.extract_structured_data',
  'ocr.extract_table',
  // data (5 tools)
  'data.write_csv',
  'data.write_sqlite',
  'data.query_sqlite',
  'data.deduplicate_records',
  'data.summarize_anomalies',
  // audit (4 tools)
  'audit.get_tool_log',
  'audit.get_session_summary',
  'audit.generate_audit_report',
  'audit.export_audit_pdf',
  // knowledge (5 tools)
  'knowledge.index_folder',
  'knowledge.search_documents',
  'knowledge.ask_about_files',
  'knowledge.update_index',
  'knowledge.get_related_chunks',
  // security (6 tools)
  'security.scan_for_pii',
  'security.scan_for_secrets',
  'security.find_duplicates',
  'security.propose_cleanup',
  'security.encrypt_file',
  'security.decrypt_file',
  // task (5 tools)
  'task.create_task',
  'task.list_tasks',
  'task.update_task',
  'task.get_overdue',
  'task.daily_briefing',
  // calendar (4 tools)
  'calendar.list_events',
  'calendar.create_event',
  'calendar.find_free_slots',
  'calendar.create_time_block',
  // email (5 tools)
  'email.draft_email',
  'email.list_drafts',
  'email.search_emails',
  'email.summarize_thread',
  'email.send_draft',
  // meeting (4 tools)
  'meeting.transcribe_audio',
  'meeting.extract_action_items',
  'meeting.extract_commitments',
  'meeting.generate_minutes',
  // clipboard (3 tools)
  'clipboard.get_clipboard',
  'clipboard.set_clipboard',
  'clipboard.clipboard_history',
  // system (5 tools)
  'system.get_system_info',
  'system.open_application',
  'system.take_screenshot',
  'system.list_processes',
  'system.open_file_with',
] as const;

/** Set for O(1) lookup of valid tool names. */
export const VALID_TOOL_SET: ReadonlySet<string> = new Set(VALID_TOOL_NAMES);

/** Result of a single step within a multi-step chain benchmark. */
export interface StepResult {
  readonly stepIndex: number;
  readonly expectedTools: readonly string[];
  readonly actualTools: readonly string[];
  readonly status: 'passed' | 'failed';
  readonly failureReason?: 'deflection' | 'wrong_tool' | 'no_tool' | 'error';
  readonly rawContent: string;
}

/** Result of a complete multi-step chain benchmark. */
export interface MultiStepResult {
  readonly testId: string;
  readonly scenario: string;
  readonly difficulty: 'easy' | 'medium' | 'hard';
  readonly status: 'passed' | 'failed' | 'skipped';
  readonly stepsCompleted: number;
  readonly totalSteps: number;
  readonly failedAtStep?: number;
  readonly failureReason?: 'deflection' | 'wrong_tool' | 'no_tool' | 'error';
  readonly stepResults: readonly StepResult[];
  readonly durationMs: number;
}
