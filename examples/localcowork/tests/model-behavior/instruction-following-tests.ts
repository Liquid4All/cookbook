/**
 * Instruction Following Tests — 50 tests verifying the model obeys
 * multi-constraint instructions precisely.
 *
 * Inspired by IFEval, adapted for our tool-calling agent context.
 * All constraints are programmatically verifiable — no LLM judge required.
 *
 * Categories: format constraints, length constraints, multi-part instructions,
 * negation / exclusion, conditional logic.
 */

import type { ConstraintDef } from './quality-scoring';

export interface InstructionFollowingTest {
  readonly id: string;
  readonly category: string;
  readonly prompt: string;
  readonly constraints: readonly ConstraintDef[];
  readonly difficulty: 'easy' | 'medium' | 'hard';
}

// ── Format Constraints (10 tests) ─────────────────────────────────────────

const formatConstraintTests: readonly InstructionFollowingTest[] = [
  {
    id: 'if-fmt-001', category: 'format-constraints', difficulty: 'easy',
    prompt: 'List my top 5 priorities for today as a numbered list',
    constraints: [
      { type: 'format_numbered_list', value: '', description: 'Response must contain a numbered list' },
      { type: 'min_length', value: 20, description: 'Response should be substantial' },
    ],
  },
  {
    id: 'if-fmt-002', category: 'format-constraints', difficulty: 'easy',
    prompt: 'Show me my upcoming meetings as bullet points',
    constraints: [
      { type: 'format_bullet_list', value: '', description: 'Response must contain a bulleted list' },
      { type: 'calls_tool', value: 'calendar.list_events', description: 'Must call calendar tool' },
    ],
  },
  {
    id: 'if-fmt-003', category: 'format-constraints', difficulty: 'medium',
    prompt: 'Give me the system information in JSON format',
    constraints: [
      { type: 'format_json', value: '', description: 'Response must contain valid JSON' },
      { type: 'calls_tool', value: 'system.get_system_info', description: 'Must call system info tool' },
    ],
  },
  {
    id: 'if-fmt-004', category: 'format-constraints', difficulty: 'medium',
    prompt: 'List all the tools you have access to as a numbered list with descriptions',
    constraints: [
      { type: 'format_numbered_list', value: '', description: 'Response must contain a numbered list' },
      { type: 'no_tool_call', value: '', description: 'Should not call any tool — just list them' },
      { type: 'min_length', value: 50, description: 'Must be comprehensive' },
    ],
  },
  {
    id: 'if-fmt-005', category: 'format-constraints', difficulty: 'hard',
    prompt: 'Show me my overdue tasks as a numbered list, and include the due date for each one in parentheses',
    constraints: [
      { type: 'format_numbered_list', value: '', description: 'Response must be a numbered list' },
      { type: 'calls_tool', value: 'task.get_overdue', description: 'Must call overdue task tool' },
    ],
  },
  {
    id: 'if-fmt-006', category: 'format-constraints', difficulty: 'easy',
    prompt: 'Summarize what my clipboard contains in a single sentence',
    constraints: [
      { type: 'max_sentences', value: 2, description: 'Should be roughly one sentence' },
      { type: 'calls_tool', value: 'clipboard.get_clipboard', description: 'Must read clipboard' },
    ],
  },
  {
    id: 'if-fmt-007', category: 'format-constraints', difficulty: 'medium',
    prompt: 'Create a bullet-point summary of my daily briefing',
    constraints: [
      { type: 'format_bullet_list', value: '', description: 'Response must use bullet points' },
      { type: 'calls_tool', value: 'task.daily_briefing', description: 'Must call daily briefing tool' },
    ],
  },
  {
    id: 'if-fmt-008', category: 'format-constraints', difficulty: 'hard',
    prompt: 'Generate my audit report for this month and present the key findings as a numbered list followed by a summary paragraph',
    constraints: [
      { type: 'format_numbered_list', value: '', description: 'Must contain a numbered list' },
      { type: 'calls_tool', value: 'audit.generate_audit_report', description: 'Must call audit report tool' },
      { type: 'min_length', value: 40, description: 'Must include both list and summary' },
    ],
  },
  {
    id: 'if-fmt-009', category: 'format-constraints', difficulty: 'medium',
    prompt: 'Show me the files in my Documents folder as a bullet list sorted by name',
    constraints: [
      { type: 'format_bullet_list', value: '', description: 'Response must use bullet points' },
      { type: 'calls_tool', value: 'filesystem.list_dir', description: 'Must call list_dir' },
    ],
  },
  {
    id: 'if-fmt-010', category: 'format-constraints', difficulty: 'hard',
    prompt: 'Search for all PDF files in my project and return the results as a JSON array of file paths',
    constraints: [
      { type: 'format_json', value: '', description: 'Response must contain valid JSON' },
      { type: 'calls_tool', value: 'filesystem.search_files', description: 'Must call search_files' },
    ],
  },
];

// ── Length Constraints (10 tests) ─────────────────────────────────────────

const lengthConstraintTests: readonly InstructionFollowingTest[] = [
  {
    id: 'if-len-001', category: 'length-constraints', difficulty: 'easy',
    prompt: 'In one sentence, tell me what time my first meeting is today',
    constraints: [
      { type: 'max_sentences', value: 2, description: 'Must be one sentence (allow 2 for tolerance)' },
      { type: 'calls_tool', value: 'calendar.list_events', description: 'Must check calendar' },
    ],
  },
  {
    id: 'if-len-002', category: 'length-constraints', difficulty: 'easy',
    prompt: 'Give me a brief one-line description of what is on my clipboard',
    constraints: [
      { type: 'max_sentences', value: 2, description: 'Must be brief — one or two sentences' },
      { type: 'calls_tool', value: 'clipboard.get_clipboard', description: 'Must read clipboard' },
    ],
  },
  {
    id: 'if-len-003', category: 'length-constraints', difficulty: 'medium',
    prompt: 'Write a detailed explanation of at least 100 words about what tools are available for file management',
    constraints: [
      { type: 'min_length', value: 100, description: 'Must be at least 100 words' },
      { type: 'no_tool_call', value: '', description: 'Should explain without calling a tool' },
    ],
  },
  {
    id: 'if-len-004', category: 'length-constraints', difficulty: 'medium',
    prompt: 'Summarize my current tasks in under 50 words',
    constraints: [
      { type: 'max_length', value: 60, description: 'Must be under 50 words (allow tolerance)' },
      { type: 'calls_tool', value: 'task.list_tasks', description: 'Must check tasks' },
    ],
  },
  {
    id: 'if-len-005', category: 'length-constraints', difficulty: 'hard',
    prompt: 'Give me a comprehensive report of at least 150 words covering my tasks, calendar, and recent emails',
    constraints: [
      { type: 'min_length', value: 150, description: 'Must be at least 150 words' },
      { type: 'addresses_all_parts', value: 'task,calendar,email', description: 'Must cover all three topics' },
    ],
  },
  {
    id: 'if-len-006', category: 'length-constraints', difficulty: 'easy',
    prompt: 'Tell me the current date and time in a single short sentence',
    constraints: [
      { type: 'max_sentences', value: 2, description: 'Must be one short sentence' },
      { type: 'no_tool_call', value: '', description: 'Should answer directly without a tool' },
    ],
  },
  {
    id: 'if-len-007', category: 'length-constraints', difficulty: 'medium',
    prompt: 'Write me a 3-sentence summary of my system information',
    constraints: [
      { type: 'max_sentences', value: 5, description: 'Should be around 3 sentences (allow tolerance)' },
      { type: 'calls_tool', value: 'system.get_system_info', description: 'Must call system info' },
    ],
  },
  {
    id: 'if-len-008', category: 'length-constraints', difficulty: 'hard',
    prompt: 'Give me a one-paragraph overview (no more than 75 words) of what happened in my last work session',
    constraints: [
      { type: 'max_length', value: 90, description: 'Must be under 75 words (with tolerance)' },
      { type: 'calls_tool', value: 'audit.get_session_summary', description: 'Must check session summary' },
    ],
  },
  {
    id: 'if-len-009', category: 'length-constraints', difficulty: 'medium',
    prompt: 'List my pending tasks — keep the response under 40 words',
    constraints: [
      { type: 'max_length', value: 50, description: 'Must be under 40 words (with tolerance)' },
      { type: 'calls_tool', value: 'task.list_tasks', description: 'Must list tasks' },
    ],
  },
  {
    id: 'if-len-010', category: 'length-constraints', difficulty: 'hard',
    prompt: 'Write a detailed 200+ word analysis of what security issues might exist in my project files',
    constraints: [
      { type: 'min_length', value: 200, description: 'Must be at least 200 words' },
      { type: 'contains_keyword', value: 'security', description: 'Must discuss security' },
    ],
  },
];

// ── Multi-Part Instructions (10 tests) ────────────────────────────────────

const multiPartTests: readonly InstructionFollowingTest[] = [
  {
    id: 'if-multi-001', category: 'multi-part', difficulty: 'medium',
    prompt: 'Search for all PDF files in my Documents folder AND tell me how many you found',
    constraints: [
      { type: 'calls_tool', value: 'filesystem.search_files', description: 'Must search for files' },
      { type: 'addresses_all_parts', value: 'pdf,found', description: 'Must mention PDFs and a count' },
    ],
  },
  {
    id: 'if-multi-002', category: 'multi-part', difficulty: 'medium',
    prompt: 'Check my calendar for today AND list my pending tasks AND tell me which should be my top priority',
    constraints: [
      { type: 'addresses_all_parts', value: 'calendar,task,priority', description: 'Must address all three parts' },
    ],
  },
  {
    id: 'if-multi-003', category: 'multi-part', difficulty: 'hard',
    prompt: 'First, list the files in my Downloads folder. Second, tell me which ones are larger than 1MB. Third, suggest which can be deleted safely.',
    constraints: [
      { type: 'calls_tool', value: 'filesystem.list_dir', description: 'Must list the folder' },
      { type: 'addresses_all_parts', value: 'download,size,delete', description: 'Must address all three steps' },
    ],
  },
  {
    id: 'if-multi-004', category: 'multi-part', difficulty: 'easy',
    prompt: 'Take a screenshot AND tell me what application is in the foreground',
    constraints: [
      { type: 'calls_tool', value: 'system.take_screenshot', description: 'Must take a screenshot' },
      { type: 'contains_keyword', value: 'application', description: 'Must mention the application' },
    ],
  },
  {
    id: 'if-multi-005', category: 'multi-part', difficulty: 'hard',
    prompt: 'Read the file at ~/Documents/report.pdf, summarize the key points, AND draft an email to my manager at boss@company.com with the summary',
    constraints: [
      { type: 'addresses_all_parts', value: 'report,summary,email,boss@company.com', description: 'Must cover read, summarize, and email' },
    ],
  },
  {
    id: 'if-multi-006', category: 'multi-part', difficulty: 'medium',
    prompt: 'Show me my overdue tasks AND create a new high-priority task called "Clear overdue backlog" with a due date of tomorrow',
    constraints: [
      { type: 'addresses_all_parts', value: 'overdue,clear overdue backlog,high', description: 'Must address both parts' },
    ],
  },
  {
    id: 'if-multi-007', category: 'multi-part', difficulty: 'hard',
    prompt: 'Search my indexed documents for "quarterly revenue", then create a summary task, then draft an email to finance@company.com with the findings',
    constraints: [
      { type: 'addresses_all_parts', value: 'revenue,task,email,finance', description: 'Must address search, task creation, and email' },
    ],
  },
  {
    id: 'if-multi-008', category: 'multi-part', difficulty: 'easy',
    prompt: 'Get my system info AND tell me how much free disk space I have',
    constraints: [
      { type: 'calls_tool', value: 'system.get_system_info', description: 'Must call system info' },
      { type: 'contains_keyword', value: 'disk', description: 'Must mention disk space' },
    ],
  },
  {
    id: 'if-multi-009', category: 'multi-part', difficulty: 'medium',
    prompt: 'List the files in ~/Projects/ AND search for any Python files AND tell me the total count',
    constraints: [
      { type: 'addresses_all_parts', value: 'project,python,count', description: 'Must list, search, and count' },
    ],
  },
  {
    id: 'if-multi-010', category: 'multi-part', difficulty: 'hard',
    prompt: 'Scan my HR folder for PII, report how many findings there are, create a task to review them, and draft an email to compliance@company.com about the findings',
    constraints: [
      { type: 'addresses_all_parts', value: 'pii,finding,task,compliance', description: 'Must address all four steps' },
    ],
  },
];

// ── Negation / Exclusion (10 tests) ───────────────────────────────────────

const negationTests: readonly InstructionFollowingTest[] = [
  {
    id: 'if-neg-001', category: 'negation-exclusion', difficulty: 'easy',
    prompt: 'List my tasks but do NOT include any completed ones',
    constraints: [
      { type: 'calls_tool', value: 'task.list_tasks', description: 'Must call task listing' },
      { type: 'excludes_keyword', value: 'completed', description: 'Must not mention completed tasks' },
    ],
  },
  {
    id: 'if-neg-002', category: 'negation-exclusion', difficulty: 'medium',
    prompt: 'Draft an email to team@startup.io about the project update. Do NOT include a greeting or sign-off.',
    constraints: [
      { type: 'calls_tool', value: 'email.draft_email', description: 'Must draft an email' },
      { type: 'excludes_keyword', value: 'dear', description: 'Must not include a greeting like Dear' },
      { type: 'excludes_keyword', value: 'sincerely', description: 'Must not include a sign-off like Sincerely' },
    ],
  },
  {
    id: 'if-neg-003', category: 'negation-exclusion', difficulty: 'medium',
    prompt: 'Search for files in my Documents folder but exclude any PDFs',
    constraints: [
      { type: 'calls_tool', value: 'filesystem.search_files', description: 'Must search for files' },
      { type: 'excludes_keyword', value: '.pdf', description: 'Must not include PDF files' },
    ],
  },
  {
    id: 'if-neg-004', category: 'negation-exclusion', difficulty: 'hard',
    prompt: 'Give me my daily briefing but do NOT mention any tasks that are low priority',
    constraints: [
      { type: 'calls_tool', value: 'task.daily_briefing', description: 'Must call daily briefing' },
      { type: 'excludes_keyword', value: 'low priority', description: 'Must not mention low priority tasks' },
    ],
  },
  {
    id: 'if-neg-005', category: 'negation-exclusion', difficulty: 'easy',
    prompt: 'Tell me about my system without mentioning the CPU or processor',
    constraints: [
      { type: 'calls_tool', value: 'system.get_system_info', description: 'Must get system info' },
      { type: 'excludes_keyword', value: 'cpu', description: 'Must not mention CPU' },
      { type: 'excludes_keyword', value: 'processor', description: 'Must not mention processor' },
    ],
  },
  {
    id: 'if-neg-006', category: 'negation-exclusion', difficulty: 'medium',
    prompt: 'Show me my calendar events for this week. Do NOT show any events before noon.',
    constraints: [
      { type: 'calls_tool', value: 'calendar.list_events', description: 'Must check calendar' },
      { type: 'excludes_keyword', value: 'morning', description: 'Should not highlight morning events' },
    ],
  },
  {
    id: 'if-neg-007', category: 'negation-exclusion', difficulty: 'hard',
    prompt: 'Summarize the meeting transcript. Do NOT include any action items — focus only on decisions made.',
    constraints: [
      { type: 'excludes_keyword', value: 'action item', description: 'Must not mention action items' },
      { type: 'contains_keyword', value: 'decision', description: 'Must focus on decisions' },
    ],
  },
  {
    id: 'if-neg-008', category: 'negation-exclusion', difficulty: 'easy',
    prompt: 'List the running processes but do NOT include system processes — only user applications',
    constraints: [
      { type: 'calls_tool', value: 'system.list_processes', description: 'Must list processes' },
      { type: 'contains_keyword', value: 'application', description: 'Must mention user applications' },
    ],
  },
  {
    id: 'if-neg-009', category: 'negation-exclusion', difficulty: 'hard',
    prompt: 'Generate an audit report for the past week. Exclude any read-only operations — only show write, create, and delete actions.',
    constraints: [
      { type: 'calls_tool', value: 'audit.generate_audit_report', description: 'Must generate audit report' },
      { type: 'excludes_keyword', value: 'read_file', description: 'Should exclude read operations' },
      { type: 'contains_keyword', value: 'write', description: 'Must mention write operations' },
    ],
  },
  {
    id: 'if-neg-010', category: 'negation-exclusion', difficulty: 'medium',
    prompt: 'Find all image files in my Pictures folder. Do NOT include screenshots — only photos.',
    constraints: [
      { type: 'calls_tool', value: 'filesystem.search_files', description: 'Must search for files' },
      { type: 'excludes_keyword', value: 'screenshot', description: 'Must exclude screenshots' },
    ],
  },
];

// ── Conditional Logic (10 tests) ──────────────────────────────────────────

const conditionalTests: readonly InstructionFollowingTest[] = [
  {
    id: 'if-cond-001', category: 'conditional-logic', difficulty: 'medium',
    prompt: 'Check if I have any overdue tasks. If yes, list them. If no, just say "All clear — no overdue tasks."',
    constraints: [
      { type: 'calls_tool', value: 'task.get_overdue', description: 'Must check for overdue tasks' },
      { type: 'conditional_branch', value: 'overdue', description: 'Must address the overdue condition' },
    ],
  },
  {
    id: 'if-cond-002', category: 'conditional-logic', difficulty: 'medium',
    prompt: 'Check my calendar for tomorrow. If I have meetings, tell me the first one. If my day is free, suggest blocking 2 hours for deep work.',
    constraints: [
      { type: 'calls_tool', value: 'calendar.list_events', description: 'Must check calendar' },
      { type: 'conditional_branch', value: 'meeting', description: 'Must address the meeting condition' },
    ],
  },
  {
    id: 'if-cond-003', category: 'conditional-logic', difficulty: 'hard',
    prompt: 'Search for a file called "budget-2026.xlsx" in my Documents. If found, read its contents. If not found, create a new task to prepare the budget.',
    constraints: [
      { type: 'calls_tool', value: 'filesystem.search_files', description: 'Must search for the file' },
      { type: 'conditional_branch', value: 'budget', description: 'Must address the budget condition' },
    ],
  },
  {
    id: 'if-cond-004', category: 'conditional-logic', difficulty: 'easy',
    prompt: 'Check if there are any email drafts. If yes, list them. If no, say "No pending drafts."',
    constraints: [
      { type: 'calls_tool', value: 'email.list_drafts', description: 'Must check drafts' },
      { type: 'conditional_branch', value: 'draft', description: 'Must address the draft condition' },
    ],
  },
  {
    id: 'if-cond-005', category: 'conditional-logic', difficulty: 'hard',
    prompt: 'Scan my project folder for secrets. If any are found, draft an email to security@company.com about the findings. If none are found, say "Clean scan — no secrets detected."',
    constraints: [
      { type: 'calls_tool', value: 'security.scan_for_secrets', description: 'Must scan for secrets' },
      { type: 'conditional_branch', value: 'secret', description: 'Must address the secret condition' },
    ],
  },
  {
    id: 'if-cond-006', category: 'conditional-logic', difficulty: 'medium',
    prompt: 'Check if my disk space is below 10GB free. If so, suggest files to clean up. Otherwise, tell me my system is healthy.',
    constraints: [
      { type: 'calls_tool', value: 'system.get_system_info', description: 'Must check system info' },
      { type: 'conditional_branch', value: 'disk', description: 'Must address disk space condition' },
    ],
  },
  {
    id: 'if-cond-007', category: 'conditional-logic', difficulty: 'hard',
    prompt: 'Look up "pricing strategy" in my knowledge base. If results are found, summarize the top findings. If nothing is found, suggest indexing my Documents folder first.',
    constraints: [
      { type: 'calls_tool', value: 'knowledge.search_documents', description: 'Must search knowledge base' },
      { type: 'conditional_branch', value: 'pricing', description: 'Must address the pricing condition' },
    ],
  },
  {
    id: 'if-cond-008', category: 'conditional-logic', difficulty: 'easy',
    prompt: 'Check my clipboard. If it contains a URL, open it. If it contains text, save it to ~/Desktop/clipboard-note.txt.',
    constraints: [
      { type: 'calls_tool', value: 'clipboard.get_clipboard', description: 'Must check clipboard' },
      { type: 'conditional_branch', value: 'clipboard', description: 'Must address the clipboard content' },
    ],
  },
  {
    id: 'if-cond-009', category: 'conditional-logic', difficulty: 'medium',
    prompt: 'List my tasks due this week. If there are more than 5, tell me I should prioritize. If there are 5 or fewer, say the workload looks manageable.',
    constraints: [
      { type: 'calls_tool', value: 'task.list_tasks', description: 'Must list tasks' },
      { type: 'conditional_branch', value: 'task', description: 'Must address the task count condition' },
    ],
  },
  {
    id: 'if-cond-010', category: 'conditional-logic', difficulty: 'hard',
    prompt: 'Check for duplicate files in ~/Documents. If duplicates are found, tell me how much space I could save by removing them. If no duplicates, say "No duplicates found — your files are clean."',
    constraints: [
      { type: 'calls_tool', value: 'security.find_duplicates', description: 'Must check for duplicates' },
      { type: 'conditional_branch', value: 'duplicate', description: 'Must address the duplicate condition' },
    ],
  },
];

// ── Exports ──────────────────────────────────────────────────────────────

export const allInstructionFollowingTests: readonly InstructionFollowingTest[] = [
  ...formatConstraintTests,
  ...lengthConstraintTests,
  ...multiPartTests,
  ...negationTests,
  ...conditionalTests,
];

export {
  formatConstraintTests,
  lengthConstraintTests,
  multiPartTests,
  negationTests,
  conditionalTests,
};
