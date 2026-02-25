/**
 * Parameter Extraction Tests — 50 tests verifying argument accuracy.
 *
 * Tests whether the model extracts correct parameters from natural language,
 * beyond just selecting the right tool. Categories: path extraction, temporal
 * reasoning, multi-param, constraint parsing, implicit params.
 */

export interface ParamExtractionTest {
  readonly id: string;
  readonly category: string;
  readonly prompt: string;
  readonly expectedTool: string;
  readonly expectedParams: Readonly<Record<string, string | number | boolean | string[]>>;
  readonly difficulty: 'easy' | 'medium' | 'hard';
}

// ── Path Extraction (10 tests) ───────────────────────────────────────────

const pathExtractionTests: readonly ParamExtractionTest[] = [
  {
    id: 'pe-path-001', category: 'path-extraction', difficulty: 'easy',
    prompt: 'Read the file at ~/Documents/report.pdf',
    expectedTool: 'filesystem.read_file',
    expectedParams: { path: '~/Documents/report.pdf' },
  },
  {
    id: 'pe-path-002', category: 'path-extraction', difficulty: 'easy',
    prompt: 'List all files in my Downloads folder',
    expectedTool: 'filesystem.list_dir',
    expectedParams: { path: '~/Downloads' },
  },
  {
    id: 'pe-path-003', category: 'path-extraction', difficulty: 'medium',
    prompt: 'Copy the budget spreadsheet from Desktop to Documents',
    expectedTool: 'filesystem.copy_file',
    expectedParams: { source: '~/Desktop/budget', destination: '~/Documents' },
  },
  {
    id: 'pe-path-004', category: 'path-extraction', difficulty: 'medium',
    prompt: 'Rename the file meeting-notes-old.txt in my Documents to meeting-notes-final.txt',
    expectedTool: 'filesystem.move_file',
    expectedParams: { source: 'meeting-notes-old.txt', destination: 'meeting-notes-final.txt' },
  },
  {
    id: 'pe-path-005', category: 'path-extraction', difficulty: 'hard',
    prompt: 'Read the Q3 Marketing Strategy document saved in my work projects folder under /Users/me/Projects/marketing/',
    expectedTool: 'filesystem.read_file',
    expectedParams: { path: '/Users/me/Projects/marketing/' },
  },
  {
    id: 'pe-path-006', category: 'path-extraction', difficulty: 'easy',
    prompt: 'Delete the file temp-output.log from /tmp/',
    expectedTool: 'filesystem.delete_file',
    expectedParams: { path: '/tmp/temp-output.log' },
  },
  {
    id: 'pe-path-007', category: 'path-extraction', difficulty: 'medium',
    prompt: 'Search for all PDF files in ~/Documents/contracts/',
    expectedTool: 'filesystem.search_files',
    expectedParams: { pattern: '*.pdf', path: '~/Documents/contracts' },
  },
  {
    id: 'pe-path-008', category: 'path-extraction', difficulty: 'hard',
    prompt: 'Move the presentation slides from ~/Desktop/Q4 Review Slides.pptx to ~/Documents/Presentations/',
    expectedTool: 'filesystem.move_file',
    expectedParams: { source: 'Q4 Review Slides.pptx', destination: '~/Documents/Presentations' },
  },
  {
    id: 'pe-path-009', category: 'path-extraction', difficulty: 'medium',
    prompt: 'Get the file size and modification date for ~/Projects/localCoWork/package.json',
    expectedTool: 'filesystem.get_metadata',
    expectedParams: { path: '~/Projects/localCoWork/package.json' },
  },
  {
    id: 'pe-path-010', category: 'path-extraction', difficulty: 'hard',
    prompt: 'Watch the folder /Users/me/Dropbox/shared-workspace for any file changes',
    expectedTool: 'filesystem.watch_folder',
    expectedParams: { path: '/Users/me/Dropbox/shared-workspace' },
  },
];

// ── Temporal Reasoning (10 tests) ────────────────────────────────────────

const temporalReasoningTests: readonly ParamExtractionTest[] = [
  {
    id: 'pe-time-001', category: 'temporal-reasoning', difficulty: 'easy',
    prompt: 'Show me my calendar events for today',
    expectedTool: 'calendar.list_events',
    expectedParams: { start: '2026-02-20' },
  },
  {
    id: 'pe-time-002', category: 'temporal-reasoning', difficulty: 'easy',
    prompt: 'Show me what tasks are due this week',
    expectedTool: 'task.list_tasks',
    expectedParams: { status: 'pending' },
  },
  {
    id: 'pe-time-003', category: 'temporal-reasoning', difficulty: 'medium',
    prompt: 'Schedule a meeting called "Sprint Planning" for tomorrow at 2pm',
    expectedTool: 'calendar.create_event',
    expectedParams: { title: 'Sprint Planning', time: '2026-02-21' },
  },
  {
    id: 'pe-time-004', category: 'temporal-reasoning', difficulty: 'medium',
    prompt: 'Create a task to submit the tax return with a due date of April 15, 2026',
    expectedTool: 'task.create_task',
    expectedParams: { title: 'submit the tax return', due_date: '2026-04-15' },
  },
  {
    id: 'pe-time-005', category: 'temporal-reasoning', difficulty: 'hard',
    prompt: 'Find me a free 90-minute slot between 9am and 5pm next Monday',
    expectedTool: 'calendar.find_free_slots',
    expectedParams: { duration: '90' },
  },
  {
    id: 'pe-time-006', category: 'temporal-reasoning', difficulty: 'medium',
    prompt: 'Block 2 hours for deep work on Friday afternoon',
    expectedTool: 'calendar.create_time_block',
    expectedParams: { title: 'deep work', duration: '120' },
  },
  {
    id: 'pe-time-007', category: 'temporal-reasoning', difficulty: 'hard',
    prompt: 'Search for all emails I received in the last 7 days from the engineering team',
    expectedTool: 'email.search_emails',
    expectedParams: { query: 'engineering' },
  },
  {
    id: 'pe-time-008', category: 'temporal-reasoning', difficulty: 'easy',
    prompt: 'Create a task to review the design mockups by end of day Friday',
    expectedTool: 'task.create_task',
    expectedParams: { title: 'review the design mockups' },
  },
  {
    id: 'pe-time-009', category: 'temporal-reasoning', difficulty: 'hard',
    prompt: 'Generate an audit report covering all tool usage from February 1 to February 15, 2026',
    expectedTool: 'audit.generate_audit_report',
    expectedParams: { start_date: '2026-02-01', end_date: '2026-02-15' },
  },
  {
    id: 'pe-time-010', category: 'temporal-reasoning', difficulty: 'medium',
    prompt: 'Show me my calendar events for the first week of March',
    expectedTool: 'calendar.list_events',
    expectedParams: { start: '2026-03-01', end: '2026-03-07' },
  },
];

// ── Multi-Param Extraction (10 tests) ────────────────────────────────────

const multiParamTests: readonly ParamExtractionTest[] = [
  {
    id: 'pe-multi-001', category: 'multi-param', difficulty: 'easy',
    prompt: 'Draft an email to sarah@company.com with the subject "Q4 Review Follow-up"',
    expectedTool: 'email.draft_email',
    expectedParams: { to: 'sarah@company.com', subject: 'Q4 Review Follow-up' },
  },
  {
    id: 'pe-multi-002', category: 'multi-param', difficulty: 'medium',
    prompt: 'Create a high-priority task called "Fix login bug" with a due date of February 25',
    expectedTool: 'task.create_task',
    expectedParams: { title: 'Fix login bug', priority: 'high', due_date: '2026-02-25' },
  },
  {
    id: 'pe-multi-003', category: 'multi-param', difficulty: 'medium',
    prompt: 'Schedule a meeting titled "Design Review" with alice@co.com and bob@co.com at 3pm tomorrow',
    expectedTool: 'calendar.create_event',
    expectedParams: { title: 'Design Review', attendees: ['alice@co.com', 'bob@co.com'] },
  },
  {
    id: 'pe-multi-004', category: 'multi-param', difficulty: 'hard',
    prompt: 'Convert the file ~/Documents/proposal.md to PDF format and save it as ~/Documents/proposal.pdf',
    expectedTool: 'document.convert_format',
    expectedParams: { source: '~/Documents/proposal.md', target_format: 'pdf', output: '~/Documents/proposal.pdf' },
  },
  {
    id: 'pe-multi-005', category: 'multi-param', difficulty: 'easy',
    prompt: 'Write "Hello World" to a new file at ~/Desktop/test.txt',
    expectedTool: 'filesystem.write_file',
    expectedParams: { path: '~/Desktop/test.txt', content: 'Hello World' },
  },
  {
    id: 'pe-multi-006', category: 'multi-param', difficulty: 'medium',
    prompt: 'Insert 3 new employee records into the HR database at ~/data/employees.db in the staff table',
    expectedTool: 'data.write_sqlite',
    expectedParams: { database: 'employees.db', table: 'staff' },
  },
  {
    id: 'pe-multi-007', category: 'multi-param', difficulty: 'hard',
    prompt: 'Draft an email to the whole team at team@startup.io with subject "Sprint Retrospective" and include a summary of what we accomplished this week',
    expectedTool: 'email.draft_email',
    expectedParams: { to: 'team@startup.io', subject: 'Sprint Retrospective' },
  },
  {
    id: 'pe-multi-008', category: 'multi-param', difficulty: 'medium',
    prompt: 'Search for emails from john.doe@gmail.com about the budget proposal',
    expectedTool: 'email.search_emails',
    expectedParams: { from: 'john.doe@gmail.com', query: 'budget proposal' },
  },
  {
    id: 'pe-multi-009', category: 'multi-param', difficulty: 'hard',
    prompt: 'Create a PDF report titled "Annual Review 2025" from the markdown content in ~/Documents/annual-review.md',
    expectedTool: 'document.create_pdf',
    expectedParams: { title: 'Annual Review 2025', source: '~/Documents/annual-review.md' },
  },
  {
    id: 'pe-multi-010', category: 'multi-param', difficulty: 'easy',
    prompt: 'Copy the file report.csv from ~/Downloads to ~/Documents/reports/',
    expectedTool: 'filesystem.copy_file',
    expectedParams: { source: '~/Downloads/report.csv', destination: '~/Documents/reports/' },
  },
];

// ── Constraint Parsing (10 tests) ────────────────────────────────────────

const constraintParsingTests: readonly ParamExtractionTest[] = [
  {
    id: 'pe-const-001', category: 'constraint-parsing', difficulty: 'medium',
    prompt: 'Find all Python files in my project directory',
    expectedTool: 'filesystem.search_files',
    expectedParams: { pattern: '*.py' },
  },
  {
    id: 'pe-const-002', category: 'constraint-parsing', difficulty: 'hard',
    prompt: 'Search for files larger than 10MB in my Downloads folder',
    expectedTool: 'filesystem.search_files',
    expectedParams: { path: '~/Downloads' },
  },
  {
    id: 'pe-const-003', category: 'constraint-parsing', difficulty: 'medium',
    prompt: 'Find duplicate files in ~/Documents that are wasting disk space',
    expectedTool: 'security.find_duplicates',
    expectedParams: { path: '~/Documents' },
  },
  {
    id: 'pe-const-004', category: 'constraint-parsing', difficulty: 'hard',
    prompt: 'Query the sales database at ~/data/sales.db for all orders over $500 in January',
    expectedTool: 'data.query_sqlite',
    expectedParams: { database: 'sales.db' },
  },
  {
    id: 'pe-const-005', category: 'constraint-parsing', difficulty: 'medium',
    prompt: 'Scan the HR folder at ~/Documents/HR/ for any personal information like SSNs or phone numbers',
    expectedTool: 'security.scan_for_pii',
    expectedParams: { path: '~/Documents/HR' },
  },
  {
    id: 'pe-const-006', category: 'constraint-parsing', difficulty: 'easy',
    prompt: 'Search for all JPEG and PNG images in my Pictures folder',
    expectedTool: 'filesystem.search_files',
    expectedParams: { path: '~/Pictures' },
  },
  {
    id: 'pe-const-007', category: 'constraint-parsing', difficulty: 'hard',
    prompt: 'Remove duplicate entries from the contacts table in ~/data/crm.db based on the email column',
    expectedTool: 'data.deduplicate_records',
    expectedParams: { database: 'crm.db', table: 'contacts', key: 'email' },
  },
  {
    id: 'pe-const-008', category: 'constraint-parsing', difficulty: 'medium',
    prompt: 'Extract the text from all pages of the scanned document at ~/Documents/contract-scan.pdf',
    expectedTool: 'ocr.extract_text_from_pdf',
    expectedParams: { path: '~/Documents/contract-scan.pdf' },
  },
  {
    id: 'pe-const-009', category: 'constraint-parsing', difficulty: 'hard',
    prompt: 'Export the top 100 customers from the analytics database as a CSV file to ~/Reports/top-customers.csv',
    expectedTool: 'data.write_csv',
    expectedParams: { path: '~/Reports/top-customers.csv' },
  },
  {
    id: 'pe-const-010', category: 'constraint-parsing', difficulty: 'medium',
    prompt: 'Encrypt the sensitive payroll file at ~/Finance/payroll-2026.xlsx with a strong password',
    expectedTool: 'security.encrypt_file',
    expectedParams: { path: '~/Finance/payroll-2026.xlsx' },
  },
];

// ── Implicit Params (10 tests) ───────────────────────────────────────────

const implicitParamTests: readonly ParamExtractionTest[] = [
  {
    id: 'pe-impl-001', category: 'implicit-params', difficulty: 'medium',
    prompt: 'Index all my project documentation for search',
    expectedTool: 'knowledge.index_folder',
    expectedParams: { path: '~/Documents' },
  },
  {
    id: 'pe-impl-002', category: 'implicit-params', difficulty: 'easy',
    prompt: 'Take a screenshot',
    expectedTool: 'system.take_screenshot',
    expectedParams: {},
  },
  {
    id: 'pe-impl-003', category: 'implicit-params', difficulty: 'medium',
    prompt: 'What is on my clipboard right now?',
    expectedTool: 'clipboard.get_clipboard',
    expectedParams: {},
  },
  {
    id: 'pe-impl-004', category: 'implicit-params', difficulty: 'easy',
    prompt: 'Give me my daily task briefing',
    expectedTool: 'task.daily_briefing',
    expectedParams: {},
  },
  {
    id: 'pe-impl-005', category: 'implicit-params', difficulty: 'hard',
    prompt: 'Check if there are any API keys or secrets accidentally left in the source code at ~/Projects/myapp/',
    expectedTool: 'security.scan_for_secrets',
    expectedParams: { path: '~/Projects/myapp' },
  },
  {
    id: 'pe-impl-006', category: 'implicit-params', difficulty: 'medium',
    prompt: 'Transcribe the recording from the client call',
    expectedTool: 'meeting.transcribe_audio',
    expectedParams: {},
  },
  {
    id: 'pe-impl-007', category: 'implicit-params', difficulty: 'easy',
    prompt: 'Show me what is running on my computer right now',
    expectedTool: 'system.list_processes',
    expectedParams: {},
  },
  {
    id: 'pe-impl-008', category: 'implicit-params', difficulty: 'hard',
    prompt: 'Find documents related to our pricing strategy in the indexed knowledge base',
    expectedTool: 'knowledge.search_documents',
    expectedParams: { query: 'pricing strategy' },
  },
  {
    id: 'pe-impl-009', category: 'implicit-params', difficulty: 'medium',
    prompt: 'What tools were used in my last work session?',
    expectedTool: 'audit.get_session_summary',
    expectedParams: {},
  },
  {
    id: 'pe-impl-010', category: 'implicit-params', difficulty: 'hard',
    prompt: 'Open the quarterly report PDF in Preview so I can review it',
    expectedTool: 'system.open_file_with',
    expectedParams: { application: 'Preview' },
  },
];

// ── Exports ──────────────────────────────────────────────────────────────

export const allParamExtractionTests: readonly ParamExtractionTest[] = [
  ...pathExtractionTests,
  ...temporalReasoningTests,
  ...multiParamTests,
  ...constraintParsingTests,
  ...implicitParamTests,
];
