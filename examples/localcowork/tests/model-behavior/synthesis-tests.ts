/**
 * Synthesis Quality Tests — 50 tests verifying the model produces coherent,
 * grounded, useful responses from tool results.
 *
 * Each test provides a user query, a tool call, a realistic mock tool result,
 * and programmatically verifiable constraints on the response.
 *
 * Categories: fact extraction, calculation, error/empty handling,
 * multi-source synthesis, raw dump avoidance.
 */

import type { SynthesisConstraintDef } from './quality-scoring';

export interface SynthesisTest {
  readonly id: string;
  readonly category: string;
  readonly userQuery: string;
  readonly toolCall: string;
  readonly toolResult: string;
  readonly constraints: readonly SynthesisConstraintDef[];
  readonly difficulty: 'easy' | 'medium' | 'hard';
}

// ── Fact Extraction (10 tests) ────────────────────────────────────────────

const factExtractionTests: readonly SynthesisTest[] = [
  {
    id: 'syn-fact-001', category: 'fact-extraction', difficulty: 'easy',
    userQuery: 'How many files are in my Downloads folder?',
    toolCall: '[filesystem.list_dir(path="~/Downloads")]',
    toolResult: '{"files": [{"name": "report.pdf", "size": 24500}, {"name": "image.png", "size": 850000}, {"name": "notes.txt", "size": 1200}, {"name": "video.mp4", "size": 52000000}, {"name": "archive.zip", "size": 3200000}]}',
    constraints: [
      { type: 'correct_count', value: '5', description: 'Must state there are 5 files' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-fact-002', category: 'fact-extraction', difficulty: 'easy',
    userQuery: 'What is my next meeting?',
    toolCall: '[calendar.list_events(start="2026-02-20")]',
    toolResult: '{"events": [{"title": "Team Standup", "start": "2026-02-20T09:00:00", "end": "2026-02-20T09:30:00", "attendees": ["alice@co.com", "bob@co.com"]}, {"title": "Design Review", "start": "2026-02-20T14:00:00", "end": "2026-02-20T15:00:00", "attendees": ["charlie@co.com"]}]}',
    constraints: [
      { type: 'mentions_key_fact', value: 'team standup', description: 'Must mention Team Standup as next meeting' },
      { type: 'mentions_key_fact', value: '9:00', description: 'Must mention the 9am start time' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-fact-003', category: 'fact-extraction', difficulty: 'medium',
    userQuery: 'How many PDF files do I have in Documents?',
    toolCall: '[filesystem.search_files(pattern="*.pdf", path="~/Documents")]',
    toolResult: '{"matches": [{"path": "~/Documents/report.pdf", "size": 24500}, {"path": "~/Documents/contracts/lease.pdf", "size": 180000}, {"path": "~/Documents/contracts/nda.pdf", "size": 45000}, {"path": "~/Documents/taxes/return-2025.pdf", "size": 320000}]}',
    constraints: [
      { type: 'correct_count', value: '4', description: 'Must state there are 4 PDF files' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-fact-004', category: 'fact-extraction', difficulty: 'medium',
    userQuery: 'Who sent me emails about the budget?',
    toolCall: '[email.search_emails(query="budget")]',
    toolResult: '{"emails": [{"from": "sarah@company.com", "subject": "Q4 Budget Review", "date": "2026-02-18"}, {"from": "finance@company.com", "subject": "Budget Approval Needed", "date": "2026-02-15"}, {"from": "sarah@company.com", "subject": "Budget Revisions", "date": "2026-02-10"}]}',
    constraints: [
      { type: 'mentions_key_fact', value: 'sarah', description: 'Must mention Sarah as a sender' },
      { type: 'mentions_key_fact', value: 'finance', description: 'Must mention finance as a sender' },
      { type: 'correct_count', value: '3', description: 'Must indicate 3 emails found' },
    ],
  },
  {
    id: 'syn-fact-005', category: 'fact-extraction', difficulty: 'hard',
    userQuery: 'What tasks are overdue and who are they assigned to?',
    toolCall: '[task.get_overdue()]',
    toolResult: '{"tasks": [{"title": "Submit expense report", "due_date": "2026-02-15", "assignee": "me", "priority": "high"}, {"title": "Review PR #42", "due_date": "2026-02-18", "assignee": "me", "priority": "medium"}, {"title": "Update documentation", "due_date": "2026-02-10", "assignee": "me", "priority": "low"}]}',
    constraints: [
      { type: 'mentions_key_fact', value: 'expense report', description: 'Must mention expense report task' },
      { type: 'mentions_key_fact', value: 'pr #42', description: 'Must mention PR review task' },
      { type: 'correct_count', value: '3', description: 'Must indicate 3 overdue tasks' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-fact-006', category: 'fact-extraction', difficulty: 'easy',
    userQuery: 'What OS am I running?',
    toolCall: '[system.get_system_info()]',
    toolResult: '{"os": "macOS 15.3", "cpu": "Apple M3 Pro", "memory": "36GB", "disk_total": "1TB", "disk_free": "456GB"}',
    constraints: [
      { type: 'mentions_key_fact', value: 'macos', description: 'Must mention macOS' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-fact-007', category: 'fact-extraction', difficulty: 'medium',
    userQuery: 'What are the key findings from the security scan?',
    toolCall: '[security.scan_for_pii(path="~/Documents/HR")]',
    toolResult: '{"findings": [{"type": "ssn", "value": "***-**-1234", "file": "employee-records.xlsx", "line": 15}, {"type": "phone", "value": "555-0123", "file": "contacts.csv", "line": 8}, {"type": "email", "value": "john.doe@personal.com", "file": "contacts.csv", "line": 3}], "total": 3, "files_scanned": 12}',
    constraints: [
      { type: 'mentions_key_fact', value: 'ssn', description: 'Must mention SSN finding' },
      { type: 'correct_count', value: '3', description: 'Must state 3 findings' },
      { type: 'references_source', value: 'employee-records', description: 'Must reference the source file' },
    ],
  },
  {
    id: 'syn-fact-008', category: 'fact-extraction', difficulty: 'hard',
    userQuery: 'What did we decide in the last meeting?',
    toolCall: '[meeting.generate_minutes(recording_id="rec-001")]',
    toolResult: '{"summary": "Team discussed Q1 priorities. Decided to postpone the mobile app launch to March 15. Alice will lead the backend migration. Budget was approved at $75K.", "action_items": ["Alice: Complete migration plan by Feb 25", "Bob: Update roadmap document", "All: Review new timeline"], "decisions": ["Postpone mobile launch to March 15", "Approve $75K budget", "Alice leads migration"]}',
    constraints: [
      { type: 'mentions_key_fact', value: 'march 15', description: 'Must mention postponement to March 15' },
      { type: 'mentions_key_fact', value: '75k', description: 'Must mention the $75K budget' },
      { type: 'mentions_key_fact', value: 'alice', description: 'Must mention Alice leading migration' },
    ],
  },
  {
    id: 'syn-fact-009', category: 'fact-extraction', difficulty: 'medium',
    userQuery: 'What is in my clipboard?',
    toolCall: '[clipboard.get_clipboard()]',
    toolResult: '{"content": "https://github.com/liquid-ai/localcowork/pull/42", "type": "text/plain"}',
    constraints: [
      { type: 'mentions_key_fact', value: 'github', description: 'Must mention it is a GitHub URL' },
      { type: 'mentions_key_fact', value: 'pull/42', description: 'Must mention pull request 42' },
    ],
  },
  {
    id: 'syn-fact-010', category: 'fact-extraction', difficulty: 'hard',
    userQuery: 'Summarize the audit trail for this week',
    toolCall: '[audit.generate_audit_report(start_date="2026-02-16", end_date="2026-02-20")]',
    toolResult: '{"report": {"total_actions": 47, "by_tool": {"filesystem.read_file": 15, "filesystem.list_dir": 8, "task.create_task": 6, "email.draft_email": 5, "calendar.list_events": 4, "knowledge.search_documents": 3, "security.scan_for_pii": 2, "document.extract_text": 2, "system.get_system_info": 2}, "by_day": {"2026-02-16": 12, "2026-02-17": 8, "2026-02-18": 15, "2026-02-19": 7, "2026-02-20": 5}}}',
    constraints: [
      { type: 'correct_count', value: '47', description: 'Must state 47 total actions' },
      { type: 'mentions_key_fact', value: 'read_file', description: 'Must mention the most used tool' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
];

// ── Calculation from Data (10 tests) ──────────────────────────────────────

const calculationTests: readonly SynthesisTest[] = [
  {
    id: 'syn-calc-001', category: 'calculation', difficulty: 'easy',
    userQuery: 'What is the total size of files in my Downloads?',
    toolCall: '[filesystem.list_dir(path="~/Downloads")]',
    toolResult: '{"files": [{"name": "report.pdf", "size": 1000000}, {"name": "image.png", "size": 2000000}, {"name": "video.mp4", "size": 7000000}]}',
    constraints: [
      { type: 'correct_calculation', value: '10', description: 'Must calculate total as ~10MB' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-002', category: 'calculation', difficulty: 'medium',
    userQuery: 'How many hours of meetings do I have today?',
    toolCall: '[calendar.list_events(start="2026-02-20")]',
    toolResult: '{"events": [{"title": "Standup", "start": "2026-02-20T09:00:00", "end": "2026-02-20T09:30:00"}, {"title": "Design Review", "start": "2026-02-20T14:00:00", "end": "2026-02-20T15:00:00"}, {"title": "Sprint Planning", "start": "2026-02-20T15:30:00", "end": "2026-02-20T17:00:00"}]}',
    constraints: [
      { type: 'correct_calculation', value: '3', description: 'Must calculate total as 3 hours (0.5 + 1 + 1.5)' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-003', category: 'calculation', difficulty: 'medium',
    userQuery: 'How many tasks do I have by priority level?',
    toolCall: '[task.list_tasks(status="pending")]',
    toolResult: '{"tasks": [{"title": "A", "priority": "high"}, {"title": "B", "priority": "high"}, {"title": "C", "priority": "medium"}, {"title": "D", "priority": "medium"}, {"title": "E", "priority": "medium"}, {"title": "F", "priority": "low"}]}',
    constraints: [
      { type: 'correct_count', value: '2', description: 'Must count 2 high priority tasks' },
      { type: 'correct_count', value: '3', description: 'Must count 3 medium priority tasks' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-004', category: 'calculation', difficulty: 'hard',
    userQuery: 'What is my total spending from the expense receipts?',
    toolCall: '[data.query_sqlite(database="expenses.db", query="SELECT * FROM receipts")]',
    toolResult: '{"rows": [{"vendor": "Acme Corp", "amount": 247.50, "date": "2026-02-01"}, {"vendor": "Cloud Services Inc", "amount": 1500.00, "date": "2026-02-05"}, {"vendor": "Office Supply Co", "amount": 89.99, "date": "2026-02-10"}, {"vendor": "Travel Express", "amount": 425.00, "date": "2026-02-15"}]}',
    constraints: [
      { type: 'correct_calculation', value: '2262', description: 'Must calculate total as $2,262.49' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-005', category: 'calculation', difficulty: 'easy',
    userQuery: 'How much free disk space do I have as a percentage?',
    toolCall: '[system.get_system_info()]',
    toolResult: '{"os": "macOS 15.3", "disk_total": "1000GB", "disk_free": "350GB", "memory": "16GB"}',
    constraints: [
      { type: 'correct_calculation', value: '35', description: 'Must calculate 35% free' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-006', category: 'calculation', difficulty: 'hard',
    userQuery: 'What percentage of my emails this week were from external senders?',
    toolCall: '[email.search_emails(date_from="2026-02-16")]',
    toolResult: '{"emails": [{"from": "alice@company.com"}, {"from": "vendor@acme.com"}, {"from": "bob@company.com"}, {"from": "newsletter@tech.io"}, {"from": "carol@company.com"}, {"from": "support@vendor.co"}, {"from": "dave@company.com"}, {"from": "partner@agency.com"}, {"from": "eve@company.com"}, {"from": "noreply@service.net"}]}',
    constraints: [
      { type: 'correct_calculation', value: '50', description: 'Must calculate 50% external (5 of 10)' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-007', category: 'calculation', difficulty: 'medium',
    userQuery: 'How many days until my earliest task is due?',
    toolCall: '[task.list_tasks(status="pending")]',
    toolResult: '{"tasks": [{"title": "Submit report", "due_date": "2026-02-25"}, {"title": "Code review", "due_date": "2026-02-22"}, {"title": "Plan workshop", "due_date": "2026-03-01"}]}',
    constraints: [
      { type: 'correct_calculation', value: '2', description: 'Must calculate 2 days until Feb 22' },
      { type: 'mentions_key_fact', value: 'code review', description: 'Must identify the earliest task' },
    ],
  },
  {
    id: 'syn-calc-008', category: 'calculation', difficulty: 'hard',
    userQuery: 'What is the average file size in my project folder?',
    toolCall: '[filesystem.list_dir(path="~/Projects/app")]',
    toolResult: '{"files": [{"name": "index.ts", "size": 4200}, {"name": "app.ts", "size": 8600}, {"name": "utils.ts", "size": 3200}, {"name": "config.ts", "size": 1800}, {"name": "types.ts", "size": 2200}]}',
    constraints: [
      { type: 'correct_calculation', value: '4000', description: 'Must calculate average as 4,000 bytes (20000/5)' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-009', category: 'calculation', difficulty: 'medium',
    userQuery: 'How many free slots do I have tomorrow that are at least 1 hour long?',
    toolCall: '[calendar.find_free_slots(date="2026-02-21", min_duration=60)]',
    toolResult: '{"free_slots": [{"start": "2026-02-21T08:00:00", "end": "2026-02-21T09:00:00", "duration_min": 60}, {"start": "2026-02-21T10:00:00", "end": "2026-02-21T12:00:00", "duration_min": 120}, {"start": "2026-02-21T13:00:00", "end": "2026-02-21T14:30:00", "duration_min": 90}]}',
    constraints: [
      { type: 'correct_count', value: '3', description: 'Must state 3 free slots' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-calc-010', category: 'calculation', difficulty: 'hard',
    userQuery: 'How much disk space could I save by removing duplicate files?',
    toolCall: '[security.find_duplicates(path="~/Documents")]',
    toolResult: '{"duplicate_groups": [{"hash": "abc123", "files": ["~/Documents/report.pdf", "~/Documents/backup/report.pdf"], "size": 245000}, {"hash": "def456", "files": ["~/Documents/photo.jpg", "~/Documents/old/photo.jpg", "~/Documents/archive/photo.jpg"], "size": 3500000}]}',
    constraints: [
      { type: 'correct_calculation', value: '3745', description: 'Must calculate saveable space (245000 + 3500000 = 3745000 bytes, or ~3.7MB)' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
];

// ── Error / Empty Result Handling (10 tests) ──────────────────────────────

const errorHandlingTests: readonly SynthesisTest[] = [
  {
    id: 'syn-err-001', category: 'error-handling', difficulty: 'easy',
    userQuery: 'Find all spreadsheets in my Desktop folder',
    toolCall: '[filesystem.search_files(pattern="*.xlsx", path="~/Desktop")]',
    toolResult: '{"matches": [], "total": 0}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge no results found' },
      { type: 'no_hallucination', value: 'spreadsheet.xlsx', description: 'Must not invent file names' },
    ],
  },
  {
    id: 'syn-err-002', category: 'error-handling', difficulty: 'easy',
    userQuery: 'Show me emails from john.doe@test.com',
    toolCall: '[email.search_emails(from="john.doe@test.com")]',
    toolResult: '{"emails": [], "total": 0}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge no emails found' },
      { type: 'no_hallucination', value: 'subject', description: 'Must not invent email subjects' },
    ],
  },
  {
    id: 'syn-err-003', category: 'error-handling', difficulty: 'medium',
    userQuery: 'Search my knowledge base for information about the merger',
    toolCall: '[knowledge.search_documents(query="merger")]',
    toolResult: '{"results": [], "total": 0, "index_status": "last_updated: 2026-01-15"}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge no results' },
      { type: 'no_hallucination', value: 'merger details', description: 'Must not invent merger information' },
    ],
  },
  {
    id: 'syn-err-004', category: 'error-handling', difficulty: 'medium',
    userQuery: 'What overdue tasks do I have?',
    toolCall: '[task.get_overdue()]',
    toolResult: '{"tasks": [], "total": 0}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge no overdue tasks' },
      { type: 'no_hallucination', value: 'submit', description: 'Must not invent task names' },
    ],
  },
  {
    id: 'syn-err-005', category: 'error-handling', difficulty: 'hard',
    userQuery: 'Find free meeting slots for a 2-hour block today',
    toolCall: '[calendar.find_free_slots(date="2026-02-20", min_duration=120)]',
    toolResult: '{"free_slots": [], "message": "No slots of 120 minutes or longer available today"}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge no slots available' },
      { type: 'no_hallucination', value: '10:00', description: 'Must not invent time slots' },
    ],
  },
  {
    id: 'syn-err-006', category: 'error-handling', difficulty: 'medium',
    userQuery: 'Scan my project for API keys or secrets',
    toolCall: '[security.scan_for_secrets(path="~/Projects/myapp")]',
    toolResult: '{"findings": [], "total": 0, "files_scanned": 84}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge clean scan' },
      { type: 'mentions_key_fact', value: '84', description: 'Should mention number of files scanned' },
    ],
  },
  {
    id: 'syn-err-007', category: 'error-handling', difficulty: 'hard',
    userQuery: 'Read the file at ~/Documents/missing-report.pdf',
    toolCall: '[filesystem.read_file(path="~/Documents/missing-report.pdf")]',
    toolResult: '{"error": "ENOENT: no such file or directory", "path": "~/Documents/missing-report.pdf"}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge the file was not found' },
      { type: 'no_hallucination', value: 'executive summary', description: 'Must not invent file contents' },
    ],
  },
  {
    id: 'syn-err-008', category: 'error-handling', difficulty: 'easy',
    userQuery: 'List my email drafts',
    toolCall: '[email.list_drafts()]',
    toolResult: '{"drafts": [], "total": 0}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge no drafts' },
      { type: 'no_hallucination', value: 'draft-', description: 'Must not invent draft IDs' },
    ],
  },
  {
    id: 'syn-err-009', category: 'error-handling', difficulty: 'hard',
    userQuery: 'Decrypt the confidential file at ~/secure/data.enc',
    toolCall: '[security.decrypt_file(path="~/secure/data.enc")]',
    toolResult: '{"error": "DecryptionError: File is not encrypted or uses an unknown encryption format", "path": "~/secure/data.enc"}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge decryption failed' },
      { type: 'no_hallucination', value: 'decrypted content', description: 'Must not invent file contents' },
    ],
  },
  {
    id: 'syn-err-010', category: 'error-handling', difficulty: 'medium',
    userQuery: 'Find duplicate files in my empty project folder',
    toolCall: '[security.find_duplicates(path="~/Projects/new-project")]',
    toolResult: '{"duplicate_groups": [], "total_duplicates": 0, "files_scanned": 0, "message": "Directory is empty or contains no files"}',
    constraints: [
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge no files to scan' },
      { type: 'no_hallucination', value: 'duplicate', description: 'Must not invent duplicate groups' },
    ],
  },
];

// ── Multi-Source Synthesis (10 tests) ─────────────────────────────────────

const multiSourceTests: readonly SynthesisTest[] = [
  {
    id: 'syn-multi-001', category: 'multi-source', difficulty: 'medium',
    userQuery: 'What should I focus on today?',
    toolCall: '[task.daily_briefing()] [calendar.list_events(start="2026-02-20")]',
    toolResult: 'TOOL_RESULT_1: {"tasks": [{"title": "Submit expense report", "priority": "high", "due_date": "2026-02-20"}, {"title": "Review PR #42", "priority": "medium", "due_date": "2026-02-22"}], "overdue": 0}\nTOOL_RESULT_2: {"events": [{"title": "Team Standup", "start": "09:00"}, {"title": "Sprint Planning", "start": "14:00"}]}',
    constraints: [
      { type: 'references_source', value: 'expense report', description: 'Must reference the task' },
      { type: 'references_source', value: 'standup', description: 'Must reference the meeting' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-multi-002', category: 'multi-source', difficulty: 'hard',
    userQuery: 'Give me a morning briefing covering tasks, calendar, and emails',
    toolCall: '[task.daily_briefing()] [calendar.list_events(start="2026-02-20")] [email.search_emails(date_from="2026-02-19")]',
    toolResult: 'TOOL_RESULT_1: {"tasks": [{"title": "Finalize proposal", "priority": "high"}]}\nTOOL_RESULT_2: {"events": [{"title": "Client Call", "start": "10:00"}, {"title": "Team Lunch", "start": "12:00"}]}\nTOOL_RESULT_3: {"emails": [{"from": "client@acme.com", "subject": "Proposal Feedback"}, {"from": "boss@company.com", "subject": "Urgent: Review needed"}]}',
    constraints: [
      { type: 'references_source', value: 'proposal', description: 'Must reference the task' },
      { type: 'references_source', value: 'client call', description: 'Must reference the meeting' },
      { type: 'references_source', value: 'urgent', description: 'Must reference the urgent email' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-multi-003', category: 'multi-source', difficulty: 'medium',
    userQuery: 'How is my system doing and are there any security concerns?',
    toolCall: '[system.get_system_info()] [security.scan_for_secrets(path="~/Projects")]',
    toolResult: 'TOOL_RESULT_1: {"os": "macOS 15.3", "cpu": "Apple M3", "memory": "16GB", "disk_free": "45GB"}\nTOOL_RESULT_2: {"findings": [{"type": "api_key", "file": "~/Projects/app/.env", "line": 3}], "total": 1}',
    constraints: [
      { type: 'references_source', value: 'macos', description: 'Must reference system info' },
      { type: 'references_source', value: 'api_key', description: 'Must reference the security finding' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-multi-004', category: 'multi-source', difficulty: 'hard',
    userQuery: 'Compare my meeting commitments with my current task list — am I overcommitted?',
    toolCall: '[meeting.extract_commitments(recording_id="rec-001")] [task.list_tasks(status="pending")]',
    toolResult: 'TOOL_RESULT_1: {"commitments": [{"person": "me", "commitment": "deliver prototype by Friday"}, {"person": "me", "commitment": "review Alice\'s design doc"}, {"person": "me", "commitment": "set up staging environment"}]}\nTOOL_RESULT_2: {"tasks": [{"title": "Fix login bug", "priority": "high"}, {"title": "Update docs", "priority": "medium"}, {"title": "Write tests", "priority": "medium"}, {"title": "Deploy to staging", "priority": "high"}]}',
    constraints: [
      { type: 'references_source', value: 'prototype', description: 'Must reference meeting commitment' },
      { type: 'references_source', value: 'login bug', description: 'Must reference existing task' },
      { type: 'answers_question', value: 'overcommit', description: 'Must address the overcommitment question' },
    ],
  },
  {
    id: 'syn-multi-005', category: 'multi-source', difficulty: 'medium',
    userQuery: 'Check my files and tasks — is there anything urgent I should know about?',
    toolCall: '[task.get_overdue()] [filesystem.search_files(pattern="*urgent*")]',
    toolResult: 'TOOL_RESULT_1: {"tasks": [{"title": "Submit tax forms", "due_date": "2026-02-18", "priority": "high"}]}\nTOOL_RESULT_2: {"matches": [{"path": "~/Documents/urgent-review.pdf", "size": 125000}]}',
    constraints: [
      { type: 'references_source', value: 'tax forms', description: 'Must reference overdue task' },
      { type: 'references_source', value: 'urgent-review', description: 'Must reference the urgent file' },
    ],
  },
  {
    id: 'syn-multi-006', category: 'multi-source', difficulty: 'hard',
    userQuery: 'Prepare a summary for my end-of-day report covering what tools I used and what tasks I completed',
    toolCall: '[audit.get_session_summary()] [task.list_tasks(status="completed")]',
    toolResult: 'TOOL_RESULT_1: {"session": {"tools_used": ["filesystem.read_file", "email.draft_email", "task.update_task"], "total_actions": 12, "duration_min": 180}}\nTOOL_RESULT_2: {"tasks": [{"title": "Review contract", "completed_at": "2026-02-20T11:00:00"}, {"title": "Send weekly update", "completed_at": "2026-02-20T15:00:00"}]}',
    constraints: [
      { type: 'references_source', value: 'read_file', description: 'Must reference tools used' },
      { type: 'references_source', value: 'review contract', description: 'Must reference completed tasks' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-multi-007', category: 'multi-source', difficulty: 'medium',
    userQuery: 'Do I have any meetings related to my current tasks?',
    toolCall: '[calendar.list_events(start="2026-02-20")] [task.list_tasks(status="pending")]',
    toolResult: 'TOOL_RESULT_1: {"events": [{"title": "Budget Review", "start": "2026-02-20T10:00:00"}, {"title": "1:1 with Manager", "start": "2026-02-20T16:00:00"}]}\nTOOL_RESULT_2: {"tasks": [{"title": "Prepare budget presentation", "priority": "high"}, {"title": "Update project timeline", "priority": "medium"}]}',
    constraints: [
      { type: 'references_source', value: 'budget', description: 'Must connect budget meeting with budget task' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-multi-008', category: 'multi-source', difficulty: 'hard',
    userQuery: 'Give me a full status report on the project — files, tasks, and recent activity',
    toolCall: '[filesystem.list_dir(path="~/Projects/app")] [task.list_tasks()] [audit.get_session_summary()]',
    toolResult: 'TOOL_RESULT_1: {"files": [{"name": "src/", "type": "dir"}, {"name": "tests/", "type": "dir"}, {"name": "package.json", "size": 1200}, {"name": "README.md", "size": 3400}]}\nTOOL_RESULT_2: {"tasks": [{"title": "Implement auth", "status": "in_progress"}, {"title": "Write tests", "status": "pending"}]}\nTOOL_RESULT_3: {"session": {"tools_used": ["filesystem.write_file"], "total_actions": 5}}',
    constraints: [
      { type: 'references_source', value: 'package.json', description: 'Must reference project files' },
      { type: 'references_source', value: 'auth', description: 'Must reference in-progress task' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
  {
    id: 'syn-multi-009', category: 'multi-source', difficulty: 'medium',
    userQuery: 'Should I index my documents? Check what is in my Documents folder and if my knowledge base has any entries.',
    toolCall: '[filesystem.list_dir(path="~/Documents")] [knowledge.search_documents(query="*")]',
    toolResult: 'TOOL_RESULT_1: {"files": [{"name": "report.pdf"}, {"name": "notes.md"}, {"name": "contracts/"}, {"name": "presentations/"}]}\nTOOL_RESULT_2: {"results": [], "total": 0, "index_status": "empty"}',
    constraints: [
      { type: 'references_source', value: 'report.pdf', description: 'Must reference existing documents' },
      { type: 'acknowledges_limitation', value: '', description: 'Must acknowledge empty knowledge base' },
    ],
  },
  {
    id: 'syn-multi-010', category: 'multi-source', difficulty: 'hard',
    userQuery: 'Help me plan my day based on meetings, tasks, and emails',
    toolCall: '[calendar.list_events(start="2026-02-20")] [task.list_tasks(status="pending")] [email.search_emails(date_from="2026-02-20")]',
    toolResult: 'TOOL_RESULT_1: {"events": [{"title": "Product Review", "start": "11:00", "end": "12:00"}, {"title": "Sync with Design", "start": "15:00", "end": "15:30"}]}\nTOOL_RESULT_2: {"tasks": [{"title": "Prepare demo", "priority": "high", "due_date": "2026-02-20"}, {"title": "Fix CSS bug", "priority": "low"}]}\nTOOL_RESULT_3: {"emails": [{"from": "cto@company.com", "subject": "Demo expectations", "date": "2026-02-20"}]}',
    constraints: [
      { type: 'references_source', value: 'product review', description: 'Must reference the meeting' },
      { type: 'references_source', value: 'demo', description: 'Must reference the demo task' },
      { type: 'references_source', value: 'cto', description: 'Must reference the CTO email' },
      { type: 'no_raw_dump', value: '', description: 'Must not dump raw JSON' },
    ],
  },
];

// ── Raw Dump Avoidance (10 tests) ─────────────────────────────────────────

const rawDumpTests: readonly SynthesisTest[] = [
  {
    id: 'syn-dump-001', category: 'raw-dump-avoidance', difficulty: 'easy',
    userQuery: 'What files are in my project?',
    toolCall: '[filesystem.list_dir(path="~/Projects/app")]',
    toolResult: '{"files": [{"name": "index.ts", "size": 4200, "modified": "2026-02-19"}, {"name": "app.ts", "size": 8600, "modified": "2026-02-20"}, {"name": "utils.ts", "size": 3200, "modified": "2026-02-18"}]}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'index.ts', description: 'Must mention file names naturally' },
    ],
  },
  {
    id: 'syn-dump-002', category: 'raw-dump-avoidance', difficulty: 'medium',
    userQuery: 'Show me the receipt details from the scanned image',
    toolCall: '[ocr.extract_structured_data(path="~/Documents/receipt.jpg")]',
    toolResult: '{"vendor": "Acme Restaurant", "date": "2026-02-18", "items": [{"name": "Lunch Special", "amount": 15.99}, {"name": "Coffee", "amount": 4.50}], "subtotal": 20.49, "tax": 1.84, "total": 22.33, "payment_method": "Visa ending 4242"}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'acme restaurant', description: 'Must mention vendor name' },
      { type: 'mentions_key_fact', value: '22.33', description: 'Must mention total amount' },
    ],
  },
  {
    id: 'syn-dump-003', category: 'raw-dump-avoidance', difficulty: 'easy',
    userQuery: 'Tell me about my system',
    toolCall: '[system.get_system_info()]',
    toolResult: '{"os": "macOS 15.3", "os_version": "15.3.1", "cpu": "Apple M3 Pro", "cpu_cores": 12, "memory": "36GB", "memory_used": "18GB", "disk_total": "1TB", "disk_free": "456GB", "uptime_hours": 72}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'macos', description: 'Must mention OS naturally' },
      { type: 'mentions_key_fact', value: 'm3', description: 'Must mention CPU naturally' },
    ],
  },
  {
    id: 'syn-dump-004', category: 'raw-dump-avoidance', difficulty: 'medium',
    userQuery: 'Summarize the meeting action items',
    toolCall: '[meeting.extract_action_items(recording_id="rec-001")]',
    toolResult: '{"action_items": [{"assignee": "Alice", "task": "Complete backend migration plan", "deadline": "2026-02-25"}, {"assignee": "Bob", "task": "Update the roadmap document", "deadline": "2026-02-22"}, {"assignee": "Carol", "task": "Schedule stakeholder review", "deadline": "2026-02-28"}, {"assignee": "Me", "task": "Send meeting notes to team", "deadline": "2026-02-20"}]}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'alice', description: 'Must mention Alice by name' },
      { type: 'mentions_key_fact', value: 'migration', description: 'Must mention the migration task' },
    ],
  },
  {
    id: 'syn-dump-005', category: 'raw-dump-avoidance', difficulty: 'hard',
    userQuery: 'What does the sales data look like?',
    toolCall: '[data.query_sqlite(database="sales.db", query="SELECT * FROM orders LIMIT 5")]',
    toolResult: '{"columns": ["id", "customer", "product", "amount", "date"], "rows": [{"id": 1, "customer": "Acme Corp", "product": "Enterprise Plan", "amount": 15000, "date": "2026-01-15"}, {"id": 2, "customer": "TechStart Inc", "product": "Pro Plan", "amount": 5000, "date": "2026-01-20"}, {"id": 3, "customer": "GlobalCo", "product": "Enterprise Plan", "amount": 15000, "date": "2026-02-01"}, {"id": 4, "customer": "SmallBiz LLC", "product": "Starter Plan", "amount": 1000, "date": "2026-02-05"}, {"id": 5, "customer": "MegaCorp", "product": "Enterprise Plan", "amount": 15000, "date": "2026-02-10"}]}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'enterprise', description: 'Must mention the enterprise plan' },
      { type: 'mentions_key_fact', value: 'acme', description: 'Must mention a customer by name' },
    ],
  },
  {
    id: 'syn-dump-006', category: 'raw-dump-avoidance', difficulty: 'easy',
    userQuery: 'Check my clipboard history',
    toolCall: '[clipboard.clipboard_history()]',
    toolResult: '{"entries": [{"content": "https://github.com/repo/pull/42", "timestamp": "2026-02-20T14:30:00", "type": "url"}, {"content": "Meeting notes from standup", "timestamp": "2026-02-20T11:00:00", "type": "text"}, {"content": "SELECT * FROM users WHERE active = true", "timestamp": "2026-02-20T09:45:00", "type": "text"}]}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'correct_count', value: '3', description: 'Must state there are 3 entries' },
    ],
  },
  {
    id: 'syn-dump-007', category: 'raw-dump-avoidance', difficulty: 'medium',
    userQuery: 'What processes are using the most resources?',
    toolCall: '[system.list_processes()]',
    toolResult: '{"processes": [{"name": "Chrome", "pid": 1234, "cpu_percent": 25.3, "memory_mb": 1800}, {"name": "Xcode", "pid": 5678, "cpu_percent": 18.7, "memory_mb": 3200}, {"name": "Docker", "pid": 9012, "cpu_percent": 12.1, "memory_mb": 2400}, {"name": "Slack", "pid": 3456, "cpu_percent": 5.2, "memory_mb": 800}]}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'chrome', description: 'Must mention Chrome as top CPU user' },
      { type: 'mentions_key_fact', value: 'xcode', description: 'Must mention Xcode' },
    ],
  },
  {
    id: 'syn-dump-008', category: 'raw-dump-avoidance', difficulty: 'hard',
    userQuery: 'Summarize the PII scan results for the HR folder',
    toolCall: '[security.scan_for_pii(path="~/Documents/HR")]',
    toolResult: '{"findings": [{"type": "ssn", "value": "***-**-1234", "file": "employees.xlsx", "line": 15}, {"type": "ssn", "value": "***-**-5678", "file": "employees.xlsx", "line": 23}, {"type": "phone", "value": "555-0123", "file": "contacts.csv", "line": 8}, {"type": "phone", "value": "555-0456", "file": "contacts.csv", "line": 12}, {"type": "email", "value": "personal@gmail.com", "file": "contacts.csv", "line": 3}, {"type": "dob", "value": "1990-05-15", "file": "employees.xlsx", "line": 15}], "total": 6, "files_scanned": 8}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'correct_count', value: '6', description: 'Must state 6 total findings' },
      { type: 'mentions_key_fact', value: 'ssn', description: 'Must mention SSN findings' },
    ],
  },
  {
    id: 'syn-dump-009', category: 'raw-dump-avoidance', difficulty: 'medium',
    userQuery: 'What is in the document I just extracted text from?',
    toolCall: '[document.extract_text(path="~/Documents/proposal.pdf")]',
    toolResult: '{"text": "Project Proposal: AI-Powered Analytics Dashboard\\n\\nObjective: Build a real-time analytics dashboard using machine learning models for predictive insights.\\n\\nBudget: $150,000\\nTimeline: 6 months\\nTeam Size: 5 engineers\\n\\nKey Milestones:\\n1. Data pipeline (Month 1-2)\\n2. ML model training (Month 2-3)\\n3. Dashboard UI (Month 3-5)\\n4. Testing and deployment (Month 5-6)", "pages": 3, "format": "pdf"}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'analytics dashboard', description: 'Must mention the project name' },
      { type: 'mentions_key_fact', value: '150', description: 'Must mention the budget' },
    ],
  },
  {
    id: 'syn-dump-010', category: 'raw-dump-avoidance', difficulty: 'hard',
    userQuery: 'Show me the anomalies detected in the sales data',
    toolCall: '[data.summarize_anomalies(database="sales.db", table="transactions")]',
    toolResult: '{"anomalies": [{"type": "outlier", "column": "amount", "value": 99999.99, "row_id": 42, "expected_range": "100-5000", "z_score": 4.2}, {"type": "missing_data", "column": "customer_id", "affected_rows": 15, "percentage": 2.3}, {"type": "duplicate", "columns": ["order_id", "date"], "duplicate_count": 3}, {"type": "outlier", "column": "quantity", "value": -5, "row_id": 87, "expected_range": "1-100", "z_score": -3.8}], "total_rows_analyzed": 650}',
    constraints: [
      { type: 'no_raw_dump', value: '', description: 'Must not paste raw JSON' },
      { type: 'mentions_key_fact', value: 'outlier', description: 'Must mention outlier anomalies' },
      { type: 'correct_count', value: '650', description: 'Must mention rows analyzed' },
    ],
  },
];

// ── Exports ──────────────────────────────────────────────────────────────

export const allSynthesisTests: readonly SynthesisTest[] = [
  ...factExtractionTests,
  ...calculationTests,
  ...errorHandlingTests,
  ...multiSourceTests,
  ...rawDumpTests,
];

export {
  factExtractionTests,
  calculationTests,
  errorHandlingTests,
  multiSourceTests,
  rawDumpTests,
};
