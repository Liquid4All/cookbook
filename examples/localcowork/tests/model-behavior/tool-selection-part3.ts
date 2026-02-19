/**
 * Tool Selection Tests — Part 3: Email, Meeting, Knowledge, System, Clipboard, Audit.
 *
 * 30 tests covering the remaining servers.
 */

import type { ToolSelectionTest } from './types';

/** Email — 8 tests covering the email server. */
export const emailTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-email-001',
    category: 'email',
    prompt: 'Draft an email to John about the project update',
    expectedTools: ['email.draft_email'],
    difficulty: 'easy',
  },
  {
    id: 'ts-email-002',
    category: 'email',
    prompt: 'Search my emails for anything about the merger',
    expectedTools: ['email.search_emails'],
    expectedParamKeys: { 'email.search_emails': ['query'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-email-003',
    category: 'email',
    prompt: 'Show me the thread summary for this email chain about the budget',
    expectedTools: ['email.summarize_thread'],
    difficulty: 'easy',
  },
  {
    id: 'ts-email-004',
    category: 'email',
    prompt: 'Show me all my draft emails',
    expectedTools: ['email.list_drafts'],
    difficulty: 'easy',
  },
  {
    id: 'ts-email-005',
    category: 'email',
    prompt: 'Send the draft email I wrote to Sarah',
    expectedTools: ['email.send_draft'],
    difficulty: 'easy',
  },
  {
    id: 'ts-email-006',
    category: 'email',
    prompt: 'Write a follow-up email to the client thanking them for the meeting',
    expectedTools: ['email.draft_email'],
    difficulty: 'easy',
  },
  {
    id: 'ts-email-007',
    category: 'email',
    prompt: 'Find all emails from my manager in the last two weeks',
    expectedTools: ['email.search_emails'],
    difficulty: 'medium',
  },
  {
    id: 'ts-email-008',
    category: 'email',
    prompt: 'Summarize the back-and-forth about the contract negotiations',
    expectedTools: ['email.summarize_thread'],
    difficulty: 'medium',
  },
];

/** Meeting & Audio — 7 tests covering the meeting server. */
export const meetingAudioTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-meet-001',
    category: 'meeting-audio',
    prompt: 'Transcribe this meeting recording',
    context: ['Recording at /Users/me/meetings/standup-2024-01-15.mp3'],
    expectedTools: ['meeting.transcribe_audio'],
    expectedParamKeys: { 'meeting.transcribe_audio': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-meet-002',
    category: 'meeting-audio',
    prompt: 'Extract action items from the meeting notes',
    context: ['Meeting transcript is available from the last transcription'],
    expectedTools: ['meeting.extract_action_items'],
    difficulty: 'easy',
  },
  {
    id: 'ts-meet-003',
    category: 'meeting-audio',
    prompt: 'Generate meeting minutes from the team standup recording',
    context: ['Recording at /Users/me/meetings/team-standup.wav'],
    expectedTools: ['meeting.generate_minutes'],
    difficulty: 'easy',
  },
  {
    id: 'ts-meet-004',
    category: 'meeting-audio',
    prompt: 'What commitments were made during the client call?',
    context: ['The transcript from the client call is available'],
    expectedTools: ['meeting.extract_commitments'],
    difficulty: 'medium',
  },
  {
    id: 'ts-meet-005',
    category: 'meeting-audio',
    prompt: 'Convert this voice memo to text',
    context: ['Voice memo at /Users/me/recordings/voice-note.m4a'],
    expectedTools: ['meeting.transcribe_audio'],
    difficulty: 'easy',
  },
  {
    id: 'ts-meet-006',
    category: 'meeting-audio',
    prompt: 'Pull out all the action items and who they were assigned to',
    context: ['We just finished transcribing the all-hands meeting'],
    expectedTools: ['meeting.extract_action_items'],
    difficulty: 'medium',
  },
  {
    id: 'ts-meet-007',
    category: 'meeting-audio',
    prompt: 'Create formal meeting minutes with attendees, decisions, and next steps',
    context: ['The board meeting recording has been transcribed'],
    expectedTools: ['meeting.generate_minutes'],
    difficulty: 'medium',
  },
];

/** Knowledge & Search — 7 tests covering the knowledge server. */
export const knowledgeSearchTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-know-001',
    category: 'knowledge-search',
    prompt: 'Index my Documents folder for search',
    expectedTools: ['knowledge.index_folder'],
    expectedParamKeys: { 'knowledge.index_folder': ['path'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-know-002',
    category: 'knowledge-search',
    prompt: 'Search my documents for information about revenue projections',
    expectedTools: ['knowledge.search_documents'],
    expectedParamKeys: { 'knowledge.search_documents': ['query'] },
    difficulty: 'easy',
  },
  {
    id: 'ts-know-003',
    category: 'knowledge-search',
    prompt: 'What do my files say about the Q3 marketing strategy?',
    expectedTools: ['knowledge.ask_about_files'],
    difficulty: 'easy',
  },
  {
    id: 'ts-know-004',
    category: 'knowledge-search',
    prompt: 'Update the search index — I added new files to the project folder',
    expectedTools: ['knowledge.update_index'],
    difficulty: 'easy',
  },
  {
    id: 'ts-know-005',
    category: 'knowledge-search',
    prompt: 'Find documents related to this paragraph about supply chain risks',
    expectedTools: ['knowledge.get_related_chunks'],
    difficulty: 'medium',
  },
  {
    id: 'ts-know-006',
    category: 'knowledge-search',
    prompt: 'Search across all my indexed files for mentions of "Project Atlas"',
    expectedTools: ['knowledge.search_documents'],
    difficulty: 'easy',
  },
  {
    id: 'ts-know-007',
    category: 'knowledge-search',
    prompt: 'Make my Reports folder searchable so I can query it later',
    expectedTools: ['knowledge.index_folder'],
    difficulty: 'easy',
  },
];

/** System & Clipboard — 5 tests covering system and clipboard servers. */
export const systemClipboardTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-sys-001',
    category: 'system-clipboard',
    prompt: 'What is on my clipboard right now?',
    expectedTools: ['clipboard.get_clipboard'],
    difficulty: 'easy',
  },
  {
    id: 'ts-sys-002',
    category: 'system-clipboard',
    prompt: 'Copy this text to my clipboard: "Meeting at 3pm in Room 201"',
    expectedTools: ['clipboard.set_clipboard'],
    difficulty: 'easy',
  },
  {
    id: 'ts-sys-003',
    category: 'system-clipboard',
    prompt: 'What processes are currently running on my system?',
    expectedTools: ['system.list_processes'],
    difficulty: 'easy',
  },
  {
    id: 'ts-sys-004',
    category: 'system-clipboard',
    prompt: 'Open this PDF in Preview',
    context: ['File at /Users/me/Documents/report.pdf'],
    expectedTools: ['system.open_file_with'],
    difficulty: 'easy',
  },
  {
    id: 'ts-sys-005',
    category: 'system-clipboard',
    prompt: 'Show me my clipboard history',
    expectedTools: ['clipboard.clipboard_history'],
    difficulty: 'easy',
  },
];

/** Audit — 3 tests covering the audit server. */
export const auditTests: readonly ToolSelectionTest[] = [
  {
    id: 'ts-audit-001',
    category: 'audit',
    prompt: 'Show me what tools were used in the last session',
    expectedTools: ['audit.get_tool_log'],
    difficulty: 'easy',
  },
  {
    id: 'ts-audit-002',
    category: 'audit',
    prompt: 'Generate an audit report for this week',
    expectedTools: ['audit.generate_audit_report'],
    difficulty: 'easy',
  },
  {
    id: 'ts-audit-003',
    category: 'audit',
    prompt: 'Give me a summary of what happened in my last work session',
    expectedTools: ['audit.get_session_summary'],
    difficulty: 'easy',
  },
];
