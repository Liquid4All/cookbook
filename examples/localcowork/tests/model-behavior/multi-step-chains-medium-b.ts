/**
 * Multi-Step Chain Tests — Medium Chains Part B (tests 11-20, 4-6 tools each).
 */

import type { MultiStepTest } from './types';

/** Medium chains part B: tests 11-20. */
export const mediumChainTestsB: readonly MultiStepTest[] = [
  {
    id: 'ms-medium-011',
    category: 'medium-chain',
    scenario: 'File organization and deduplication',
    steps: [
      {
        description: 'List the target folder',
        prompt: 'Show me everything in the project archive folder',
        expectedTools: ['filesystem.list_dir'],
      },
      {
        description: 'Find duplicates',
        prompt: 'Find any duplicate files in there',
        expectedTools: ['security.find_duplicates'],
      },
      {
        description: 'Get metadata on large files',
        prompt: 'What are the sizes of the largest files?',
        expectedTools: ['filesystem.get_metadata'],
      },
      {
        description: 'Propose cleanup',
        prompt: 'Suggest which files I can safely remove',
        expectedTools: ['security.propose_cleanup'],
      },
      {
        description: 'Delete the recommended files',
        prompt: 'Go ahead and delete the duplicates you identified',
        expectedTools: ['filesystem.delete_file'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-012',
    category: 'medium-chain',
    scenario: 'Contract review and action item extraction',
    steps: [
      {
        description: 'Extract contract text',
        prompt: 'Extract the text from the new vendor contract PDF',
        expectedTools: ['document.extract_text'],
      },
      {
        description: 'Search knowledge base for similar contracts',
        prompt: 'Have we had similar terms in previous vendor contracts?',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Diff with previous contract',
        prompt: 'Compare this with our previous vendor contract',
        expectedTools: ['document.diff_documents'],
      },
      {
        description: 'Create task for legal review',
        prompt: 'Create a task for legal to review the new terms by Friday',
        expectedTools: ['task.create_task'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-013',
    category: 'medium-chain',
    scenario: 'Email triage and task creation',
    steps: [
      {
        description: 'Search for unread project emails',
        prompt: 'Find all emails about Project Phoenix from this week',
        expectedTools: ['email.search_emails'],
      },
      {
        description: 'Summarize the thread',
        prompt: 'Summarize the key discussion points',
        expectedTools: ['email.summarize_thread'],
      },
      {
        description: 'Create tasks from the email discussion',
        prompt: 'Create tasks for each action item mentioned in the emails',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Draft a status update reply',
        prompt: 'Draft a reply with our status update',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-014',
    category: 'medium-chain',
    scenario: 'Scanned document archival pipeline',
    steps: [
      {
        description: 'OCR the scanned PDF',
        prompt: 'Extract text from the scanned multi-page document',
        expectedTools: ['ocr.extract_text_from_pdf'],
      },
      {
        description: 'Extract tables from specific pages',
        prompt: 'Extract the data table on page 3',
        expectedTools: ['ocr.extract_table'],
      },
      {
        description: 'Save the table data',
        prompt: 'Save the extracted table data to a CSV',
        expectedTools: ['data.write_csv'],
      },
      {
        description: 'Write the full text to a file',
        prompt: 'Save the full OCR text to a searchable text file',
        expectedTools: ['filesystem.write_file'],
      },
      {
        description: 'Index for future search',
        prompt: 'Index this document for search',
        expectedTools: ['knowledge.index_folder'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-015',
    category: 'medium-chain',
    scenario: 'Morning routine automation',
    steps: [
      {
        description: 'Get daily briefing',
        prompt: 'Start my morning briefing',
        expectedTools: ['task.daily_briefing'],
      },
      {
        description: 'Check calendar',
        prompt: 'What meetings do I have today?',
        expectedTools: ['calendar.list_events'],
      },
      {
        description: 'Check for overdue tasks',
        prompt: 'Any overdue tasks I need to address?',
        expectedTools: ['task.get_overdue'],
      },
      {
        description: 'Search for emails needing response',
        prompt: 'Show me any urgent emails from yesterday',
        expectedTools: ['email.search_emails'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-016',
    category: 'medium-chain',
    scenario: 'Spreadsheet to database migration',
    steps: [
      {
        description: 'Read the spreadsheet',
        prompt: 'Read the data from the master inventory spreadsheet',
        expectedTools: ['document.read_spreadsheet'],
      },
      {
        description: 'Deduplicate entries',
        prompt: 'Remove any duplicate entries from the data',
        expectedTools: ['data.deduplicate_records'],
      },
      {
        description: 'Check for anomalies',
        prompt: 'Flag any entries with suspicious values',
        expectedTools: ['data.summarize_anomalies'],
      },
      {
        description: 'Write to SQLite',
        prompt: 'Import the clean data into the inventory database',
        expectedTools: ['data.write_sqlite'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-017',
    category: 'medium-chain',
    scenario: 'Client meeting follow-up',
    steps: [
      {
        description: 'Transcribe meeting',
        prompt: 'Transcribe the client meeting recording from this morning',
        expectedTools: ['meeting.transcribe_audio'],
      },
      {
        description: 'Extract commitments',
        prompt: 'What commitments did we make to the client?',
        expectedTools: ['meeting.extract_commitments'],
      },
      {
        description: 'Create tasks for commitments',
        prompt: 'Create a task for each commitment with the agreed deadlines',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Draft follow-up email',
        prompt: 'Draft a follow-up email to the client summarizing our commitments',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-018',
    category: 'medium-chain',
    scenario: 'End-of-day audit and cleanup',
    steps: [
      {
        description: 'Get the session audit log',
        prompt: 'Show me what I did today using the tool log',
        expectedTools: ['audit.get_tool_log'],
      },
      {
        description: 'Generate session summary',
        prompt: "Generate a summary of today's session",
        expectedTools: ['audit.get_session_summary'],
      },
      {
        description: 'Check open tasks',
        prompt: 'What tasks are still open?',
        expectedTools: ['task.list_tasks'],
      },
      {
        description: 'Generate daily report',
        prompt: 'Create a daily activity report as a PDF',
        expectedTools: ['document.create_pdf'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-019',
    category: 'medium-chain',
    scenario: 'PDF form filling from database',
    steps: [
      {
        description: 'Query database for employee details',
        prompt: 'Look up the employee record for John Smith in the HR database',
        expectedTools: ['data.query_sqlite'],
      },
      {
        description: 'Read the form template',
        prompt: 'Show me what fields are in the onboarding form PDF',
        expectedTools: ['document.extract_text'],
      },
      {
        description: 'Fill the form',
        prompt: "Fill in the onboarding PDF form with John's information",
        expectedTools: ['document.fill_pdf_form'],
      },
      {
        description: 'Move to the completed forms folder',
        prompt: 'Move the completed form to the HR/Completed folder',
        expectedTools: ['filesystem.move_file'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-020',
    category: 'medium-chain',
    scenario: 'Research topic across multiple sources',
    steps: [
      {
        description: 'Search knowledge base',
        prompt: 'Search my indexed documents for information on market trends',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Find related content',
        prompt: 'Find documents related to these initial results',
        expectedTools: ['knowledge.get_related_chunks'],
      },
      {
        description: 'Search emails for additional context',
        prompt: 'Also check my emails for any market analysis reports',
        expectedTools: ['email.search_emails'],
      },
      {
        description: 'Create a summary document',
        prompt: 'Create a Word document summarizing what I found',
        expectedTools: ['document.create_docx'],
      },
    ],
    difficulty: 'medium',
  },
];
