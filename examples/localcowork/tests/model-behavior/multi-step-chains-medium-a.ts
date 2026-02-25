/**
 * Multi-Step Chain Tests — Medium Chains Part A (tests 1-10, 4-6 tools each).
 */

import type { MultiStepTest } from './types';

/** Medium chains part A: tests 1-10. */
export const mediumChainTestsA: readonly MultiStepTest[] = [
  {
    id: 'ms-medium-001',
    category: 'medium-chain',
    scenario: 'Scan for secrets, generate report, and email it',
    steps: [
      {
        description: 'Scan the project for secrets',
        prompt: 'Scan my codebase for exposed credentials',
        expectedTools: ['security.scan_for_secrets'],
      },
      {
        description: 'Generate an audit report of the findings',
        prompt: 'Create a detailed audit report of the security scan',
        expectedTools: ['audit.generate_audit_report'],
      },
      {
        description: 'Export the report as PDF',
        prompt: 'Export the audit report as a PDF file',
        expectedTools: ['audit.export_audit_pdf'],
      },
      {
        description: 'Draft an email with the report',
        prompt: 'Draft an email to the security team with the findings attached',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-002',
    category: 'medium-chain',
    scenario: 'Transcribe meeting, create tasks, and add to calendar',
    steps: [
      {
        description: 'Transcribe the meeting',
        prompt: 'Transcribe the team standup recording',
        expectedTools: ['meeting.transcribe_audio'],
      },
      {
        description: 'Extract action items',
        prompt: 'Pull out all the action items from the transcript',
        expectedTools: ['meeting.extract_action_items'],
      },
      {
        description: 'Create tasks for each action item',
        prompt: 'Create tasks for each of these action items',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Schedule follow-up meeting',
        prompt: 'Schedule a follow-up meeting for next Tuesday at 10am',
        expectedTools: ['calendar.create_event'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-003',
    category: 'medium-chain',
    scenario: 'OCR receipts and create expense report',
    steps: [
      {
        description: 'List receipt images',
        prompt: 'Show me all the receipt images in my Receipts folder',
        expectedTools: ['filesystem.search_files'],
      },
      {
        description: 'OCR the first receipt',
        prompt: 'Extract the text from the first receipt image',
        expectedTools: ['ocr.extract_text_from_image'],
      },
      {
        description: 'Extract structured data from OCR text',
        prompt: 'Extract the vendor, date, and total from the receipt text',
        expectedTools: ['ocr.extract_structured_data'],
      },
      {
        description: 'Write the data to CSV',
        prompt: 'Add this receipt data to my expense tracking CSV',
        expectedTools: ['data.write_csv'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-004',
    category: 'medium-chain',
    scenario: 'Document comparison and stakeholder notification',
    steps: [
      {
        description: 'Extract text from both documents',
        prompt: 'Extract text from the old and new policy documents',
        expectedTools: ['document.extract_text'],
      },
      {
        description: 'Diff the documents',
        prompt: 'Compare the two versions and show me what changed',
        expectedTools: ['document.diff_documents'],
      },
      {
        description: 'Create a summary PDF',
        prompt: 'Create a PDF highlighting the key changes',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Draft notification email',
        prompt: 'Draft an email to the team about the policy updates',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-005',
    category: 'medium-chain',
    scenario: 'Knowledge base setup and question answering',
    steps: [
      {
        description: 'List the project documentation',
        prompt: 'Show me what docs are in the project Documentation folder',
        expectedTools: ['filesystem.list_dir'],
      },
      {
        description: 'Index the documentation',
        prompt: 'Index all these documents for search',
        expectedTools: ['knowledge.index_folder'],
      },
      {
        description: 'Search for specific topic',
        prompt: 'Search for information about the API authentication flow',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Ask a detailed question',
        prompt: 'How does the token refresh mechanism work according to these docs?',
        expectedTools: ['knowledge.ask_about_files'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-006',
    category: 'medium-chain',
    scenario: 'Database cleanup and anomaly detection',
    steps: [
      {
        description: 'Query the database for recent entries',
        prompt: 'Show me all transactions from the last 30 days',
        expectedTools: ['data.query_sqlite'],
      },
      {
        description: 'Check for duplicates',
        prompt: 'Are there any duplicate transactions?',
        expectedTools: ['data.deduplicate_records'],
      },
      {
        description: 'Analyze for anomalies',
        prompt: 'Look for any unusual or suspicious transactions',
        expectedTools: ['data.summarize_anomalies'],
      },
      {
        description: 'Export clean data',
        prompt: 'Export the clean deduplicated data to a CSV',
        expectedTools: ['data.write_csv'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-007',
    category: 'medium-chain',
    scenario: 'Security audit with cleanup recommendations',
    steps: [
      {
        description: 'Scan for PII',
        prompt: 'Scan the shared drive for files with personal information',
        expectedTools: ['security.scan_for_pii'],
      },
      {
        description: 'Find duplicate files',
        prompt: 'Also find any duplicate files taking up space',
        expectedTools: ['security.find_duplicates'],
      },
      {
        description: 'Propose cleanup actions',
        prompt: 'What files can I safely clean up?',
        expectedTools: ['security.propose_cleanup'],
      },
      {
        description: 'Encrypt sensitive files',
        prompt: 'Encrypt the files that contain PII',
        expectedTools: ['security.encrypt_file'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-008',
    category: 'medium-chain',
    scenario: 'Meeting preparation workflow',
    steps: [
      {
        description: 'Check calendar for meeting details',
        prompt: 'What is my next meeting today?',
        expectedTools: ['calendar.list_events'],
      },
      {
        description: 'Search for related documents',
        prompt: 'Find any documents related to this meeting topic',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Get overdue tasks for this project',
        prompt: 'Show me any overdue tasks related to this project',
        expectedTools: ['task.get_overdue'],
      },
      {
        description: 'Draft agenda email',
        prompt: 'Draft an email with the meeting agenda and status updates',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-009',
    category: 'medium-chain',
    scenario: 'Invoice processing from scan to database',
    steps: [
      {
        description: 'OCR the invoice scan',
        prompt: 'Extract text from the scanned invoice image',
        expectedTools: ['ocr.extract_text_from_image'],
      },
      {
        description: 'Extract structured invoice data',
        prompt: 'Parse the invoice number, date, line items, and total',
        expectedTools: ['ocr.extract_structured_data'],
      },
      {
        description: 'Save to database',
        prompt: 'Save the invoice data to the invoices database',
        expectedTools: ['data.write_sqlite'],
      },
      {
        description: 'Create a task for payment',
        prompt: 'Create a task to pay this invoice by the due date',
        expectedTools: ['task.create_task'],
      },
    ],
    difficulty: 'medium',
  },
  {
    id: 'ms-medium-010',
    category: 'medium-chain',
    scenario: 'Weekly report generation',
    steps: [
      {
        description: 'Get completed tasks',
        prompt: 'Show me all tasks I completed this week',
        expectedTools: ['task.list_tasks'],
      },
      {
        description: 'Get session summary',
        prompt: 'Summarize the tool usage for this week',
        expectedTools: ['audit.get_session_summary'],
      },
      {
        description: 'Create the report as PDF',
        prompt: 'Create a weekly status report PDF with this information',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Draft email with report',
        prompt: 'Draft an email to my manager with the weekly report',
        expectedTools: ['email.draft_email'],
      },
      {
        description: 'Send the email',
        prompt: 'Send it',
        expectedTools: ['email.send_draft'],
      },
    ],
    difficulty: 'medium',
  },
];
