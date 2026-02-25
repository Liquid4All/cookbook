/**
 * Multi-Step Chain Tests — Complex Chains Part A (tests 1-5, 6+ tools each).
 */

import type { MultiStepTest } from './types';

/** Complex chains part A: tests 1-5. */
export const complexChainTestsA: readonly MultiStepTest[] = [
  {
    id: 'ms-complex-001',
    category: 'complex-chain',
    scenario: 'Full receipt reconciliation pipeline',
    steps: [
      {
        description: 'List receipt files in the folder',
        prompt: 'Show me all receipt images in my Receipts folder',
        expectedTools: ['filesystem.search_files'],
      },
      {
        description: 'OCR the first receipt',
        prompt: 'Extract text from the first receipt image',
        expectedTools: ['ocr.extract_text_from_image'],
      },
      {
        description: 'Extract structured data from OCR',
        prompt: 'Parse the vendor, date, items, and total from the OCR text',
        expectedTools: ['ocr.extract_structured_data'],
      },
      {
        description: 'Deduplicate against existing records',
        prompt: 'Check if this receipt is already in the system',
        expectedTools: ['data.deduplicate_records'],
      },
      {
        description: 'Export reconciled data to CSV',
        prompt: 'Export all the reconciled receipt data to a CSV report',
        expectedTools: ['data.write_csv'],
      },
      {
        description: 'Summarize anomalies',
        prompt: 'Flag any receipts with unusual amounts or missing data',
        expectedTools: ['data.summarize_anomalies'],
      },
      {
        description: 'Create a final PDF report',
        prompt: 'Generate a PDF expense reconciliation report',
        expectedTools: ['document.create_pdf'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-002',
    category: 'complex-chain',
    scenario: 'Full meeting pipeline — transcribe to email',
    steps: [
      {
        description: 'Transcribe the meeting audio',
        prompt: 'Transcribe the board meeting recording',
        expectedTools: ['meeting.transcribe_audio'],
      },
      {
        description: 'Extract action items',
        prompt: 'Pull out all action items with owners and deadlines',
        expectedTools: ['meeting.extract_action_items'],
      },
      {
        description: 'Extract commitments',
        prompt: 'What commitments were made?',
        expectedTools: ['meeting.extract_commitments'],
      },
      {
        description: 'Create tasks for each action item',
        prompt: 'Create tasks for each action item',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Schedule follow-up events',
        prompt: 'Schedule the follow-up meetings that were discussed',
        expectedTools: ['calendar.create_event'],
      },
      {
        description: 'Generate formal meeting minutes',
        prompt: 'Generate formal meeting minutes',
        expectedTools: ['meeting.generate_minutes'],
      },
      {
        description: 'Draft distribution email',
        prompt: 'Draft an email to all attendees with the minutes attached',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-003',
    category: 'complex-chain',
    scenario: 'Full security audit and remediation',
    steps: [
      {
        description: 'Scan for PII in the shared drive',
        prompt: 'Scan the shared drive for personal information exposure',
        expectedTools: ['security.scan_for_pii'],
      },
      {
        description: 'Scan for hardcoded secrets',
        prompt: 'Also check all code repos for hardcoded secrets',
        expectedTools: ['security.scan_for_secrets'],
      },
      {
        description: 'Find duplicate sensitive files',
        prompt: 'Find any duplicate copies of sensitive files',
        expectedTools: ['security.find_duplicates'],
      },
      {
        description: 'Encrypt the most critical files',
        prompt: 'Encrypt all files that contain unprotected PII',
        expectedTools: ['security.encrypt_file'],
      },
      {
        description: 'Generate audit report',
        prompt: 'Generate a comprehensive security audit report',
        expectedTools: ['audit.generate_audit_report'],
      },
      {
        description: 'Export report as PDF',
        prompt: 'Export the report as a formal PDF document',
        expectedTools: ['audit.export_audit_pdf'],
      },
      {
        description: 'Email the report to the security team',
        prompt: 'Send the security audit report to the CISO',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-004',
    category: 'complex-chain',
    scenario: 'Document digitization and knowledge base creation',
    steps: [
      {
        description: 'Search for scanned documents',
        prompt: 'Find all scanned PDF files in the Legacy Documents folder',
        expectedTools: ['filesystem.search_files'],
      },
      {
        description: 'OCR the first scanned PDF',
        prompt: 'Extract text from the first scanned document',
        expectedTools: ['ocr.extract_text_from_pdf'],
      },
      {
        description: 'Extract any tables',
        prompt: 'Extract the data tables from the scanned pages',
        expectedTools: ['ocr.extract_table'],
      },
      {
        description: 'Save extracted text',
        prompt: 'Save the extracted text to a searchable text file',
        expectedTools: ['filesystem.write_file'],
      },
      {
        description: 'Save table data to CSV',
        prompt: 'Export the table data to CSV files',
        expectedTools: ['data.write_csv'],
      },
      {
        description: 'Index for knowledge base',
        prompt: 'Index all the extracted content for search',
        expectedTools: ['knowledge.index_folder'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-005',
    category: 'complex-chain',
    scenario: 'Quarterly business review preparation',
    steps: [
      {
        description: 'Query financial data',
        prompt: 'Pull Q4 revenue and expense data from the database',
        expectedTools: ['data.query_sqlite'],
      },
      {
        description: 'Analyze for anomalies',
        prompt: 'Are there any unusual trends in the Q4 numbers?',
        expectedTools: ['data.summarize_anomalies'],
      },
      {
        description: 'Search for relevant context in documents',
        prompt: 'Find any documents about Q4 performance drivers',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Read the previous QBR for comparison',
        prompt: 'Read the Q3 QBR document for comparison',
        expectedTools: ['document.extract_text'],
      },
      {
        description: 'Create the QBR PDF',
        prompt: 'Create the Q4 Quarterly Business Review as a PDF',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Draft distribution email',
        prompt: 'Draft an email to the leadership team with the QBR',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
];
