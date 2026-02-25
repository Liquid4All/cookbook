/**
 * Multi-Step Chain Tests — Simple Chains Part B (tests 9-15).
 */

import type { MultiStepTest } from './types';

/** Simple chains part B: last 7 of 15 tests. */
export const simpleChainTestsB: readonly MultiStepTest[] = [
  {
    id: 'ms-simple-009',
    category: 'simple-chain',
    scenario: 'Scan for secrets and generate audit report',
    steps: [
      {
        description: 'Scan project for secrets',
        prompt: 'Scan my project directory for any exposed secrets or API keys',
        expectedTools: ['security.scan_for_secrets'],
      },
      {
        description: 'Generate an audit report of the findings',
        prompt: 'Generate an audit report summarizing the scan results',
        expectedTools: ['audit.generate_audit_report'],
      },
      {
        description: 'Export the report as PDF',
        prompt: 'Export that audit report as a PDF',
        expectedTools: ['audit.export_audit_pdf'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-010',
    category: 'simple-chain',
    scenario: 'Read spreadsheet data and export to CSV',
    steps: [
      {
        description: 'Read the spreadsheet',
        prompt: 'Read the data from the expenses spreadsheet',
        expectedTools: ['document.read_spreadsheet'],
      },
      {
        description: 'Check for duplicates',
        prompt: 'Check if there are any duplicate entries',
        expectedTools: ['data.deduplicate_records'],
      },
      {
        description: 'Export clean data to CSV',
        prompt: 'Export the deduplicated data to a clean CSV file',
        expectedTools: ['data.write_csv'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-011',
    category: 'simple-chain',
    scenario: 'Take a screenshot and extract text from it',
    steps: [
      {
        description: 'Take a screenshot',
        prompt: 'Take a screenshot of my current screen',
        expectedTools: ['system.take_screenshot'],
      },
      {
        description: 'OCR the screenshot',
        prompt: 'Extract the text from that screenshot',
        expectedTools: ['ocr.extract_text_from_image'],
      },
      {
        description: 'Copy extracted text to clipboard',
        prompt: 'Copy that text to my clipboard',
        expectedTools: ['clipboard.set_clipboard'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-012',
    category: 'simple-chain',
    scenario: 'Compare documents and create summary PDF',
    steps: [
      {
        description: 'Diff the two documents',
        prompt: 'Compare contract-v1.pdf and contract-v2.pdf',
        expectedTools: ['document.diff_documents'],
      },
      {
        description: 'Create a PDF of the differences',
        prompt: 'Create a PDF summarizing the changes between the two versions',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Move the diff report to the contracts folder',
        prompt: 'Move the diff report to the Contracts folder',
        expectedTools: ['filesystem.move_file'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-013',
    category: 'simple-chain',
    scenario: 'Get daily briefing and schedule tasks',
    steps: [
      {
        description: 'Get daily briefing',
        prompt: 'Show me my daily briefing',
        expectedTools: ['task.daily_briefing'],
      },
      {
        description: 'Check calendar for the day',
        prompt: 'What meetings do I have today?',
        expectedTools: ['calendar.list_events'],
      },
      {
        description: 'Create a task for preparation',
        prompt: 'Add a task to prepare the slides for the 2pm meeting',
        expectedTools: ['task.create_task'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-014',
    category: 'simple-chain',
    scenario: 'Find PII in documents and encrypt them',
    steps: [
      {
        description: 'Scan for PII',
        prompt: 'Scan the HR folder for files containing personal information',
        expectedTools: ['security.scan_for_pii'],
      },
      {
        description: 'Encrypt the sensitive file',
        prompt: 'Encrypt the file that contains social security numbers',
        expectedTools: ['security.encrypt_file'],
      },
      {
        description: 'Log the action',
        prompt: 'Show me the audit log of what we just did',
        expectedTools: ['audit.get_tool_log'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-015',
    category: 'simple-chain',
    scenario: 'Search files and ask questions about them',
    steps: [
      {
        description: 'Search for relevant files',
        prompt: 'Find all files related to the marketing campaign',
        expectedTools: ['filesystem.search_files'],
      },
      {
        description: 'Index the found files',
        prompt: 'Index these files so I can ask questions about them',
        expectedTools: ['knowledge.index_folder'],
      },
      {
        description: 'Ask a question',
        prompt: 'What was the total campaign budget mentioned in these documents?',
        expectedTools: ['knowledge.ask_about_files'],
      },
    ],
    difficulty: 'easy',
  },
];
