/**
 * Multi-Step Chain Tests — Complex Chains Part C (tests 11-15, 6+ tools each).
 */

import type { MultiStepTest } from './types';

/** Complex chains part C: tests 11-15. */
export const complexChainTestsC: readonly MultiStepTest[] = [
  {
    id: 'ms-complex-011',
    category: 'complex-chain',
    scenario: 'Compliance document review pipeline',
    steps: [
      {
        description: 'Extract the new regulation text',
        prompt: 'Extract text from the new compliance regulation PDF',
        expectedTools: ['document.extract_text'],
      },
      {
        description: 'Diff with current policy',
        prompt: 'Compare with our current compliance policy',
        expectedTools: ['document.diff_documents'],
      },
      {
        description: 'Scan for PII handling requirements',
        prompt: 'Scan our systems for any PII that might be affected',
        expectedTools: ['security.scan_for_pii'],
      },
      {
        description: 'Search for related internal docs',
        prompt: 'Find all internal documents about data handling practices',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Create gap analysis report',
        prompt: 'Create a PDF report of the compliance gaps',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Create tasks for remediation',
        prompt: 'Create tasks for each compliance gap that needs to be addressed',
        expectedTools: ['task.create_task'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-012',
    category: 'complex-chain',
    scenario: 'Multi-source data consolidation',
    steps: [
      {
        description: 'Read spreadsheet data',
        prompt: 'Read the data from the regional sales spreadsheets',
        expectedTools: ['document.read_spreadsheet'],
      },
      {
        description: 'Query the master database',
        prompt: 'Pull the corresponding data from the master sales database',
        expectedTools: ['data.query_sqlite'],
      },
      {
        description: 'Deduplicate the combined data',
        prompt: 'Merge and deduplicate the records from both sources',
        expectedTools: ['data.deduplicate_records'],
      },
      {
        description: 'Detect anomalies',
        prompt: 'Flag any inconsistencies between the two data sources',
        expectedTools: ['data.summarize_anomalies'],
      },
      {
        description: 'Write consolidated data back',
        prompt: 'Save the consolidated data to the master database',
        expectedTools: ['data.write_sqlite'],
      },
      {
        description: 'Export summary CSV',
        prompt: 'Also export a summary CSV for the finance team',
        expectedTools: ['data.write_csv'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-013',
    category: 'complex-chain',
    scenario: 'Whistleblower report processing',
    steps: [
      {
        description: 'Read the anonymous report',
        prompt: 'Read the anonymous report file that was submitted',
        expectedTools: ['filesystem.read_file'],
      },
      {
        description: 'Scan for any identifying PII',
        prompt: 'Check if the report accidentally contains PII that could identify the reporter',
        expectedTools: ['security.scan_for_pii'],
      },
      {
        description: 'Encrypt the original file',
        prompt: 'Encrypt the original report for secure storage',
        expectedTools: ['security.encrypt_file'],
      },
      {
        description: 'Create a redacted version',
        prompt: 'Create a redacted PDF version safe for distribution',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Create investigation tasks',
        prompt: 'Create tasks for the investigation committee',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Log all actions taken',
        prompt: 'Show me the audit trail of everything we did with this report',
        expectedTools: ['audit.get_tool_log'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-014',
    category: 'complex-chain',
    scenario: 'Client deliverable preparation',
    steps: [
      {
        description: 'Search for project deliverables',
        prompt: 'Find all files tagged as deliverables for Client XYZ',
        expectedTools: ['filesystem.search_files'],
      },
      {
        description: 'Read and verify each document',
        prompt: 'Read the first deliverable document',
        expectedTools: ['filesystem.read_file'],
      },
      {
        description: 'Scan for sensitive internal data',
        prompt: 'Make sure none of these files contain internal-only information',
        expectedTools: ['security.scan_for_pii'],
      },
      {
        description: 'Convert to client-preferred format',
        prompt: 'Convert all documents to PDF format',
        expectedTools: ['document.convert_format'],
      },
      {
        description: 'Merge into single deliverable',
        prompt: 'Merge all the PDFs into a single deliverable package',
        expectedTools: ['document.merge_pdfs'],
      },
      {
        description: 'Draft delivery email',
        prompt: 'Draft the delivery email to the client with a summary',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-015',
    category: 'complex-chain',
    scenario: 'Full workspace setup for new project',
    steps: [
      {
        description: 'Create project directory structure',
        prompt: 'Create the folder structure for the new Project Beta',
        expectedTools: ['filesystem.write_file'],
      },
      {
        description: 'Copy template files',
        prompt: 'Copy the project template files into the new structure',
        expectedTools: ['filesystem.copy_file'],
      },
      {
        description: 'Create project tasks',
        prompt: 'Create the initial project milestone tasks',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Schedule kickoff meeting',
        prompt: 'Schedule a project kickoff meeting for next Wednesday',
        expectedTools: ['calendar.create_event'],
      },
      {
        description: 'Block focus time for project work',
        prompt: 'Block off 2 hours daily next week for this project',
        expectedTools: ['calendar.create_time_block'],
      },
      {
        description: 'Draft project announcement',
        prompt: 'Draft an announcement email to the team about the new project',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
];
