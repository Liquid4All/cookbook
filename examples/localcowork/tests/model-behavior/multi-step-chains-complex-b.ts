/**
 * Multi-Step Chain Tests — Complex Chains Part B (tests 6-10, 6+ tools each).
 */

import type { MultiStepTest } from './types';

/** Complex chains part B: tests 6-10. */
export const complexChainTestsB: readonly MultiStepTest[] = [
  {
    id: 'ms-complex-006',
    category: 'complex-chain',
    scenario: 'Employee onboarding document pipeline',
    steps: [
      {
        description: 'Query employee database',
        prompt: 'Look up the new hire details in the HR database',
        expectedTools: ['data.query_sqlite'],
      },
      {
        description: 'Fill the offer letter template',
        prompt: 'Fill in the offer letter PDF with their details',
        expectedTools: ['document.fill_pdf_form'],
      },
      {
        description: 'Fill the NDA form',
        prompt: 'Also fill in the NDA form',
        expectedTools: ['document.fill_pdf_form'],
      },
      {
        description: 'Merge into onboarding packet',
        prompt: 'Merge the offer letter and NDA into a single onboarding PDF',
        expectedTools: ['document.merge_pdfs'],
      },
      {
        description: 'Create onboarding tasks',
        prompt: 'Create onboarding tasks: setup laptop, badge access, team intro',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Schedule orientation',
        prompt: 'Schedule a 2-hour orientation for their first day',
        expectedTools: ['calendar.create_event'],
      },
      {
        description: 'Email HR with the packet',
        prompt: 'Draft an email to HR with the onboarding packet',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-007',
    category: 'complex-chain',
    scenario: 'End-of-week comprehensive review',
    steps: [
      {
        description: 'Get completed tasks',
        prompt: 'Show me all tasks completed this week',
        expectedTools: ['task.list_tasks'],
      },
      {
        description: 'Check remaining overdue items',
        prompt: 'What is still overdue?',
        expectedTools: ['task.get_overdue'],
      },
      {
        description: 'Get full audit log',
        prompt: 'Show me all tool usage from this week',
        expectedTools: ['audit.get_tool_log'],
      },
      {
        description: 'Get session summary',
        prompt: 'Generate a session summary',
        expectedTools: ['audit.get_session_summary'],
      },
      {
        description: 'Create the weekly report',
        prompt: 'Create a comprehensive weekly report PDF',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Draft email to manager',
        prompt: 'Draft my weekly status email to my manager',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-008',
    category: 'complex-chain',
    scenario: 'Cross-referencing invoices with purchase orders',
    steps: [
      {
        description: 'Search for invoice files',
        prompt: 'Find all invoice PDFs from this month',
        expectedTools: ['filesystem.search_files'],
      },
      {
        description: 'Extract text from an invoice',
        prompt: 'Extract the text from the first invoice',
        expectedTools: ['document.extract_text'],
      },
      {
        description: 'Query PO database',
        prompt: 'Look up the matching purchase order in the database',
        expectedTools: ['data.query_sqlite'],
      },
      {
        description: 'Check for discrepancies',
        prompt: 'Compare the invoice amount with the PO amount',
        expectedTools: ['data.summarize_anomalies'],
      },
      {
        description: 'Write reconciliation data',
        prompt: 'Save the reconciliation results to the database',
        expectedTools: ['data.write_sqlite'],
      },
      {
        description: 'Create reconciliation report',
        prompt: 'Create a PDF report of the invoice-PO reconciliation',
        expectedTools: ['document.create_pdf'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-009',
    category: 'complex-chain',
    scenario: 'Research paper compilation from multiple sources',
    steps: [
      {
        description: 'Search knowledge base for topic',
        prompt: 'Search my indexed documents for AI governance research',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Find related chunks',
        prompt: 'Find more related content across my document collection',
        expectedTools: ['knowledge.get_related_chunks'],
      },
      {
        description: 'Read specific source documents',
        prompt: 'Read the full text of the most relevant paper',
        expectedTools: ['filesystem.read_file'],
      },
      {
        description: 'Search emails for expert opinions',
        prompt: 'Check my emails for any correspondence about AI governance',
        expectedTools: ['email.search_emails'],
      },
      {
        description: 'Create the research document',
        prompt: 'Compile everything into a research summary Word document',
        expectedTools: ['document.create_docx'],
      },
      {
        description: 'Copy key quote to clipboard',
        prompt: 'Copy the key conclusion to my clipboard for the presentation',
        expectedTools: ['clipboard.set_clipboard'],
      },
    ],
    difficulty: 'hard',
  },
  {
    id: 'ms-complex-010',
    category: 'complex-chain',
    scenario: 'Project handoff documentation',
    steps: [
      {
        description: 'List all project files',
        prompt: 'Show me all files in the Project Alpha directory',
        expectedTools: ['filesystem.list_dir'],
      },
      {
        description: 'Get task status',
        prompt: 'List all tasks associated with Project Alpha',
        expectedTools: ['task.list_tasks'],
      },
      {
        description: 'Search knowledge base for project docs',
        prompt: 'Search my documents for any Project Alpha decisions and rationale',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Get audit history',
        prompt: 'Show the tool usage history for Project Alpha',
        expectedTools: ['audit.get_tool_log'],
      },
      {
        description: 'Create handoff PDF',
        prompt: 'Create a comprehensive project handoff document as PDF',
        expectedTools: ['document.create_pdf'],
      },
      {
        description: 'Draft handoff email',
        prompt: 'Draft a handoff email to the new project lead with all the context',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'hard',
  },
];
