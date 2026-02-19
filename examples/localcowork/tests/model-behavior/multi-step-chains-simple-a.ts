/**
 * Multi-Step Chain Tests — Simple Chains Part A (tests 1-8).
 */

import type { MultiStepTest } from './types';

/** Simple chains part A: first 8 of 15 tests. */
export const simpleChainTestsA: readonly MultiStepTest[] = [
  {
    id: 'ms-simple-001',
    category: 'simple-chain',
    scenario: 'List Downloads and find all receipts',
    steps: [
      {
        description: 'List the Downloads directory',
        prompt: 'Show me what is in my Downloads folder',
        expectedTools: ['filesystem.list_dir'],
      },
      {
        description: 'Search for receipt files',
        prompt: 'Now search for files with "receipt" in the name',
        expectedTools: ['filesystem.search_files'],
      },
      {
        description: 'Read a specific receipt',
        prompt: 'Open the first receipt file',
        expectedTools: ['filesystem.read_file'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-002',
    category: 'simple-chain',
    scenario: 'Read a PDF and copy summary to clipboard',
    steps: [
      {
        description: 'Extract text from the PDF',
        prompt: 'Extract the text from report.pdf in my Documents',
        expectedTools: ['document.extract_text'],
      },
      {
        description: 'Copy the key findings to clipboard',
        prompt: 'Copy the executive summary section to my clipboard',
        expectedTools: ['clipboard.set_clipboard'],
      },
      {
        description: 'Verify clipboard contents',
        prompt: 'Show me what is on my clipboard now',
        expectedTools: ['clipboard.get_clipboard'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-003',
    category: 'simple-chain',
    scenario: 'Create a task from meeting notes',
    steps: [
      {
        description: 'Read the meeting notes file',
        prompt: 'Read the meeting notes from today',
        expectedTools: ['filesystem.read_file'],
      },
      {
        description: 'Create a task for the first action item',
        prompt: 'Create a task for the first action item: prepare proposal by Wednesday',
        expectedTools: ['task.create_task'],
      },
      {
        description: 'Create a calendar event for the follow-up',
        prompt: 'Also schedule a follow-up meeting next Monday at 10am',
        expectedTools: ['calendar.create_event'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-004',
    category: 'simple-chain',
    scenario: 'Search emails and draft a reply',
    steps: [
      {
        description: 'Search for the original email thread',
        prompt: 'Find emails about the vendor contract renewal',
        expectedTools: ['email.search_emails'],
      },
      {
        description: 'Summarize the thread',
        prompt: 'Summarize this email thread for me',
        expectedTools: ['email.summarize_thread'],
      },
      {
        description: 'Draft a response',
        prompt: 'Draft a reply accepting the renewal terms',
        expectedTools: ['email.draft_email'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-005',
    category: 'simple-chain',
    scenario: 'Scan a document with OCR and save as text',
    steps: [
      {
        description: 'OCR the scanned document',
        prompt: 'Extract text from the scanned page image',
        expectedTools: ['ocr.extract_text_from_image'],
      },
      {
        description: 'Save the extracted text to a file',
        prompt: 'Save the extracted text to a new file called scanned-output.txt',
        expectedTools: ['filesystem.write_file'],
      },
      {
        description: 'Verify the file was created',
        prompt: 'Show me the contents of the file we just created',
        expectedTools: ['filesystem.read_file'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-006',
    category: 'simple-chain',
    scenario: 'Check overdue tasks and send reminder email',
    steps: [
      {
        description: 'Get overdue tasks',
        prompt: 'What tasks are overdue?',
        expectedTools: ['task.get_overdue'],
      },
      {
        description: 'Draft a reminder email',
        prompt: 'Draft an email to the team reminding them about these overdue items',
        expectedTools: ['email.draft_email'],
      },
      {
        description: 'Send the draft',
        prompt: 'Go ahead and send that email',
        expectedTools: ['email.send_draft'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-007',
    category: 'simple-chain',
    scenario: 'Index a folder and search for specific info',
    steps: [
      {
        description: 'Index the documents folder',
        prompt: 'Index my project Documents folder for search',
        expectedTools: ['knowledge.index_folder'],
      },
      {
        description: 'Search for specific content',
        prompt: 'Search for anything about budget allocations',
        expectedTools: ['knowledge.search_documents'],
      },
      {
        description: 'Ask a follow-up question about the results',
        prompt: 'What do the documents say about Q4 budget increases?',
        expectedTools: ['knowledge.ask_about_files'],
      },
    ],
    difficulty: 'easy',
  },
  {
    id: 'ms-simple-008',
    category: 'simple-chain',
    scenario: 'Check calendar and create time block',
    steps: [
      {
        description: "List today's events",
        prompt: 'Show me my calendar for today',
        expectedTools: ['calendar.list_events'],
      },
      {
        description: 'Find a free slot',
        prompt: 'When am I free for an hour this afternoon?',
        expectedTools: ['calendar.find_free_slots'],
      },
      {
        description: 'Block the time',
        prompt: 'Block off that time for focused work',
        expectedTools: ['calendar.create_time_block'],
      },
    ],
    difficulty: 'easy',
  },
];
