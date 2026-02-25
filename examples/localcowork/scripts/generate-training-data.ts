#!/usr/bin/env npx tsx
/**
 * Training Data Generator for LFM2-1.2B-Tool Fine-Tuning
 *
 * Generates ChatML-format JSONL training data from:
 * 1. Existing benchmark tests (100 single-step + 222 multi-step steps)
 * 2. Synthetic contrastive examples targeting known failure modes
 * 3. Paraphrased variants for robustness
 *
 * Output: training-data/{train,eval,test}.jsonl + metadata.json
 *
 * Usage:
 *   npx tsx scripts/generate-training-data.ts [--output-dir training-data]
 */

import * as fs from 'fs';
import * as path from 'path';

// ─── Import benchmark data ──────────────────────────────────────────────────

import { allToolSelectionTests } from '../tests/model-behavior/tool-selection';
import { TOOL_DESCRIPTIONS, buildFilteredToolDefinitions } from '../tests/model-behavior/benchmark-shared';
import { VALID_TOOL_NAMES } from '../tests/model-behavior/types';
import type { ToolSelectionTest, MultiStepTest, MultiStepEntry } from '../tests/model-behavior/types';

// Import multi-step chains (using actual export names from each file)
import { simpleChainTestsA as simpleA } from '../tests/model-behavior/multi-step-chains-simple-a';
import { simpleChainTestsB as simpleB } from '../tests/model-behavior/multi-step-chains-simple-b';
import { mediumChainTestsA as mediumA } from '../tests/model-behavior/multi-step-chains-medium-a';
import { mediumChainTestsB as mediumB } from '../tests/model-behavior/multi-step-chains-medium-b';
import { complexChainTestsA as complexA } from '../tests/model-behavior/multi-step-chains-complex-a';
import { complexChainTestsB as complexB } from '../tests/model-behavior/multi-step-chains-complex-b';
import { complexChainTestsC as complexC } from '../tests/model-behavior/multi-step-chains-complex-c';

// ─── Types ──────────────────────────────────────────────────────────────────

interface TrainingExample {
  messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  metadata: {
    source: string;         // 'benchmark' | 'synthetic-sibling' | 'synthetic-crossserver' | etc.
    category: string;       // Tool category
    expectedTool: string;   // Ground truth tool
    difficulty: string;     // easy | medium | hard
    failureMode?: string;   // Which FM this targets
  };
}

interface GenerationStats {
  total: number;
  bySource: Record<string, number>;
  byCategory: Record<string, number>;
  byDifficulty: Record<string, number>;
  trainCount: number;
  evalCount: number;
  testCount: number;
  generatedAt: string;
}

// ─── Constants ──────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are LocalCowork, a desktop AI assistant that runs entirely on-device. You have access to the following tools. ALWAYS call exactly one tool using bracket syntax: [server.tool(param="value")]. NEVER ask questions. NEVER say you cannot help. ALWAYS select the most appropriate tool.`;

const OUTPUT_DIR = process.argv.includes('--output-dir')
  ? process.argv[process.argv.indexOf('--output-dir') + 1]
  : path.join(process.cwd(), 'training-data');

// All multi-step chains combined
const ALL_MULTI_STEP: readonly MultiStepTest[] = [
  ...simpleA, ...simpleB,
  ...mediumA, ...mediumB,
  ...complexA, ...complexB, ...complexC,
];

// ─── Tool Candidate Selection ───────────────────────────────────────────────

/** Get K=15 candidate tools for a given target tool.
 *  In production, this would use the RAG pre-filter. Here we simulate
 *  by including the correct tool + semantically related tools. */
function getRealisticCandidates(targetTool: string, k: number = 15): string[] {
  const server = targetTool.split('.')[0];
  const allTools = [...VALID_TOOL_NAMES];

  // Always include all tools from the target server
  const sameServer = allTools.filter((t) => t.startsWith(`${server}.`));

  // Add tools from semantically related servers
  const relatedServers = getRelatedServers(server);
  const relatedTools = allTools.filter((t) => {
    const s = t.split('.')[0];
    return relatedServers.includes(s);
  });

  // Combine: same server + related + random fill
  const candidates = new Set<string>(sameServer);
  for (const t of relatedTools) {
    if (candidates.size >= k) break;
    candidates.add(t);
  }

  // Fill remaining slots with random tools from other servers
  const remaining = allTools.filter((t) => !candidates.has(t));
  shuffleArray(remaining);
  for (const t of remaining) {
    if (candidates.size >= k) break;
    candidates.add(t);
  }

  // Ensure target tool is always included
  candidates.add(targetTool);

  return [...candidates].slice(0, k);
}

/** Map server to semantically related servers (confusion sources). */
function getRelatedServers(server: string): string[] {
  const relations: Record<string, string[]> = {
    filesystem: ['document', 'data', 'security'],
    document: ['filesystem', 'ocr', 'data'],
    ocr: ['document', 'filesystem'],
    data: ['document', 'filesystem', 'knowledge'],
    audit: ['task', 'system'],
    knowledge: ['document', 'filesystem', 'data'],
    security: ['filesystem', 'audit'],
    task: ['calendar', 'audit'],
    calendar: ['task', 'meeting'],
    email: ['meeting', 'knowledge', 'task'],
    meeting: ['email', 'calendar', 'task'],
    clipboard: ['system', 'filesystem'],
    system: ['clipboard', 'filesystem'],
  };
  return relations[server] ?? [];
}

function shuffleArray<T>(arr: T[]): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ─── System Prompt Builder ──────────────────────────────────────────────────

function buildSystemPrompt(candidateTools: string[]): string {
  const toolList = candidateTools
    .map((name, i) => {
      const desc = TOOL_DESCRIPTIONS[name] ?? name;
      return `${i + 1}. ${name} — ${desc}`;
    })
    .join('\n');

  return `${SYSTEM_PROMPT}\n\nAvailable tools:\n${toolList}`;
}

// ─── Format Tool Call ───────────────────────────────────────────────────────

function formatToolCall(toolName: string, params?: Record<string, string>): string {
  if (!params || Object.keys(params).length === 0) {
    return `[${toolName}()]`;
  }
  const paramStr = Object.entries(params)
    .map(([k, v]) => `${k}="${v}"`)
    .join(', ');
  return `[${toolName}(${paramStr})]`;
}

// ─── Source 1: Existing Benchmark Tests ─────────────────────────────────────

function generateFromBenchmarks(): TrainingExample[] {
  const examples: TrainingExample[] = [];

  for (const test of allToolSelectionTests) {
    const targetTool = test.expectedTools[0];
    const candidates = getRealisticCandidates(targetTool);
    const systemPrompt = buildSystemPrompt(candidates);

    // Generate realistic params based on tool and prompt
    const params = inferParams(targetTool, test.prompt);

    examples.push({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: test.prompt },
        { role: 'assistant', content: formatToolCall(targetTool, params) },
      ],
      metadata: {
        source: 'benchmark',
        category: test.category,
        expectedTool: targetTool,
        difficulty: test.difficulty,
      },
    });
  }

  return examples;
}

// ─── Source 2: Multi-Step Chain Steps (Isolated) ────────────────────────────

function generateFromMultiStepChains(): TrainingExample[] {
  const examples: TrainingExample[] = [];

  for (const chain of ALL_MULTI_STEP) {
    for (const step of chain.steps) {
      const targetTool = step.expectedTools[0];
      const candidates = getRealisticCandidates(targetTool);
      const systemPrompt = buildSystemPrompt(candidates);
      const params = inferParams(targetTool, step.prompt);

      examples.push({
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: step.prompt },
          { role: 'assistant', content: formatToolCall(targetTool, params) },
        ],
        metadata: {
          source: 'multi-step-isolated',
          category: chain.category,
          expectedTool: targetTool,
          difficulty: chain.difficulty,
        },
      });
    }
  }

  return examples;
}

// ─── Source 3: Sibling Confusion Contrastive Pairs ──────────────────────────

interface SiblingPair {
  toolA: string;
  toolB: string;
  promptsForA: string[];
  promptsForB: string[];
}

const SIBLING_PAIRS: SiblingPair[] = [
  {
    toolA: 'calendar.list_events',
    toolB: 'calendar.find_free_slots',
    promptsForA: [
      'Show me what meetings I have this Wednesday',
      'What\'s on my calendar for tomorrow?',
      'List all my appointments for next week',
      'Do I have any meetings after 3pm today?',
      'Show me my schedule for Monday',
      'What events are scheduled for this afternoon?',
      'Check my calendar for February 20th',
      'Am I busy on Thursday morning?',
      'What meetings do I have with Sarah this week?',
      'Show me all my team standups this month',
      'List events between March 1st and March 15th',
      'Is there anything on my calendar for Friday?',
      'What\'s my meeting schedule look like today?',
      'Show me what I have booked for next Tuesday',
      'Do I have any calls scheduled for this evening?',
    ],
    promptsForB: [
      'When am I free this Wednesday for a 30 minute call?',
      'Find me an open time slot tomorrow afternoon',
      'When can I schedule a 1-hour meeting this week?',
      'What times am I available on Friday?',
      'Find a free slot for a team meeting next Monday',
      'When\'s the next available 45-minute block?',
      'Check my availability for a lunch meeting Thursday',
      'I need to find time for a 2-hour planning session',
      'What open slots do I have next week?',
      'When could I schedule a call with the client?',
      'Find me some free time on Wednesday morning',
      'What\'s my availability look like for a quick sync?',
      'Is there any open time this Friday for a review?',
      'Find a gap in my schedule for a dentist appointment',
      'When am I not in meetings tomorrow?',
    ],
  },
  {
    toolA: 'calendar.create_event',
    toolB: 'calendar.create_time_block',
    promptsForA: [
      'Schedule a meeting with John at 2pm tomorrow',
      'Book a team standup for Monday at 9am',
      'Create a lunch appointment with Sarah on Friday',
      'Set up a project review meeting for next week',
      'Add a call with the client at 4pm today',
      'Schedule a 1-on-1 with my manager on Thursday',
      'Book a demo session with the product team',
      'Create an interview slot for 11am Wednesday',
      'Set up a retrospective for the end of the sprint',
      'Schedule a brainstorming session with design',
      'Book a sync with the engineering team at 3pm',
      'Create a meeting for the budget review next Monday',
      'Schedule a presentation for stakeholders on Friday',
      'Add a training session for new hires at 10am',
      'Set up a coffee chat with the new team member',
    ],
    promptsForB: [
      'Block off 2 hours for deep work this afternoon',
      'Reserve focus time tomorrow morning for coding',
      'Block my calendar from 9-11am for writing',
      'Set aside some personal time on Friday afternoon',
      'Block focus time for working on the proposal',
      'Reserve 3 hours for uninterrupted research',
      'Mark my calendar as busy for gym time at 6pm',
      'Block time for thesis writing tomorrow',
      'Reserve the morning for heads-down coding',
      'Set a do-not-disturb block from 2-5pm',
      'Block off lunch hour every day this week',
      'Reserve some quiet time for report writing',
      'Mark 4-6pm as focus time for the rest of the week',
      'Block personal time for a doctor visit on Tuesday',
      'Set aside 90 minutes for strategic planning',
    ],
  },
  {
    toolA: 'email.draft_email',
    toolB: 'email.send_draft',
    promptsForA: [
      'Write an email to the team about the project update',
      'Draft a follow-up email to the client',
      'Compose an email to HR about my time-off request',
      'Write a thank-you email to the interviewer',
      'Draft a message to the vendor about pricing',
      'Compose a newsletter for the engineering team',
      'Write an email summarizing the meeting notes',
      'Draft a response to the customer complaint',
      'Compose an invitation email for the team dinner',
      'Write an apology email about the delayed shipment',
      'Draft a proposal email for the new partnership',
      'Compose a welcome email for the new hire',
      'Write an email asking for feedback on the design',
      'Draft a status update email for management',
      'Compose a reminder email about the deadline',
    ],
    promptsForB: [
      'Send the draft email I wrote to the team',
      'Go ahead and send that email to the client',
      'Send my drafted response to HR',
      'Dispatch the follow-up email now',
      'Send the newsletter draft to the mailing list',
      'Send out the meeting summary email',
      'Deliver that customer response I drafted',
      'Send the invitation email I composed earlier',
      'Go ahead and send the proposal email',
      'Send the welcome email to the new team member',
      'Dispatch the status update to management',
      'Send the reminder email about Friday\'s deadline',
      'Send that apology email I wrote',
      'Go ahead and send my feedback request email',
      'Deliver the pricing email to the vendor',
    ],
  },
  {
    toolA: 'email.list_drafts',
    toolB: 'email.search_emails',
    promptsForA: [
      'Show me my email drafts',
      'What drafts do I have saved?',
      'List all my unsent emails',
      'Show me the emails I haven\'t sent yet',
      'What draft messages do I have?',
      'Check my draft folder',
      'List my saved email drafts',
      'Show any pending email drafts',
      'What emails are sitting in my drafts?',
      'Do I have any unsent draft emails?',
    ],
    promptsForB: [
      'Search my inbox for emails from John',
      'Find the email about the project proposal',
      'Look for emails with the subject "Q4 Report"',
      'Search for emails I received last week',
      'Find messages from the client about pricing',
      'Search emails for the word "invoice"',
      'Look up emails from sarah@company.com',
      'Find the email thread about the budget',
      'Search for emails with attachments from this month',
      'Look for that email about the meeting reschedule',
    ],
  },
  {
    toolA: 'data.summarize_anomalies',
    toolB: 'data.query_sqlite',
    promptsForA: [
      'Check for anomalies in the sales database',
      'Find unusual patterns in the transaction data',
      'Detect any outliers in the expense records',
      'Are there any anomalies in this month\'s data?',
      'Look for irregularities in the revenue figures',
      'Scan the database for unusual activity',
      'Find any suspicious patterns in the logs',
      'Check for data anomalies in the user metrics',
      'Detect outliers in the performance data',
      'Are there any unusual spikes in the traffic data?',
    ],
    promptsForB: [
      'Query the database for total sales this quarter',
      'Run a SQL query to get customer counts by region',
      'Look up the order details for order #1234',
      'Get the average transaction amount from the database',
      'Query the employees table for the engineering team',
      'Run a SELECT query on the inventory table',
      'Get the top 10 products by revenue',
      'Query for all records where status is "active"',
      'Look up the user with ID 42 in the database',
      'Get the count of orders by month from the database',
    ],
  },
  {
    toolA: 'document.extract_text',
    toolB: 'document.read_spreadsheet',
    promptsForA: [
      'Extract the text from the project proposal PDF',
      'Read the content of the contract document',
      'Get the text from the meeting notes DOCX',
      'Extract text from the annual report',
      'Read the content of the research paper PDF',
      'Get the text from the NDA document',
      'Extract the content from the invoice PDF',
      'Read the lease agreement document',
      'Get the text from the employee handbook',
      'Extract content from the design spec document',
    ],
    promptsForB: [
      'Read the data from the quarterly report spreadsheet',
      'Open the employee list Excel file',
      'Load the budget spreadsheet data',
      'Read the inventory CSV file',
      'Get the data from the sales tracking XLSX',
      'Open the customer database CSV',
      'Read the project timeline spreadsheet',
      'Load the financial data from the Excel file',
      'Get the expense report spreadsheet contents',
      'Read the pricing matrix from the CSV',
    ],
  },
  {
    toolA: 'filesystem.move_file',
    toolB: 'filesystem.copy_file',
    promptsForA: [
      'Move the report to the archive folder',
      'Rename screenshot.png to meeting-notes.png',
      'Move all PDFs from Downloads to Documents',
      'Relocate the project folder to the backup drive',
      'Move the config file to the new directory',
      'Rename this file to a more descriptive name',
      'Move the log file to the logs directory',
      'Relocate the database backup to cold storage',
      'Move the draft to the published folder',
      'Rename the file from old-name.txt to new-name.txt',
    ],
    promptsForB: [
      'Make a copy of the report for the client',
      'Duplicate the config file as a backup',
      'Copy the template to start a new project',
      'Make a backup copy of the database',
      'Copy the presentation for offline editing',
      'Duplicate the spreadsheet for Sarah',
      'Make a copy of the source code before changes',
      'Copy the contract to the shared folder',
      'Duplicate this file as a safety backup',
      'Copy the image to the assets folder',
    ],
  },
  {
    toolA: 'security.encrypt_file',
    toolB: 'security.scan_for_pii',
    promptsForA: [
      'Encrypt the salary spreadsheet for secure storage',
      'Lock down the confidential report',
      'Password-protect the financial data file',
      'Encrypt my tax documents',
      'Secure the client database file with encryption',
      'Lock the personnel records file',
      'Encrypt the backup before uploading',
      'Password-protect the contracts folder',
      'Secure the medical records file',
      'Encrypt the trade secrets document',
    ],
    promptsForB: [
      'Scan the document for personal information',
      'Check if there are any SSNs in this file',
      'Look for exposed email addresses in the data',
      'Scan for credit card numbers in the spreadsheet',
      'Check the report for any personal data that needs redacting',
      'Find any phone numbers or addresses in the document',
      'Scan this file for PII before sharing it',
      'Check for exposed personal data in the export',
      'Look for names and social security numbers',
      'Scan the customer data for sensitive information',
    ],
  },
  {
    toolA: 'knowledge.search_documents',
    toolB: 'knowledge.ask_about_files',
    promptsForA: [
      'Search my indexed docs for mentions of the Q4 deadline',
      'Find documents that discuss the migration plan',
      'Search the knowledge base for API documentation',
      'Look through my notes for the meeting with the CTO',
      'Search indexed files for the deployment checklist',
      'Find documents mentioning the new pricing model',
      'Search my files for references to Project Phoenix',
      'Look for documents about the security audit',
      'Search the index for onboarding materials',
      'Find files that discuss the architecture decision',
    ],
    promptsForB: [
      'What was the decision on the database migration?',
      'What did the Q4 report say about revenue growth?',
      'What are the steps in the deployment process?',
      'What was discussed in the last architecture review?',
      'What is our current policy on remote work?',
      'What were the key findings from the security audit?',
      'What is the timeline for Project Phoenix?',
      'What did the CTO say about the product roadmap?',
      'What are the requirements for the API redesign?',
      'What was the conclusion of the performance review?',
    ],
  },
  {
    toolA: 'meeting.extract_action_items',
    toolB: 'meeting.generate_minutes',
    promptsForA: [
      'What are the action items from the standup transcript?',
      'Pull out the to-dos from the meeting recording',
      'Extract the tasks assigned in the planning session',
      'What action items came out of the retrospective?',
      'List the follow-ups from the client call transcript',
      'Extract the deliverables from the project meeting',
      'What tasks were assigned in today\'s sync?',
      'Pull the action items from the board meeting notes',
      'Extract the next steps from the design review',
      'What were the takeaways and action items from the all-hands?',
    ],
    promptsForB: [
      'Generate meeting minutes from the standup transcript',
      'Create formal minutes for the board meeting recording',
      'Write up the meeting notes from the planning session',
      'Generate a formatted summary of the client call',
      'Create meeting minutes from the retrospective audio',
      'Write formal minutes for the project kickoff',
      'Generate a structured summary of today\'s sync',
      'Create minutes from the design review recording',
      'Write up the all-hands meeting as formal minutes',
      'Generate the official minutes from the strategy session',
    ],
  },
  {
    toolA: 'task.list_tasks',
    toolB: 'task.daily_briefing',
    promptsForA: [
      'Show me all my tasks',
      'List my to-dos filtered by high priority',
      'What tasks are assigned to me?',
      'Show all pending tasks for this project',
      'List tasks due this week',
      'What are my open items?',
      'Show me tasks sorted by due date',
      'List all tasks with "review" in the title',
      'What tasks are in progress?',
      'Show me completed tasks from last week',
    ],
    promptsForB: [
      'Give me my daily briefing',
      'What\'s my overview for today?',
      'Summarize what I need to focus on today',
      'What\'s on my plate for today?',
      'Give me a morning summary of priorities',
      'What should I tackle first today?',
      'Briefing for today please',
      'What\'s my day look like task-wise?',
      'Morning overview of deadlines and priorities',
      'Daily summary of what needs my attention',
    ],
  },
];

function generateSiblingContrastive(): TrainingExample[] {
  const examples: TrainingExample[] = [];

  for (const pair of SIBLING_PAIRS) {
    const category = pair.toolA.split('.')[0];

    for (const prompt of pair.promptsForA) {
      const candidates = getRealisticCandidates(pair.toolA);
      // Ensure the confusing sibling is in the candidate set
      if (!candidates.includes(pair.toolB)) {
        candidates[candidates.length - 1] = pair.toolB;
      }
      const params = inferParams(pair.toolA, prompt);
      examples.push({
        messages: [
          { role: 'system', content: buildSystemPrompt(candidates) },
          { role: 'user', content: prompt },
          { role: 'assistant', content: formatToolCall(pair.toolA, params) },
        ],
        metadata: {
          source: 'synthetic-sibling',
          category,
          expectedTool: pair.toolA,
          difficulty: 'medium',
          failureMode: 'FM-sibling-confusion',
        },
      });
    }

    for (const prompt of pair.promptsForB) {
      const candidates = getRealisticCandidates(pair.toolB);
      if (!candidates.includes(pair.toolA)) {
        candidates[candidates.length - 1] = pair.toolA;
      }
      const params = inferParams(pair.toolB, prompt);
      examples.push({
        messages: [
          { role: 'system', content: buildSystemPrompt(candidates) },
          { role: 'user', content: prompt },
          { role: 'assistant', content: formatToolCall(pair.toolB, params) },
        ],
        metadata: {
          source: 'synthetic-sibling',
          category,
          expectedTool: pair.toolB,
          difficulty: 'medium',
          failureMode: 'FM-sibling-confusion',
        },
      });
    }
  }

  return examples;
}

// ─── Source 4: Cross-Server Disambiguation ──────────────────────────────────

interface CrossServerPair {
  correct: string;
  confused: string;
  prompts: string[];
}

const CROSS_SERVER_PAIRS: CrossServerPair[] = [
  {
    correct: 'ocr.extract_text_from_image',
    confused: 'document.extract_text',
    prompts: [
      'Read the text from this receipt photo on my Desktop',
      'Extract text from the screenshot I just took',
      'OCR this image of the whiteboard notes',
      'Get the text from the scanned business card',
      'Read what it says in this photo of the sign',
      'Extract the text from the product label image',
      'Read the handwritten note in this picture',
      'OCR the menu photo I took at the restaurant',
      'Get the text from this image of a receipt',
      'Extract text from the photo of the document',
      'Read the text in this PNG screenshot',
      'What does this image of the error message say?',
      'Extract text from the JPEG photo of the form',
      'Read the text from the captured screen image',
      'OCR this photograph of the page',
    ],
  },
  {
    correct: 'document.extract_text',
    confused: 'ocr.extract_text_from_image',
    prompts: [
      'Extract the text from the project proposal PDF',
      'Read the content of the contract DOCX file',
      'Get the text from the README.md file',
      'Extract text from the annual report PDF',
      'Read the content of the meeting-notes.docx',
      'Get the text from the requirements.txt file',
      'Extract content from the design-spec.pdf',
      'Read the employee handbook PDF',
      'Get the text from this Word document',
      'Extract content from the lease-agreement.pdf',
      'Read the text from the RTF file',
      'Get the content of this TXT document',
      'Extract text from the research-paper.pdf',
      'Read the markdown documentation file',
      'Get text from the NDA document',
    ],
  },
  {
    correct: 'data.query_sqlite',
    confused: 'filesystem.read_file',
    prompts: [
      'Query the database for total sales this quarter',
      'Look up order #1234 in the database',
      'Run a SQL query on the inventory table',
      'Get the average price from the products database',
      'Query the users table for active accounts',
      'Look up the top customers by revenue',
      'Run a SELECT on the transactions table',
      'Get the count of orders this month from the DB',
      'Query for all employees in the engineering team',
      'Look up the shipping status from the orders database',
    ],
  },
  {
    correct: 'data.query_sqlite',
    confused: 'document.read_spreadsheet',
    prompts: [
      'Analyze the data in the SQLite database',
      'Get a summary of records from the database',
      'Count entries in the database by category',
      'Look at the database for revenue numbers',
      'Check the database for duplicate entries',
      'Pull financial data from the SQLite file',
      'Get statistics from the metrics database',
      'Query the analytics database for page views',
      'Look at the transaction history in the DB',
      'Get the latest entries from the log database',
    ],
  },
  {
    correct: 'knowledge.search_documents',
    confused: 'filesystem.search_files',
    prompts: [
      'Search my notes for information about the migration',
      'Find documents that discuss the pricing strategy',
      'Search the knowledge base for deployment guides',
      'Look through my indexed files for API docs',
      'Search for notes about the architecture decision',
      'Find indexed documents mentioning the deadline',
      'Search my knowledge base for onboarding info',
      'Look for information about the security policy',
      'Search indexed docs for the design rationale',
      'Find notes about the Q4 planning meeting',
    ],
  },
  {
    correct: 'filesystem.search_files',
    confused: 'knowledge.search_documents',
    prompts: [
      'Find all PDF files in my Documents folder',
      'Search for files named "report" on my Desktop',
      'Find files larger than 100MB in Downloads',
      'Search for .py files in the project directory',
      'Find all images in the screenshots folder',
      'Search for files modified today',
      'Find any .env files in the repository',
      'Search for CSV files in the data directory',
      'Find all log files on the system',
      'Search for files with "invoice" in the name',
    ],
  },
];

function generateCrossServerExamples(): TrainingExample[] {
  const examples: TrainingExample[] = [];

  for (const pair of CROSS_SERVER_PAIRS) {
    for (const prompt of pair.prompts) {
      const candidates = getRealisticCandidates(pair.correct);
      if (!candidates.includes(pair.confused)) {
        candidates[candidates.length - 1] = pair.confused;
      }
      const params = inferParams(pair.correct, prompt);
      examples.push({
        messages: [
          { role: 'system', content: buildSystemPrompt(candidates) },
          { role: 'user', content: prompt },
          { role: 'assistant', content: formatToolCall(pair.correct, params) },
        ],
        metadata: {
          source: 'synthetic-crossserver',
          category: pair.correct.split('.')[0],
          expectedTool: pair.correct,
          difficulty: 'hard',
          failureMode: 'FM-cross-server',
        },
      });
    }
  }

  return examples;
}

// ─── Source 5: Anti-Refusal Examples ────────────────────────────────────────

interface AntiRefusalExample {
  tool: string;
  prompts: string[];
}

const ANTI_REFUSAL_TOOLS: AntiRefusalExample[] = [
  {
    tool: 'filesystem.delete_file',
    prompts: [
      'Delete the old draft from my Desktop',
      'Remove the temporary file from Downloads',
      'Delete screenshot-2024.png from the screenshots folder',
      'Remove the outdated backup file',
      'Delete the duplicate copy of the report',
      'Remove the cached file from .cache directory',
      'Delete the old log files in /tmp',
      'Remove the failed download file',
      'Delete the test output files',
      'Remove the stale lock file',
    ],
  },
  {
    tool: 'filesystem.write_file',
    prompts: [
      'Create a new file with the meeting summary',
      'Write the project notes to a file',
      'Save this configuration to config.yaml',
      'Create a TODO.md file with the task list',
      'Write the extracted data to output.txt',
      'Save the template to a new file',
      'Create a new Python script file',
      'Write the error log to debug.txt',
      'Save the formatted report as report.md',
      'Create a changelog entry file',
    ],
  },
  {
    tool: 'security.encrypt_file',
    prompts: [
      'Encrypt the tax documents folder',
      'Lock down the salary spreadsheet',
      'Password-protect the client database',
      'Encrypt the medical records file',
      'Secure the financial reports with encryption',
      'Lock the personnel file for safekeeping',
      'Encrypt the backup before cloud upload',
      'Protect the contract documents with encryption',
      'Encrypt sensitive data in the export file',
      'Lock down the API credentials file',
    ],
  },
  {
    tool: 'clipboard.set_clipboard',
    prompts: [
      'Copy this text to my clipboard',
      'Put the extracted data on my clipboard',
      'Copy the file path to clipboard',
      'Set the clipboard to this URL',
      'Copy the summary to my clipboard',
      'Put this code snippet on the clipboard',
      'Copy the selected text to clipboard',
      'Set clipboard content to the output',
      'Copy the generated password to clipboard',
      'Put the formatted text on my clipboard',
    ],
  },
  {
    tool: 'security.scan_for_secrets',
    prompts: [
      'Check this project for exposed API keys',
      'Scan the codebase for leaked passwords',
      'Look for secrets in the configuration files',
      'Check if any tokens are exposed in the repo',
      'Scan for credential leaks in the source code',
      'Find exposed private keys in the project',
      'Check for hardcoded passwords in the config',
      'Scan the environment files for exposed secrets',
      'Look for leaked credentials in the codebase',
      'Check if any API keys are committed to git',
    ],
  },
  {
    tool: 'audit.get_tool_log',
    prompts: [
      'Show me which tools were used today',
      'What tools have been called in this session?',
      'Get the tool execution history',
      'Show the audit log of tool usage',
      'What tool calls happened in the last hour?',
      'List the tools that were executed recently',
      'Show me the tool activity log',
      'What has been done in this session so far?',
      'Get the history of tool calls',
      'Show the log of executed tools',
    ],
  },
];

function generateAntiRefusalExamples(): TrainingExample[] {
  const examples: TrainingExample[] = [];

  for (const entry of ANTI_REFUSAL_TOOLS) {
    for (const prompt of entry.prompts) {
      const candidates = getRealisticCandidates(entry.tool);
      const params = inferParams(entry.tool, prompt);
      examples.push({
        messages: [
          { role: 'system', content: buildSystemPrompt(candidates) },
          { role: 'user', content: prompt },
          { role: 'assistant', content: formatToolCall(entry.tool, params) },
        ],
        metadata: {
          source: 'synthetic-antirefusal',
          category: entry.tool.split('.')[0],
          expectedTool: entry.tool,
          difficulty: 'medium',
          failureMode: 'FM-refusal',
        },
      });
    }
  }

  return examples;
}

// ─── Source 6: Weak Category Boosting ───────────────────────────────────────

interface CategoryBoost {
  tool: string;
  prompts: string[];
}

const WEAK_CATEGORY_BOOSTS: CategoryBoost[] = [
  // file-operations (60% accuracy) — extra examples
  {
    tool: 'filesystem.get_metadata',
    prompts: [
      'How big is the report.pdf file?',
      'When was this file last modified?',
      'What are the permissions on the config file?',
      'Get the creation date of the project folder',
      'How large is the Downloads directory?',
      'What type of file is document.bin?',
      'When was this photo taken? Check the file metadata',
      'Show me the file details for the backup',
    ],
  },
  {
    tool: 'filesystem.watch_folder',
    prompts: [
      'Monitor my Downloads folder for new files',
      'Watch the inbox directory for incoming documents',
      'Set up a notification when files change in the project folder',
      'Track changes in the shared drive directory',
      'Watch the log directory for new entries',
      'Monitor the upload folder for incoming files',
      'Set up file change tracking on the config directory',
      'Watch for new screenshots in the screenshots folder',
    ],
  },
  // email (62.5% accuracy) — extra examples
  {
    tool: 'email.summarize_thread',
    prompts: [
      'Summarize the email thread about the project deadline',
      'Give me a recap of the conversation with the client',
      'What was the email discussion about the budget?',
      'Summarize the back-and-forth about the design changes',
      'Recap the email chain regarding the contract renewal',
      'What was said in the email thread about hiring?',
      'Summarize the correspondence with the vendor',
      'Give me the highlights from the email discussion',
    ],
  },
  // document-processing (66.7% accuracy) — extra examples
  {
    tool: 'document.diff_documents',
    prompts: [
      'Compare these two versions of the contract',
      'Show the differences between draft v1 and v2',
      'What changed between the original and revised proposal?',
      'Diff the old and new versions of the report',
      'Compare the two policy documents for changes',
      'Show what was modified between these two files',
      'Find the differences between the two specifications',
      'Compare the original and updated requirements',
    ],
  },
  {
    tool: 'document.merge_pdfs',
    prompts: [
      'Merge these three PDF files into one',
      'Combine the report sections into a single PDF',
      'Join the individual chapter PDFs together',
      'Merge all the invoice PDFs into one document',
      'Combine the presentation slides into one PDF',
      'Join the application form pages into a single file',
      'Merge the scanned pages into one PDF document',
      'Combine all the contract appendices into one PDF',
    ],
  },
  // knowledge-search (71.4% accuracy) — extra examples
  {
    tool: 'knowledge.index_folder',
    prompts: [
      'Index all the documents in my project folder',
      'Build a search index for my research papers',
      'Index the company wiki folder for searching',
      'Set up semantic search on my notes directory',
      'Index the documentation folder for RAG',
      'Build a knowledge base from my Documents folder',
      'Index all the markdown files in the docs directory',
      'Set up search indexing on the shared knowledge base',
    ],
  },
  {
    tool: 'knowledge.get_related_chunks',
    prompts: [
      'Find related passages about database optimization',
      'Get text chunks related to the authentication flow',
      'Find relevant sections about error handling patterns',
      'Get related content about the deployment process',
      'Find passages related to the pricing strategy',
      'Get relevant chunks about the API design',
      'Find related text about the migration plan',
      'Get passages that discuss performance tuning',
    ],
  },
];

function generateWeakCategoryBoosts(): TrainingExample[] {
  const examples: TrainingExample[] = [];

  for (const boost of WEAK_CATEGORY_BOOSTS) {
    for (const prompt of boost.prompts) {
      const candidates = getRealisticCandidates(boost.tool);
      const params = inferParams(boost.tool, prompt);
      examples.push({
        messages: [
          { role: 'system', content: buildSystemPrompt(candidates) },
          { role: 'user', content: prompt },
          { role: 'assistant', content: formatToolCall(boost.tool, params) },
        ],
        metadata: {
          source: 'synthetic-category-boost',
          category: boost.tool.split('.')[0],
          expectedTool: boost.tool,
          difficulty: 'medium',
          failureMode: 'FM-weak-category',
        },
      });
    }
  }

  return examples;
}

// ─── Param Inference ────────────────────────────────────────────────────────

/** Infer realistic parameters from the tool name and user prompt. */
function inferParams(toolName: string, prompt: string): Record<string, string> {
  const lower = prompt.toLowerCase();

  // Extract file paths mentioned in the prompt
  const pathMatch = prompt.match(/(?:\/[\w./-]+|~\/[\w./-]+|my\s+(\w+)\s+folder)/i);

  switch (toolName) {
    case 'filesystem.list_dir':
      if (lower.includes('download')) return { path: '/Users/user/Downloads' };
      if (lower.includes('desktop')) return { path: '/Users/user/Desktop' };
      if (lower.includes('document')) return { path: '/Users/user/Documents' };
      if (lower.includes('project')) return { path: '/Users/user/Projects' };
      return { path: '/Users/user/Documents' };

    case 'filesystem.read_file':
    case 'filesystem.delete_file':
    case 'filesystem.get_metadata':
      if (pathMatch) return { path: pathMatch[0] };
      return { path: '/Users/user/Documents/file.txt' };

    case 'filesystem.move_file':
    case 'filesystem.copy_file':
      return { source: '/Users/user/Documents/file.txt', destination: '/Users/user/Archive/file.txt' };

    case 'filesystem.search_files':
      return { path: '/Users/user', pattern: '*' };

    case 'filesystem.watch_folder':
      return { path: '/Users/user/Downloads' };

    case 'document.extract_text':
      return { file_path: '/Users/user/Documents/document.pdf' };

    case 'ocr.extract_text_from_image':
      return { source: '/Users/user/Desktop/screenshot.png' };

    case 'data.query_sqlite':
      return { database: '/Users/user/data/app.db', query: 'SELECT * FROM records' };

    case 'calendar.list_events':
      return { date_range: 'today' };
    case 'calendar.find_free_slots':
      return { date: 'today', duration_minutes: '30' };
    case 'calendar.create_event':
      return { title: 'Meeting', date: 'tomorrow' };
    case 'calendar.create_time_block':
      return { title: 'Focus Time', duration_hours: '2' };

    case 'email.draft_email':
      return { to: 'team@example.com', subject: 'Update' };
    case 'email.send_draft':
      return { draft_id: 'draft-001' };
    case 'email.search_emails':
      return { query: 'project update' };

    case 'security.encrypt_file':
    case 'security.decrypt_file':
      return { file_path: '/Users/user/Documents/sensitive.pdf' };
    case 'security.scan_for_pii':
    case 'security.scan_for_secrets':
      return { path: '/Users/user/Projects' };

    case 'knowledge.search_documents':
    case 'knowledge.ask_about_files':
      return { query: 'project deadline' };
    case 'knowledge.index_folder':
      return { path: '/Users/user/Documents' };

    case 'clipboard.set_clipboard':
      return { content: 'copied text' };

    default:
      return {};
  }
}

// ─── Paraphrase Generation ──────────────────────────────────────────────────

const PARAPHRASE_TEMPLATES: Array<(prompt: string) => string | null> = [
  // Add "please" / "can you"
  (p) => `Can you ${p.charAt(0).toLowerCase()}${p.slice(1)}`,
  (p) => `Please ${p.charAt(0).toLowerCase()}${p.slice(1)}`,
  // "I need to" / "I want to"
  (p) => {
    if (p.toLowerCase().startsWith('show') || p.toLowerCase().startsWith('list'))
      return `I need to see ${p.replace(/^(show|list)\s+(me\s+)?/i, '')}`;
    return null;
  },
  // Imperative to question
  (p) => {
    if (!p.endsWith('?')) return `${p}?`;
    return null;
  },
];

function generateParaphrases(examples: TrainingExample[], maxNew: number): TrainingExample[] {
  const paraphrased: TrainingExample[] = [];
  const shuffled = [...examples];
  shuffleArray(shuffled);

  for (const ex of shuffled) {
    if (paraphrased.length >= maxNew) break;

    for (const template of PARAPHRASE_TEMPLATES) {
      if (paraphrased.length >= maxNew) break;

      const userMsg = ex.messages.find((m) => m.role === 'user');
      if (!userMsg) continue;

      const newPrompt = template(userMsg.content);
      if (!newPrompt || newPrompt === userMsg.content) continue;

      paraphrased.push({
        messages: [
          ex.messages[0], // system prompt
          { role: 'user', content: newPrompt },
          ex.messages[2], // assistant response
        ],
        metadata: {
          ...ex.metadata,
          source: `paraphrase-${ex.metadata.source}`,
        },
      });
    }
  }

  return paraphrased;
}

// ─── Train/Eval/Test Split ──────────────────────────────────────────────────

interface DataSplit {
  train: TrainingExample[];
  eval: TrainingExample[];
  test: TrainingExample[];
}

function stratifiedSplit(examples: TrainingExample[]): DataSplit {
  // Group by category
  const byCategory: Record<string, TrainingExample[]> = {};
  for (const ex of examples) {
    const cat = ex.metadata.category;
    if (!byCategory[cat]) byCategory[cat] = [];
    byCategory[cat].push(ex);
  }

  const train: TrainingExample[] = [];
  const evalSet: TrainingExample[] = [];
  const test: TrainingExample[] = [];

  // For each category, do 80/10/10 split
  for (const [, catExamples] of Object.entries(byCategory)) {
    shuffleArray(catExamples);
    const n = catExamples.length;
    const evalSize = Math.max(1, Math.round(n * 0.1));
    const testSize = Math.max(1, Math.round(n * 0.1));

    test.push(...catExamples.slice(0, testSize));
    evalSet.push(...catExamples.slice(testSize, testSize + evalSize));
    train.push(...catExamples.slice(testSize + evalSize));
  }

  // Shuffle each split
  shuffleArray(train);
  shuffleArray(evalSet);
  shuffleArray(test);

  return { train, eval: evalSet, test };
}

// ─── Format as JSONL ────────────────────────────────────────────────────────

function formatAsJsonl(examples: TrainingExample[]): string {
  return examples.map((ex) => JSON.stringify({
    messages: ex.messages,
    metadata: ex.metadata,
  })).join('\n');
}

// ─── Main ───────────────────────────────────────────────────────────────────

function main(): void {
  console.log('=== LocalCowork Training Data Generator ===\n');

  // Generate from all sources
  const benchmarkExamples = generateFromBenchmarks();
  console.log(`Source 1 (benchmarks): ${benchmarkExamples.length} examples`);

  const multiStepExamples = generateFromMultiStepChains();
  console.log(`Source 2 (multi-step isolated): ${multiStepExamples.length} examples`);

  const siblingExamples = generateSiblingContrastive();
  console.log(`Source 3 (sibling contrastive): ${siblingExamples.length} examples`);

  const crossServerExamples = generateCrossServerExamples();
  console.log(`Source 4 (cross-server): ${crossServerExamples.length} examples`);

  const antiRefusalExamples = generateAntiRefusalExamples();
  console.log(`Source 5 (anti-refusal): ${antiRefusalExamples.length} examples`);

  const weakCategoryExamples = generateWeakCategoryBoosts();
  console.log(`Source 6 (weak categories): ${weakCategoryExamples.length} examples`);

  // Combine all non-paraphrase examples
  const coreExamples = [
    ...benchmarkExamples,
    ...multiStepExamples,
    ...siblingExamples,
    ...crossServerExamples,
    ...antiRefusalExamples,
    ...weakCategoryExamples,
  ];
  console.log(`\nCore examples: ${coreExamples.length}`);

  // Generate paraphrases (target: ~40% of core)
  const targetParaphrases = Math.round(coreExamples.length * 0.4);
  const paraphraseExamples = generateParaphrases(coreExamples, targetParaphrases);
  console.log(`Paraphrases: ${paraphraseExamples.length}`);

  const allExamples = [...coreExamples, ...paraphraseExamples];
  console.log(`Total: ${allExamples.length}\n`);

  // Split
  const { train, eval: evalSet, test } = stratifiedSplit(allExamples);
  console.log(`Train: ${train.length} | Eval: ${evalSet.length} | Test: ${test.length}`);

  // Write output
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.writeFileSync(path.join(OUTPUT_DIR, 'train.jsonl'), formatAsJsonl(train));
  fs.writeFileSync(path.join(OUTPUT_DIR, 'eval.jsonl'), formatAsJsonl(evalSet));
  fs.writeFileSync(path.join(OUTPUT_DIR, 'test.jsonl'), formatAsJsonl(test));

  // Compute stats
  const stats: GenerationStats = {
    total: allExamples.length,
    bySource: {},
    byCategory: {},
    byDifficulty: {},
    trainCount: train.length,
    evalCount: evalSet.length,
    testCount: test.length,
    generatedAt: new Date().toISOString(),
  };

  for (const ex of allExamples) {
    stats.bySource[ex.metadata.source] = (stats.bySource[ex.metadata.source] ?? 0) + 1;
    stats.byCategory[ex.metadata.category] = (stats.byCategory[ex.metadata.category] ?? 0) + 1;
    stats.byDifficulty[ex.metadata.difficulty] = (stats.byDifficulty[ex.metadata.difficulty] ?? 0) + 1;
  }

  fs.writeFileSync(path.join(OUTPUT_DIR, 'metadata.json'), JSON.stringify(stats, null, 2));

  console.log('\nBy source:');
  for (const [source, count] of Object.entries(stats.bySource).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${source}: ${count}`);
  }

  console.log('\nBy category:');
  for (const [cat, count] of Object.entries(stats.byCategory).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${cat}: ${count}`);
  }

  console.log(`\nOutput written to ${OUTPUT_DIR}/`);
  console.log('  train.jsonl');
  console.log('  eval.jsonl');
  console.log('  test.jsonl');
  console.log('  metadata.json');
}

main();
