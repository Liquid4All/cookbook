/**
 * Shared utilities for LFM model benchmarks.
 *
 * Extracted from benchmark-lfm.ts and benchmark-multi-step.ts to avoid
 * duplication across the single-step, multi-step, and orchestrator benchmarks.
 */

import { VALID_TOOL_NAMES, VALID_TOOL_SET } from './types';

// ─── Tool Descriptions (contrastive, synonym-augmented) ────────────────────

export const TOOL_DESCRIPTIONS: Record<string, string> = {
  'filesystem.list_dir': 'List all files and folders in a single directory. Use for browsing, not searching',
  'filesystem.read_file': 'Read the text contents of a file by path. For spreadsheets use document.read_spreadsheet instead',
  'filesystem.write_file': 'Write, create, or save text content to a file on disk',
  'filesystem.move_file': 'Move or rename a file. Handles renaming by moving to the same directory with a new name',
  'filesystem.copy_file': 'Copy or duplicate a file to a new location or backup',
  'filesystem.delete_file': 'Delete, remove, or trash a file from disk',
  'filesystem.search_files': 'Search, find, or locate files by name pattern, extension, size, or content match. Use for recursive or filtered file searches',
  'filesystem.get_metadata': 'Get file metadata: size, creation date, modification date, permissions, and file type',
  'filesystem.watch_folder': 'Watch a folder for real-time file change notifications. Only for monitoring, not for listing',
  'document.extract_text': 'Extract text from document files: PDF, DOCX, TXT, RTF, MD. NOT for images or screenshots — use ocr tools instead',
  'document.convert_format': 'Convert a document between formats: PDF to DOCX, Markdown to HTML, DOCX to PDF, etc.',
  'document.diff_documents': 'Compare two document versions and show the differences between them. Use for tracking changes',
  'document.create_pdf': 'Create a new PDF document from text or markdown content. Use for generating reports or summaries as PDF',
  'document.fill_pdf_form': 'Fill in specific fields of an existing PDF form. Requires an existing PDF with form fields',
  'document.merge_pdfs': 'Merge or combine multiple PDF files into a single PDF',
  'document.create_docx': 'Create a new Word DOCX document from text or markdown content',
  'document.read_spreadsheet': 'Read data from spreadsheet files: CSV, XLSX, XLS. Use for tabular data, NOT for reading text files',
  'ocr.extract_text_from_image': 'OCR: Extract text from images, screenshots, photos, or scanned pictures. NOT for PDF or document files',
  'ocr.extract_text_from_pdf': 'OCR: Extract text from scanned PDFs where text is embedded as images. For normal PDFs use document.extract_text',
  'ocr.extract_structured_data': 'OCR: Extract structured fields from images of receipts, invoices, business cards, or forms. Returns named fields like amount, date, vendor',
  'ocr.extract_table': 'OCR: Extract table or grid data from images or scanned documents into rows and columns',
  'data.write_csv': 'Write or export structured data to a CSV file',
  'data.write_sqlite': 'Write, insert, or import records into a SQLite database table',
  'data.query_sqlite': 'Query or read from a SQLite database using SQL. Use for database lookups, reports, and analytics',
  'data.deduplicate_records': 'Find and remove duplicate records in a database table based on key columns',
  'data.summarize_anomalies': 'Detect anomalies, outliers, unusual patterns, or irregularities in database data',
  'audit.get_tool_log': 'Get the log of which tools were used, executed, or called in previous sessions',
  'audit.get_session_summary': 'Get a summary or recap of what happened in a previous work session',
  'audit.generate_audit_report': 'Generate a comprehensive audit report covering tool usage over a date range',
  'audit.export_audit_pdf': 'Export an existing audit report as a PDF file',
  'knowledge.index_folder': 'Index a folder of documents for semantic search and RAG question-answering',
  'knowledge.search_documents': 'Semantic search across previously indexed documents by meaning, not just keywords',
  'knowledge.ask_about_files': 'Ask a natural language question about previously indexed files and get an AI-generated answer',
  'knowledge.update_index': 'Update the search index when documents have been added or changed since last indexing',
  'knowledge.get_related_chunks': 'Retrieve related text passages or chunks from indexed documents for a given topic',
  'security.scan_for_pii': 'Scan for personal identity data: names, SSNs, addresses, phone numbers, dates of birth, email addresses',
  'security.scan_for_secrets': 'Scan for credentials and secrets: API keys, passwords, tokens, private keys, connection strings',
  'security.find_duplicates': 'Find duplicate or identical files in a directory to free up storage space',
  'security.propose_cleanup': 'Suggest files that can be safely deleted or archived to clean up a directory',
  'security.encrypt_file': 'Encrypt, lock down, or password-protect a file for secure storage',
  'security.decrypt_file': 'Decrypt or unlock a previously encrypted file',
  'task.create_task': 'Create a new task or to-do item with optional due date and priority',
  'task.list_tasks': 'List tasks or to-dos with optional filters by status, priority, or date',
  'task.update_task': 'Update, modify, or mark a task as complete, change its status, due date, or priority',
  'task.get_overdue': 'Get a list of tasks that are past their due date',
  'task.daily_briefing': 'Generate a daily briefing summarizing upcoming tasks, deadlines, and priorities',
  'calendar.list_events': 'View, show, or list your existing scheduled meetings and calendar events for a date range',
  'calendar.create_event': 'Schedule a new meeting, appointment, or event with other people on the calendar',
  'calendar.find_free_slots': 'Find open, available, or free time slots when you have no events scheduled',
  'calendar.create_time_block': 'Block personal focus time or deep work time for yourself with no attendees',
  'email.draft_email': 'Compose, draft, or write a new email message. Does NOT send it — creates a draft only',
  'email.list_drafts': 'List previously drafted email messages',
  'email.search_emails': 'Search through emails by keyword, sender, subject, or date',
  'email.summarize_thread': 'Summarize or recap an email conversation thread',
  'email.send_draft': 'Send a previously drafted email. Requires a draft_id from draft_email or list_drafts',
  'meeting.transcribe_audio': 'Transcribe speech from an audio or video recording file into text',
  'meeting.extract_action_items': 'Extract action items and to-dos from meeting notes or a transcript',
  'meeting.extract_commitments': 'Extract promises, commitments, and agreements made during a meeting',
  'meeting.generate_minutes': 'Generate complete formatted meeting minutes from a transcript. Includes summary, decisions, and action items',
  'clipboard.get_clipboard': 'Get or read the current system clipboard contents',
  'clipboard.set_clipboard': 'Copy text to the system clipboard. Sets or writes to the clipboard',
  'clipboard.clipboard_history': 'Get the history of recent clipboard entries',
  'system.get_system_info': 'Get system information: OS, CPU, memory, disk space, and hardware details',
  'system.open_application': 'Launch or open a desktop application by name',
  'system.take_screenshot': 'Capture a screenshot of the screen',
  'system.list_processes': 'List currently running processes and applications',
  'system.open_file_with': 'Open a file using a specific desktop application (e.g., open PDF in Preview)',
};

// ─── Tool Definition Builders ──────────────────────────────────────────────

export interface ToolDef {
  name: string;
  description: string;
  parameters: { type: string; properties: Record<string, unknown> };
}

export function buildToolDefinitions(): ToolDef[] {
  return VALID_TOOL_NAMES.map((name) => ({
    name,
    description: TOOL_DESCRIPTIONS[name] ?? name.replace(/[._]/g, ' '),
    parameters: { type: 'object', properties: {} },
  }));
}

export function buildFilteredToolDefinitions(toolNames: string[]): ToolDef[] {
  return toolNames.map((name) => ({
    name,
    description: TOOL_DESCRIPTIONS[name] ?? name.replace(/[._]/g, ' '),
    parameters: { type: 'object', properties: {} },
  }));
}

// ─── Embedding & Pre-Filter ────────────────────────────────────────────────

interface RawEmbeddingItem {
  index: number;
  embedding: number[] | number[][];
}

export function meanPoolEmbedding(embedding: number[] | number[][]): number[] {
  if (embedding.length === 0) return [];
  if (typeof embedding[0] === 'number') return embedding as number[];
  const tokens = embedding as number[][];
  const nDim = tokens[0].length;
  const result = new Array<number>(nDim).fill(0);
  for (const token of tokens) {
    for (let d = 0; d < nDim; d++) result[d] += token[d];
  }
  for (let d = 0; d < nDim; d++) result[d] /= tokens.length;
  return result;
}

export async function embedTexts(endpoint: string, texts: string[]): Promise<number[][]> {
  const response = await fetch(`${endpoint}/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: texts }),
  });
  if (!response.ok) {
    throw new Error(`Embedding request failed (HTTP ${response.status}): ${await response.text()}`);
  }
  const result = (await response.json()) as RawEmbeddingItem[];
  const sorted = result.sort((a, b) => a.index - b.index);
  return sorted.map((d) => meanPoolEmbedding(d.embedding));
}

export function l2Normalize(vec: number[]): number[] {
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  }
  return vec;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export interface ToolEmbeddingIndex {
  toolNames: string[];
  embeddingTexts: string[];
  embeddings: number[][];
}

export async function buildToolEmbeddingIndex(
  endpoint: string,
  toolDefs: ToolDef[],
): Promise<ToolEmbeddingIndex> {
  const texts = toolDefs.map((t) => `${t.name}: ${t.description}`);
  const raw = await embedTexts(endpoint, texts);
  const embeddings = raw.map((vec) => l2Normalize(vec));
  return { toolNames: toolDefs.map((t) => t.name), embeddingTexts: texts, embeddings };
}

export async function filterToolsByRelevance(
  endpoint: string,
  query: string,
  index: ToolEmbeddingIndex,
  topK: number,
): Promise<{ selectedTools: string[]; scores: Array<{ name: string; score: number }> }> {
  const [rawQueryEmb] = await embedTexts(endpoint, [query]);
  const queryEmb = l2Normalize(rawQueryEmb);
  const scored = index.toolNames.map((name, i) => ({
    name,
    score: cosineSimilarity(queryEmb, index.embeddings[i]),
  }));
  scored.sort((a, b) => b.score - a.score);
  return { selectedTools: scored.slice(0, topK).map((s) => s.name), scores: scored };
}

// ─── LFM Bracket Parser ────────────────────────────────────────────────────

const TOOL_CALL_START = '<|tool_call_start|>';
const TOOL_CALL_END = '<|tool_call_end|>';
const TOOL_NAME_SET: ReadonlySet<string> = VALID_TOOL_SET;

export function parseLfmToolCalls(content: string): string[] {
  const tools: string[] = [];

  // Mode 1: Special token markers
  let searchFrom = 0;
  while (true) {
    const startIdx = content.indexOf(TOOL_CALL_START, searchFrom);
    if (startIdx === -1) break;
    const afterStart = startIdx + TOOL_CALL_START.length;
    const endIdx = content.indexOf(TOOL_CALL_END, afterStart);
    if (endIdx === -1) break;
    const block = content.slice(afterStart, endIdx).trim();
    searchFrom = endIdx + TOOL_CALL_END.length;
    const inner = block.startsWith('[') && block.endsWith(']') ? block.slice(1, -1) : block;
    if (!inner) continue;
    const parenIdx = inner.indexOf('(');
    const funcName = parenIdx >= 0 ? inner.slice(0, parenIdx).trim() : inner.trim();
    if (funcName) tools.push(funcName);
  }

  // Mode 2: Bare bracket [tool.name(args)]
  if (tools.length === 0) {
    const bracketPattern = /\[([a-z_]+\.[a-z_]+)\(([^)]*)\)\]/g;
    let match: RegExpExecArray | null;
    while ((match = bracketPattern.exec(content)) !== null) {
      if (TOOL_NAME_SET.has(match[1])) tools.push(match[1]);
    }
  }

  // Mode 3: Lenient bracket
  if (tools.length === 0) {
    const lenientBracket = /\[(?:[a-z]+\.)?([a-z_]+\.[a-z_]+)(?:\(.*?\))?\]/g;
    let match: RegExpExecArray | null;
    while ((match = lenientBracket.exec(content)) !== null) {
      if (TOOL_NAME_SET.has(match[1])) tools.push(match[1]);
    }
  }

  // Mode 4: Backtick or bare mention
  if (tools.length === 0) {
    const backtickPattern = /`([a-z_]+\.[a-z_]+)`/g;
    let match: RegExpExecArray | null;
    while ((match = backtickPattern.exec(content)) !== null) {
      if (TOOL_NAME_SET.has(match[1])) { tools.push(match[1]); break; }
    }
  }
  if (tools.length === 0) {
    const mentions: Array<{ name: string; pos: number }> = [];
    for (const toolName of TOOL_NAME_SET) {
      const pos = content.indexOf(toolName);
      if (pos >= 0) mentions.push({ name: toolName, pos });
    }
    if (mentions.length > 0) {
      mentions.sort((a, b) => a.pos - b.pos);
      tools.push(mentions[0].name);
    }
  }

  return tools;
}

// ─── Deflection Detection ──────────────────────────────────────────────────

export function isDeflection(text: string): boolean {
  const lower = text.toLowerCase();
  const patterns = [
    'what would you like', 'how would you like', 'how should i',
    'what should i', 'would you like me to', 'shall i',
    'do you want me to', 'let me know', 'please let me know',
    'i can help you', 'i can assist', 'what do you think',
    'which one', 'which files', 'here are some options',
    'i see the files', 'i see the following', 'i found the following',
    'i notice', 'here are the files', 'i can see',
  ];
  for (const p of patterns) {
    if (lower.includes(p)) return true;
  }
  return text.length < 300 && text.includes('?');
}

// ─── Mock Tool Results ─────────────────────────────────────────────────────

export const MOCK_RESULTS: Record<string, string> = {
  'filesystem': '{"files": [{"name": "report.pdf", "size": 24500}, {"name": "notes.txt", "size": 1200}, {"name": "screenshot.png", "size": 850000}]}',
  'document': '{"text": "Executive Summary: Q4 revenue increased 15% year-over-year to $2.3M."}',
  'ocr': '{"text": "Invoice #2026-0342\\nDate: March 15, 2026\\nAmount: $1,247.50\\nVendor: Acme Corp"}',
  'data': '{"rows_affected": 3, "status": "success"}',
  'audit': '{"entries": [{"tool": "filesystem.list_dir", "timestamp": "2026-02-15T08:30:00Z"}]}',
  'knowledge': '{"results": [{"chunk": "The project deadline is April 30, 2026. Budget: $50K.", "score": 0.92}]}',
  'security': '{"findings": [{"type": "email", "value": "john@example.com", "line": 42}], "total": 1}',
  'task': '{"task_id": "task-001", "title": "Review quarterly report", "status": "created"}',
  'calendar': '{"events": [{"title": "Team Standup", "start": "2026-02-16T09:00:00"}]}',
  'email': '{"draft_id": "draft-001", "to": "team@example.com", "subject": "Follow-up: Q4 Review"}',
  'meeting': '{"transcript": "Alice: We need to finalize the proposal by Friday.\\nBob: I will handle the technical section."}',
  'clipboard': '{"content": "Copied text content from clipboard"}',
  'system': '{"os": "macOS 15.3", "cpu": "Apple M3", "memory": "24GB", "disk_free": "156GB"}',
};

export function getMockResult(toolName: string): string {
  const server = toolName.split('.')[0];
  return MOCK_RESULTS[server] ?? '{"status": "success"}';
}

// ─── Model Communication ───────────────────────────────────────────────────

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
}

interface ChatCompletionResponse {
  choices: Array<{
    message: { content?: string | null; tool_calls?: Array<{ function: { name: string; arguments: string } }> };
    finish_reason?: string;
  }>;
}

export async function queryModel(
  endpoint: string,
  messages: ChatMessage[],
  options?: { temperature?: number; topP?: number; maxTokens?: number; model?: string },
): Promise<{ content: string; toolCalls: string[] }> {
  const body: Record<string, unknown> = {
    messages,
    temperature: options?.temperature ?? 0.1,
    top_p: options?.topP ?? 0.1,
    max_tokens: options?.maxTokens ?? 512,
    stream: false,
  };
  if (options?.model) {
    body.model = options.model;
  }
  const response = await fetch(`${endpoint}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Model query failed: ${response.status} ${await response.text()}`);
  }

  const data = (await response.json()) as ChatCompletionResponse;
  const content = data.choices?.[0]?.message?.content ?? '';
  let toolCalls = parseLfmToolCalls(content);
  if (toolCalls.length === 0 && data.choices[0]?.message?.tool_calls) {
    toolCalls = data.choices[0].message.tool_calls.map((tc) => tc.function.name);
  }
  return { content, toolCalls };
}
