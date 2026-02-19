/**
 * Email MCP Server — Entry Point
 *
 * Registers all email tools and starts the JSON-RPC listener.
 * This server provides local SQLite-backed email draft management,
 * email archive search, and thread summarization.
 *
 * Tools (5):
 *   email.draft_email       — create an email draft (confirm)
 *   email.list_drafts       — list saved drafts
 *   email.search_emails     — search local email archive
 *   email.summarize_thread  — summarize an email thread
 *   email.send_draft        — send a draft (confirm)
 */

import os from 'os';
import path from 'path';

import { MCPServer } from '../../_shared/ts/mcp-base';
import { initSandbox } from '../../_shared/ts/validation';
import { closeDb } from './db';
import { draftEmail } from './tools/draft_email';
import { listDrafts } from './tools/list_drafts';
import { searchEmails } from './tools/search_emails';
import { summarizeThread } from './tools/summarize_thread';
import { sendDraft } from './tools/send_draft';

// ─── Sandbox Initialization ─────────────────────────────────────────────────

const allowedPaths = process.env.LOCALCOWORK_ALLOWED_PATHS
  ? process.env.LOCALCOWORK_ALLOWED_PATHS.split(path.delimiter)
  : [os.homedir()];

initSandbox(allowedPaths);

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'email',
  version: '1.0.0',
  tools: [draftEmail, listDrafts, searchEmails, summarizeThread, sendDraft],
});

// ─── Graceful Shutdown ──────────────────────────────────────────────────────

process.on('SIGINT', () => {
  closeDb();
  process.exit(0);
});

process.on('SIGTERM', () => {
  closeDb();
  process.exit(0);
});

// ─── Start ──────────────────────────────────────────────────────────────────

server.start();
