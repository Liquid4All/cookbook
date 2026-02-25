/**
 * Audit MCP Server — Entry Point
 *
 * Registers all audit tools and starts the JSON-RPC listener.
 * This server provides read-only access to the audit log and
 * generates compliance reports.
 *
 * Tools (4):
 *   audit.get_tool_log           — query audit log entries
 *   audit.get_session_summary    — session aggregate summary
 *   audit.generate_audit_report  — text report generation
 *   audit.export_audit_pdf       — PDF export (confirm)
 */

import os from 'os';
import path from 'path';

import { MCPServer } from '../../_shared/ts/mcp-base';
import { initSandbox } from '../../_shared/ts/validation';
import { closeDb } from './db';
import { getToolLog } from './tools/get_tool_log';
import { getSessionSummary } from './tools/get_session_summary';
import { generateAuditReport } from './tools/generate_audit_report';
import { exportAuditPdf } from './tools/export_audit_pdf';

// ─── Sandbox Initialization ─────────────────────────────────────────────────

const allowedPaths = process.env.LOCALCOWORK_ALLOWED_PATHS
  ? process.env.LOCALCOWORK_ALLOWED_PATHS.split(path.delimiter)
  : [os.homedir()];

initSandbox(allowedPaths);

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'audit',
  version: '1.0.0',
  tools: [getToolLog, getSessionSummary, generateAuditReport, exportAuditPdf],
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
