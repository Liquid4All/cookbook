/**
 * Data MCP Server — Entry Point
 *
 * Registers all data tools and starts the JSON-RPC listener.
 * This server provides CSV and SQLite operations for LocalCowork.
 *
 * Tools (5):
 *   data.write_csv             — write structured data to CSV file (confirm)
 *   data.write_sqlite          — write data to SQLite table (confirm)
 *   data.query_sqlite          — read-only SQL query on SQLite DB
 *   data.deduplicate_records   — find duplicate records
 *   data.summarize_anomalies   — detect anomalies in a dataset
 */

import os from 'os';
import path from 'path';

import { MCPServer } from '../../_shared/ts/mcp-base';
import { initSandbox } from '../../_shared/ts/validation';
import { writeCsv } from './tools/write_csv';
import { writeSqlite } from './tools/write_sqlite';
import { querySqlite } from './tools/query_sqlite';
import { deduplicateRecords } from './tools/deduplicate_records';
import { summarizeAnomalies } from './tools/summarize_anomalies';

// ─── Sandbox Initialization ─────────────────────────────────────────────────
// Allowed paths come from the Agent Core via environment variable.
// Default to home directory for development.

const allowedPaths = process.env.LOCALCOWORK_ALLOWED_PATHS
  ? process.env.LOCALCOWORK_ALLOWED_PATHS.split(path.delimiter)
  : [os.homedir()];

initSandbox(allowedPaths);

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'data',
  version: '1.0.0',
  tools: [writeCsv, writeSqlite, querySqlite, deduplicateRecords, summarizeAnomalies],
});

// ─── Graceful Shutdown ──────────────────────────────────────────────────────

process.on('SIGINT', () => {
  process.exit(0);
});

process.on('SIGTERM', () => {
  process.exit(0);
});

// ─── Start ──────────────────────────────────────────────────────────────────

server.start();
