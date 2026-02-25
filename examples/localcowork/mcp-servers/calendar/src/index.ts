/**
 * Calendar MCP Server -- Entry Point
 *
 * Registers all calendar tools and starts the JSON-RPC listener.
 * This server provides local SQLite-backed calendar management
 * with event CRUD and intelligent time-block scheduling.
 *
 * Tools (4):
 *   calendar.list_events       -- list events in a date range
 *   calendar.create_event      -- create a new event (confirm)
 *   calendar.find_free_slots   -- find free time blocks in a day
 *   calendar.create_time_block -- create a focused time block (confirm)
 */

import os from 'os';
import path from 'path';

import { MCPServer } from '../../_shared/ts/mcp-base';
import { initSandbox } from '../../_shared/ts/validation';
import { closeDb } from './db';
import { listEvents } from './tools/list_events';
import { createEvent } from './tools/create_event';
import { findFreeSlots } from './tools/find_free_slots';
import { createTimeBlock } from './tools/create_time_block';

// ─── Sandbox Initialization ─────────────────────────────────────────────────

const allowedPaths = process.env.LOCALCOWORK_ALLOWED_PATHS
  ? process.env.LOCALCOWORK_ALLOWED_PATHS.split(path.delimiter)
  : [os.homedir()];

initSandbox(allowedPaths);

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'calendar',
  version: '1.0.0',
  tools: [listEvents, createEvent, findFreeSlots, createTimeBlock],
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
