/**
 * Task MCP Server — Entry Point
 *
 * Registers all task tools and starts the JSON-RPC listener.
 * This server provides local SQLite-backed task management
 * with CRUD operations and daily briefings.
 *
 * Tools (5):
 *   task.create_task     — create a new task (confirm)
 *   task.list_tasks      — list tasks with filters
 *   task.update_task     — update an existing task (confirm)
 *   task.get_overdue     — get overdue tasks
 *   task.daily_briefing  — generate daily briefing
 */

import os from 'os';
import path from 'path';

import { MCPServer } from '../../_shared/ts/mcp-base';
import { initSandbox } from '../../_shared/ts/validation';
import { closeDb } from './db';
import { createTask } from './tools/create_task';
import { listTasks } from './tools/list_tasks';
import { updateTask } from './tools/update_task';
import { getOverdue } from './tools/get_overdue';
import { dailyBriefing } from './tools/daily_briefing';

// ─── Sandbox Initialization ─────────────────────────────────────────────────

const allowedPaths = process.env.LOCALCOWORK_ALLOWED_PATHS
  ? process.env.LOCALCOWORK_ALLOWED_PATHS.split(path.delimiter)
  : [os.homedir()];

initSandbox(allowedPaths);

// ─── Server Setup ───────────────────────────────────────────────────────────

const server = new MCPServer({
  name: 'task',
  version: '1.0.0',
  tools: [createTask, listTasks, updateTask, getOverdue, dailyBriefing],
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
