/**
 * Test helpers for the audit MCP server.
 *
 * Provides in-memory SQLite database setup with seed data.
 * Schema matches the Agent Core's agent.db audit_log table.
 */

import Database from 'better-sqlite3';
import * as os from 'os';
import { setDb } from '../src/db';
import { initSandbox } from '../../_shared/ts/validation';

/** Seed data for audit tests */
interface SeedOptions {
  sessionId?: string;
  entryCount?: number;
}

/** Create an in-memory audit database with optional seed data. */
export function setupTestDb(opts?: SeedOptions): Database.Database {
  const db = new Database(':memory:');
  setDb(db);

  // Allow temp paths for sandbox
  initSandbox([os.tmpdir(), '/private/var/folders', '/private/tmp', '/tmp']);

  const sessionId = opts?.sessionId ?? 'test-session-001';
  const count = opts?.entryCount ?? 5;

  // Seed the sessions table (required by foreign key)
  db.prepare(
    `INSERT INTO sessions (id) VALUES (?)`,
  ).run(sessionId);

  const insert = db.prepare(`
    INSERT INTO audit_log
      (session_id, timestamp, tool_name, arguments, result, result_status, user_confirmed, execution_time_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);

  const tools = [
    'filesystem.read_file',
    'filesystem.write_file',
    'filesystem.list_dir',
    'document.extract_text',
    'ocr.extract_text_from_image',
  ];

  for (let i = 0; i < count; i++) {
    const tool = tools[i % tools.length];
    const userConfirmed = tool.includes('write') ? 1 : 0;
    const hour = String(10 + i).padStart(2, '0');
    const timestamp = `2026-02-12T${hour}:00:00Z`;
    const args = JSON.stringify({ path: `/home/user/docs/file${i}.txt` });
    const result = JSON.stringify({ success: true });

    insert.run(sessionId, timestamp, tool, args, result, 'success', userConfirmed, 100 + i * 50);
  }

  // Add a failed entry
  insert.run(
    sessionId,
    '2026-02-12T15:00:00Z',
    'filesystem.delete_file',
    JSON.stringify({ path: '/home/user/docs/important.txt' }),
    null,
    'error',
    0,
    20,
  );

  return db;
}

/** Close and clean up test database. */
export function teardownTestDb(db: Database.Database): void {
  db.close();
}
