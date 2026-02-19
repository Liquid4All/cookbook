/**
 * Test helpers for the task MCP server.
 *
 * Provides in-memory SQLite database setup with seed data.
 */

import Database from 'better-sqlite3';
import { setDb } from '../src/db';
import { initSandbox } from '../../_shared/ts/validation';
import * as os from 'os';

/** Seed options for task tests. */
interface SeedOptions {
  readonly taskCount?: number;
  readonly withOverdue?: boolean;
  readonly withCompleted?: boolean;
}

/** Create an in-memory task database with optional seed data. */
export function setupTestDb(opts?: SeedOptions): Database.Database {
  const db = new Database(':memory:');
  setDb(db);

  // Allow temp paths for sandbox
  initSandbox([os.tmpdir(), '/private/var/folders', '/private/tmp', '/tmp']);

  const count = opts?.taskCount ?? 5;
  const withOverdue = opts?.withOverdue ?? false;
  const withCompleted = opts?.withCompleted ?? false;

  const insert = db.prepare(`
    INSERT INTO tasks (title, description, source, priority, due_date, completed_at, created_at)
    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
  `);

  // Insert standard pending tasks
  for (let i = 0; i < count; i++) {
    const priority = (i % 5) + 1;
    const dueDate = `2026-12-${String(10 + i).padStart(2, '0')}`;
    insert.run(
      `Task ${i + 1}`,
      `Description for task ${i + 1}`,
      i % 2 === 0 ? 'manual' : 'email',
      priority,
      dueDate,
      null,
    );
  }

  // Insert overdue tasks (due date in the past)
  if (withOverdue) {
    insert.run('Overdue task A', 'Past deadline', 'manual', 1, '2025-01-15', null);
    insert.run('Overdue task B', 'Way past deadline', 'email', 2, '2025-06-01', null);
  }

  // Insert completed tasks
  if (withCompleted) {
    insert.run(
      'Completed task X',
      'Already done',
      'manual',
      3,
      '2026-12-01',
      '2026-11-30T10:00:00Z',
    );
    insert.run(
      'Completed task Y',
      'Also done',
      'meeting',
      2,
      '2026-11-15',
      '2026-11-14T15:30:00Z',
    );
  }

  return db;
}

/** Close and clean up test database. */
export function teardownTestDb(db: Database.Database): void {
  db.close();
}
