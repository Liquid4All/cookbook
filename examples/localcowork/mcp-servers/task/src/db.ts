/**
 * Task Database — SQLite schema and access.
 *
 * Stores tasks with priority, due dates, and completion status.
 * Uses better-sqlite3 with WAL mode for concurrent reads.
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import * as os from 'os';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface Task {
  id: number;
  title: string;
  description: string | null;
  source: string | null;
  source_ref: string | null;
  priority: number;
  due_date: string | null;
  completed_at: string | null;
  created_at: string;
}

// ─── Database Singleton ─────────────────────────────────────────────────────

let db: Database.Database | null = null;

const DATA_DIR =
  process.env.LOCALCOWORK_DATA_DIR ?? path.join(os.homedir(), '.localcowork');

const DB_PATH =
  process.env.LOCALCOWORK_TASK_DB ?? path.join(DATA_DIR, 'tasks.db');

/** Get or create the task database connection. */
export function getDb(): Database.Database {
  if (db) return db;

  db = new Database(DB_PATH);

  // Enable WAL mode for concurrent reads
  db.pragma('journal_mode = WAL');

  // Create tables if they don't exist
  db.exec(`
    CREATE TABLE IF NOT EXISTS tasks (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      title        TEXT NOT NULL,
      description  TEXT,
      source       TEXT,
      source_ref   TEXT,
      priority     INTEGER NOT NULL DEFAULT 3,
      due_date     TEXT,
      completed_at TEXT,
      created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_tasks_priority
      ON tasks(priority);
    CREATE INDEX IF NOT EXISTS idx_tasks_due_date
      ON tasks(due_date);
    CREATE INDEX IF NOT EXISTS idx_tasks_completed
      ON tasks(completed_at);
  `);

  return db;
}

/** Close the database connection. */
export function closeDb(): void {
  if (db) {
    db.close();
    db = null;
  }
}

/**
 * Set a custom database instance (for testing).
 * Runs the schema migration on the provided database.
 */
export function setDb(customDb: Database.Database): void {
  customDb.exec(`
    CREATE TABLE IF NOT EXISTS tasks (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      title        TEXT NOT NULL,
      description  TEXT,
      source       TEXT,
      source_ref   TEXT,
      priority     INTEGER NOT NULL DEFAULT 3,
      due_date     TEXT,
      completed_at TEXT,
      created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_tasks_priority
      ON tasks(priority);
    CREATE INDEX IF NOT EXISTS idx_tasks_due_date
      ON tasks(due_date);
    CREATE INDEX IF NOT EXISTS idx_tasks_completed
      ON tasks(completed_at);
  `);
  db = customDb;
}
