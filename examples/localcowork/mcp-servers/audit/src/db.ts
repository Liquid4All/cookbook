/**
 * Audit Database — SQLite schema and access.
 *
 * All audit tools read from this database. The Agent Core writes to it
 * whenever a tool is invoked.
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import * as os from 'os';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface AuditEntry {
  id: number;
  session_id: string;
  timestamp: string;
  tool_name: string;
  status: string | null;
  file_path: string | null;
  file_hash: string | null;
  params_json: string | null;
  result_json: string | null;
  error_message: string | null;
  duration_ms: number | null;
}

// ─── Database Singleton ─────────────────────────────────────────────────────

let db: Database.Database | null = null;

const DATA_DIR =
  process.env.LOCALCOWORK_DATA_DIR ?? path.join(os.homedir(), '.localcowork');

const DB_PATH =
  process.env.LOCALCOWORK_AUDIT_DB ?? path.join(DATA_DIR, 'audit.db');

/** Get or create the audit database connection. */
export function getDb(): Database.Database {
  if (db) return db;

  db = new Database(DB_PATH);

  // Enable WAL mode for concurrent reads
  db.pragma('journal_mode = WAL');

  // Create tables if they don't exist
  db.exec(`
    CREATE TABLE IF NOT EXISTS audit_log (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id   TEXT NOT NULL,
      timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
      tool_name    TEXT NOT NULL,
      status       TEXT,
      file_path    TEXT,
      file_hash    TEXT,
      params_json  TEXT,
      result_json  TEXT,
      error_message TEXT,
      duration_ms  INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_audit_session
      ON audit_log(session_id);
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp
      ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_tool
      ON audit_log(tool_name);
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
    CREATE TABLE IF NOT EXISTS audit_log (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id   TEXT NOT NULL,
      timestamp    TEXT NOT NULL DEFAULT (datetime('now')),
      tool_name    TEXT NOT NULL,
      status       TEXT,
      file_path    TEXT,
      file_hash    TEXT,
      params_json  TEXT,
      result_json  TEXT,
      error_message TEXT,
      duration_ms  INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_audit_session
      ON audit_log(session_id);
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp
      ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_tool
      ON audit_log(tool_name);
  `);
  db = customDb;
}
