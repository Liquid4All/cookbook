/**
 * Email Database — SQLite schema and access.
 *
 * Stores email drafts and a local email archive index.
 * Uses better-sqlite3 with WAL mode for concurrent reads.
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import * as os from 'os';

// ─── Types ──────────────────────────────────────────────────────────────────

/** A draft email stored in the drafts table. */
export interface EmailDraft {
  id: string;
  to_addresses: string;
  cc_addresses: string | null;
  subject: string;
  body: string;
  status: string;
  sent_at: string | null;
  message_id: string | null;
  created_at: string;
}

/** An email record from the local archive index. */
export interface EmailRecord {
  id: string;
  thread_id: string | null;
  folder: string;
  from_address: string;
  to_addresses: string;
  subject: string;
  body: string;
  received_at: string;
  is_read: number;
}

// ─── Schema SQL ─────────────────────────────────────────────────────────────

const SCHEMA_SQL = `
  CREATE TABLE IF NOT EXISTS drafts (
    id            TEXT PRIMARY KEY,
    to_addresses  TEXT NOT NULL,
    cc_addresses  TEXT,
    subject       TEXT NOT NULL,
    body          TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'draft',
    sent_at       TEXT,
    message_id    TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS emails (
    id            TEXT PRIMARY KEY,
    thread_id     TEXT,
    folder        TEXT DEFAULT 'inbox',
    from_address  TEXT NOT NULL,
    to_addresses  TEXT NOT NULL,
    subject       TEXT NOT NULL,
    body          TEXT NOT NULL,
    received_at   TEXT NOT NULL,
    is_read       INTEGER DEFAULT 0
  );

  CREATE INDEX IF NOT EXISTS idx_emails_thread_id
    ON emails(thread_id);
  CREATE INDEX IF NOT EXISTS idx_emails_folder
    ON emails(folder);
  CREATE INDEX IF NOT EXISTS idx_emails_subject
    ON emails(subject);
  CREATE INDEX IF NOT EXISTS idx_drafts_status
    ON drafts(status);
`;

// ─── Database Singleton ─────────────────────────────────────────────────────

let db: Database.Database | null = null;

const DATA_DIR =
  process.env.LOCALCOWORK_DATA_DIR ?? path.join(os.homedir(), '.localcowork');

const DB_PATH =
  process.env.LOCALCOWORK_EMAIL_DB ?? path.join(DATA_DIR, 'email.db');

/** Get or create the email database connection. */
export function getDb(): Database.Database {
  if (db) return db;

  db = new Database(DB_PATH);

  // Enable WAL mode for concurrent reads
  db.pragma('journal_mode = WAL');

  // Create tables if they don't exist
  db.exec(SCHEMA_SQL);

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
  customDb.exec(SCHEMA_SQL);
  db = customDb;
}
