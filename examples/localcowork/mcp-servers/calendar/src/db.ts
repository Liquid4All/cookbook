/**
 * Calendar Database -- SQLite schema and access.
 *
 * Stores calendar events with start/end times, calendar names, and descriptions.
 * Uses better-sqlite3 with WAL mode for concurrent reads.
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import * as os from 'os';

// ─── Types ──────────────────────────────────────────────────────────────────

export interface CalendarEvent {
  id: number;
  title: string;
  description: string | null;
  calendar_name: string;
  start_time: string;
  end_time: string;
  all_day: number;
  created_at: string;
}

export interface TimeSlot {
  start: string;
  end: string;
  duration_minutes: number;
}

// ─── Schema SQL ─────────────────────────────────────────────────────────────

const SCHEMA_SQL = `
  CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    description   TEXT,
    calendar_name TEXT NOT NULL DEFAULT 'default',
    start_time    TEXT NOT NULL,
    end_time      TEXT NOT NULL,
    all_day       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
  );

  CREATE INDEX IF NOT EXISTS idx_events_start_time
    ON events(start_time);
  CREATE INDEX IF NOT EXISTS idx_events_calendar_name
    ON events(calendar_name);
`;

// ─── Database Singleton ─────────────────────────────────────────────────────

let db: Database.Database | null = null;

const DATA_DIR =
  process.env.LOCALCOWORK_DATA_DIR ?? path.join(os.homedir(), '.localcowork');

const DB_PATH =
  process.env.LOCALCOWORK_CALENDAR_DB ?? path.join(DATA_DIR, 'calendar.db');

/** Get or create the calendar database connection. */
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
