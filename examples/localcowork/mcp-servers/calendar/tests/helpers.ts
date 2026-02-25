/**
 * Test helpers for the calendar MCP server.
 *
 * Provides in-memory SQLite database setup with seed events.
 */

import Database from 'better-sqlite3';
import { setDb } from '../src/db';
import { initSandbox } from '../../_shared/ts/validation';
import * as os from 'os';

/** Seed options for calendar tests. */
interface SeedOptions {
  /** Number of events to seed (default: 0 -- no auto-seeded events). */
  readonly eventCount?: number;
  /** If true, seed events spread across a specific test date. */
  readonly withDayEvents?: boolean;
  /** The date to seed day events on (default: 2026-03-15). */
  readonly dayEventsDate?: string;
  /** If true, seed events on multiple calendars. */
  readonly withMultipleCalendars?: boolean;
}

/** Create an in-memory calendar database with optional seed data. */
export function setupTestDb(opts?: SeedOptions): Database.Database {
  const db = new Database(':memory:');
  setDb(db);

  // Allow temp paths for sandbox
  initSandbox([os.tmpdir(), '/private/var/folders', '/private/tmp', '/tmp']);

  const withDayEvents = opts?.withDayEvents ?? false;
  const dayDate = opts?.dayEventsDate ?? '2026-03-15';
  const withMultipleCalendars = opts?.withMultipleCalendars ?? false;
  const eventCount = opts?.eventCount ?? 0;

  const insert = db.prepare(`
    INSERT INTO events (title, description, calendar_name, start_time, end_time, all_day, created_at)
    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
  `);

  // Insert generic events spread across a month
  for (let i = 0; i < eventCount; i++) {
    const day = String(1 + (i % 28)).padStart(2, '0');
    const hour = String(9 + (i % 8)).padStart(2, '0');
    insert.run(
      `Event ${i + 1}`,
      `Description for event ${i + 1}`,
      withMultipleCalendars && i % 2 === 1 ? 'work' : 'default',
      `2026-03-${day}T${hour}:00:00`,
      `2026-03-${day}T${hour}:30:00`,
      0,
    );
  }

  // Insert events on a specific day for free-slot testing
  if (withDayEvents) {
    // Morning standup: 09:00-09:30
    insert.run(
      'Morning standup',
      'Daily team sync',
      'default',
      `${dayDate}T09:00:00`,
      `${dayDate}T09:30:00`,
      0,
    );

    // Design review: 10:00-11:00
    insert.run(
      'Design review',
      'Review new mockups',
      'default',
      `${dayDate}T10:00:00`,
      `${dayDate}T11:00:00`,
      0,
    );

    // Lunch meeting: 12:00-13:00
    insert.run(
      'Lunch meeting',
      'Client lunch',
      'default',
      `${dayDate}T12:00:00`,
      `${dayDate}T13:00:00`,
      0,
    );

    // Afternoon workshop: 14:00-16:00
    insert.run(
      'Afternoon workshop',
      'Architecture deep dive',
      'default',
      `${dayDate}T14:00:00`,
      `${dayDate}T16:00:00`,
      0,
    );
  }

  return db;
}

/** Close and clean up test database. */
export function teardownTestDb(db: Database.Database): void {
  db.close();
}
