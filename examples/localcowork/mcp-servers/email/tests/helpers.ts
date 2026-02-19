/**
 * Test helpers for the email MCP server.
 *
 * Provides in-memory SQLite database setup with seed data
 * for drafts and email archive entries.
 */

import Database from 'better-sqlite3';
import crypto from 'crypto';
import { setDb } from '../src/db';
import { initSandbox } from '../../_shared/ts/validation';
import * as os from 'os';

// ─── Seed Options ───────────────────────────────────────────────────────────

/** Options for seeding draft data. */
interface DraftSeedOptions {
  readonly count?: number;
  readonly withSent?: boolean;
}

/** Options for seeding email archive data. */
interface EmailSeedOptions {
  readonly count?: number;
  readonly withThreads?: boolean;
  readonly folders?: string[];
}

/** Combined seed options. */
export interface SeedOptions {
  readonly drafts?: DraftSeedOptions;
  readonly emails?: EmailSeedOptions;
}

// ─── Setup / Teardown ───────────────────────────────────────────────────────

/** Create an in-memory email database with optional seed data. */
export function setupTestDb(opts?: SeedOptions): Database.Database {
  const db = new Database(':memory:');
  setDb(db);

  // Allow temp paths for sandbox
  initSandbox([os.tmpdir(), '/private/var/folders', '/private/tmp', '/tmp']);

  seedDrafts(db, opts?.drafts);
  seedEmails(db, opts?.emails);

  return db;
}

/** Close and clean up test database. */
export function teardownTestDb(db: Database.Database): void {
  db.close();
}

// ─── Draft Seeding ──────────────────────────────────────────────────────────

function seedDrafts(db: Database.Database, opts?: DraftSeedOptions): void {
  const count = opts?.count ?? 0;
  const withSent = opts?.withSent ?? false;

  const insert = db.prepare(`
    INSERT INTO drafts (id, to_addresses, cc_addresses, subject, body, status, sent_at, message_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-' || ? || ' minutes'))
  `);

  for (let i = 0; i < count; i++) {
    const id = crypto.randomUUID();
    insert.run(
      id,
      JSON.stringify([`user${i}@example.com`]),
      i % 2 === 0 ? JSON.stringify([`cc${i}@example.com`]) : null,
      `Draft subject ${i + 1}`,
      `This is the body of draft ${i + 1}. It contains some content.`,
      'draft',
      null,
      null,
      count - i, // Stagger created_at so newest is first
    );
  }

  if (withSent) {
    const sentId = crypto.randomUUID();
    insert.run(
      sentId,
      JSON.stringify(['sent-recipient@example.com']),
      null,
      'Already sent email',
      'This draft has been sent.',
      'sent',
      '2026-01-15T10:00:00Z',
      `<${crypto.randomUUID()}@localcowork.local>`,
      100, // Created 100 minutes ago
    );
  }
}

// ─── Email Seeding ──────────────────────────────────────────────────────────

function seedEmails(db: Database.Database, opts?: EmailSeedOptions): void {
  const count = opts?.count ?? 0;
  const withThreads = opts?.withThreads ?? false;
  const folders = opts?.folders ?? ['inbox'];

  const insert = db.prepare(`
    INSERT INTO emails (id, thread_id, folder, from_address, to_addresses, subject, body, received_at, is_read)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);

  for (let i = 0; i < count; i++) {
    const folder = folders[i % folders.length];
    const receivedAt = `2026-01-${String(10 + i).padStart(2, '0')}T09:00:00Z`;
    insert.run(
      crypto.randomUUID(),
      null, // No thread by default
      folder,
      `sender${i}@example.com`,
      JSON.stringify([`recipient${i}@example.com`]),
      `Email subject ${i + 1}`,
      `Body of email ${i + 1}. Contains various content for searching.`,
      receivedAt,
      i % 3 === 0 ? 1 : 0,
    );
  }

  if (withThreads) {
    seedThread(db, insert);
  }
}

/** Seed a multi-message thread for thread summarization tests. */
function seedThread(
  db: Database.Database,
  insert: Database.Statement,
): void {
  const threadId = 'thread-abc-123';

  insert.run(
    crypto.randomUUID(),
    threadId,
    'inbox',
    'alice@example.com',
    JSON.stringify(['bob@example.com']),
    'Project kickoff meeting',
    'Hi Bob, let us schedule the kickoff meeting for next week.',
    '2026-02-01T09:00:00Z',
    1,
  );

  insert.run(
    crypto.randomUUID(),
    threadId,
    'inbox',
    'bob@example.com',
    JSON.stringify(['alice@example.com']),
    'Re: Project kickoff meeting',
    'Sure, how about Tuesday at 2pm?',
    '2026-02-01T10:30:00Z',
    1,
  );

  insert.run(
    crypto.randomUUID(),
    threadId,
    'inbox',
    'alice@example.com',
    JSON.stringify(['bob@example.com']),
    'Re: Project kickoff meeting',
    'Tuesday at 2pm works. I will send the calendar invite.',
    '2026-02-01T11:00:00Z',
    0,
  );
}
