import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { draftEmail } from '../src/tools/draft_email';
import { setupTestDb, teardownTestDb } from './helpers';
import type { EmailDraft } from '../src/db';

describe('email.draft_email', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb();
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should create a draft with required fields only', async () => {
    const result = await draftEmail.execute({
      to: ['alice@example.com'],
      subject: 'Hello',
      body: 'Hi Alice, how are you?',
    });

    expect(result.success).toBe(true);
    expect(result.data.draft_id).toBeDefined();
    expect(typeof result.data.draft_id).toBe('string');
    expect(result.data.preview).toBe('Hi Alice, how are you?');

    // Verify in database
    const row = db
      .prepare('SELECT * FROM drafts WHERE id = ?')
      .get(result.data.draft_id) as EmailDraft;
    expect(row.subject).toBe('Hello');
    expect(row.body).toBe('Hi Alice, how are you?');
    expect(row.status).toBe('draft');
    expect(JSON.parse(row.to_addresses)).toEqual(['alice@example.com']);
    expect(row.cc_addresses).toBeNull();
    expect(row.sent_at).toBeNull();
    expect(row.message_id).toBeNull();
  });

  it('should create a draft with CC addresses', async () => {
    const result = await draftEmail.execute({
      to: ['bob@example.com'],
      subject: 'Meeting notes',
      body: 'Please find the meeting notes attached.',
      cc: ['charlie@example.com', 'dave@example.com'],
    });

    expect(result.success).toBe(true);

    const row = db
      .prepare('SELECT * FROM drafts WHERE id = ?')
      .get(result.data.draft_id) as EmailDraft;
    expect(JSON.parse(row.cc_addresses!)).toEqual([
      'charlie@example.com',
      'dave@example.com',
    ]);
  });

  it('should create a draft with multiple recipients', async () => {
    const result = await draftEmail.execute({
      to: ['a@example.com', 'b@example.com', 'c@example.com'],
      subject: 'Group email',
      body: 'Hello everyone!',
    });

    expect(result.success).toBe(true);

    const row = db
      .prepare('SELECT * FROM drafts WHERE id = ?')
      .get(result.data.draft_id) as EmailDraft;
    expect(JSON.parse(row.to_addresses)).toEqual([
      'a@example.com',
      'b@example.com',
      'c@example.com',
    ]);
  });

  it('should truncate preview to 200 chars for long bodies', async () => {
    const longBody = 'A'.repeat(300);
    const result = await draftEmail.execute({
      to: ['long@example.com'],
      subject: 'Long email',
      body: longBody,
    });

    expect(result.success).toBe(true);
    expect(result.data.preview.length).toBe(203); // 200 + '...'
    expect(result.data.preview).toBe('A'.repeat(200) + '...');
  });

  it('should not truncate preview for short bodies', async () => {
    const shortBody = 'Short body.';
    const result = await draftEmail.execute({
      to: ['short@example.com'],
      subject: 'Short email',
      body: shortBody,
    });

    expect(result.success).toBe(true);
    expect(result.data.preview).toBe('Short body.');
  });

  it('should generate unique draft IDs', async () => {
    const result1 = await draftEmail.execute({
      to: ['x@example.com'],
      subject: 'Draft 1',
      body: 'Body 1',
    });
    const result2 = await draftEmail.execute({
      to: ['y@example.com'],
      subject: 'Draft 2',
      body: 'Body 2',
    });

    expect(result1.data.draft_id).not.toBe(result2.data.draft_id);
  });

  it('has correct metadata', () => {
    expect(draftEmail.name).toBe('email.draft_email');
    expect(draftEmail.confirmationRequired).toBe(true);
    expect(draftEmail.undoSupported).toBe(false);
  });
});
