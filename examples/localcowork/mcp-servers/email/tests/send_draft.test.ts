import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type Database from 'better-sqlite3';
import { sendDraft } from '../src/tools/send_draft';
import { draftEmail } from '../src/tools/draft_email';
import { setupTestDb, teardownTestDb } from './helpers';
import type { EmailDraft } from '../src/db';

describe('email.send_draft', () => {
  let db: Database.Database;

  beforeEach(() => {
    db = setupTestDb({
      drafts: { count: 2, withSent: true },
    });
  });

  afterEach(() => {
    teardownTestDb(db);
  });

  it('should send a draft successfully', async () => {
    // Create a fresh draft to send
    const createResult = await draftEmail.execute({
      to: ['recipient@example.com'],
      subject: 'Test send',
      body: 'This email will be sent.',
    });
    const draftId = createResult.data.draft_id as string;

    const result = await sendDraft.execute({ draft_id: draftId });
    expect(result.success).toBe(true);
    expect(result.data.success).toBe(true);
    expect(result.data.message_id).toBeDefined();
    expect(typeof result.data.message_id).toBe('string');
    expect(result.data.message_id).toContain('@localcowork.local');
  });

  it('should update draft status to sent', async () => {
    const createResult = await draftEmail.execute({
      to: ['test@example.com'],
      subject: 'Status check',
      body: 'Checking status update.',
    });
    const draftId = createResult.data.draft_id as string;

    await sendDraft.execute({ draft_id: draftId });

    const row = db
      .prepare('SELECT * FROM drafts WHERE id = ?')
      .get(draftId) as EmailDraft;
    expect(row.status).toBe('sent');
    expect(row.sent_at).not.toBeNull();
    expect(row.message_id).not.toBeNull();
  });

  it('should throw for non-existent draft', async () => {
    await expect(
      sendDraft.execute({ draft_id: 'nonexistent-id-12345' }),
    ).rejects.toThrow('Draft not found');
  });

  it('should throw when trying to send an already-sent draft', async () => {
    // Create and send a draft
    const createResult = await draftEmail.execute({
      to: ['once@example.com'],
      subject: 'Send once',
      body: 'Can only send once.',
    });
    const draftId = createResult.data.draft_id as string;

    await sendDraft.execute({ draft_id: draftId });

    // Try to send again
    await expect(
      sendDraft.execute({ draft_id: draftId }),
    ).rejects.toThrow('has already been sent');
  });

  it('should generate a unique message_id each time', async () => {
    const create1 = await draftEmail.execute({
      to: ['a@example.com'],
      subject: 'Unique 1',
      body: 'Body 1',
    });
    const create2 = await draftEmail.execute({
      to: ['b@example.com'],
      subject: 'Unique 2',
      body: 'Body 2',
    });

    const send1 = await sendDraft.execute({
      draft_id: create1.data.draft_id as string,
    });
    const send2 = await sendDraft.execute({
      draft_id: create2.data.draft_id as string,
    });

    expect(send1.data.message_id).not.toBe(send2.data.message_id);
  });

  it('has correct metadata', () => {
    expect(sendDraft.name).toBe('email.send_draft');
    expect(sendDraft.confirmationRequired).toBe(true);
    expect(sendDraft.undoSupported).toBe(false);
  });
});
