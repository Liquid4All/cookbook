import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { listDrafts } from '../src/tools/list_drafts';
import { setupTestDb, teardownTestDb } from './helpers';

describe('email.list_drafts', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({
      drafts: { count: 5, withSent: true },
    });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should return all drafts (not sent)', async () => {
    const result = await listDrafts.execute({});
    expect(result.success).toBe(true);
    // 5 drafts, 1 sent — only drafts returned
    expect(result.data.length).toBe(5);
    for (const draft of result.data) {
      expect(draft.id).toBeDefined();
      expect(draft.subject).toBeDefined();
      expect(draft.to).toBeInstanceOf(Array);
    }
  });

  it('should not include sent drafts', async () => {
    const result = await listDrafts.execute({ limit: 100 });
    expect(result.success).toBe(true);
    for (const draft of result.data) {
      expect(draft.subject).not.toBe('Already sent email');
    }
  });

  it('should respect the limit parameter', async () => {
    const result = await listDrafts.execute({ limit: 2 });
    expect(result.success).toBe(true);
    expect(result.data.length).toBe(2);
  });

  it('should order by created_at DESC (newest first)', async () => {
    const result = await listDrafts.execute({});
    expect(result.success).toBe(true);
    // Drafts are seeded with staggered created_at — newest has the smallest offset
    // so "Draft subject 5" was created most recently
    expect(result.data[0].subject).toBe('Draft subject 5');
  });

  it('should parse to and cc addresses from JSON', async () => {
    const result = await listDrafts.execute({});
    expect(result.success).toBe(true);
    for (const draft of result.data) {
      expect(Array.isArray(draft.to)).toBe(true);
      expect(draft.to.length).toBeGreaterThan(0);
      expect(Array.isArray(draft.cc)).toBe(true);
    }
  });

  it('should default limit to 20', async () => {
    // We only have 5 drafts, so this just confirms no error with default
    const result = await listDrafts.execute({});
    expect(result.success).toBe(true);
    expect(result.data.length).toBeLessThanOrEqual(20);
  });

  it('should return empty array when no drafts exist', async () => {
    const emptyDb = setupTestDb();
    const result = await listDrafts.execute({});
    expect(result.success).toBe(true);
    expect(result.data).toEqual([]);
    teardownTestDb(emptyDb);

    // Restore original db for remaining tests
    setupTestDb({ drafts: { count: 5, withSent: true } });
  });

  it('has correct metadata', () => {
    expect(listDrafts.name).toBe('email.list_drafts');
    expect(listDrafts.confirmationRequired).toBe(false);
    expect(listDrafts.undoSupported).toBe(false);
  });
});
