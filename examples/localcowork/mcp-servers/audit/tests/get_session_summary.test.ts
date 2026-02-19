import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { getSessionSummary } from '../src/tools/get_session_summary';
import { setupTestDb, teardownTestDb } from './helpers';

describe('audit.get_session_summary', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ sessionId: 'sess-summary-001', entryCount: 5 });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should return session summary with all fields', async () => {
    const result = await getSessionSummary.execute({
      session_id: 'sess-summary-001',
    });
    expect(result.success).toBe(true);
    expect(result.data).toHaveProperty('documents_touched');
    expect(result.data).toHaveProperty('tools_called');
    expect(result.data).toHaveProperty('confirmations');
    expect(result.data).toHaveProperty('rejections');
  });

  it('should count confirmations correctly', async () => {
    const result = await getSessionSummary.execute({
      session_id: 'sess-summary-001',
    });
    // write_file entries are 'confirmed'
    expect(result.data.confirmations).toBeGreaterThanOrEqual(1);
  });

  it('should count rejections correctly', async () => {
    const result = await getSessionSummary.execute({
      session_id: 'sess-summary-001',
    });
    // 1 rejected entry in seed data
    expect(result.data.rejections).toBe(1);
  });

  it('should list unique documents touched', async () => {
    const result = await getSessionSummary.execute({
      session_id: 'sess-summary-001',
    });
    expect(Array.isArray(result.data.documents_touched)).toBe(true);
    expect(result.data.documents_touched.length).toBeGreaterThan(0);
  });

  it('has correct metadata', () => {
    expect(getSessionSummary.name).toBe('audit.get_session_summary');
    expect(getSessionSummary.confirmationRequired).toBe(false);
    expect(getSessionSummary.undoSupported).toBe(false);
  });
});
