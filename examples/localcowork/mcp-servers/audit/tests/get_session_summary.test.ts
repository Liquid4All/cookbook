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
    expect(result.data).toHaveProperty('tools_called');
    expect(result.data).toHaveProperty('succeeded');
    expect(result.data).toHaveProperty('failed');
    expect(result.data).toHaveProperty('user_confirmed');
    expect(result.data).toHaveProperty('total_execution_ms');
  });

  it('should count succeeded correctly', async () => {
    const result = await getSessionSummary.execute({
      session_id: 'sess-summary-001',
    });
    // 5 seed entries are 'Success'
    expect(result.data.succeeded).toBe(5);
  });

  it('should count failed correctly', async () => {
    const result = await getSessionSummary.execute({
      session_id: 'sess-summary-001',
    });
    // 1 Error entry in seed data
    expect(result.data.failed).toBe(1);
  });

  it('should count user_confirmed correctly', async () => {
    const result = await getSessionSummary.execute({
      session_id: 'sess-summary-001',
    });
    // write_file entries have user_confirmed = 1
    expect(result.data.user_confirmed).toBeGreaterThanOrEqual(1);
  });

  it('has correct metadata', () => {
    expect(getSessionSummary.name).toBe('audit.get_session_summary');
    expect(getSessionSummary.confirmationRequired).toBe(false);
    expect(getSessionSummary.undoSupported).toBe(false);
  });
});
