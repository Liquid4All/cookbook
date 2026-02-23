import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { getToolLog } from '../src/tools/get_tool_log';
import { setupTestDb, teardownTestDb } from './helpers';

describe('audit.get_tool_log', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ sessionId: 'sess-log-001', entryCount: 5 });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should return all entries when no filter', async () => {
    const result = await getToolLog.execute({});
    expect(result.success).toBe(true);
    // 5 seed entries + 1 Error = 6
    expect(result.data.length).toBeGreaterThanOrEqual(5);
  });

  it('should filter by session_id', async () => {
    const result = await getToolLog.execute({ session_id: 'sess-log-001' });
    expect(result.success).toBe(true);
    expect(result.data.length).toBe(6); // 5 Success + 1 Error
  });

  it('should filter by tool_name', async () => {
    const result = await getToolLog.execute({ tool_name: 'filesystem.read_file' });
    expect(result.success).toBe(true);
    for (const entry of result.data) {
      expect(entry.tool_name).toBe('filesystem.read_file');
    }
  });

  it('should filter by time range', async () => {
    const result = await getToolLog.execute({
      start_time: '2026-02-12T11:00:00Z',
      end_time: '2026-02-12T13:00:00Z',
    });
    expect(result.success).toBe(true);
    expect(result.data.length).toBeGreaterThan(0);
  });

  it('has correct metadata', () => {
    expect(getToolLog.name).toBe('audit.get_tool_log');
    expect(getToolLog.confirmationRequired).toBe(false);
    expect(getToolLog.undoSupported).toBe(false);
  });
});
