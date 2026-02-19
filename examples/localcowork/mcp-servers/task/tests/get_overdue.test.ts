import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { getOverdue } from '../src/tools/get_overdue';
import { setupTestDb, teardownTestDb } from './helpers';

describe('task.get_overdue', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ taskCount: 3, withOverdue: true, withCompleted: false });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should return overdue tasks', async () => {
    const result = await getOverdue.execute({});
    expect(result.success).toBe(true);
    // 2 overdue tasks seeded (due 2025-01-15 and 2025-06-01)
    expect(result.data.length).toBeGreaterThanOrEqual(2);
  });

  it('should only return incomplete tasks', async () => {
    const result = await getOverdue.execute({});
    expect(result.success).toBe(true);
    for (const task of result.data) {
      expect(task.completed_at).toBeNull();
    }
  });

  it('should sort by due_date ascending', async () => {
    const result = await getOverdue.execute({});
    expect(result.success).toBe(true);
    for (let i = 1; i < result.data.length; i++) {
      const prev = result.data[i - 1].due_date;
      const curr = result.data[i].due_date;
      if (prev && curr) {
        expect(prev <= curr).toBe(true);
      }
    }
  });

  it('has correct metadata', () => {
    expect(getOverdue.name).toBe('task.get_overdue');
    expect(getOverdue.confirmationRequired).toBe(false);
    expect(getOverdue.undoSupported).toBe(false);
  });
});
