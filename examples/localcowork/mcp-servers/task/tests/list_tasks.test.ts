import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { listTasks } from '../src/tools/list_tasks';
import { setupTestDb, teardownTestDb } from './helpers';

describe('task.list_tasks', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ taskCount: 5, withOverdue: true, withCompleted: true });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should return all tasks when no filter', async () => {
    const result = await listTasks.execute({});
    expect(result.success).toBe(true);
    // 5 standard + 2 overdue + 2 completed = 9
    expect(result.data.length).toBe(9);
  });

  it('should filter pending tasks', async () => {
    const result = await listTasks.execute({ status: 'pending' });
    expect(result.success).toBe(true);
    // 5 standard + 2 overdue = 7 (completed_at IS NULL)
    expect(result.data.length).toBe(7);
    for (const task of result.data) {
      expect(task.completed_at).toBeNull();
    }
  });

  it('should filter completed tasks', async () => {
    const result = await listTasks.execute({ status: 'completed' });
    expect(result.success).toBe(true);
    expect(result.data.length).toBe(2);
    for (const task of result.data) {
      expect(task.completed_at).not.toBeNull();
    }
  });

  it('should filter overdue tasks', async () => {
    const result = await listTasks.execute({ status: 'overdue' });
    expect(result.success).toBe(true);
    // Tasks with due_date in the past and not completed
    expect(result.data.length).toBeGreaterThanOrEqual(2);
    for (const task of result.data) {
      expect(task.completed_at).toBeNull();
    }
  });

  it('should filter by priority', async () => {
    const result = await listTasks.execute({ priority: 1 });
    expect(result.success).toBe(true);
    for (const task of result.data) {
      expect(task.priority).toBe(1);
    }
  });

  it('should respect limit', async () => {
    const result = await listTasks.execute({ limit: 3 });
    expect(result.success).toBe(true);
    expect(result.data.length).toBe(3);
  });

  it('should sort by priority then due_date', async () => {
    const result = await listTasks.execute({});
    expect(result.success).toBe(true);
    for (let i = 1; i < result.data.length; i++) {
      const prev = result.data[i - 1];
      const curr = result.data[i];
      // Lower priority number = higher urgency = should come first
      expect(prev.priority).toBeLessThanOrEqual(curr.priority);
    }
  });

  it('has correct metadata', () => {
    expect(listTasks.name).toBe('task.list_tasks');
    expect(listTasks.confirmationRequired).toBe(false);
    expect(listTasks.undoSupported).toBe(false);
  });
});
