import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { updateTask } from '../src/tools/update_task';
import { setupTestDb, teardownTestDb } from './helpers';
import type { Task } from '../src/db';

describe('task.update_task', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ taskCount: 3 });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should update title', async () => {
    const result = await updateTask.execute({
      task_id: 1,
      updates: { title: 'Updated title' },
    });
    expect(result.success).toBe(true);

    const row = db.prepare('SELECT * FROM tasks WHERE id = 1').get() as Task;
    expect(row.title).toBe('Updated title');
  });

  it('should update multiple fields', async () => {
    const result = await updateTask.execute({
      task_id: 2,
      updates: {
        title: 'New title',
        description: 'New description',
        priority: 1,
        due_date: '2026-12-25',
      },
    });
    expect(result.success).toBe(true);

    const row = db.prepare('SELECT * FROM tasks WHERE id = 2').get() as Task;
    expect(row.title).toBe('New title');
    expect(row.description).toBe('New description');
    expect(row.priority).toBe(1);
    expect(row.due_date).toBe('2026-12-25');
  });

  it('should mark a task as completed', async () => {
    const result = await updateTask.execute({
      task_id: 3,
      updates: { completed_at: '2026-12-10T10:00:00Z' },
    });
    expect(result.success).toBe(true);

    const row = db.prepare('SELECT * FROM tasks WHERE id = 3').get() as Task;
    expect(row.completed_at).toBe('2026-12-10T10:00:00Z');
  });

  it('should return success with message when no updates provided', async () => {
    const result = await updateTask.execute({
      task_id: 1,
      updates: {},
    });
    expect(result.success).toBe(true);
    expect(result.data.message).toBe('No updates provided');
  });

  it('should throw for non-existent task', async () => {
    await expect(
      updateTask.execute({ task_id: 9999, updates: { title: 'Ghost' } }),
    ).rejects.toThrow(/not found/i);
  });

  it('has correct metadata', () => {
    expect(updateTask.name).toBe('task.update_task');
    expect(updateTask.confirmationRequired).toBe(true);
    expect(updateTask.undoSupported).toBe(false);
  });
});
