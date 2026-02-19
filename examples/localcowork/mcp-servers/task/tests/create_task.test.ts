import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { createTask } from '../src/tools/create_task';
import { setupTestDb, teardownTestDb } from './helpers';
import type { Task } from '../src/db';

describe('task.create_task', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ taskCount: 0 });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should create a task with required fields only', async () => {
    const result = await createTask.execute({ title: 'Buy groceries' });
    expect(result.success).toBe(true);
    expect(result.data.task_id).toBeDefined();

    const row = db.prepare('SELECT * FROM tasks WHERE id = ?').get(result.data.task_id) as Task;
    expect(row.title).toBe('Buy groceries');
    expect(row.priority).toBe(3); // default
    expect(row.description).toBeNull();
    expect(row.source).toBeNull();
    expect(row.due_date).toBeNull();
  });

  it('should create a task with all fields', async () => {
    const result = await createTask.execute({
      title: 'Review contract',
      description: 'NDA for vendor partnership',
      source: 'email',
      priority: 1,
      due_date: '2026-12-31',
    });
    expect(result.success).toBe(true);

    const row = db.prepare('SELECT * FROM tasks WHERE id = ?').get(result.data.task_id) as Task;
    expect(row.title).toBe('Review contract');
    expect(row.description).toBe('NDA for vendor partnership');
    expect(row.source).toBe('email');
    expect(row.priority).toBe(1);
    expect(row.due_date).toBe('2026-12-31');
    expect(row.completed_at).toBeNull();
  });

  it('should auto-assign priority 3 when not provided', async () => {
    const result = await createTask.execute({ title: 'Default priority' });
    expect(result.success).toBe(true);

    const row = db.prepare('SELECT * FROM tasks WHERE id = ?').get(result.data.task_id) as Task;
    expect(row.priority).toBe(3);
  });

  it('has correct metadata', () => {
    expect(createTask.name).toBe('task.create_task');
    expect(createTask.confirmationRequired).toBe(true);
    expect(createTask.undoSupported).toBe(false);
  });
});
