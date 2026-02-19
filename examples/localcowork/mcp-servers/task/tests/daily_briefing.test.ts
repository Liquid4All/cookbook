import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { dailyBriefing } from '../src/tools/daily_briefing';
import { setupTestDb, teardownTestDb } from './helpers';

describe('task.daily_briefing', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ taskCount: 5, withOverdue: true, withCompleted: true });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should generate a briefing for today', async () => {
    const result = await dailyBriefing.execute({});
    expect(result.success).toBe(true);
    expect(result.data.briefing).toContain('Daily Briefing');
    expect(result.data.tasks).toBeDefined();
    expect(result.data.overdue).toBeDefined();
    expect(result.data.due_today).toBeDefined();
  });

  it('should generate a briefing for a specific date', async () => {
    const result = await dailyBriefing.execute({ date: '2026-12-10' });
    expect(result.success).toBe(true);
    expect(result.data.briefing).toContain('2026-12-10');
  });

  it('should include overdue tasks in briefing', async () => {
    const result = await dailyBriefing.execute({});
    expect(result.success).toBe(true);
    expect(result.data.overdue.length).toBeGreaterThanOrEqual(2);
    expect(result.data.briefing).toContain('OVERDUE');
  });

  it('should include pending tasks in briefing', async () => {
    const result = await dailyBriefing.execute({});
    expect(result.success).toBe(true);
    // pending = not completed
    expect(result.data.tasks.length).toBeGreaterThan(0);
    expect(result.data.briefing).toContain('PENDING');
  });

  it('should show due-today tasks when date matches', async () => {
    // Seed has tasks due on 2026-12-10 through 2026-12-14
    const result = await dailyBriefing.execute({ date: '2026-12-10' });
    expect(result.success).toBe(true);
    expect(result.data.due_today.length).toBeGreaterThanOrEqual(1);
    expect(result.data.briefing).toContain('DUE TODAY');
  });

  it('has correct metadata', () => {
    expect(dailyBriefing.name).toBe('task.daily_briefing');
    expect(dailyBriefing.confirmationRequired).toBe(false);
    expect(dailyBriefing.undoSupported).toBe(false);
  });
});
