import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { createTimeBlock } from '../src/tools/create_time_block';
import { setupTestDb, teardownTestDb } from './helpers';
import type { CalendarEvent } from '../src/db';
import { MCPError } from '../../_shared/ts/mcp-base';

describe('calendar.create_time_block', () => {
  let db: Database.Database;

  beforeAll(() => {
    // Seed with day events on 2026-03-15:
    //   09:00-09:30  Morning standup
    //   10:00-11:00  Design review
    //   12:00-13:00  Lunch meeting
    //   14:00-16:00  Afternoon workshop
    db = setupTestDb({ withDayEvents: true, dayEventsDate: '2026-03-15' });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should create a time block on a day with free slots', async () => {
    const result = await createTimeBlock.execute({
      title: 'Deep work',
      date: '2026-03-15',
      duration_minutes: 60,
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };
    expect(data.event_id).toBeDefined();
    expect(data.scheduled_at).toBeDefined();

    // Verify the event was created in the database
    const row = db
      .prepare('SELECT * FROM events WHERE id = ?')
      .get(Number(data.event_id)) as CalendarEvent;
    expect(row.title).toBe('Deep work');
  });

  it('should prefer morning slot when requested', async () => {
    // Clear any previously created time blocks for this test
    const result = await createTimeBlock.execute({
      title: 'Morning focus',
      date: '2026-03-15',
      duration_minutes: 30,
      preferred_time: 'morning',
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };

    // Should schedule in a morning slot (8:00-12:00)
    const scheduledHour = new Date(data.scheduled_at).getHours();
    expect(scheduledHour).toBeGreaterThanOrEqual(8);
    expect(scheduledHour).toBeLessThan(12);
  });

  it('should prefer afternoon slot when requested', async () => {
    const result = await createTimeBlock.execute({
      title: 'Afternoon review',
      date: '2026-03-15',
      duration_minutes: 30,
      preferred_time: 'afternoon',
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };

    // Should schedule in an afternoon slot (12:00-17:00)
    const scheduledHour = new Date(data.scheduled_at).getHours();
    expect(scheduledHour).toBeGreaterThanOrEqual(12);
    expect(scheduledHour).toBeLessThan(17);
  });

  it('should prefer evening slot when requested', async () => {
    const result = await createTimeBlock.execute({
      title: 'Evening wrap-up',
      date: '2026-03-15',
      duration_minutes: 30,
      preferred_time: 'evening',
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };

    // Working hours end at 18:00, evening range is 17-20
    // Free slot after workshop: 16:00-18:00, evening starts at 17
    const scheduledHour = new Date(data.scheduled_at).getHours();
    expect(scheduledHour).toBeGreaterThanOrEqual(16);
  });

  it('should create a time block on an empty day', async () => {
    const result = await createTimeBlock.execute({
      title: 'Full day focus',
      date: '2026-06-01',
      duration_minutes: 120,
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };
    expect(data.event_id).toBeDefined();

    // Should schedule at the start of working hours (08:00)
    const scheduledHour = new Date(data.scheduled_at).getHours();
    expect(scheduledHour).toBe(8);
  });

  it('should fall back when preferred time has no space', async () => {
    // Block the entire morning on a test date
    const insert = db.prepare(`
      INSERT INTO events (title, calendar_name, start_time, end_time)
      VALUES (?, ?, ?, ?)
    `);
    insert.run('Block morning', 'default', '2026-04-10T08:00:00', '2026-04-10T12:00:00');

    const result = await createTimeBlock.execute({
      title: 'Forced afternoon',
      date: '2026-04-10',
      duration_minutes: 60,
      preferred_time: 'morning',
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };

    // Morning is blocked, should fall back to first available slot (12:00+)
    const scheduledHour = new Date(data.scheduled_at).getHours();
    expect(scheduledHour).toBeGreaterThanOrEqual(12);
  });

  it('should throw when no slots are available', async () => {
    // Block the entire working day
    db.prepare(`
      INSERT INTO events (title, calendar_name, start_time, end_time)
      VALUES (?, ?, ?, ?)
    `).run('Full day', 'default', '2026-04-20T07:00:00', '2026-04-20T19:00:00');

    await expect(
      createTimeBlock.execute({
        title: 'Impossible block',
        date: '2026-04-20',
        duration_minutes: 30,
      }),
    ).rejects.toThrow(MCPError);
  });

  it('should store time block with description', async () => {
    const result = await createTimeBlock.execute({
      title: 'Writing session',
      date: '2026-06-02',
      duration_minutes: 45,
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };

    const row = db
      .prepare('SELECT * FROM events WHERE id = ?')
      .get(Number(data.event_id)) as CalendarEvent;
    expect(row.description).toBe('Time block created by calendar.create_time_block');
  });

  it('has correct metadata', () => {
    expect(createTimeBlock.name).toBe('calendar.create_time_block');
    expect(createTimeBlock.confirmationRequired).toBe(true);
    expect(createTimeBlock.undoSupported).toBe(false);
  });
});
