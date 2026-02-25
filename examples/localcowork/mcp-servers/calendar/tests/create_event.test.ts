import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { createEvent } from '../src/tools/create_event';
import { setupTestDb, teardownTestDb } from './helpers';
import type { CalendarEvent } from '../src/db';
import { MCPError } from '../../_shared/ts/mcp-base';

describe('calendar.create_event', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({ eventCount: 0 });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should create an event with required fields only', async () => {
    const result = await createEvent.execute({
      title: 'Team standup',
      start: '2026-04-01T09:00:00',
      end: '2026-04-01T09:30:00',
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string };
    expect(data.event_id).toBeDefined();

    const row = db
      .prepare('SELECT * FROM events WHERE id = ?')
      .get(Number(data.event_id)) as CalendarEvent;
    expect(row.title).toBe('Team standup');
    expect(row.calendar_name).toBe('default');
    expect(row.description).toBeNull();
    expect(row.start_time).toBe('2026-04-01T09:00:00');
    expect(row.end_time).toBe('2026-04-01T09:30:00');
  });

  it('should create an event with all fields', async () => {
    const result = await createEvent.execute({
      title: 'Design review',
      start: '2026-04-02T14:00:00',
      end: '2026-04-02T15:00:00',
      description: 'Review Q2 design specs',
      calendar: 'work',
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string };

    const row = db
      .prepare('SELECT * FROM events WHERE id = ?')
      .get(Number(data.event_id)) as CalendarEvent;
    expect(row.title).toBe('Design review');
    expect(row.description).toBe('Review Q2 design specs');
    expect(row.calendar_name).toBe('work');
    expect(row.start_time).toBe('2026-04-02T14:00:00');
    expect(row.end_time).toBe('2026-04-02T15:00:00');
  });

  it('should reject event where end is before start', async () => {
    await expect(
      createEvent.execute({
        title: 'Bad event',
        start: '2026-04-01T15:00:00',
        end: '2026-04-01T14:00:00',
      }),
    ).rejects.toThrow(MCPError);
  });

  it('should reject event where end equals start', async () => {
    await expect(
      createEvent.execute({
        title: 'Zero duration',
        start: '2026-04-01T10:00:00',
        end: '2026-04-01T10:00:00',
      }),
    ).rejects.toThrow(MCPError);
  });

  it('should default calendar to "default"', async () => {
    const result = await createEvent.execute({
      title: 'Default cal event',
      start: '2026-04-03T08:00:00',
      end: '2026-04-03T08:30:00',
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string };

    const row = db
      .prepare('SELECT * FROM events WHERE id = ?')
      .get(Number(data.event_id)) as CalendarEvent;
    expect(row.calendar_name).toBe('default');
  });

  it('has correct metadata', () => {
    expect(createEvent.name).toBe('calendar.create_event');
    expect(createEvent.confirmationRequired).toBe(true);
    expect(createEvent.undoSupported).toBe(false);
  });
});
