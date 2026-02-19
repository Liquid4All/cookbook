import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { listEvents } from '../src/tools/list_events';
import { setupTestDb, teardownTestDb } from './helpers';
import type { CalendarEvent } from '../src/db';

describe('calendar.list_events', () => {
  let db: Database.Database;

  beforeAll(() => {
    db = setupTestDb({
      eventCount: 10,
      withDayEvents: true,
      dayEventsDate: '2026-03-15',
      withMultipleCalendars: true,
    });
  });

  afterAll(() => {
    teardownTestDb(db);
  });

  it('should return events within a date range', async () => {
    const result = await listEvents.execute({
      start_date: '2026-03-15',
      end_date: '2026-03-15',
    });
    expect(result.success).toBe(true);
    const events = result.data as CalendarEvent[];
    // Should include the 4 day events plus any generic events on the 15th
    expect(events.length).toBeGreaterThanOrEqual(4);
  });

  it('should return empty array when no events in range', async () => {
    const result = await listEvents.execute({
      start_date: '2030-01-01',
      end_date: '2030-01-31',
    });
    expect(result.success).toBe(true);
    const events = result.data as CalendarEvent[];
    expect(events.length).toBe(0);
  });

  it('should filter by calendar name', async () => {
    const result = await listEvents.execute({
      start_date: '2026-03-01',
      end_date: '2026-03-31',
      calendar: 'work',
    });
    expect(result.success).toBe(true);
    const events = result.data as CalendarEvent[];
    for (const event of events) {
      expect(event.calendar_name).toBe('work');
    }
  });

  it('should sort events by start_time ascending', async () => {
    const result = await listEvents.execute({
      start_date: '2026-03-01',
      end_date: '2026-03-31',
    });
    expect(result.success).toBe(true);
    const events = result.data as CalendarEvent[];
    for (let i = 1; i < events.length; i++) {
      expect(events[i - 1].start_time <= events[i].start_time).toBe(true);
    }
  });

  it('should return events spanning a full month range', async () => {
    const result = await listEvents.execute({
      start_date: '2026-03-01',
      end_date: '2026-03-31',
    });
    expect(result.success).toBe(true);
    const events = result.data as CalendarEvent[];
    // 10 generic events + 4 day events = 14
    expect(events.length).toBe(14);
  });

  it('should return all calendars when no calendar filter', async () => {
    const result = await listEvents.execute({
      start_date: '2026-03-01',
      end_date: '2026-03-31',
    });
    expect(result.success).toBe(true);
    const events = result.data as CalendarEvent[];
    const calendars = new Set(events.map((e) => e.calendar_name));
    expect(calendars.size).toBeGreaterThanOrEqual(2);
  });

  it('has correct metadata', () => {
    expect(listEvents.name).toBe('calendar.list_events');
    expect(listEvents.confirmationRequired).toBe(false);
    expect(listEvents.undoSupported).toBe(false);
  });
});
