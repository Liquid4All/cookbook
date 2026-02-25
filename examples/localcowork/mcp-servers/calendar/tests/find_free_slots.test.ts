import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import type Database from 'better-sqlite3';
import { findFreeSlots } from '../src/tools/find_free_slots';
import { setupTestDb, teardownTestDb } from './helpers';
import type { TimeSlot } from '../src/db';

describe('calendar.find_free_slots', () => {
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

  it('should find free slots on a day with events', async () => {
    const result = await findFreeSlots.execute({ date: '2026-03-15' });
    expect(result.success).toBe(true);
    const data = result.data as { slots: TimeSlot[] };

    // Working hours: 08:00-18:00
    // Busy: 09:00-09:30, 10:00-11:00, 12:00-13:00, 14:00-16:00
    // Free slots (>= 30 min default):
    //   08:00-09:00 (60 min)
    //   09:30-10:00 (30 min)
    //   11:00-12:00 (60 min)
    //   13:00-14:00 (60 min)
    //   16:00-18:00 (120 min)
    expect(data.slots.length).toBe(5);
  });

  it('should respect min_duration_minutes filter', async () => {
    const result = await findFreeSlots.execute({
      date: '2026-03-15',
      min_duration_minutes: 60,
    });
    expect(result.success).toBe(true);
    const data = result.data as { slots: TimeSlot[] };

    // Slots >= 60 min:
    //   08:00-09:00 (60 min)
    //   11:00-12:00 (60 min)
    //   13:00-14:00 (60 min)
    //   16:00-18:00 (120 min)
    expect(data.slots.length).toBe(4);

    for (const slot of data.slots) {
      expect(slot.duration_minutes).toBeGreaterThanOrEqual(60);
    }
  });

  it('should return full working hours when no events', async () => {
    const result = await findFreeSlots.execute({ date: '2026-06-01' });
    expect(result.success).toBe(true);
    const data = result.data as { slots: TimeSlot[] };

    // Entire working day is free: 08:00-18:00
    expect(data.slots.length).toBe(1);
    expect(data.slots[0].duration_minutes).toBe(600);
  });

  it('should return slots with correct start and end times', async () => {
    const result = await findFreeSlots.execute({ date: '2026-03-15' });
    expect(result.success).toBe(true);
    const data = result.data as { slots: TimeSlot[] };

    for (const slot of data.slots) {
      // Verify duration matches start/end
      const start = new Date(slot.start);
      const end = new Date(slot.end);
      const computedMinutes = (end.getTime() - start.getTime()) / 60000;
      expect(slot.duration_minutes).toBe(computedMinutes);

      // All slots should be within working hours
      expect(start.getHours()).toBeGreaterThanOrEqual(8);
      expect(end.getHours()).toBeLessThanOrEqual(18);
    }
  });

  it('should return empty slots when day is fully booked', async () => {
    // Add an all-day blocking event
    db.prepare(`
      INSERT INTO events (title, calendar_name, start_time, end_time)
      VALUES (?, ?, ?, ?)
    `).run('All day block', 'default', '2026-03-20T07:00:00', '2026-03-20T19:00:00');

    const result = await findFreeSlots.execute({
      date: '2026-03-20',
      min_duration_minutes: 30,
    });
    expect(result.success).toBe(true);
    const data = result.data as { slots: TimeSlot[] };
    expect(data.slots.length).toBe(0);
  });

  it('should handle overlapping events correctly', async () => {
    // Add overlapping events on a new date
    const insert = db.prepare(`
      INSERT INTO events (title, calendar_name, start_time, end_time)
      VALUES (?, ?, ?, ?)
    `);
    insert.run('Event A', 'default', '2026-03-22T09:00:00', '2026-03-22T10:30:00');
    insert.run('Event B', 'default', '2026-03-22T10:00:00', '2026-03-22T11:00:00');

    const result = await findFreeSlots.execute({ date: '2026-03-22' });
    expect(result.success).toBe(true);
    const data = result.data as { slots: TimeSlot[] };

    // Merged busy: 09:00-11:00
    // Free: 08:00-09:00 (60), 11:00-18:00 (420)
    expect(data.slots.length).toBe(2);
  });

  it('has correct metadata', () => {
    expect(findFreeSlots.name).toBe('calendar.find_free_slots');
    expect(findFreeSlots.confirmationRequired).toBe(false);
    expect(findFreeSlots.undoSupported).toBe(false);
  });
});
