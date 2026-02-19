/**
 * UC-7: Personal Ops
 * Servers: task, calendar, email
 * Flow: create_task -> list_tasks -> list_events -> get_overdue -> daily_briefing
 *        -> find_free_slots -> create_time_block
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";
import Database from "better-sqlite3";
import { setDb as setTaskDb, closeDb as closeTaskDb } from "../../mcp-servers/task/src/db";
import { setDb as setCalDb, closeDb as closeCalDb } from "../../mcp-servers/calendar/src/db";

describe("UC-7: Personal Ops", () => {
  const harness = new TestHarness("uc7");

  beforeAll(async () => {
    await harness.setup();
    setTaskDb(new Database(":memory:"));
    setCalDb(new Database(":memory:"));
  });
  afterAll(async () => {
    closeTaskDb(); closeCalDb();
    await harness.teardown();
  });

  it("should seed tasks with various priorities and due dates", async () => {
    const tasks = [
      { title: "Review PR #42", priority: 1, due_date: "2026-02-10" },
      { title: "Write blog post", priority: 3, due_date: "2026-02-14" },
      { title: "Expense report", priority: 2, due_date: "2026-02-11" },
      { title: "Update docs", priority: 4, due_date: "2026-02-20" },
    ];
    for (const t of tasks) {
      const result = await harness.callTsTool("task", "create_task", t);
      expect(result.success).toBe(true);
    }
  });

  it("should list pending tasks", async () => {
    const result = await harness.callTsTool("task", "list_tasks", { status: "pending" });
    expect(result.success).toBe(true);
    const tasks = result.data as Array<{ title: string; priority: number }>;
    expect(tasks.length).toBe(4);
    expect(tasks[0].priority).toBeLessThanOrEqual(tasks[1].priority);
  });

  it("should seed calendar events", async () => {
    const events = [
      { title: "Team Standup", start: "2026-02-12T09:00:00", end: "2026-02-12T09:30:00" },
      { title: "1:1 with Manager", start: "2026-02-12T14:00:00", end: "2026-02-12T14:30:00" },
    ];
    for (const e of events) {
      const result = await harness.callTsTool("calendar", "create_event", e);
      expect(result.success).toBe(true);
    }
  });

  it("should list events for a date range", async () => {
    const result = await harness.callTsTool("calendar", "list_events", {
      start_date: "2026-02-12", end_date: "2026-02-12",
    });
    expect(result.success).toBe(true);
    const events = result.data as Array<{ title: string }>;
    expect(events.length).toBe(2);
  });

  it("should get overdue tasks", async () => {
    const result = await harness.callTsTool("task", "get_overdue", {});
    expect(result.success).toBe(true);
    const overdue = result.data as Array<{ title: string }>;
    const overdueNames = overdue.map((t) => t.title);
    expect(overdueNames).toContain("Review PR #42");
    expect(overdueNames).toContain("Expense report");
  });

  it("should generate a daily briefing", async () => {
    const result = await harness.callTsTool("task", "daily_briefing", { date: "2026-02-12" });
    expect(result.success).toBe(true);
    const data = result.data as { briefing: string; tasks: unknown[]; overdue: unknown[] };
    expect(data.briefing).toContain("Daily Briefing");
    expect(data.briefing).toContain("OVERDUE");
  });

  it("should find free slots on a given date", async () => {
    const result = await harness.callTsTool("calendar", "find_free_slots", {
      date: "2026-02-12", min_duration_minutes: 30,
    });
    expect(result.success).toBe(true);
    const data = result.data as { slots: Array<{ start: string; end: string; duration_minutes: number }> };
    expect(data.slots.length).toBeGreaterThan(0);
    for (const slot of data.slots) {
      expect(slot.duration_minutes).toBeGreaterThanOrEqual(30);
    }
  });

  it("should create a time block for focused work", async () => {
    const result = await harness.callTsTool("calendar", "create_time_block", {
      title: "Deep Work: Blog Post", date: "2026-02-12",
      duration_minutes: 60, preferred_time: "morning",
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string; scheduled_at: string };
    expect(data.event_id).toBeDefined();
    expect(data.scheduled_at).toContain("2026-02-12");
  });
});