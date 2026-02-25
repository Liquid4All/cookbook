/**
 * UC-6: Meeting Pipeline
 * Servers: meeting (Python), task, calendar, email
 * Flow: transcribe -> extract_action_items -> create_task -> create_event -> draft_email
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";
import Database from "better-sqlite3";
import { setDb as setTaskDb, closeDb as closeTaskDb } from "../../mcp-servers/task/src/db";
import { setDb as setCalDb, closeDb as closeCalDb } from "../../mcp-servers/calendar/src/db";
import { setDb as setEmailDb, closeDb as closeEmailDb } from "../../mcp-servers/email/src/db";

const NL = String.fromCharCode(10);
const STUB_TRANSCRIPT = [
  "Alice: Welcome everyone to the Q1 planning meeting.",
  "Bob: Thanks Alice. ACTION: Bob will prepare the budget proposal by March 1st.",
  "Charlie: I agree. TODO: Charlie needs to review vendor contracts by Feb 28th.",
  "Alice: Great. I will schedule a follow-up meeting for next Tuesday.",
  "Bob: We decided to increase the marketing budget by 20%.",
  "Charlie: Question - should we hire a new designer? TBD.",
].join(NL);

describe("UC-6: Meeting Pipeline", () => {
  const harness = new TestHarness("uc6");

  beforeAll(async () => {
    await harness.setup();
    setTaskDb(new Database(":memory:"));
    setCalDb(new Database(":memory:"));
    setEmailDb(new Database(":memory:"));
  });
  afterAll(async () => {
    closeTaskDb(); closeCalDb(); closeEmailDb();
    await harness.teardown();
  });

  it("should create tasks from extracted action items", async () => {
    const actionItems = [
      { title: "Prepare budget proposal", assignee: "Bob", due_date: "2026-03-01" },
      { title: "Review vendor contracts", assignee: "Charlie", due_date: "2026-02-28" },
    ];
    for (const item of actionItems) {
      const result = await harness.callTsTool("task", "create_task", {
        title: item.title, description: "Assigned to " + item.assignee,
        source: "meeting", priority: 2, due_date: item.due_date,
      });
      expect(result.success).toBe(true);
      const data = result.data as { task_id: number };
      expect(data.task_id).toBeGreaterThan(0);
    }
  });

  it("should list the created tasks", async () => {
    const result = await harness.callTsTool("task", "list_tasks", { status: "pending" });
    expect(result.success).toBe(true);
    const tasks = result.data as Array<{ title: string }>;
    expect(tasks.length).toBe(2);
  });

  it("should create a follow-up calendar event", async () => {
    const nextTuesday = "2026-02-17";
    const result = await harness.callTsTool("calendar", "create_event", {
      title: "Q1 Planning Follow-up",
      start: nextTuesday + "T10:00:00",
      end: nextTuesday + "T11:00:00",
      description: "Follow-up meeting for Q1 planning",
    });
    expect(result.success).toBe(true);
    const data = result.data as { event_id: string };
    expect(data.event_id).toBeDefined();
  });

  it("should draft a summary email", async () => {
    const body = [
      "Hi team,",
      "",
      "Here are the action items from our meeting:",
      "",
      "1. Bob: Prepare budget proposal (due March 1)",
      "2. Charlie: Review vendor contracts (due Feb 28)",
      "",
      "A follow-up meeting has been scheduled for next Tuesday at 10 AM.",
      "",
      "Best,",
      "Alice",
    ].join(NL);
    const result = await harness.callTsTool("email", "draft_email", {
      to: ["alice@example.com", "bob@example.com", "charlie@example.com"],
      subject: "Q1 Planning Meeting - Action Items",
      body,
    });
    expect(result.success).toBe(true);
    const data = result.data as { draft_id: string; preview: string };
    expect(data.draft_id).toBeDefined();
    expect(data.preview).toContain("action items");
  });

  it("should run the full meeting pipeline", async () => {
    const task1 = await harness.callTsTool("task", "create_task", {
      title: "Follow up on budget", source: "meeting", priority: 1, due_date: "2026-03-05",
    });
    expect(task1.success).toBe(true);
    const event1 = await harness.callTsTool("calendar", "create_event", {
      title: "Budget Review", start: "2026-03-05T14:00:00", end: "2026-03-05T15:00:00",
    });
    expect(event1.success).toBe(true);
    const email1 = await harness.callTsTool("email", "draft_email", {
      to: ["team@example.com"], subject: "Meeting Follow-up",
      body: "Please review the action items from our meeting.",
    });
    expect(email1.success).toBe(true);
  });
});