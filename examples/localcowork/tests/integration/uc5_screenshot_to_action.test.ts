/**
 * UC-5: Screenshot to Action
 * Servers: clipboard, data
 * Flow: set_clipboard -> get_clipboard -> process -> write_csv
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";
import { setBridge, MockClipboardBridge } from "../../mcp-servers/clipboard/src/bridge";

const NL = String.fromCharCode(10);
const SAMPLE_TEXT = [
  "Meeting Notes",
  "Date: 2026-02-10",
  "Attendees: Alice, Bob, Charlie",
  "Action: Review Q4 budget by Friday",
].join(NL);

describe("UC-5: Screenshot to Action", () => {
  const harness = new TestHarness("uc5");

  beforeAll(async () => {
    await harness.setup();
    setBridge(new MockClipboardBridge());
  });
  afterAll(async () => { await harness.teardown(); });

  it("should set clipboard content", async () => {
    const result = await harness.callTsTool("clipboard", "set_clipboard", { content: SAMPLE_TEXT });
    expect(result.success).toBe(true);
  });

  it("should get clipboard content", async () => {
    await harness.callTsTool("clipboard", "set_clipboard", { content: SAMPLE_TEXT });
    const result = await harness.callTsTool("clipboard", "get_clipboard", {});
    expect(result.success).toBe(true);
    const data = result.data as { content: string; type: string };
    expect(data.content).toContain("Meeting Notes");
    expect(data.content).toContain("Action:");
  });

  it("should process clipboard text and write structured CSV", async () => {
    await harness.callTsTool("clipboard", "set_clipboard", { content: SAMPLE_TEXT });
    const clipRes = await harness.callTsTool("clipboard", "get_clipboard", {});
    const content = (clipRes.data as { content: string }).content;
    const date = content.match(/Date:\s*(.+)/)?.[1]?.trim() ?? "Unknown";
    const attendees = content.match(/Attendees:\s*(.+)/)?.[1]?.trim() ?? "";
    const action = content.match(/Action:\s*(.+)/)?.[1]?.trim() ?? "";
    const structured = [{ date, attendees, action, source: "clipboard" }];
    const csvPath = harness.tempPath("output", "clipboard_extract.csv");
    const csvRes = await harness.callTsTool("data", "write_csv", {
      data: structured, output_path: csvPath,
    });
    expect(csvRes.success).toBe(true);
    const csvContent = await harness.readTemp("output/clipboard_extract.csv");
    expect(csvContent).toContain("date,attendees,action,source");
    expect(csvContent).toContain("2026-02-10");
    expect(csvContent).toContain("Review Q4 budget by Friday");
  });

  it("should update clipboard with processed content", async () => {
    const processed = "EXTRACTED: Action Item - Review Q4 budget by Friday (Due: Friday)";
    const setRes = await harness.callTsTool("clipboard", "set_clipboard", { content: processed });
    expect(setRes.success).toBe(true);
    const getRes = await harness.callTsTool("clipboard", "get_clipboard", {});
    const data = getRes.data as { content: string };
    expect(data.content).toBe(processed);
  });
});