/**
 * UC-9: Codebase Navigator
 * Servers: filesystem, clipboard
 * Flow: search_files -> read_file -> set_clipboard
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";
import { setBridge, MockClipboardBridge } from "../../mcp-servers/clipboard/src/bridge";

describe("UC-9: Codebase Navigator", () => {
  const harness = new TestHarness("uc9");

  beforeAll(async () => {
    await harness.setup();
    setBridge(new MockClipboardBridge());
  });
  afterAll(async () => { await harness.teardown(); });

  it("should search for TypeScript files in the mini codebase", async () => {
    const dir = harness.tempPath("mini_codebase");
    const result = await harness.callTsTool("filesystem", "search_files", {
      path: dir, pattern: "*.ts", type: "file",
    });
    expect(result.success).toBe(true);
    const files = result.data as Array<{ name: string; path: string }>;
    expect(files.length).toBe(2);
    const names = files.map((f) => f.name).sort();
    expect(names).toContain("main.ts");
    expect(names).toContain("utils.ts");
  });

  it("should read a specific source file", async () => {
    const filePath = harness.tempPath("mini_codebase", "utils.ts");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const data = result.data as { content: string };
    expect(data.content).toContain("export function greet");
    expect(data.content).toContain("export function add");
  });

  it("should search for a specific function name", async () => {
    const dir = harness.tempPath("mini_codebase");
    const searchRes = await harness.callTsTool("filesystem", "search_files", {
      path: dir, pattern: "*.ts", type: "file",
    });
    const files = searchRes.data as Array<{ path: string }>;
    let foundFile = "";
    for (const file of files) {
      const readRes = await harness.callTsTool("filesystem", "read_file", { path: file.path });
      const content = (readRes.data as { content: string }).content;
      if (content.includes("function greet")) { foundFile = file.path; break; }
    }
    expect(foundFile).toContain("utils.ts");
  });

  it("should copy code to clipboard", async () => {
    const filePath = harness.tempPath("mini_codebase", "utils.ts");
    const readRes = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    const content = (readRes.data as { content: string }).content;
    const setRes = await harness.callTsTool("clipboard", "set_clipboard", { content });
    expect(setRes.success).toBe(true);
    const getRes = await harness.callTsTool("clipboard", "get_clipboard", {});
    const clipData = getRes.data as { content: string };
    expect(clipData.content).toContain("export function greet");
  });

  it("should read the config file", async () => {
    const filePath = harness.tempPath("mini_codebase", "config.json");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const data = result.data as { content: string };
    const config = JSON.parse(data.content);
    expect(config.app_name).toBe("LocalCowork");
  });
});