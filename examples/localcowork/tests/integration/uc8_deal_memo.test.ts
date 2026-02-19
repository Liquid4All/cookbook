/**
 * UC-8: Deal Memo
 * Servers: filesystem, data, task
 * Flow: read files -> structure data -> write_csv -> create_task (diligence items)
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";
import Database from "better-sqlite3";
import { setDb as setTaskDb, closeDb as closeTaskDb } from "../../mcp-servers/task/src/db";

describe("UC-8: Deal Memo", () => {
  const harness = new TestHarness("uc8");

  beforeAll(async () => {
    await harness.setup();
    setTaskDb(new Database(":memory:"));
  });
  afterAll(async () => { closeTaskDb(); await harness.teardown(); });

  it("should read the deal memo file", async () => {
    const filePath = harness.tempPath("deal_memo.txt");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const data = result.data as { content: string };
    expect(data.content).toContain("Project Phoenix");
    expect(data.content).toContain("TechStart Inc");
  });

  it("should read the financials file", async () => {
    const filePath = harness.tempPath("financials.txt");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const data = result.data as { content: string };
    expect(data.content).toContain("Annual Revenue");
  });

  it("should extract and structure key metrics into CSV", async () => {
    const metrics = [
      { metric: "Valuation", value: "15000000", unit: "USD" },
      { metric: "ARR", value: "2100000", unit: "USD" },
      { metric: "MRR Growth", value: "15", unit: "percent" },
      { metric: "Customers", value: "47", unit: "count" },
      { metric: "Churn", value: "3.2", unit: "percent" },
      { metric: "EBITDA", value: "470000", unit: "USD" },
    ];
    const csvPath = harness.tempPath("output", "deal_metrics.csv");
    const result = await harness.callTsTool("data", "write_csv", {
      data: metrics, output_path: csvPath,
    });
    expect(result.success).toBe(true);
    const csvContent = await harness.readTemp("output/deal_metrics.csv");
    expect(csvContent).toContain("metric,value,unit");
    expect(csvContent).toContain("Valuation,15000000,USD");
    expect(csvContent).toContain("ARR,2100000,USD");
  });

  it("should create due diligence tasks", async () => {
    const diligenceItems = [
      "Financial audit (Q1-Q4 2025)",
      "IP review and patent search",
      "Customer reference calls (5 minimum)",
      "Technical architecture review",
      "Team background checks",
    ];
    for (const item of diligenceItems) {
      const result = await harness.callTsTool("task", "create_task", {
        title: item, description: "Due diligence for Project Phoenix",
        source: "manual", priority: 2,
      });
      expect(result.success).toBe(true);
    }
    const listRes = await harness.callTsTool("task", "list_tasks", { status: "pending" });
    const tasks = listRes.data as Array<{ title: string }>;
    expect(tasks.length).toBe(5);
    expect(tasks.some((t) => t.title === "Financial audit (Q1-Q4 2025)")).toBe(true);
  });

  it("should run the full deal memo pipeline", async () => {
    const memoRes = await harness.callTsTool("filesystem", "read_file", { path: harness.tempPath("deal_memo.txt") });
    expect(memoRes.success).toBe(true);
    const finRes = await harness.callTsTool("filesystem", "read_file", { path: harness.tempPath("financials.txt") });
    expect(finRes.success).toBe(true);
    const summaryPath = harness.tempPath("output", "deal_summary.csv");
    const csvRes = await harness.callTsTool("data", "write_csv", {
      data: [{ field: "Company", value: "TechStart Inc" }, { field: "Deal", value: "Series A" }],
      output_path: summaryPath,
    });
    expect(csvRes.success).toBe(true);
    const exists = await harness.existsTemp("output/deal_summary.csv");
    expect(exists).toBe(true);
  });
});