/**
 * UC-10: Compliance Pack
 * Servers: audit, filesystem
 * Flow: seed audit -> get_tool_log -> get_session_summary -> generate_audit_report
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";
import Database from "better-sqlite3";
import { setDb as setAuditDb, closeDb as closeAuditDb } from "../../mcp-servers/audit/src/db";

const SESSION_ID = "test-session-001";

function seedAuditData(db: InstanceType<typeof Database>): void {
  const insert = db.prepare(
    "INSERT INTO audit_log (session_id, timestamp, tool_name, status, file_path, params_json, duration_ms) VALUES (?,?,?,?,?,?,?)"
  );
  insert.run(SESSION_ID, "2026-02-12T10:00:00", "filesystem.read_file", "confirmed", "/tmp/doc.txt", "{}", 12);
  insert.run(SESSION_ID, "2026-02-12T10:00:05", "filesystem.write_file", "confirmed", "/tmp/out.csv", "{}", 25);
  insert.run(SESSION_ID, "2026-02-12T10:00:10", "data.write_csv", "confirmed", "/tmp/out.csv", "{}", 18);
  insert.run(SESSION_ID, "2026-02-12T10:00:15", "filesystem.delete_file", "rejected", "/tmp/secret.txt", "{}", 5);
  insert.run(SESSION_ID, "2026-02-12T10:00:20", "task.create_task", "confirmed", null, "{}", 30);
}

describe("UC-10: Compliance Pack", () => {
  const harness = new TestHarness("uc10");
  let auditDb: InstanceType<typeof Database>;

  beforeAll(async () => {
    await harness.setup();
    auditDb = new Database(":memory:");
    setAuditDb(auditDb);
    seedAuditData(auditDb);
  });
  afterAll(async () => { closeAuditDb(); await harness.teardown(); });

  it("should retrieve tool execution log", async () => {
    const result = await harness.callTsTool("audit", "get_tool_log", { session_id: SESSION_ID });
    expect(result.success).toBe(true);
    const entries = result.data as Array<{ tool_name: string; status: string }>;
    expect(entries.length).toBe(5);
  });

  it("should filter log by tool name", async () => {
    const result = await harness.callTsTool("audit", "get_tool_log", {
      session_id: SESSION_ID, tool_name: "filesystem.read_file",
    });
    expect(result.success).toBe(true);
    const entries = result.data as Array<{ tool_name: string }>;
    expect(entries.length).toBe(1);
    expect(entries[0].tool_name).toBe("filesystem.read_file");
  });

  it("should get session summary with aggregated stats", async () => {
    const result = await harness.callTsTool("audit", "get_session_summary", { session_id: SESSION_ID });
    expect(result.success).toBe(true);
    const data = result.data as {
      documents_touched: Array<{ path: string }>;
      tools_called: Array<{ tool_name: string; call_count: number }>;
      confirmations: number;
      rejections: number;
    };
    expect(data.confirmations).toBe(4);
    expect(data.rejections).toBe(1);
    expect(data.documents_touched.length).toBeGreaterThan(0);
    expect(data.tools_called.length).toBeGreaterThan(0);
  });

  it("should generate a text audit report", async () => {
    const result = await harness.callTsTool("audit", "generate_audit_report", { session_id: SESSION_ID });
    expect(result.success).toBe(true);
    const data = result.data as { report: string };
    expect(data.report).toContain("AUDIT REPORT");
    expect(data.report).toContain(SESSION_ID);
    expect(data.report).toContain("Total tool calls: 5");
    expect(data.report).toContain("Confirmed: 4");
    expect(data.report).toContain("Rejected: 1");
  });

  it("should save the audit report to a file", async () => {
    const reportRes = await harness.callTsTool("audit", "generate_audit_report", { session_id: SESSION_ID });
    const report = (reportRes.data as { report: string }).report;
    const reportPath = harness.tempPath("output", "audit_report.txt");
    const writeRes = await harness.callTsTool("filesystem", "write_file", {
      path: reportPath, content: report,
    });
    expect(writeRes.success).toBe(true);
    const saved = await harness.readTemp("output/audit_report.txt");
    expect(saved).toContain("AUDIT REPORT");
    expect(saved).toContain("Timeline");
  });
});