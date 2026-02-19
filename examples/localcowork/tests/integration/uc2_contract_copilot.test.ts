/**
 * UC-2: Contract Copilot
 * Servers: filesystem, document (Python)
 * Flow: read files -> diff -> verify changes
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";

describe("UC-2: Contract Copilot", () => {
  const harness = new TestHarness("uc2");

  beforeAll(async () => { await harness.setup(); });
  afterAll(async () => { await harness.teardown(); });

  it("should read the original contract", async () => {
    const filePath = harness.tempPath("original_contract.txt");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const data = result.data as { content: string };
    expect(data.content).toContain("SERVICE AGREEMENT");
    expect(data.content).toContain("Section 1: Scope of Work");
  });

  it("should read the revised contract", async () => {
    const filePath = harness.tempPath("revised_contract.txt");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const data = result.data as { content: string };
    expect(data.content).toContain("bi-weekly status reports");
    expect(data.content).toContain("Section 5: Liability");
  });

  it("should detect differences between the two contracts", async () => {
    const origPath = harness.tempPath("original_contract.txt");
    const revPath = harness.tempPath("revised_contract.txt");
    const origRes = await harness.callTsTool("filesystem", "read_file", { path: origPath });
    const revRes = await harness.callTsTool("filesystem", "read_file", { path: revPath });
    const origText = (origRes.data as { content: string }).content;
    const revText = (revRes.data as { content: string }).content;
    const splitter = String.fromCharCode(10) + String.fromCharCode(10);
    const origParagraphs = origText.split(splitter).filter((s: string) => s.trim());
    const revParagraphs = revText.split(splitter).filter((s: string) => s.trim());
    expect(revParagraphs.length).toBeGreaterThan(origParagraphs.length);
    const origSection2 = origParagraphs.find((pg: string) => pg.includes("Section 2")) ?? "";
    const revSection2 = revParagraphs.find((pg: string) => pg.includes("Section 2")) ?? "";
    expect(origSection2).toContain("5,000");
    expect(revSection2).toContain("6,500");
    expect(revSection2).toContain("Late payments");
    const hasLiability = revParagraphs.some((pg: string) => pg.includes("Section 5: Liability"));
    expect(hasLiability).toBe(true);
    const origHasLiability = origParagraphs.some((pg: string) => pg.includes("Section 5"));
    expect(origHasLiability).toBe(false);
  });

  it("should write a diff summary report", async () => {
    const NL = String.fromCharCode(10);
    const changes = [
      "CHANGED: Section 2 - Compensation increased from 5000 to 6500",
      "CHANGED: Section 3 - Term extended from 12 to 24 months",
      "ADDED: Section 5 - Liability clause",
      "CHANGED: Section 1 - Added bi-weekly status reports",
    ];
    const reportContent = "Contract Diff Summary" + NL + changes.join(NL) + NL;
    const reportPath = harness.tempPath("output", "contract_diff.txt");
    const result = await harness.callTsTool("filesystem", "write_file", {
      path: reportPath, content: reportContent,
    });
    expect(result.success).toBe(true);
    const saved = await harness.readTemp("output/contract_diff.txt");
    expect(saved).toContain("Contract Diff Summary");
    expect(saved).toContain("Liability");
  });
});