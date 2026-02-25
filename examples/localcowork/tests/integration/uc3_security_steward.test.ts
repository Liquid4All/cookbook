/**
 * UC-3: Security Steward
 * Servers: filesystem, security (Python)
 * Flow: list_dir -> scan for PII/secrets -> propose_cleanup
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";

describe("UC-3: Security Steward", () => {
  const harness = new TestHarness("uc3");

  beforeAll(async () => { await harness.setup(); });
  afterAll(async () => { await harness.teardown(); });

  it("should list files in the sample_files directory", async () => {
    const dir = harness.tempPath("sample_files");
    const result = await harness.callTsTool("filesystem", "list_dir", { path: dir, recursive: false });
    expect(result.success).toBe(true);
    const files = result.data as Array<{ name: string }>;
    expect(files.length).toBe(3);
    const names = files.map((f) => f.name).sort();
    expect(names).toContain("has_ssn.txt");
    expect(names).toContain("has_api_key.env");
    expect(names).toContain("clean_file.txt");
  });

  it("should detect SSN patterns in has_ssn.txt", async () => {
    const filePath = harness.tempPath("sample_files", "has_ssn.txt");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const content = (result.data as { content: string }).content;
    const ssnPattern = /\d{3}-\d{2}-\d{4}/g;
    const matches = content.match(ssnPattern);
    expect(matches).not.toBeNull();
    expect(matches!.length).toBe(2);
    expect(matches).toContain("123-45-6789");
    expect(matches).toContain("987-65-4321");
  });

  it("should detect AWS key patterns in has_api_key.env", async () => {
    const filePath = harness.tempPath("sample_files", "has_api_key.env");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const content = (result.data as { content: string }).content;
    const awsKeyPattern = /AKIA[0-9A-Z]{16}/;
    expect(awsKeyPattern.test(content)).toBe(true);
    const stripePattern = /sk_test_[0-9a-zA-Z]{24,}/;
    expect(stripePattern.test(content)).toBe(true);
  });

  it("should confirm clean_file.txt has no PII or secrets", async () => {
    const filePath = harness.tempPath("sample_files", "clean_file.txt");
    const result = await harness.callTsTool("filesystem", "read_file", { path: filePath });
    expect(result.success).toBe(true);
    const content = (result.data as { content: string }).content;
    const ssnPattern = /\d{3}-\d{2}-\d{4}/g;
    const awsKeyPattern = /AKIA[0-9A-Z]{16}/;
    expect(ssnPattern.test(content)).toBe(false);
    expect(awsKeyPattern.test(content)).toBe(false);
  });

  it("should generate a cleanup proposal from findings", async () => {
    const findings = [
      { file: "has_ssn.txt", type: "ssn", count: 2, severity: "high" },
      { file: "has_api_key.env", type: "aws_key", count: 1, severity: "high" },
      { file: "has_api_key.env", type: "stripe_key", count: 1, severity: "high" },
    ];
    const NL = String.fromCharCode(10);
    const reportLines = ["Security Scan Report", ""];
    for (const f of findings) {
      reportLines.push(f.file + ": " + f.count + " " + f.type + " finding(s) [" + f.severity + "]");
    }
    reportLines.push("", "Recommended actions:");
    reportLines.push("- Redact SSNs from has_ssn.txt");
    reportLines.push("- Rotate AWS key in has_api_key.env");
    reportLines.push("- Rotate Stripe key in has_api_key.env");
    const reportPath = harness.tempPath("output", "security_report.txt");
    const writeRes = await harness.callTsTool("filesystem", "write_file", {
      path: reportPath, content: reportLines.join(NL),
    });
    expect(writeRes.success).toBe(true);
    const saved = await harness.readTemp("output/security_report.txt");
    expect(saved).toContain("Security Scan Report");
    expect(saved).toContain("Rotate AWS key");
  });
});