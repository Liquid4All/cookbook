/**
 * UC-4: Download Triage
 * Servers: filesystem, data
 * Flow: list_dir -> get_metadata -> classify -> move_file -> write_csv log
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { TestHarness } from "../helpers/test-harness";

describe("UC-4: Download Triage", () => {
  const harness = new TestHarness("uc4");

  beforeAll(async () => { await harness.setup(); });
  afterAll(async () => { await harness.teardown(); });

  it("should list files in the downloads directory", async () => {
    const dir = harness.tempPath("downloads");
    const result = await harness.callTsTool("filesystem", "list_dir", { path: dir, recursive: false });
    expect(result.success).toBe(true);
    const files = result.data as Array<{ name: string }>;
    expect(files.length).toBe(3);
  });

  it("should get metadata for each file", async () => {
    const dir = harness.tempPath("downloads");
    const listRes = await harness.callTsTool("filesystem", "list_dir", { path: dir, recursive: false });
    const files = listRes.data as Array<{ path: string; name: string }>;
    for (const file of files) {
      const metaRes = await harness.callTsTool("filesystem", "get_metadata", { path: file.path });
      expect(metaRes.success).toBe(true);
      const meta = metaRes.data as { size: number; type: string; extension: string };
      expect(meta.size).toBeGreaterThan(0);
      expect(meta.extension).toBe(".txt");
    }
  });

  it("should classify and move files into category directories", async () => {
    const dir = harness.tempPath("downloads");
    const listRes = await harness.callTsTool("filesystem", "list_dir", { path: dir, recursive: false });
    const files = listRes.data as Array<{ path: string; name: string }>;
    const categories: Record<string, string> = {
      "quarterly_report.txt": "reports",
      "photo_readme.txt": "photos",
      "receipt_amazon.txt": "receipts",
    };
    const manifest: Array<Record<string, string>> = [];
    for (const file of files) {
      const category = categories[file.name] ?? "other";
      const destDir = harness.tempPath("sorted", category);
      const destPath = destDir + "/" + file.name;
      const moveRes = await harness.callTsTool("filesystem", "move_file", {
        source: file.path, destination: destPath, create_dirs: true,
      });
      expect(moveRes.success).toBe(true);
      manifest.push({ file: file.name, category, destination: destPath });
    }
    expect(manifest.length).toBe(3);
    const reportsExists = await harness.existsTemp("sorted/reports/quarterly_report.txt");
    expect(reportsExists).toBe(true);
    const photosExists = await harness.existsTemp("sorted/photos/photo_readme.txt");
    expect(photosExists).toBe(true);
    const receiptsExists = await harness.existsTemp("sorted/receipts/receipt_amazon.txt");
    expect(receiptsExists).toBe(true);
  });

  it("should write a CSV manifest of sorted files", async () => {
    const manifest = [
      { file: "quarterly_report.txt", category: "reports" },
      { file: "photo_readme.txt", category: "photos" },
      { file: "receipt_amazon.txt", category: "receipts" },
    ];
    const csvPath = harness.tempPath("output", "triage_manifest.csv");
    const result = await harness.callTsTool("data", "write_csv", {
      data: manifest, output_path: csvPath,
    });
    expect(result.success).toBe(true);
    const csvContent = await harness.readTemp("output/triage_manifest.csv");
    expect(csvContent).toContain("file,category");
    expect(csvContent).toContain("quarterly_report.txt,reports");
  });
});