import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { writeCsv } from '../src/tools/write_csv';
import { setupTestDir, teardownTestDir } from './helpers';

describe('data.write_csv', () => {
  let testDir: string;

  beforeAll(() => {
    testDir = setupTestDir();
  });

  afterAll(() => {
    teardownTestDir(testDir);
  });

  it('should write CSV with auto-detected headers', async () => {
    const outPath = path.join(testDir, 'output.csv');
    const result = await writeCsv.execute({
      data: [
        { name: 'Alice', age: 30 },
        { name: 'Bob', age: 25 },
      ],
      output_path: outPath,
    });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({ path: outPath, rows: 2 });

    const content = fs.readFileSync(outPath, 'utf-8');
    expect(content).toContain('name,age');
    expect(content).toContain('Alice,30');
    expect(content).toContain('Bob,25');
  });

  it('should write CSV with explicit headers', async () => {
    const outPath = path.join(testDir, 'explicit.csv');
    const result = await writeCsv.execute({
      data: [
        { name: 'Alice', age: 30, city: 'NYC' },
        { name: 'Bob', age: 25, city: 'LA' },
      ],
      output_path: outPath,
      headers: ['name', 'city'],
    });

    expect(result.success).toBe(true);
    const content = fs.readFileSync(outPath, 'utf-8');
    expect(content).toContain('name,city');
    expect(content).not.toContain('age');
  });

  it('should escape special characters in CSV fields', async () => {
    const outPath = path.join(testDir, 'escaped.csv');
    await writeCsv.execute({
      data: [
        { name: 'O"Brien', notes: 'has,comma' },
        { name: 'Normal', notes: 'clean' },
      ],
      output_path: outPath,
    });

    const content = fs.readFileSync(outPath, 'utf-8');
    expect(content).toContain('"O""Brien"');
    expect(content).toContain('"has,comma"');
  });

  it('should create parent directories if needed', async () => {
    const outPath = path.join(testDir, 'nested', 'dir', 'file.csv');
    const result = await writeCsv.execute({
      data: [{ x: 1 }],
      output_path: outPath,
    });

    expect(result.success).toBe(true);
    expect(fs.existsSync(outPath)).toBe(true);
  });

  it('has correct metadata', () => {
    expect(writeCsv.name).toBe('data.write_csv');
    expect(writeCsv.confirmationRequired).toBe(true);
    expect(writeCsv.undoSupported).toBe(false);
  });
});
