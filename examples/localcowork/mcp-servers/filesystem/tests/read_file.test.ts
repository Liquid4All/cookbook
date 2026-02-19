import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { readFile } from '../src/tools/read_file';
import { setupTestDir, teardownTestDir } from './helpers';

describe('filesystem.read_file', () => {
  let testDir: string;

  beforeAll(async () => {
    testDir = await setupTestDir({ files: ['hello.txt'] });
  });

  afterAll(async () => {
    await teardownTestDir(testDir);
  });

  it('should read file contents', async () => {
    const result = await readFile.execute({ path: `${testDir}/hello.txt` });
    expect(result.success).toBe(true);
    expect(result.data.content).toBe('content of hello.txt');
    expect(result.data.size).toBeGreaterThan(0);
  });

  it('should throw for non-existent file', async () => {
    await expect(
      readFile.execute({ path: `${testDir}/missing.txt` }),
    ).rejects.toThrow();
  });

  it('should throw for directory path', async () => {
    await expect(readFile.execute({ path: testDir })).rejects.toThrow();
  });

  it('has correct metadata', () => {
    expect(readFile.name).toBe('filesystem.read_file');
    expect(readFile.confirmationRequired).toBe(false);
    expect(readFile.undoSupported).toBe(false);
  });
});
