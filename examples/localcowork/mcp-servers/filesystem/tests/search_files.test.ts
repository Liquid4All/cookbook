import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { searchFiles } from '../src/tools/search_files';
import { setupTestDir, teardownTestDir } from './helpers';

describe('filesystem.search_files', () => {
  let testDir: string;

  beforeAll(async () => {
    testDir = await setupTestDir({
      files: ['report.pdf', 'notes.md', 'data.csv', 'sub/deep.txt'],
    });
  });

  afterAll(async () => {
    await teardownTestDir(testDir);
  });

  it('should find files matching glob pattern', async () => {
    const result = await searchFiles.execute({
      path: testDir,
      pattern: '*.pdf',
    });
    expect(result.success).toBe(true);
    expect(result.data).toHaveLength(1);
    expect(result.data[0].name).toBe('report.pdf');
  });

  it('should search recursively by default', async () => {
    const result = await searchFiles.execute({
      path: testDir,
      pattern: '*.txt',
    });
    expect(result.success).toBe(true);
    expect(result.data.length).toBeGreaterThanOrEqual(1);
  });

  it('should respect max_results', async () => {
    const result = await searchFiles.execute({
      path: testDir,
      pattern: '*',
      max_results: 2,
    });
    expect(result.data.length).toBeLessThanOrEqual(2);
  });

  it('should filter by file type', async () => {
    const result = await searchFiles.execute({
      path: testDir,
      pattern: '*',
      type: 'dir',
    });
    const types = result.data.map((e: { type: string }) => e.type);
    expect(types.every((t: string) => t === 'dir')).toBe(true);
  });

  it('has correct metadata', () => {
    expect(searchFiles.name).toBe('filesystem.search_files');
    expect(searchFiles.confirmationRequired).toBe(false);
    expect(searchFiles.undoSupported).toBe(false);
  });
});
