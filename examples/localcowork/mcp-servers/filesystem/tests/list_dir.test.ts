import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { listDir } from '../src/tools/list_dir';
import { setupTestDir, teardownTestDir } from './helpers';

describe('filesystem.list_dir', () => {
  let testDir: string;

  beforeAll(async () => {
    testDir = await setupTestDir({
      files: ['a.txt', 'b.pdf', 'c.md'],
      subdirs: ['subdir'],
    });
  });

  afterAll(async () => {
    await teardownTestDir(testDir);
  });

  it('should list files in a directory', async () => {
    const result = await listDir.execute({ path: testDir, recursive: false });
    expect(result.success).toBe(true);
    expect(result.data).toHaveLength(4); // 3 files + 1 subdir
  });

  it('should include name, path, type, size, modified fields', async () => {
    const result = await listDir.execute({ path: testDir, recursive: false });
    const entry = result.data[0];
    expect(entry).toHaveProperty('name');
    expect(entry).toHaveProperty('path');
    expect(entry).toHaveProperty('type');
    expect(entry).toHaveProperty('size');
    expect(entry).toHaveProperty('modified');
  });

  it('should identify directories vs files', async () => {
    const result = await listDir.execute({ path: testDir, recursive: false });
    const dir = result.data.find((e: { name: string }) => e.name === 'subdir');
    expect(dir?.type).toBe('dir');
    const file = result.data.find((e: { name: string }) => e.name === 'a.txt');
    expect(file?.type).toBe('file');
  });

  it('should support recursive listing', async () => {
    const result = await listDir.execute({ path: testDir, recursive: true });
    expect(result.success).toBe(true);
    // recursive includes the subdir entry itself
    expect(result.data.length).toBeGreaterThanOrEqual(4);
  });

  it('should support glob filter', async () => {
    const result = await listDir.execute({
      path: testDir,
      recursive: false,
      filter: '*.txt',
    });
    expect(result.success).toBe(true);
    expect(result.data).toHaveLength(1);
    expect(result.data[0].name).toBe('a.txt');
  });

  it('should throw for non-existent directory', async () => {
    await expect(
      listDir.execute({ path: testDir + '/nonexistent', recursive: false }),
    ).rejects.toThrow();
  });

  it('has correct metadata', () => {
    expect(listDir.name).toBe('filesystem.list_dir');
    expect(listDir.confirmationRequired).toBe(false);
    expect(listDir.undoSupported).toBe(false);
  });
});
