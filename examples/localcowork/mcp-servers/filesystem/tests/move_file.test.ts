import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as fs from 'fs/promises';
import * as path from 'path';
import { moveFile } from '../src/tools/move_file';
import { setupTestDir, teardownTestDir } from './helpers';

describe('filesystem.move_file', () => {
  let testDir: string;

  beforeAll(async () => {
    testDir = await setupTestDir({ files: ['source.txt'] });
  });

  afterAll(async () => {
    await teardownTestDir(testDir);
  });

  it('should move a file', async () => {
    const src = path.join(testDir, 'source.txt');
    const dest = path.join(testDir, 'moved.txt');

    const result = await moveFile.execute({ source: src, destination: dest });
    expect(result.success).toBe(true);
    expect(result.data.original_path).toBe(src);
    expect(result.data.new_path).toBe(dest);

    // Source should no longer exist
    await expect(fs.access(src)).rejects.toThrow();
    // Destination should exist
    const content = await fs.readFile(dest, 'utf-8');
    expect(content).toBe('content of source.txt');
  });

  it('should create parent directories when create_dirs is true', async () => {
    // Write a fresh file to move
    const src = path.join(testDir, 'to-move.txt');
    await fs.writeFile(src, 'data', 'utf-8');

    const dest = path.join(testDir, 'deep', 'nested', 'moved.txt');
    const result = await moveFile.execute({
      source: src,
      destination: dest,
      create_dirs: true,
    });

    expect(result.success).toBe(true);
    const content = await fs.readFile(dest, 'utf-8');
    expect(content).toBe('data');
  });

  it('should throw for non-existent source', async () => {
    await expect(
      moveFile.execute({
        source: path.join(testDir, 'nope.txt'),
        destination: path.join(testDir, 'dest.txt'),
      }),
    ).rejects.toThrow();
  });

  it('has correct metadata', () => {
    expect(moveFile.name).toBe('filesystem.move_file');
    expect(moveFile.confirmationRequired).toBe(true);
    expect(moveFile.undoSupported).toBe(true);
  });
});
