import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as fs from 'fs/promises';
import * as path from 'path';
import { writeFile } from '../src/tools/write_file';
import { setupTestDir, teardownTestDir } from './helpers';

describe('filesystem.write_file', () => {
  let testDir: string;

  beforeAll(async () => {
    testDir = await setupTestDir();
  });

  afterAll(async () => {
    await teardownTestDir(testDir);
  });

  it('should write content to a new file', async () => {
    const filePath = path.join(testDir, 'output.txt');
    const result = await writeFile.execute({
      path: filePath,
      content: 'hello world',
    });

    expect(result.success).toBe(true);
    expect(result.data.path).toBe(filePath);

    const contents = await fs.readFile(filePath, 'utf-8');
    expect(contents).toBe('hello world');
  });

  it('should create parent directories', async () => {
    const filePath = path.join(testDir, 'nested', 'deep', 'file.txt');
    const result = await writeFile.execute({
      path: filePath,
      content: 'nested content',
    });

    expect(result.success).toBe(true);
    const contents = await fs.readFile(filePath, 'utf-8');
    expect(contents).toBe('nested content');
  });

  it('should overwrite existing file', async () => {
    const filePath = path.join(testDir, 'overwrite.txt');
    await writeFile.execute({ path: filePath, content: 'version 1' });
    await writeFile.execute({ path: filePath, content: 'version 2' });

    const contents = await fs.readFile(filePath, 'utf-8');
    expect(contents).toBe('version 2');
  });

  it('has correct metadata', () => {
    expect(writeFile.name).toBe('filesystem.write_file');
    expect(writeFile.confirmationRequired).toBe(true);
    expect(writeFile.undoSupported).toBe(false);
  });
});
