/**
 * LocalCowork — Performance Benchmarks
 *
 * Quick performance tests that verify key filesystem and data operations
 * complete within acceptable time bounds on the current platform.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { createTestTempDir, cleanupTestDir } from './platform-helpers';
import { initSandbox } from '../../mcp-servers/_shared/ts/validation';
import { listDir } from '../../mcp-servers/filesystem/src/tools/list_dir';
import { readFile } from '../../mcp-servers/filesystem/src/tools/read_file';
import { searchFiles } from '../../mcp-servers/filesystem/src/tools/search_files';
import { writeCsv } from '../../mcp-servers/data/src/tools/write_csv';

let benchDir: string;

beforeAll(async () => {
  benchDir = await createTestTempDir('bench');
  initSandbox([benchDir, os.tmpdir(), '/private/var/folders', '/private/tmp', '/tmp']);
});

afterAll(async () => {
  await cleanupTestDir(benchDir);
});

// --- Helper: time an async operation ---

async function timeMs(fn: () => Promise<unknown>): Promise<number> {
  const start = performance.now();
  await fn();
  return performance.now() - start;
}

// --- 1. File List Speed ---

describe('File List Speed', () => {
  it('list_dir on 100 files completes in < 500ms', async () => {
    // Create 100 files
    const listTestDir = path.join(benchDir, 'list-speed');
    await fs.mkdir(listTestDir, { recursive: true });
    const writePromises: Promise<void>[] = [];
    for (let i = 0; i < 100; i++) {
      writePromises.push(
        fs.writeFile(path.join(listTestDir, `file-${String(i).padStart(3, '0')}.txt`), `content ${i}`, 'utf-8'),
      );
    }
    await Promise.all(writePromises);

    const elapsed = await timeMs(async () => {
      await listDir.execute({ path: listTestDir, recursive: false });
    });

    expect(elapsed).toBeLessThan(500);
  });
});

// --- 2. File Read Speed ---

describe('File Read Speed', () => {
  it('read_file on a 1MB file completes in < 200ms', async () => {
    const bigFilePath = path.join(benchDir, 'bigfile-1mb.txt');
    // Create a ~1MB file
    const content = 'A'.repeat(1024 * 1024);
    await fs.writeFile(bigFilePath, content, 'utf-8');

    const elapsed = await timeMs(async () => {
      await readFile.execute({ path: bigFilePath, encoding: 'utf-8' });
    });

    expect(elapsed).toBeLessThan(200);
  });
});

// --- 3. CSV Write Speed ---

describe('CSV Write Speed', () => {
  it('write_csv with 1000 rows completes in < 500ms', async () => {
    const csvPath = path.join(benchDir, 'bench-output.csv');
    const rows: Record<string, unknown>[] = [];
    for (let i = 0; i < 1000; i++) {
      rows.push({
        id: i,
        name: `item-${i}`,
        value: Math.random() * 1000,
        category: ['A', 'B', 'C'][i % 3],
      });
    }

    const elapsed = await timeMs(async () => {
      await writeCsv.execute({
        data: rows,
        output_path: csvPath,
        headers: ['id', 'name', 'value', 'category'],
      });
    });

    expect(elapsed).toBeLessThan(500);
  });
});

// --- 4. SQLite Query Speed ---

describe('SQLite Query Speed', () => {
  it('query_sqlite with 1000 rows completes in < 500ms', async () => {
    // Use better-sqlite3 directly since query_sqlite requires an existing DB
    const Database = (await import('better-sqlite3')).default;
    const dbPath = path.join(benchDir, 'bench.db');
    const db = new Database(dbPath);

    db.exec('CREATE TABLE bench (id INTEGER PRIMARY KEY, name TEXT, value REAL)');
    const insert = db.prepare('INSERT INTO bench (name, value) VALUES (?, ?)');
    const insertMany = db.transaction((rows: Array<{ name: string; value: number }>) => {
      for (const row of rows) {
        insert.run(row.name, row.value);
      }
    });

    const rows: Array<{ name: string; value: number }> = [];
    for (let i = 0; i < 1000; i++) {
      rows.push({ name: `item-${i}`, value: Math.random() * 1000 });
    }
    insertMany(rows);
    db.close();

    // Now query via tool
    const { querySqlite } = await import('../../mcp-servers/data/src/tools/query_sqlite');

    const elapsed = await timeMs(async () => {
      await querySqlite.execute({
        query: 'SELECT * FROM bench WHERE value > 500',
        db_path: dbPath,
      });
    });

    expect(elapsed).toBeLessThan(500);
  });
});

// --- 5. Search Speed ---

describe('Search Speed', () => {
  it('search_files across 50 files completes in < 1000ms', async () => {
    const searchDir = path.join(benchDir, 'search-speed');
    await fs.mkdir(searchDir, { recursive: true });
    const writePromises: Promise<void>[] = [];
    for (let i = 0; i < 50; i++) {
      const ext = i % 3 === 0 ? '.ts' : i % 3 === 1 ? '.txt' : '.md';
      writePromises.push(
        fs.writeFile(
          path.join(searchDir, `search-file-${String(i).padStart(3, '0')}${ext}`),
          `content for file ${i}\n`,
          'utf-8',
        ),
      );
    }
    await Promise.all(writePromises);

    const elapsed = await timeMs(async () => {
      await searchFiles.execute({
        path: searchDir,
        pattern: '*.ts',
        max_results: 100,
      });
    });

    expect(elapsed).toBeLessThan(1000);
  });
});
