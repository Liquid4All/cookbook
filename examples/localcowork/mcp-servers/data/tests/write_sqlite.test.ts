import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import Database from 'better-sqlite3';
import * as path from 'path';
import { writeSqlite } from '../src/tools/write_sqlite';
import { setupTestDir, teardownTestDir } from './helpers';

describe('data.write_sqlite', () => {
  let testDir: string;

  beforeAll(() => {
    testDir = setupTestDir();
  });

  afterAll(() => {
    teardownTestDir(testDir);
  });

  it('should create table and insert rows', async () => {
    const dbPath = path.join(testDir, 'test.db');
    const result = await writeSqlite.execute({
      data: [
        { name: 'Alice', age: 30 },
        { name: 'Bob', age: 25 },
      ],
      table: 'users',
      db_path: dbPath,
    });

    expect(result.success).toBe(true);
    expect(result.data).toEqual({ rows_inserted: 2 });

    // Verify data in DB
    const db = new Database(dbPath, { readonly: true });
    const rows = db.prepare('SELECT * FROM users').all() as { name: string; age: number }[];
    expect(rows).toHaveLength(2);
    expect(rows[0].name).toBe('Alice');
    expect(rows[1].age).toBe(25);
    db.close();
  });

  it('should handle null values', async () => {
    const dbPath = path.join(testDir, 'nulls.db');
    const result = await writeSqlite.execute({
      data: [
        { name: 'Alice', score: null },
        { name: 'Bob', score: 95 },
      ],
      table: 'scores',
      db_path: dbPath,
    });

    expect(result.success).toBe(true);

    const db = new Database(dbPath, { readonly: true });
    const rows = db.prepare('SELECT * FROM scores').all() as { name: string; score: number | null }[];
    expect(rows[0].score).toBeNull();
    expect(rows[1].score).toBe(95);
    db.close();
  });

  it('should handle boolean values', async () => {
    const dbPath = path.join(testDir, 'booleans.db');
    await writeSqlite.execute({
      data: [{ name: 'Alice', active: true }, { name: 'Bob', active: false }],
      table: 'flags',
      db_path: dbPath,
    });

    const db = new Database(dbPath, { readonly: true });
    const rows = db.prepare('SELECT * FROM flags').all() as { name: string; active: number }[];
    expect(rows[0].active).toBe(1);
    expect(rows[1].active).toBe(0);
    db.close();
  });

  it('should reject invalid table names', async () => {
    const dbPath = path.join(testDir, 'bad.db');
    await expect(
      writeSqlite.execute({
        data: [{ x: 1 }],
        table: 'DROP TABLE users; --',
        db_path: dbPath,
      }),
    ).rejects.toThrow();
  });

  it('has correct metadata', () => {
    expect(writeSqlite.name).toBe('data.write_sqlite');
    expect(writeSqlite.confirmationRequired).toBe(true);
    expect(writeSqlite.undoSupported).toBe(false);
  });
});
