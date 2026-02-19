import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import Database from 'better-sqlite3';
import * as path from 'path';
import { querySqlite } from '../src/tools/query_sqlite';
import { setupTestDir, teardownTestDir } from './helpers';

describe('data.query_sqlite', () => {
  let testDir: string;
  let dbPath: string;

  beforeAll(() => {
    testDir = setupTestDir();
    dbPath = path.join(testDir, 'query-test.db');

    // Seed a database
    const db = new Database(dbPath);
    db.exec(`
      CREATE TABLE products (id INTEGER, name TEXT, price REAL);
      INSERT INTO products VALUES (1, 'Widget', 9.99);
      INSERT INTO products VALUES (2, 'Gadget', 24.99);
      INSERT INTO products VALUES (3, 'Doohickey', 4.50);
    `);
    db.close();
  });

  afterAll(() => {
    teardownTestDir(testDir);
  });

  it('should execute a SELECT query', async () => {
    const result = await querySqlite.execute({
      query: 'SELECT * FROM products ORDER BY id',
      db_path: dbPath,
    });

    expect(result.success).toBe(true);
    expect(result.data.results).toHaveLength(3);
    expect(result.data.columns).toEqual(['id', 'name', 'price']);
    expect(result.data.results[0]).toEqual({ id: 1, name: 'Widget', price: 9.99 });
  });

  it('should support WHERE clauses', async () => {
    const result = await querySqlite.execute({
      query: "SELECT name FROM products WHERE price > 10",
      db_path: dbPath,
    });

    expect(result.success).toBe(true);
    expect(result.data.results).toHaveLength(1);
    expect(result.data.results[0]).toEqual({ name: 'Gadget' });
  });

  it('should reject non-SELECT queries', async () => {
    await expect(
      querySqlite.execute({
        query: "INSERT INTO products VALUES (4, 'Thingamajig', 15.00)",
        db_path: dbPath,
      }),
    ).rejects.toThrow('read-only SELECT');

    await expect(
      querySqlite.execute({
        query: 'DROP TABLE products',
        db_path: dbPath,
      }),
    ).rejects.toThrow('read-only SELECT');
  });

  it('should allow WITH (CTE) queries', async () => {
    const result = await querySqlite.execute({
      query: 'WITH cheap AS (SELECT * FROM products WHERE price < 10) SELECT name FROM cheap',
      db_path: dbPath,
    });

    expect(result.success).toBe(true);
    expect(result.data.results).toHaveLength(2);
  });

  it('should return empty results for no matches', async () => {
    const result = await querySqlite.execute({
      query: "SELECT * FROM products WHERE price > 1000",
      db_path: dbPath,
    });

    expect(result.success).toBe(true);
    expect(result.data.results).toHaveLength(0);
    expect(result.data.columns).toEqual([]);
  });

  it('has correct metadata', () => {
    expect(querySqlite.name).toBe('data.query_sqlite');
    expect(querySqlite.confirmationRequired).toBe(false);
    expect(querySqlite.undoSupported).toBe(false);
  });
});
