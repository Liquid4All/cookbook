/**
 * data.write_sqlite — Write structured data to a SQLite table.
 *
 * Mutable: requires user confirmation (modifies a database).
 * Creates the table if it does not exist, auto-detecting column types.
 */

import Database from 'better-sqlite3';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  data: z
    .array(z.record(z.string(), z.unknown()))
    .min(1)
    .describe('Array of row objects'),
  table: z
    .string()
    .regex(/^[a-zA-Z_]\w{0,63}$/, 'Invalid table name')
    .describe('Table name'),
  db_path: z.string().describe('Path to SQLite database'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Infer SQLite column type from a JS value */
function inferSqliteType(value: unknown): string {
  if (typeof value === 'number') {
    return Number.isInteger(value) ? 'INTEGER' : 'REAL';
  }
  if (typeof value === 'boolean') return 'INTEGER';
  return 'TEXT';
}

/** Infer SQLite column type by scanning all rows (skips null/undefined) */
function inferColumnType(data: Record<string, unknown>[], col: string): string {
  for (const row of data) {
    const v = row[col];
    if (v !== null && v !== undefined) {
      return inferSqliteType(v);
    }
  }
  return 'TEXT'; // Default if all null
}

/** Validate table name against SQL injection */
function validateTableName(name: string): void {
  if (!/^[a-zA-Z_]\w{0,63}$/.test(name)) {
    throw new MCPError(
      ErrorCodes.INVALID_PARAMS,
      `Invalid table name: "${name}". Must start with a letter or underscore and contain only word characters.`,
    );
  }
}

/** Sanitize a column name for use in SQL */
function sanitizeColumnName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, '_');
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const writeSqlite: MCPTool<Params> = {
  name: 'data.write_sqlite',
  description: 'Write structured data to a SQLite table',
  paramsSchema,
  confirmationRequired: true,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.db_path, 'db_path');
    assertSandboxed(params.db_path);
    validateTableName(params.table);

    let db: Database.Database | null = null;

    try {
      const firstRow = params.data[0] ?? {};
      const columns = Object.keys(firstRow);

      if (columns.length === 0) {
        throw new MCPError(ErrorCodes.INVALID_PARAMS, 'No columns detected in data');
      }

      db = new Database(params.db_path);
      db.pragma('journal_mode = WAL');

      // Create table if not exists — infer types by scanning all rows
      const colDefs = columns.map((col) => {
        const safeCol = sanitizeColumnName(col);
        const sqlType = inferColumnType(params.data, col);
        return `"${safeCol}" ${sqlType}`;
      });

      db.exec(`CREATE TABLE IF NOT EXISTS "${params.table}" (${colDefs.join(', ')})`);

      // Prepare INSERT statement
      const safeCols = columns.map(sanitizeColumnName);
      const placeholders = safeCols.map(() => '?').join(', ');
      const sql = `INSERT INTO "${params.table}" (${safeCols.map((c) => `"${c}"`).join(', ')}) VALUES (${placeholders})`;

      const insert = db.prepare(sql);

      // Insert all rows in a transaction
      const insertMany = db.transaction((rows: Record<string, unknown>[]) => {
        for (const row of rows) {
          const values = columns.map((col) => {
            const v = row[col];
            if (v === null || v === undefined) return null;
            if (typeof v === 'boolean') return v ? 1 : 0;
            return v;
          });
          insert.run(...values);
        }
      });

      insertMany(params.data);

      return {
        success: true,
        data: { rows_inserted: params.data.length },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to write SQLite: ${msg}`);
    } finally {
      db?.close();
    }
  },
};
