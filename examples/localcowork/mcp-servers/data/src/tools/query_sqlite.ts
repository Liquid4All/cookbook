/**
 * data.query_sqlite — Run a read-only SQL query against a SQLite database.
 *
 * Non-destructive: executes immediately, no confirmation needed.
 * Only SELECT statements are allowed.
 */

import Database from 'better-sqlite3';
import { z } from 'zod';
import type { MCPTool, MCPResult } from '../../../_shared/ts/mcp-base';
import { MCPError, ErrorCodes } from '../../../_shared/ts/mcp-base';
import { assertSandboxed, assertAbsolutePath } from '../../../_shared/ts/validation';

// ─── Params Schema ──────────────────────────────────────────────────────────

const paramsSchema = z.object({
  query: z.string().describe('SQL SELECT query'),
  db_path: z.string().describe('Path to SQLite database'),
});

type Params = z.infer<typeof paramsSchema>;

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Check whether a SQL string is a read-only SELECT query */
function isReadOnlyQuery(sql: string): boolean {
  const normalized = sql.trim().toUpperCase();
  // Must start with SELECT or WITH (for CTEs)
  if (!normalized.startsWith('SELECT') && !normalized.startsWith('WITH')) {
    return false;
  }
  // Reject DML/DDL keywords
  const forbidden = [
    'INSERT',
    'UPDATE',
    'DELETE',
    'DROP',
    'ALTER',
    'CREATE',
    'REPLACE',
    'ATTACH',
    'DETACH',
    'PRAGMA',
  ];
  // Only check outside of string literals (simplified: split on single quotes)
  const outsideStrings = sql.replace(/'[^']*'/g, '').toUpperCase();
  return !forbidden.some((kw) => outsideStrings.includes(kw));
}

// ─── Tool Definition ────────────────────────────────────────────────────────

export const querySqlite: MCPTool<Params> = {
  name: 'data.query_sqlite',
  description: 'Run a read-only SQL query against a SQLite database',
  paramsSchema,
  confirmationRequired: false,
  undoSupported: false,

  async execute(params: Params): Promise<MCPResult> {
    assertAbsolutePath(params.db_path, 'db_path');
    assertSandboxed(params.db_path);

    let db: Database.Database | null = null;

    try {
      if (!isReadOnlyQuery(params.query)) {
        throw new MCPError(
          ErrorCodes.INVALID_PARAMS,
          'Only read-only SELECT queries are allowed. Use data.write_sqlite for mutations.',
        );
      }

      db = new Database(params.db_path, { readonly: true });

      const stmt = db.prepare(params.query);
      const rows = stmt.all() as Record<string, unknown>[];

      // Extract column names from the first row or statement
      const columns = rows.length > 0 ? Object.keys(rows[0]) : [];

      return {
        success: true,
        data: { results: rows, columns },
      };
    } catch (err) {
      if (err instanceof MCPError) throw err;
      const msg = err instanceof Error ? err.message : String(err);
      throw new MCPError(ErrorCodes.INTERNAL_ERROR, `Failed to query SQLite: ${msg}`);
    } finally {
      db?.close();
    }
  },
};
