/**
 * Audit Database — reads from the Agent Core's `agent.db`.
 *
 * The Agent Core (Rust) writes tool execution entries to the `audit_log`
 * table inside `agent.db`. This MCP server opens that same database in
 * read-only mode so audit tools can query real execution history.
 *
 * Schema (agent core):
 *   id, session_id, timestamp, tool_name, arguments, result,
 *   result_status, user_confirmed, execution_time_ms
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import * as os from 'os';

// ─── Types ──────────────────────────────────────────────────────────────────

/** Matches the agent core's audit_log schema in agent.db. */
export interface AuditEntry {
  id: number;
  session_id: string;
  timestamp: string;
  tool_name: string;
  arguments: string | null;
  result: string | null;
  result_status: string;
  user_confirmed: number;
  execution_time_ms: number;
}

// ─── Database Singleton ─────────────────────────────────────────────────────

let db: Database.Database | null = null;

/** Resolve the agent.db path used by the Tauri Agent Core. */
function resolveAgentDbPath(): string {
  // Explicit override for testing
  if (process.env.LOCALCOWORK_AUDIT_DB) {
    return process.env.LOCALCOWORK_AUDIT_DB;
  }

  // Platform-specific data dirs — must match Tauri's dirs::data_dir()
  const platform = process.platform;

  // macOS: ~/Library/Application Support/com.localcowork.app/agent.db
  if (platform === 'darwin') {
    return path.join(
      os.homedir(),
      'Library',
      'Application Support',
      'com.localcowork.app',
      'agent.db',
    );
  }

  // Windows: %APPDATA%\com.localcowork.app\agent.db
  // Tauri uses dirs::data_dir() which maps to FOLDERID_RoamingAppData on Windows.
  if (platform === 'win32') {
    const appData = process.env.APPDATA ?? path.join(os.homedir(), 'AppData', 'Roaming');
    return path.join(appData, 'com.localcowork.app', 'agent.db');
  }

  // Linux: $XDG_DATA_HOME/com.localcowork.app/agent.db
  if (platform === 'linux') {
    const xdgData = process.env.XDG_DATA_HOME ?? path.join(os.homedir(), '.local', 'share');
    return path.join(xdgData, 'com.localcowork.app', 'agent.db');
  }

  // Fallback for unknown platforms: LOCALCOWORK_DATA_DIR or ~/.localcowork
  const dataDir =
    process.env.LOCALCOWORK_DATA_DIR ?? path.join(os.homedir(), '.localcowork');
  return path.join(dataDir, 'agent.db');
}

const DB_PATH = resolveAgentDbPath();

/** Get the audit database connection (opens agent.db for reading). */
export function getDb(): Database.Database {
  if (db) return db;

  // Open in normal (read-write) mode, NOT readonly.
  // SQLite WAL mode requires shared memory access (agent.db-shm) to read
  // the latest committed data from the WAL. A readonly connection cannot
  // acquire the shared memory lock and will only see data from the last
  // checkpoint — missing all recent writes from the agent core.
  //
  // Opening in read-write mode with WAL is safe for concurrent access:
  // the Rust agent core and this Node.js reader can coexist. We only
  // perform SELECT queries, never INSERT/UPDATE/DELETE.
  db = new Database(DB_PATH);

  // Ensure WAL mode (should already be set by agent core, but harmless to repeat)
  db.pragma('journal_mode = WAL');

  return db;
}

/** Close the database connection. */
export function closeDb(): void {
  if (db) {
    db.close();
    db = null;
  }
}

/**
 * Set a custom database instance (for testing).
 * Creates the agent-core-compatible schema on the provided database.
 */
export function setDb(customDb: Database.Database): void {
  customDb.exec(`
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      last_activity TEXT NOT NULL DEFAULT (datetime('now')),
      summary TEXT,
      files_touched TEXT DEFAULT '[]',
      decisions_made TEXT DEFAULT '[]'
    );

    CREATE TABLE IF NOT EXISTS audit_log (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id        TEXT NOT NULL,
      timestamp         TEXT NOT NULL DEFAULT (datetime('now')),
      tool_name         TEXT NOT NULL,
      arguments         TEXT,
      result            TEXT,
      result_status     TEXT NOT NULL,
      user_confirmed    INTEGER NOT NULL DEFAULT 0,
      execution_time_ms INTEGER NOT NULL DEFAULT 0,
      FOREIGN KEY (session_id) REFERENCES sessions(id)
    );

    CREATE INDEX IF NOT EXISTS idx_audit_session
      ON audit_log(session_id);
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp
      ON audit_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_tool
      ON audit_log(tool_name);
  `);
  db = customDb;
}
