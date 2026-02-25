"""
Knowledge server database — SQLite schema and connection management.

Manages the documents and chunks tables used for the RAG pipeline.
WAL mode is enabled for concurrent reads during indexing.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Final

# ─── Constants ────────────────────────────────────────────────────────────────

_DB_FILENAME: Final[str] = "knowledge.db"

# ─── Module-level connection ─────────────────────────────────────────────────

_connection: sqlite3.Connection | None = None


def _data_dir() -> Path:
    """Return the platform-standard data directory (injected by Tauri host)."""
    env_dir = os.environ.get("LOCALCOWORK_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".localcowork"


def _default_db_path() -> str:
    """Return the default database path under the data directory."""
    db_dir = _data_dir()
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / _DB_FILENAME)


def get_db(db_path: str | None = None) -> sqlite3.Connection:
    """
    Get or create the SQLite database connection.

    If a connection already exists, return it. Otherwise, open a new one
    at ``db_path`` (or the default location) and initialise the schema.
    """
    global _connection

    if _connection is not None:
        return _connection

    path = db_path or _default_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    _connection = conn
    return _connection


def set_db(conn: sqlite3.Connection) -> None:
    """
    Inject an externally created connection (useful for testing with :memory:).
    The caller is responsible for calling ``_init_schema`` if needed.
    """
    global _connection
    _connection = conn


def close_db() -> None:
    """Close the current connection if open."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


# ─── Schema ──────────────────────────────────────────────────────────────────


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they don't already exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            path        TEXT    NOT NULL UNIQUE,
            filename    TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            file_hash   TEXT    NOT NULL,
            indexed_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            content      TEXT    NOT NULL,
            chunk_index  INTEGER NOT NULL,
            embedding    BLOB,
            UNIQUE(document_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
        CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
        """
    )
    conn.commit()
