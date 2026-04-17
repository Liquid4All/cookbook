"""SQLite persistence layer for wildfire predictions."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def init_db(path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the predictions table exists."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            lon         REAL    NOT NULL,
            lat         REAL    NOT NULL,
            timestamp   TEXT    NOT NULL,
            size_km     REAL    NOT NULL,
            source      TEXT    NOT NULL,
            rgb_path    TEXT,
            swir_path   TEXT,
            risk_level  TEXT,
            dry_vegetation_present  INTEGER,
            urban_interface         INTEGER,
            steep_terrain           INTEGER,
            water_body_present      INTEGER,
            image_quality_limited   INTEGER,
            model       TEXT    NOT NULL,
            created_at  TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def insert_prediction(
    conn: sqlite3.Connection,
    lon: float,
    lat: float,
    timestamp: str,
    size_km: float,
    source: str,
    rgb_path: str | None,
    swir_path: str | None,
    prediction: dict[str, object],
    model: str,
) -> int:
    """Insert a prediction row and return the new row id."""
    created_at = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        """
        INSERT INTO predictions (
            lon, lat, timestamp, size_km, source,
            rgb_path, swir_path,
            risk_level, dry_vegetation_present, urban_interface,
            steep_terrain, water_body_present, image_quality_limited,
            model, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            lon, lat, timestamp, size_km, source,
            rgb_path, swir_path,
            prediction.get("risk_level"),
            int(bool(prediction.get("dry_vegetation_present"))),
            int(bool(prediction.get("urban_interface"))),
            int(bool(prediction.get("steep_terrain"))),
            int(bool(prediction.get("water_body_present"))),
            int(bool(prediction.get("image_quality_limited"))),
            model,
            created_at,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)  # type: ignore[arg-type]


def fetch_all(conn: sqlite3.Connection) -> list[dict[str, object]]:
    """Return all predictions as a list of dicts, newest first."""
    cursor = conn.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def fetch_recent(conn: sqlite3.Connection, hours: int) -> list[dict[str, object]]:
    """Return predictions created within the last N hours, newest first."""
    cursor = conn.execute(
        """
        SELECT * FROM predictions
        WHERE created_at >= datetime('now', ?)
        ORDER BY created_at DESC
        """,
        (f"-{hours} hours",),
    )
    return [dict(row) for row in cursor.fetchall()]
