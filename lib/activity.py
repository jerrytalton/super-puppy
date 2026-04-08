"""Persistent activity logging for Super Puppy.

SQLite-backed request history that survives server restarts. Both the MCP
server and the profile server write to the same DB (WAL mode handles
concurrent access from separate processes).
"""

import sqlite3
import time
from pathlib import Path

from lib.models import ACTIVITY_DB

_PRUNE_DAYS = 90


def _connect() -> sqlite3.Connection:
    ACTIVITY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ACTIVITY_DB), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool TEXT NOT NULL,
            model TEXT NOT NULL,
            backend TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            error_msg TEXT,
            duration_ms INTEGER NOT NULL,
            started_at REAL NOT NULL,
            completed_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_completed_at ON requests(completed_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool ON requests(tool)")
    cutoff = time.time() - (_PRUNE_DAYS * 86400)
    conn.execute("DELETE FROM requests WHERE completed_at < ?", (cutoff,))
    conn.commit()
    conn.close()


def log_request(
    tool: str,
    model: str,
    backend: str,
    source: str,
    status: str,
    duration_ms: int,
    started_at: float,
    completed_at: float,
    error_msg: str | None = None,
) -> None:
    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO requests (tool, model, backend, source, status, error_msg, duration_ms, started_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tool, model, backend, source, status, error_msg, duration_ms, started_at, completed_at),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # never let logging failures break tool execution


def query_activity(period_seconds: int, limit: int = 200) -> dict:
    conn = _connect()
    cutoff = time.time() - period_seconds

    history = [
        dict(r) for r in conn.execute(
            "SELECT tool, model, backend, source, status, error_msg, duration_ms, started_at, completed_at "
            "FROM requests WHERE completed_at > ? ORDER BY completed_at DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    ]

    tool_stats = [
        dict(r) for r in conn.execute(
            "SELECT tool, COUNT(*) as count, CAST(AVG(duration_ms) AS INTEGER) as avg_ms, "
            "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors "
            "FROM requests WHERE completed_at > ? GROUP BY tool ORDER BY count DESC",
            (cutoff,),
        ).fetchall()
    ]

    totals = conn.execute(
        "SELECT COUNT(*) as total, SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors "
        "FROM requests WHERE completed_at > ?",
        (cutoff,),
    ).fetchone()

    conn.close()
    return {
        "history": history,
        "tool_stats": tool_stats,
        "total": totals["total"] or 0,
        "errors": totals["errors"] or 0,
    }
