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
_JUNK_TOOLS = ("test", "first_task", "second_task", "a", "b", "c",
               "failing", "test_tool", "task1", "task2")
_JUNK_LIKE = ("task\\_%",)  # matches task_0, task_1, ...


def _connect() -> sqlite3.Connection:
    ACTIVITY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ACTIVITY_DB), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def prune_junk_rows() -> None:
    """Delete rows with test-pollution tool names (one-time cleanup)."""
    conn = _connect()
    placeholders = ",".join("?" * len(_JUNK_TOOLS))
    conn.execute(f"DELETE FROM requests WHERE tool IN ({placeholders})", _JUNK_TOOLS)
    for pattern in _JUNK_LIKE:
        conn.execute("DELETE FROM requests WHERE tool LIKE ? ESCAPE '\\'", (pattern,))
    conn.commit()
    conn.close()


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
            completed_at REAL NOT NULL,
            machine TEXT NOT NULL DEFAULT ''
        )
    """)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
        if "machine" not in cols:
            conn.execute("ALTER TABLE requests ADD COLUMN machine TEXT NOT NULL DEFAULT ''")
        conn.execute("PRAGMA user_version = 1")
    if version < 2:
        conn.commit()
        conn.close()
        prune_junk_rows()
        conn = _connect()
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
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
    machine: str = "",
) -> None:
    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO requests (tool, model, backend, source, status, error_msg, "
            "duration_ms, started_at, completed_at, machine) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tool, model, backend, source, status, error_msg, duration_ms,
             started_at, completed_at, machine),
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
            "SELECT tool, model, backend, source, status, error_msg, duration_ms, started_at, completed_at, machine "
            "FROM requests WHERE completed_at > ? AND source != 'session' ORDER BY completed_at DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    ]

    tool_stats = [
        dict(r) for r in conn.execute(
            "SELECT tool, COUNT(*) as count, CAST(AVG(duration_ms) AS INTEGER) as avg_ms, "
            "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors "
            "FROM requests WHERE completed_at > ? AND source != 'session' GROUP BY tool ORDER BY count DESC",
            (cutoff,),
        ).fetchall()
    ]

    totals = conn.execute(
        "SELECT COUNT(*) as total, SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors "
        "FROM requests WHERE completed_at > ? AND source != 'session'",
        (cutoff,),
    ).fetchone()

    sessions = conn.execute(
        "SELECT COUNT(*) AS c FROM requests WHERE completed_at > ? AND source='session'",
        (cutoff,),
    ).fetchone()["c"]

    conn.close()
    return {
        "history": history,
        "tool_stats": tool_stats,
        "total": totals["total"] or 0,
        "errors": totals["errors"] or 0,
        "sessions": sessions or 0,
    }


def last_activity_at() -> float | None:
    conn = _connect()
    row = conn.execute(
        "SELECT MAX(completed_at) AS m FROM requests WHERE source != 'session'"
    ).fetchone()
    conn.close()
    return row["m"]
