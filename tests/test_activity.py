"""Tests for lib/activity.py — persistent activity logging."""

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def tmp_db(tmp_path):
    """Redirect ACTIVITY_DB to a temp path for every test."""
    db_path = tmp_path / "activity.db"
    with patch("lib.activity.ACTIVITY_DB", db_path):
        yield db_path


from lib import activity


class TestInitDb:
    def test_creates_table_and_indexes(self, tmp_db):
        activity.init_db()
        conn = sqlite3.connect(str(tmp_db))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='requests'"
        ).fetchall()
        assert len(tables) == 1
        indexes = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        ]
        assert "idx_completed_at" in indexes
        assert "idx_tool" in indexes
        conn.close()

    def test_idempotent(self, tmp_db):
        activity.init_db()
        activity.init_db()  # should not raise

    def test_prunes_old_rows(self, tmp_db):
        activity.init_db()
        conn = sqlite3.connect(str(tmp_db))
        old_time = time.time() - (91 * 86400)
        conn.execute(
            "INSERT INTO requests (tool, model, backend, source, status, duration_ms, started_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("chat", "llama", "ollama", "mcp", "ok", 100, old_time, old_time),
        )
        conn.commit()
        conn.close()
        activity.init_db()
        conn = sqlite3.connect(str(tmp_db))
        count = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
        assert count == 0
        conn.close()


class TestLogRequest:
    def test_inserts_row(self, tmp_db):
        activity.init_db()
        now = time.time()
        activity.log_request(
            tool="chat", model="llama3", backend="ollama", source="mcp",
            status="ok", duration_ms=1500, started_at=now - 1.5, completed_at=now,
        )
        conn = sqlite3.connect(str(tmp_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM requests").fetchall()
        assert len(rows) == 1
        assert rows[0]["tool"] == "chat"
        assert rows[0]["model"] == "llama3"
        assert rows[0]["backend"] == "ollama"
        assert rows[0]["source"] == "mcp"
        assert rows[0]["status"] == "ok"
        assert rows[0]["duration_ms"] == 1500
        assert rows[0]["error_msg"] is None
        conn.close()

    def test_logs_error_with_message(self, tmp_db):
        activity.init_db()
        now = time.time()
        activity.log_request(
            tool="vision", model="qwen", backend="mlx", source="playground",
            status="error", duration_ms=500, started_at=now - 0.5, completed_at=now,
            error_msg="model not found",
        )
        conn = sqlite3.connect(str(tmp_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM requests").fetchone()
        assert row["status"] == "error"
        assert row["error_msg"] == "model not found"
        conn.close()

    def test_never_raises(self, tmp_db):
        # Even without init_db, log_request should silently fail
        activity.log_request(
            tool="chat", model="x", backend="ollama", source="mcp",
            status="ok", duration_ms=100, started_at=0, completed_at=0,
        )


class TestQueryActivity:
    def _seed(self, tmp_db, count=5, tool="chat", status="ok", age_offset=0):
        conn = sqlite3.connect(str(tmp_db))
        now = time.time()
        for i in range(count):
            t = now - age_offset - i
            conn.execute(
                "INSERT INTO requests (tool, model, backend, source, status, duration_ms, started_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tool, "llama3", "ollama", "mcp", status, 1000 + i * 100, t - 1, t),
            )
        conn.commit()
        conn.close()

    def test_returns_history_in_period(self, tmp_db):
        activity.init_db()
        self._seed(tmp_db, count=3)
        result = activity.query_activity(3600)
        assert result["total"] == 3
        assert len(result["history"]) == 3
        # Newest first
        assert result["history"][0]["completed_at"] >= result["history"][1]["completed_at"]

    def test_excludes_old_history(self, tmp_db):
        activity.init_db()
        self._seed(tmp_db, count=3, age_offset=7200)  # 2 hours old
        result = activity.query_activity(3600)  # 1 hour window
        assert result["total"] == 0
        assert len(result["history"]) == 0

    def test_tool_stats_with_avg(self, tmp_db):
        activity.init_db()
        self._seed(tmp_db, count=3, tool="chat")
        self._seed(tmp_db, count=2, tool="vision")
        result = activity.query_activity(3600)
        assert result["total"] == 5
        stats_by_tool = {s["tool"]: s for s in result["tool_stats"]}
        assert stats_by_tool["chat"]["count"] == 3
        assert stats_by_tool["vision"]["count"] == 2
        assert stats_by_tool["chat"]["avg_ms"] > 0

    def test_error_count(self, tmp_db):
        activity.init_db()
        self._seed(tmp_db, count=3, status="ok")
        self._seed(tmp_db, count=2, status="error")
        result = activity.query_activity(3600)
        assert result["total"] == 5
        assert result["errors"] == 2

    def test_tool_stats_sorted_by_count(self, tmp_db):
        activity.init_db()
        self._seed(tmp_db, count=5, tool="chat")
        self._seed(tmp_db, count=1, tool="tts")
        self._seed(tmp_db, count=3, tool="vision")
        result = activity.query_activity(3600)
        tools = [s["tool"] for s in result["tool_stats"]]
        assert tools == ["chat", "vision", "tts"]

    def test_respects_limit(self, tmp_db):
        activity.init_db()
        self._seed(tmp_db, count=10)
        result = activity.query_activity(3600, limit=3)
        assert len(result["history"]) == 3
        assert result["total"] == 10  # total count unaffected by limit

    def test_empty_db(self, tmp_db):
        activity.init_db()
        result = activity.query_activity(3600)
        assert result["total"] == 0
        assert result["errors"] == 0
        assert result["history"] == []
        assert result["tool_stats"] == []
