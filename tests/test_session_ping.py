import os
import sqlite3
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PING = REPO / "bin" / "sp-session-ping"


def _run(agent, db):
    env = {**os.environ, "SP_ACTIVITY_DB": str(db)}
    return subprocess.run([str(PING), agent], env=env, capture_output=True, text=True)


def test_ping_inserts_session_row(tmp_path):
    db = tmp_path / "activity.db"
    r = _run("claude-code", db)
    assert r.returncode == 0
    rows = list(sqlite3.connect(str(db)).execute(
        "SELECT tool, source, model FROM requests"))
    assert rows == [("session", "session", "claude-code")]


def test_ping_rejects_bad_agent_falls_back(tmp_path):
    db = tmp_path / "activity.db"
    r = _run("x'); DROP TABLE requests;--", db)
    assert r.returncode == 0
    # table still exists and holds one clean row with the default agent
    rows = list(sqlite3.connect(str(db)).execute("SELECT model FROM requests"))
    assert rows == [("claude-code",)]


def test_ping_never_fails_on_locked_db(tmp_path):
    # A non-writable path must not crash the hook.
    r = _run("claude-code", tmp_path / "nonexistent-dir" / "x.db")
    assert r.returncode == 0


def test_ping_row_matches_library_schema(tmp_path):
    """The hand-rolled INSERT must stay compatible with lib.activity's schema."""
    import importlib
    db_path = tmp_path / "activity.db"
    os.environ["SP_ACTIVITY_DB"] = str(db_path)
    import lib.models
    import lib.activity as act
    importlib.reload(lib.models)
    importlib.reload(act)
    act.init_db()  # library creates the table
    _run("claude-code", db_path)  # script inserts into it
    data = act.query_activity(3600)
    assert data["sessions"] == 1
    del os.environ["SP_ACTIVITY_DB"]
