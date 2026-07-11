"""Ensure the repo root is on sys.path so `import app.menubar` works."""

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolate_activity_db(tmp_path, monkeypatch):
    """No test may touch the real ~/.config/local-models/activity.db.

    Points SP_ACTIVITY_DB at a per-test tmp file. lib.activity reads
    lib.models.ACTIVITY_DB at call time (via _connect), so setting the
    env var before the DB is opened is sufficient; modules that cached
    the old path re-resolve through lib.models.ACTIVITY_DB.
    """
    db = tmp_path / "activity.db"
    monkeypatch.setenv("SP_ACTIVITY_DB", str(db))
    import lib.models
    importlib.reload(lib.models)
    import lib.activity
    importlib.reload(lib.activity)
    yield db
