"""Ensure the repo root is on sys.path so `import app.menubar` works."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _isolate_activity_db(tmp_path, monkeypatch):
    """No test may touch the real ~/.config/local-models/activity.db.

    Points SP_ACTIVITY_DB at a per-test tmp file, then reloads lib.models
    (which re-resolves ACTIVITY_DB from the env var) and lib.activity
    (which binds ACTIVITY_DB by value at import via `from lib.models import
    ACTIVITY_DB` — so the reload is what actually redirects it, NOT any
    call-time lookup; do not drop the lib.activity reload).
    """
    db = tmp_path / "activity.db"
    monkeypatch.setenv("SP_ACTIVITY_DB", str(db))
    import lib.models
    importlib.reload(lib.models)
    import lib.activity
    importlib.reload(lib.activity)
    yield db
