"""Smoke test for the fleet compat orchestrator's failure handling."""
import contextlib
import subprocess
from pathlib import Path

import tests.fleet.run_compat as rc


def test_latest_tag_returns_a_tag():
    tag = rc._latest_tag()
    assert tag.startswith("v"), f"expected a vX.Y.Z tag, got {tag!r}"


@contextlib.contextmanager
def _nullctx():
    yield


def test_worktree_is_removed_even_when_a_probe_fails(monkeypatch):
    """A failing probe must not leak a git worktree."""
    removed = {"called": False}
    real_call = subprocess.call

    def fake_call(cmd, *a, **k):
        if cmd[:2] == ["git", "-C"] and "worktree" in cmd and "remove" in cmd:
            removed["called"] = True
            return 0
        return real_call(cmd, *a, **k)

    monkeypatch.setattr(rc, "_run_probe", lambda repo: 1)        # force probe failure
    monkeypatch.setattr(rc, "_server", lambda repo: _nullctx())  # skip real server
    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)  # stub worktree add

    rc_code = rc.main()
    assert rc_code == 1, "gate must fail when a probe fails"
    assert removed["called"], "worktree must be removed even on probe failure"
