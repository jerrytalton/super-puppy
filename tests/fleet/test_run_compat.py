"""Smoke test for the fleet compat orchestrator's failure handling."""
import contextlib
import http.server
import subprocess
import threading

import pytest

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


class _FakeMcp(http.server.BaseHTTPRequestHandler):
    """Stands in for a stale MCP server left on PORT by a crashed prior run:
    answers 200 on /api/mcp-models so a naive readiness poll thinks it's ready."""

    def do_GET(self):
        body = b'{"models": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # silence the test server
        pass


def test_server_refuses_when_port_is_already_in_use():
    """If something is already serving on PORT (a leftover server from a crashed
    run), _server() must refuse — adopting it would run the probe against the
    wrong version and false-pass. The fix is a pre-flight port check; without it,
    the readiness poll gets 200 from the squatter and yields a false 'ready'."""
    squatter = http.server.HTTPServer(("127.0.0.1", rc.PORT), _FakeMcp)
    t = threading.Thread(target=squatter.serve_forever, daemon=True)
    t.start()
    try:
        with pytest.raises(SystemExit):
            with rc._server(rc.REPO):
                pass
    finally:
        squatter.shutdown()
        squatter.server_close()
