#!/usr/bin/env python3
"""Fleet cross-version compatibility gate.

Worktrees the previous released tag and checks both directions of version skew:
  - Direction B (always): new client probe  vs  old server  (HEAD probe → prev server)
  - Direction A (when prev has a probe): old client probe vs new server

Exits 0 if every available direction passes; non-zero otherwise. Designed to be
called from bin/release.sh BEFORE a tag is pushed (the fleet auto-pulls tags).
"""
from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TOKEN = "fleet-compat-test-token"
FQDN = "test.fqdn"
HOST_HEADER = f"{FQDN}:8100"
PORT = 8199  # offset from production MCP port (8100) to avoid colliding with a running instance


def _latest_tag() -> str:
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "tag", "--sort=-v:refname"], text=True)
    tags = [t for t in out.split() if t.strip()]
    if not tags:
        raise SystemExit("no existing tags — nothing to check compatibility against")
    return tags[0]


@contextlib.contextmanager
def _server(server_repo: Path):
    """Start that worktree's MCP server on PORT with a correct (FQDN-allowlisted)
    launch env, yield once /api/mcp-models answers, always tear down."""
    # Refuse if PORT is already serving: a leftover server from a crashed prior
    # run would answer the readiness poll, and the probe would run against THAT
    # (wrong-version) server — a false pass, the worst outcome for the gate.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _probe:
        if _probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(
                f"port {PORT} is already in use — a leftover compat server would "
                f"cause a false pass; kill it before re-running the gate")
    env = {**os.environ, "MCP_AUTH_TOKEN": TOKEN, "MCP_HOST": "127.0.0.1",
           "MCP_PORT": str(PORT), "MCP_ALLOWED_HOSTS": f"{FQDN}:*",
           "OLLAMA_URL": "http://127.0.0.1:1", "MLX_URL": "http://127.0.0.1:1"}
    stderr_f = tempfile.NamedTemporaryFile(
        mode="w+", prefix="sp-compat-srv-", suffix=".log", delete=False)
    # --python 3.12 like the production launchers: a bare `uv run` honors any
    # .python-version found walking up from cwd (e.g. a stale alpha pin in
    # $HOME), and the server's deps don't resolve on prerelease interpreters.
    proc = subprocess.Popen(
        ["uv", "run", "--python", "3.12", "mcp/local-models-server.py"],
        cwd=str(server_repo), env=env,
        stdout=subprocess.DEVNULL, stderr=stderr_f)

    def _stderr_tail(n=20):
        try:
            stderr_f.flush()
            with open(stderr_f.name) as fh:
                return "".join(fh.readlines()[-n:]).strip()
        except Exception:
            return "(stderr unavailable)"

    try:
        # Generous: the prev-tag worktree's FIRST `uv run` does a cold dep
        # resolve (downloads the mcp SDK, uvicorn, …). 180s covers a cold cache.
        deadline = time.time() + 180
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                raise SystemExit(
                    f"server from {server_repo} exited during startup "
                    f"(rc={proc.returncode}):\n{_stderr_tail()}")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/api/mcp-models", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(0.5)
        if not ready:
            raise SystemExit(
                f"server from {server_repo} not ready within 180s "
                f"(rc={proc.poll()}):\n{_stderr_tail()}")
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            stderr_f.close()
            os.unlink(stderr_f.name)


def _run_probe(probe_repo: Path) -> int:
    """Run probe_repo's contract probe against the server on PORT."""
    return subprocess.call(
        ["uv", "run", "--python", "3.12", "tests/fleet/contract_probe.py",
         "--base", f"http://127.0.0.1:{PORT}",
         "--token", TOKEN, "--host-header", HOST_HEADER],
        cwd=str(probe_repo))


def main() -> int:
    prev = _latest_tag()
    print(f"== Fleet compat gate: HEAD vs {prev} ==")
    wt = Path(tempfile.mkdtemp(prefix="sp-compat-"))
    failures = []
    try:
        subprocess.check_call(
            ["git", "-C", str(REPO), "worktree", "add", "--detach", str(wt), prev])

        print(f"\n-- Direction B: new client (HEAD) vs old server ({prev}) --")
        with _server(wt):  # old server
            if _run_probe(REPO) != 0:  # HEAD probe
                failures.append("Direction B (HEAD probe vs old server)")

        prev_probe = wt / "tests" / "fleet" / "contract_probe.py"
        if prev_probe.exists():
            print(f"\n-- Direction A: old client ({prev}) vs new server (HEAD) --")
            with _server(REPO):  # new server
                if _run_probe(wt) != 0:  # prev probe
                    failures.append("Direction A (old probe vs new server)")
        else:
            print(f"\n-- Direction A skipped: {prev} predates the contract probe --")
    finally:
        subprocess.call(["git", "-C", str(REPO), "worktree", "remove", "--force", str(wt)])
        shutil.rmtree(wt, ignore_errors=True)

    if failures:
        print("\nFLEET COMPAT GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nfleet compat gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
