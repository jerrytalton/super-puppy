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
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TOKEN = "fleet-compat-test-token"
FQDN = "test.fqdn"
HOST_HEADER = f"{FQDN}:8100"
PORT = 8199


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
    env = {**os.environ, "MCP_AUTH_TOKEN": TOKEN, "MCP_HOST": "127.0.0.1",
           "MCP_PORT": str(PORT), "MCP_ALLOWED_HOSTS": f"{FQDN}:*"}
    proc = subprocess.Popen(
        ["uv", "run", "mcp/local-models-server.py"],
        cwd=str(server_repo), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Generous: the prev-tag worktree's FIRST `uv run` does a cold dep
        # resolve (downloads the mcp SDK, uvicorn, …). 180s covers a cold cache.
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/api/mcp-models", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.5)
            if proc.poll() is not None:
                raise SystemExit(f"server from {server_repo} exited during startup")
        else:
            raise SystemExit(f"server from {server_repo} not ready within 180s")
        yield
    finally:
        proc.terminate()
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)


def _run_probe(probe_repo: Path) -> int:
    """Run probe_repo's contract probe against the server on PORT."""
    return subprocess.call(
        ["uv", "run", "tests/fleet/contract_probe.py",
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
