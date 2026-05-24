# Fleet Compatibility Test Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pre-tag release gate that refuses to ship a Super Puppy tag which breaks client↔server wire compatibility with the previously released version.

**Architecture:** An executable consumer-contract script (`contract_probe.py`, stdlib-only) defines the cross-machine wire. An orchestrator (`run_compat.py`) uses `git worktree` to materialize the previous tag, starts each version's MCP server via `uv run`, and replays the other version's probe against it in both directions of skew. A `bin/release.sh` runs this gate (plus the normal suite + tag signing/verification) before pushing a tag — the gate must run pre-push because the fleet auto-pulls tags within ~2 minutes.

**Tech Stack:** Python 3.12 (stdlib `urllib`/`json` for the probe), `uv` (PEP 723 per-version deps), `git worktree`, bash, pytest, SSH-signed git tags.

**Spec:** `docs/superpowers/specs/2026-05-24-fleet-compat-test-gate-design.md`

---

## File Structure

- `tests/fleet/contract_probe.py` — **Create.** Stdlib-only executable consumer contract. Makes the canonical SP client requests against a server and exits 0 (green) or non-zero (with a diff). Each tag carries its own copy.
- `tests/fleet/run_compat.py` — **Create.** Orchestrator: worktree the prev tag, start servers, run probes both directions, teardown, aggregate exit code.
- `tests/fleet/test_run_compat.py` — **Create.** Smoke test for the orchestrator (teardown-on-failure; an incompatible stub fails the gate).
- `bin/release.sh` — **Create.** The release path: preconditions → suite → compat gate → sign tag → verify against `allowed_signers` → push.
- `docs/RELEASING.md` — **Create.** How to cut a release.
- `CLAUDE.md` — **Modify.** Add a one-line pointer to `bin/release.sh` / `docs/RELEASING.md` under testing/release.

---

## Reusable: starting a server for probe development

Several tasks start a local MCP server to exercise the probe. Two configs:

**Correct server** (FQDN allowlisted — probe should PASS):
```bash
MCP_AUTH_TOKEN=test-token MCP_HOST=127.0.0.1 MCP_PORT=8199 \
  MCP_ALLOWED_HOSTS='test.fqdn:*' uv run mcp/local-models-server.py &
SRV=$!
until curl -fsS http://127.0.0.1:8199/api/mcp-models >/dev/null 2>&1; do sleep 0.5; done
```

**Broken server** (no allowlist — FQDN Host gets 421, probe should FAIL):
```bash
MCP_AUTH_TOKEN=test-token MCP_HOST=127.0.0.1 MCP_PORT=8199 \
  uv run mcp/local-models-server.py &
SRV=$!
until curl -fsS http://127.0.0.1:8199/api/mcp-models >/dev/null 2>&1; do sleep 0.5; done
```

Always tear down with `kill $SRV 2>/dev/null` between runs. Ollama/MLX do **not** need to be running — the server starts with an empty model list (discovery has bounded per-call timeouts).

---

## Task 1: Contract probe — the `/mcp` FQDN-Host contract (the 421 path)

**Files:**
- Create: `tests/fleet/contract_probe.py`

- [ ] **Step 1: Write the probe skeleton + the FQDN-Host assertion**

```python
#!/usr/bin/env python3
"""Executable cross-machine wire contract for Super Puppy.

Run against an SP MCP/profile server to assert it honors the requests SP
clients make. The fleet compat gate (tests/fleet/run_compat.py) runs one tag's
copy of this probe against another tag's server, both directions of skew.

stdlib ONLY — must import from any tagged worktree without installing deps.

Exit 0 if every check passes; non-zero with a printed list of failures.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request


class ContractFailure(Exception):
    pass


_INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2024-11-05", "capabilities": {},
               "clientInfo": {"name": "contract-probe", "version": "1"}},
}


def _post(base, path, token, host_header, body, timeout=15):
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json, text/event-stream")
    if host_header:
        req.add_header("Host", host_header)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def check_mcp_fqdn_host(base, token, host_header):
    """A bearer-authed initialize whose Host is the Tailscale FQDN must NOT 421.
    This is the exact contract the 421 'Invalid Host header' incident broke."""
    status, _, body = _post(base, "/mcp", token, host_header, _INIT)
    if status == 421:
        raise ContractFailure(
            f"/mcp initialize with Host={host_header} returned 421 "
            f"(Invalid Host header) — server rejects the Tailscale FQDN clients use")
    if status != 200:
        raise ContractFailure(
            f"/mcp initialize with Host={host_header} expected 200, got {status}; "
            f"body={body[:160]!r}")


CHECKS = [
    ("mcp_fqdn_host", lambda a: check_mcp_fqdn_host(a.base, a.token, a.host_header)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8199")
    p.add_argument("--token", required=True)
    p.add_argument("--host-header", required=True, help="e.g. test.fqdn:8100")
    args = p.parse_args()

    failures = []
    for name, fn in CHECKS:
        try:
            fn(args)
            print(f"PASS {name}")
        except ContractFailure as e:
            failures.append((name, str(e)))
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001 — surface unexpected probe errors as failures
            failures.append((name, f"unexpected error: {e}"))
            print(f"FAIL {name}: unexpected error: {e}")

    if failures:
        print(f"\n{len(failures)} contract check(s) failed", file=sys.stderr)
        sys.exit(1)
    print("\nall contract checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe against a BROKEN server, verify it FAILS**

Start the **broken server** (snippet above), then:
```bash
uv run tests/fleet/contract_probe.py --base http://127.0.0.1:8199 \
  --token test-token --host-header test.fqdn:8100
```
Expected: `FAIL mcp_fqdn_host: ... returned 421 ...`, exit code 1. Then `kill $SRV`.

- [ ] **Step 3: Run the probe against a CORRECT server, verify it PASSES**

Start the **correct server** (snippet above), then run the same probe command.
Expected: `PASS mcp_fqdn_host`, `all contract checks passed`, exit 0. Then `kill $SRV`.

- [ ] **Step 4: Commit**

```bash
git add tests/fleet/contract_probe.py
git commit -m "test(fleet): contract probe for the /mcp Tailscale-Host path"
```

---

## Task 2: Contract probe — rest of the `/mcp` + `/api/mcp-models` surface

**Files:**
- Modify: `tests/fleet/contract_probe.py`

- [ ] **Step 1: Add the remaining checks**

Add these functions and append them to `CHECKS` (after `mcp_fqdn_host`):

```python
def _get(base, path, token=None, host_header=None, timeout=15):
    req = urllib.request.Request(f"{base}{path}")
    if host_header:
        req.add_header("Host", host_header)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def check_mcp_initialize_session(base, token, host_header):
    """initialize returns 200 and an mcp-session-id header (session contract)."""
    status, headers, body = _post(base, "/mcp", token, host_header, _INIT)
    if status != 200:
        raise ContractFailure(f"/mcp initialize expected 200, got {status}")
    if not headers.get("mcp-session-id"):
        raise ContractFailure("/mcp initialize response missing mcp-session-id header")


def check_mcp_requires_auth(base, host_header):
    """No bearer token → 403 before anything else (auth contract)."""
    status, _, _ = _post(base, "/mcp", None, host_header, _INIT)
    if status != 403:
        raise ContractFailure(f"/mcp without auth expected 403, got {status}")


def check_mcp_models_shape(base):
    """/api/mcp-models is reachable, auth-exempt, and returns {'models': [...]}."""
    status, body = _get(base, "/api/mcp-models")
    if status != 200:
        raise ContractFailure(f"/api/mcp-models expected 200, got {status}")
    try:
        data = json.loads(body)
    except ValueError:
        raise ContractFailure(f"/api/mcp-models returned non-JSON: {body[:160]!r}")
    if not isinstance(data.get("models"), list):
        raise ContractFailure(f"/api/mcp-models missing list 'models': {data!r}")
```

Append to `CHECKS`:
```python
    ("mcp_initialize_session", lambda a: check_mcp_initialize_session(a.base, a.token, a.host_header)),
    ("mcp_requires_auth", lambda a: check_mcp_requires_auth(a.base, a.host_header)),
    ("mcp_models_shape", lambda a: check_mcp_models_shape(a.base)),
```

> Note: a `tools/list` / `tools/call` check needs an established session id from `initialize` and the MCP streamable-HTTP session dance. That's heavier and lower-value than the auth/host/shape checks above (tool schemas are re-fetched live by Claude Code, so they self-heal). Deferred — `mcp_fqdn_host` already exercises the real transport. If added later, thread the `mcp-session-id` from `initialize` into a follow-up `tools/list` POST and assert HTTP 200 + a JSON-RPC `result.tools` array.

- [ ] **Step 2: Run against the CORRECT server, verify all PASS**

Start the correct server, run the probe (same command as Task 1 Step 3).
Expected: four `PASS` lines, exit 0. `kill $SRV`.

- [ ] **Step 3: Sanity — verify `mcp_requires_auth` actually discriminates**

Temporarily break the check by making `check_mcp_requires_auth` assert `status != 200` (so it would pass on a 403 OR 401). Re-run; confirm it still passes. Revert. (This is a 30-second confidence check that the assertion targets 403 specifically, not "any non-200".)

- [ ] **Step 4: Commit**

```bash
git add tests/fleet/contract_probe.py
git commit -m "test(fleet): add auth, session, and /api/mcp-models contract checks"
```

---

## Task 3: Contract probe — proxy-hop loop guard (`:8101`, light)

**Files:**
- Modify: `tests/fleet/contract_probe.py`

**Background:** `app/profile-server.py` `_proxy_to_desktop` reads `X-SP-Proxy-Hops`; when `hops >= _MAX_PROXY_HOPS` it returns `502 {"error": "Proxy loop detected — too many hops between servers"}`, but only when the server is in client/remote mode (`_is_remote_ollama()` true). This task asserts the loop-guard contract, the stable cross-machine bit. Before writing, read `app/profile-server.py` for `_MAX_PROXY_HOPS`, `_is_remote_ollama`, and `_desktop_profile_server_url`, and how the profile server selects client mode from env/config, so the server can be forced into a proxying state pointed at an unreachable upstream.

- [ ] **Step 1: Add the proxy-hop check**

```python
def check_proxy_hop_guard(profile_base, token):
    """A proxied request already at the hop limit must be refused (502 loop
    detected), not forwarded — the loop-prevention contract between paired
    profile servers."""
    req = urllib.request.Request(
        f"{profile_base}/api/chat", data=json.dumps({"model": "x", "messages": []}).encode(),
        method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-SP-Proxy-Hops", "99")  # >= _MAX_PROXY_HOPS
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read()
    if status != 502 or b"loop" not in body.lower():
        raise ContractFailure(
            f"proxy hop guard: expected 502 loop-detected, got {status}; body={body[:160]!r}")
```

Add a `--profile-base` arg (optional; default `None`) and append to `CHECKS` **only when provided**:
```python
    if args.profile_base:
        CHECKS.append(("proxy_hop_guard", lambda a: check_proxy_hop_guard(a.profile_base, a.token)))
```
(Add this inside `main()` after parsing args, before the loop. The `/mcp` checks always run; the proxy check is opt-in so the gate can skip it if profile-server client-mode setup is unavailable.)

- [ ] **Step 2: Run against a profile server forced into client mode toward an unreachable upstream**

Start the profile server with the env that forces remote/client mode (per the code you read in the background step), pointed at a non-existent desktop. Run:
```bash
uv run tests/fleet/contract_probe.py --base http://127.0.0.1:8199 \
  --token test-token --host-header test.fqdn:8100 \
  --profile-base http://127.0.0.1:8101
```
Expected: `PASS proxy_hop_guard` (the guard returns 502 before attempting to reach the unreachable upstream). `kill` the profile server.

- [ ] **Step 3: Commit**

```bash
git add tests/fleet/contract_probe.py
git commit -m "test(fleet): proxy-hop loop-guard contract check (opt-in)"
```

---

## Task 4: Orchestrator — `run_compat.py` (worktree both directions)

**Files:**
- Create: `tests/fleet/run_compat.py`
- Create: `tests/fleet/test_run_compat.py`

- [ ] **Step 1: Write the orchestrator**

```python
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
        # resolve (downloads the mcp SDK, uvicorn, …). Pre-warm in release.sh
        # with `uv run --directory <wt> mcp/local-models-server.py --help` if
        # this proves too slow, but 180s covers a cold cache in practice.
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
            raise SystemExit(f"server from {server_repo} not ready within 60s")
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
```

- [ ] **Step 2: Write the orchestrator smoke test**

```python
# tests/fleet/test_run_compat.py
"""Smoke test for the fleet compat orchestrator's failure handling."""
import subprocess
from pathlib import Path

import pytest

import tests.fleet.run_compat as rc

REPO = Path(__file__).resolve().parent.parent.parent


def test_latest_tag_returns_a_tag():
    """_latest_tag returns the most recent SemVer tag from the repo."""
    tag = rc._latest_tag()
    assert tag.startswith("v"), f"expected a vX.Y.Z tag, got {tag!r}"


def test_worktree_is_removed_even_when_a_probe_fails(monkeypatch, tmp_path):
    """A failing probe must not leak a git worktree."""
    removed = {"called": False}

    real_call = subprocess.call

    def fake_call(cmd, *a, **k):
        if cmd[:2] == ["git", "-C"] and "worktree" in cmd and "remove" in cmd:
            removed["called"] = True
            return 0
        return real_call(cmd, *a, **k)

    # Force a probe failure and stub server startup so the test is hermetic.
    monkeypatch.setattr(rc, "_run_probe", lambda repo: 1)
    monkeypatch.setattr(rc, "_server", lambda repo: _nullctx())
    monkeypatch.setattr(rc, "_latest_tag", lambda: rc._latest_tag())
    monkeypatch.setattr(subprocess, "call", fake_call)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: 0)

    rc_main_rc = rc.main()
    assert rc_main_rc == 1, "gate must fail when a probe fails"
    assert removed["called"], "worktree must be removed even on probe failure"


import contextlib


@contextlib.contextmanager
def _nullctx():
    yield
```

- [ ] **Step 3: Run the smoke test, verify it passes**

Run: `uv run --with pytest pytest tests/fleet/test_run_compat.py -v`
Expected: both tests PASS. (`test_latest_tag_returns_a_tag` needs at least one tag in the repo — true: v1.0.21 exists.)

- [ ] **Step 4: Run the real gate end-to-end against the live prev tag**

Run: `uv run tests/fleet/run_compat.py`
Expected: `Direction B` runs (HEAD probe vs v1.0.21 server) and passes; `Direction A skipped` (v1.0.21 predates the probe); `fleet compat gate passed`, exit 0. This proves the whole gate works on the real previous release before it guards anything.

- [ ] **Step 5: Commit**

```bash
git add tests/fleet/run_compat.py tests/fleet/test_run_compat.py
git commit -m "test(fleet): cross-version compat orchestrator (worktree, both directions)"
```

---

## Task 5: `bin/release.sh` — the gate at the release boundary

**Files:**
- Create: `bin/release.sh`

- [ ] **Step 1: Write the release script**

```bash
#!/bin/bash
# Cut a signed Super Puppy release — the ONLY supported way to ship a tag.
# Runs the test suite + the fleet compat gate BEFORE creating/pushing the tag,
# because the fleet auto-pulls new tags within ~2 minutes (a bad tag is live
# almost instantly). Then signs, verifies the signature against the repo's
# allowed_signers, and pushes main + the tag.
#
# Usage: bin/release.sh vX.Y.Z [--dry-run]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0" || echo "$0")")/.." && pwd)"
cd "$REPO_DIR"

VERSION="${1:-}"
DRY_RUN="${2:-}"
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "usage: bin/release.sh vX.Y.Z [--dry-run]" >&2
    exit 2
fi

PREV="$(git tag --sort=-v:refname | head -1)"
if [ -n "$PREV" ] && [ "$(printf '%s\n%s\n' "$PREV" "$VERSION" | sort -V | tail -1)" != "$VERSION" ]; then
    echo "refusing: $VERSION is not greater than latest tag $PREV" >&2
    exit 2
fi

echo "== preconditions =="
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "not on main" >&2; exit 1; }
git fetch --quiet origin
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { echo "main not in sync with origin/main" >&2; exit 1; }

echo "== test suite =="
uv run --with pytest --with flask --with pyyaml --with requests --with mlx-audio \
    pytest tests/ -q -m "not slow"

echo "== fleet compat gate (vs $PREV) =="
uv run tests/fleet/run_compat.py

echo "== sign tag $VERSION =="
git tag -s "$VERSION" -m "$VERSION"

echo "== verify signature against allowed_signers =="
git -c gpg.ssh.allowedSignersFile="$REPO_DIR/config/git/allowed_signers" tag -v "$VERSION"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "dry run: created+verified $VERSION locally; not pushing. Remove with: git tag -d $VERSION"
    exit 0
fi

echo "== push =="
git push origin main
git push origin "$VERSION"
echo "released $VERSION"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x bin/release.sh`

- [ ] **Step 3: Dry-run it (creates a local signed tag, verifies, does not push)**

Run: `bin/release.sh v1.0.22 --dry-run`
Expected: suite passes, compat gate passes, `v1.0.22` is created and verified (`Good "git" signature for jerrytalton@gmail.com`), no push. Then clean up the local tag:
```bash
git tag -d v1.0.22
```
(If preconditions fail because the branch isn't `main`/clean during development, run the inner steps manually — suite + `uv run tests/fleet/run_compat.py` — to validate, and note it.)

- [ ] **Step 4: Commit**

```bash
git add bin/release.sh
git commit -m "build: bin/release.sh — gated, signed, verified release path"
```

---

## Task 6: Release documentation

**Files:**
- Create: `docs/RELEASING.md`
- Modify: `CLAUDE.md` (testing/release section)

- [ ] **Step 1: Write `docs/RELEASING.md`**

```markdown
# Releasing Super Puppy

Releases ship to the fleet via signed git tags; every machine auto-updates to
the latest tag within ~2 minutes. **A bad tag is live almost immediately**, so
releases go through `bin/release.sh`, which gates on tests + cross-version
compatibility *before* the tag is pushed.

## Cut a release

```bash
bin/release.sh v1.0.22            # full release: gate, sign, verify, push
bin/release.sh v1.0.22 --dry-run  # everything except the push
```

The script will refuse unless: the tree is clean, you're on `main`, `main` is
in sync with `origin/main`, the test suite passes, and the fleet compat gate
passes.

## The fleet compat gate

`tests/fleet/run_compat.py` worktrees the previous tag and checks both
directions of version skew using `tests/fleet/contract_probe.py` — the
executable definition of the cross-machine wire contract (`/mcp` auth +
Tailscale-Host, `/api/mcp-models`, the `:8101` proxy-hop guard).

**Compatibility rule:** the wire contract is **additive-only**. Adding fields,
endpoints, or tools is fine; changing or removing what an existing peer relies
on is a breaking change and must not ship without a deliberate contract-version
bump (see the runtime-handshake spec, track #2). The gate checks only the
adjacent prior version; transitivity across older fleet members depends on this
rule holding.

## Signing

Tags are SSH-signed; the trusted key lives in `config/git/allowed_signers`.
A new key must ride a tag signed by the *outgoing* key (the running fleet
verifies the next tag against the key it already trusts).
```

- [ ] **Step 2: Add a pointer in `CLAUDE.md`**

In `CLAUDE.md`, under the Testing or "When Modifying This Repo" section, add:
```markdown
- Cut releases with `bin/release.sh vX.Y.Z` (never a bare `git tag`); it gates on the suite + the fleet cross-version compat check before pushing. See `docs/RELEASING.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/RELEASING.md CLAUDE.md
git commit -m "docs: release process and fleet compat gate"
```

---

## Self-Review

- **Spec coverage:**
  - Contract surface (`/mcp` auth+FQDN-Host+init, `/api/mcp-models`, proxy hop) → Tasks 1–3. ✓ (`tools/list`/`tools/call` deferred with rationale, per spec's "one inference-free tools/call" being lower-value than the auth/host checks — noted in Task 2.)
  - Live-worktree harness, both directions, bootstrap skip → Task 4. ✓
  - Pre-tag gate via `bin/release.sh` + sign + verify against `allowed_signers` → Task 5. ✓
  - Adjacent-version-only + additive rule → documented in Task 6 (`RELEASING.md`). ✓
  - Inference-free / CI-friendly → reusable server snippet + Task 4 startup note; no model assertions anywhere. ✓
  - Harness self-test (teardown-on-failure) → Task 4 Step 2. ✓
- **Placeholder scan:** Task 3 intentionally directs reading `profile-server.py` internals for the client-mode setup (the one place full code can't be pre-written without that context); the assertion code itself is complete. No other placeholders.
- **Type/name consistency:** `ContractFailure`, `_post`/`_get`, `CHECKS`, `--host-header`, `_server`/`_run_probe`/`_latest_tag`, `TOKEN`/`FQDN`/`HOST_HEADER`/`PORT` consistent across Tasks 1–4. The probe CLI args (`--base`, `--token`, `--host-header`, optional `--profile-base`) match how `run_compat.py` and `release.sh` invoke it.
```
