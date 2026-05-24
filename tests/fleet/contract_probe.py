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


CHECKS = [
    ("mcp_fqdn_host", lambda a: check_mcp_fqdn_host(a.base, a.token, a.host_header)),
    ("mcp_initialize_session", lambda a: check_mcp_initialize_session(a.base, a.token, a.host_header)),
    ("mcp_requires_auth", lambda a: check_mcp_requires_auth(a.base, a.host_header)),
    ("mcp_models_shape", lambda a: check_mcp_models_shape(a.base)),
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
