# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mcp[cli]==1.26.0", "httpx==0.28.1", "pyyaml==6.0.3"]
# ///
"""Over-the-wire proof that X-SP-Client attribution survives FastMCP's
streamable-HTTP task boundary.

Run standalone (not under pytest's mocked session, which stubs mcp/httpx):

    uv run --python 3.12 tests/_attribution_probe.py

Exits 0 on success, non-zero with a diagnostic on failure. Wrapped by
tests/test_mcp_attribution_e2e.py, which runs it via `uv run` so it gets the
real transport deps in a clean interpreter.

Faithfulness: this drives the REAL streamable_http_app + BearerAuthMiddleware +
session machinery + low-level receive loop + _gpu_request + _current_client
through a real MCP initialize -> tools/call handshake over an in-process ASGI
transport, asserting the activity DB row. The only stand-in is the tool body: a
no-op instead of a live Ollama/MLX call. The attribution path (transport threads
the current POST's Request -> request_ctx on the receive-loop task ->
_current_client reads X-SP-Client off it -> _gpu_request stamps the DB) is
identical to production, and that path is exactly what the contextvar bug broke.
"""

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The receive loop that runs tool handlers is forked during the *initialize*
# POST and snapshots that task's context. To prove per-call attribution across
# that task boundary (and to catch the original contextvar bug, which would
# report the stale init-time value), we initialize with one client and then
# issue TWO tool calls on the same session with two DIFFERENT clients. Correct
# attribution yields {machine-A, machine-B}; the contextvar no-op yields
# {init-machine} (or {unknown-client}).
INIT_CLIENT = "init-machine"
CLIENT_A = "machine-A"
CLIENT_B = "machine-B"
TOKEN = "probe-token"


def _load_server(db_path: str):
    os.environ["MCP_AUTH_TOKEN"] = TOKEN
    os.environ["SP_ACTIVITY_DB"] = db_path
    os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
    os.environ.setdefault("MLX_URL", "http://localhost:8000")
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "local_models_server", str(REPO_ROOT / "mcp" / "local-models-server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["local_models_server"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_sse_json(text: str) -> dict:
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    return {}


async def _run(server) -> None:
    import httpx
    from lib import activity
    activity.init_db()

    @server.mcp.tool(name="probe_attribution",
                     description="No-op tool that exercises _gpu_request.")
    async def probe_attribution() -> str:
        with server._gpu_request("ollama", "probe:probe-model"):
            pass
        return "ok"

    app = server.mcp.streamable_http_app()
    app.add_middleware(server.BearerAuthMiddleware)

    def _headers(client_id: str, session_id: str | None = None) -> dict:
        h = {
            "Authorization": f"Bearer {TOKEN}",
            "X-SP-Client": client_id,
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if session_id:
            h["mcp-session-id"] = session_id
        return h

    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), \
            httpx.AsyncClient(transport=transport,
                              base_url="http://localhost:8100") as client:
        # 1. initialize — forks the receive loop, captures the session id.
        init = await client.post("/mcp", headers=_headers(INIT_CLIENT), json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "probe", "version": "1"},
            },
        })
        if init.status_code != 200:
            raise SystemExit(f"initialize failed: {init.status_code} {init.text}")
        session_id = init.headers.get("mcp-session-id")
        if not session_id:
            raise SystemExit("no mcp-session-id header on initialize response")

        # 2. initialized notification (still the init client).
        note = await client.post("/mcp", headers=_headers(INIT_CLIENT, session_id),
                                 json={"jsonrpc": "2.0",
                                       "method": "notifications/initialized"})
        if note.status_code not in (200, 202):
            raise SystemExit(f"initialized failed: {note.status_code} {note.text}")

        # 3. two tool calls on the SAME session, DIFFERENT clients each.
        for req_id, client_id in ((2, CLIENT_A), (3, CLIENT_B)):
            call = await client.post(
                "/mcp", headers=_headers(client_id, session_id), json={
                    "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
                    "params": {"name": "probe_attribution", "arguments": {}},
                })
            if call.status_code != 200:
                raise SystemExit(
                    f"tools/call ({client_id}) failed: "
                    f"{call.status_code} {call.text}")
            result = _parse_sse_json(call.text)
            if result.get("error"):
                raise SystemExit(f"tools/call ({client_id}) error: {result['error']}")

    rows = activity.query_activity(60)["history"]
    machines = {r["machine"] for r in rows if r.get("tool") == "probe"}
    expected = {CLIENT_A, CLIENT_B}
    if machines != expected:
        raise SystemExit(
            f"attribution wrong: expected per-call {expected}, got {machines}. "
            f"(A stale value like {{{INIT_CLIENT!r}}} means the handler read a "
            f"snapshot from session creation, not the current request.)")
    print(f"PASS: two same-session tool calls attributed per-call to {machines} "
          f"over the wire (init client {INIT_CLIENT!r} correctly not seen)")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "activity.db")
        server = _load_server(db_path)
        asyncio.run(_run(server))


if __name__ == "__main__":
    main()
