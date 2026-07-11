# Usage Telemetry & Fleet Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give any Super Puppy install a truthful, fleet-wide view of how much its agents actually use the local-model tools, and an audit that verifies (and on request fixes) the MCP config + agent guidance that drives that usage — all within the user's own tailnet.

**Architecture:** Extend the existing SQLite activity pipeline (`lib/activity.py`, written by both the MCP server and profile server) rather than forking a new one. Add machine attribution to every request, a session-count denominator via a Claude Code hook, a 15-minute menubar heartbeat that pushes per-machine aggregates + audit results to the fleet server, a Fleet view on the existing Activity page, and an audit library surfaced as a CLI / menubar item / install.sh step. Security mitigations from the red-team pass (spec §Security) are folded into the relevant tasks, not bolted on at the end.

**Tech Stack:** Python 3.12 (stdlib `sqlite3`, `contextvars`, `socket`, `json`), Flask (profile server), `mcp`/FastMCP + Starlette middleware (MCP server), rumps (menubar), vanilla JS (dashboard), bash + a Python bound-parameter helper (session hook), pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-usage-telemetry-fleet-audit-design.md`

## Global Constraints

- **Python pinned to 3.12** (pyobjc-core doesn't build on 3.14+). Menubar app uses PEP 723 inline metadata — no requirements.txt/venv. Pin exact dependency versions (no `>=`).
- **Shared constants and logic go in `lib/models.py`**, not duplicated across menubar/MCP/profile server.
- **All remote URLs use `https://{tailscale_fqdn}:{port}`** — never `http://{ip}`. Tailscale serve rejects plain HTTP and presents remote requests as `127.0.0.1`, so loopback is never trusted; the bearer token is the only authn.
- **Both MCP and profile servers fail closed without a token.** `SP_ALLOW_NO_AUTH=1` is the test/dev escape hatch only.
- **Telemetry never leaves the user's fleet.** No external phone-home. Data flows only client → the user's own fleet server.
- **Telemetry paths are best-effort and silent on failure** (existing `log_request` philosophy): a logging/heartbeat/hook failure must never break a tool call, a Claude session, or the menubar. Audit **fixes** are the opposite — fail loud, touch nothing on a parse error.
- **Dashboard renders every server-supplied string with `textContent`/`esc()` — never raw `innerHTML` interpolation** (spec §S1). Existing `esc()` helper lives in `app/activity.html`.
- **Config-file mutations are atomic** (temp file + `rename`), **merge into arrays/objects by key** (never replace), and write a `.bak` first (spec §S6).
- **Conventional Commits**, one logical change per commit. Run `uv run --with pytest pytest tests/ -v` before pushing.
- **New `bin/` scripts must be registered in `bin/post-update.sh`** (the shared symlink routine, called by both install.sh and the auto-updater) AND in the uninstall list in `install.sh`.
- Test invocation (full deps): `uv run --with pytest --with flask --with pyyaml --with requests --with pillow --with "transformers==5.12.1" --with "mlx-audio[tts] @ git+https://github.com/Blaizzy/mlx-audio.git@e42e1431fcf89af313375296c46d03a0153c4aa7" pytest tests/ -v`. For pure-Python tasks here, `uv run --with pytest --with flask --with requests pytest tests/<file> -v` suffices.

---

## Phase 1 — Foundation: schema v2 + test isolation

This phase is the standalone bug fix (tests pollute the prod DB) plus the schema every later phase builds on. Ship it first even if nothing else lands.

### Task 1: Make the activity DB path overridable and isolate tests

**Files:**
- Modify: `lib/models.py:22` (ACTIVITY_DB)
- Create: `tests/conftest.py`
- Test: `tests/test_activity.py` (new)

**Interfaces:**
- Produces: `lib.models.ACTIVITY_DB` resolves from `$SP_ACTIVITY_DB` when set, else `CONFIG_DIR / "activity.db"`. An autouse fixture points it at a tmp DB for every test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_activity.py
import importlib
import os


def test_activity_db_honors_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SP_ACTIVITY_DB", str(tmp_path / "custom.db"))
    import lib.models as models
    importlib.reload(models)
    assert str(models.ACTIVITY_DB) == str(tmp_path / "custom.db")
    # restore default for other tests
    monkeypatch.delenv("SP_ACTIVITY_DB")
    importlib.reload(models)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest pytest tests/test_activity.py::test_activity_db_honors_env_override -v`
Expected: FAIL — `ACTIVITY_DB` is a fixed `CONFIG_DIR / "activity.db"`, ignores the env var.

- [ ] **Step 3: Implement the override in `lib/models.py`**

Replace line 22:

```python
ACTIVITY_DB = Path(os.environ["SP_ACTIVITY_DB"]).expanduser() if os.environ.get("SP_ACTIVITY_DB") else CONFIG_DIR / "activity.db"
```

Add `import os` to the imports at the top of `lib/models.py` if not already present (it imports `re`; add `os` alongside).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --with pytest pytest tests/test_activity.py -v`
Expected: PASS

- [ ] **Step 5: Add the autouse isolation fixture**

```python
# tests/conftest.py
import os
import pytest


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
    import importlib
    import lib.models
    importlib.reload(lib.models)
    import lib.activity
    importlib.reload(lib.activity)
    yield db
```

- [ ] **Step 6: Verify no test writes to the real DB**

Run: `uv run --with pytest --with flask --with requests pytest tests/test_activity.py tests/test_mcp_server.py -v`
Expected: PASS. Manually confirm the real DB's mtime is unchanged:
`ls -l ~/.config/local-models/activity.db` before and after — timestamp must not move.

- [ ] **Step 7: Commit**

```bash
git add lib/models.py tests/conftest.py tests/test_activity.py
git commit -m "test(activity): isolate tests from prod DB via SP_ACTIVITY_DB override"
```

### Task 2: Schema v2 — `machine` column + `PRAGMA user_version` migration

**Files:**
- Modify: `lib/activity.py:26-47` (init_db), `:50-71` (log_request), `:74-107` (query_activity)
- Test: `tests/test_activity.py`

**Interfaces:**
- Consumes: `SP_ACTIVITY_DB` isolation from Task 1.
- Produces:
  - `activity.init_db()` — creates tables and runs migrations idempotently; `PRAGMA user_version` gates each step (0→1 adds `machine`).
  - `activity.log_request(..., machine: str = "")` — new trailing kwarg; stamps the `machine` column.
  - `activity.query_activity(period_seconds, limit=200)` — unchanged signature; `history`/`tool_stats`/`total` now **exclude** `source='session'` rows; adds `"sessions": <int>` (count of session rows in window).
  - `activity.last_activity_at() -> float | None` — max `completed_at` across non-session rows, or None.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_activity.py
import time
from lib import activity


def test_migration_adds_machine_column_to_existing_db(tmp_path, monkeypatch):
    # Simulate a v0 DB: create the old schema by hand, then migrate.
    import sqlite3
    db = tmp_path / "old.db"
    monkeypatch.setenv("SP_ACTIVITY_DB", str(db))
    import importlib, lib.models, lib.activity as act
    importlib.reload(lib.models); importlib.reload(act)
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, tool TEXT NOT NULL,
        model TEXT NOT NULL, backend TEXT NOT NULL, source TEXT NOT NULL,
        status TEXT NOT NULL, error_msg TEXT, duration_ms INTEGER NOT NULL,
        started_at REAL NOT NULL, completed_at REAL NOT NULL)""")
    conn.execute("INSERT INTO requests (tool,model,backend,source,status,duration_ms,started_at,completed_at) "
                 "VALUES ('code','x','ollama','mcp','ok',10,1,2)")
    conn.commit(); conn.close()
    act.init_db()  # must not raise; must add the column
    cols = {r[1] for r in sqlite3.connect(str(db)).execute("PRAGMA table_info(requests)")}
    assert "machine" in cols


def test_log_and_query_records_machine(tmp_path):
    activity.init_db()
    now = time.time()
    activity.log_request(tool="vision", model="qwen", backend="ollama",
                         source="mcp", status="ok", duration_ms=42,
                         started_at=now-1, completed_at=now, machine="jerry-laptop")
    data = activity.query_activity(3600)
    assert data["history"][0]["machine"] == "jerry-laptop"


def test_query_excludes_sessions_from_history_but_counts_them(tmp_path):
    activity.init_db()
    now = time.time()
    activity.log_request(tool="code", model="x", backend="ollama", source="mcp",
                         status="ok", duration_ms=5, started_at=now-1, completed_at=now)
    activity.log_request(tool="session", model="claude-code", backend="", source="session",
                         status="ok", duration_ms=0, started_at=now, completed_at=now)
    data = activity.query_activity(3600)
    assert data["total"] == 1               # session not counted as a request
    assert data["sessions"] == 1
    assert all(r["source"] != "session" for r in data["history"])


def test_last_activity_at_ignores_sessions(tmp_path):
    activity.init_db()
    now = time.time()
    activity.log_request(tool="code", model="x", backend="ollama", source="mcp",
                         status="ok", duration_ms=5, started_at=now-10, completed_at=now-10)
    activity.log_request(tool="session", model="claude-code", backend="", source="session",
                         status="ok", duration_ms=0, started_at=now, completed_at=now)
    assert abs(activity.last_activity_at() - (now-10)) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest tests/test_activity.py -v`
Expected: FAIL — `machine` column doesn't exist, `log_request` has no `machine` kwarg, `sessions`/`last_activity_at` undefined.

- [ ] **Step 3: Implement migration + machine column in `init_db`**

Replace the body of `init_db()` (lines 26-47) so the `CREATE TABLE` includes `machine TEXT NOT NULL DEFAULT ''`, and add a version-gated migration for pre-existing DBs:

```python
def init_db() -> None:
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool TEXT NOT NULL,
            model TEXT NOT NULL,
            backend TEXT NOT NULL,
            source TEXT NOT NULL,
            status TEXT NOT NULL,
            error_msg TEXT,
            duration_ms INTEGER NOT NULL,
            started_at REAL NOT NULL,
            completed_at REAL NOT NULL,
            machine TEXT NOT NULL DEFAULT ''
        )
    """)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < 1:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
        if "machine" not in cols:
            conn.execute("ALTER TABLE requests ADD COLUMN machine TEXT NOT NULL DEFAULT ''")
        conn.execute("PRAGMA user_version = 1")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_completed_at ON requests(completed_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tool ON requests(tool)")
    cutoff = time.time() - (_PRUNE_DAYS * 86400)
    conn.execute("DELETE FROM requests WHERE completed_at < ?", (cutoff,))
    conn.commit()
    conn.close()
```

- [ ] **Step 4: Add `machine` kwarg to `log_request`**

Modify `log_request` (lines 50-71): add `machine: str = ""` as the final parameter and include it in the INSERT column list and values tuple:

```python
def log_request(
    tool: str, model: str, backend: str, source: str, status: str,
    duration_ms: int, started_at: float, completed_at: float,
    error_msg: str | None = None, machine: str = "",
) -> None:
    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO requests (tool, model, backend, source, status, error_msg, "
            "duration_ms, started_at, completed_at, machine) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tool, model, backend, source, status, error_msg, duration_ms,
             started_at, completed_at, machine),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # never let logging failures break tool execution
```

- [ ] **Step 5: Exclude sessions from `query_activity`, add `sessions` + `last_activity_at`**

In `query_activity`, add `AND source != 'session'` to the `history`, `tool_stats`, and `totals` WHERE clauses, add `machine` to the history SELECT column list, compute a `sessions` count, and add it to the returned dict:

```python
    sessions = conn.execute(
        "SELECT COUNT(*) AS c FROM requests WHERE completed_at > ? AND source='session'",
        (cutoff,),
    ).fetchone()["c"]
```

History SELECT becomes:
`"SELECT tool, model, backend, source, status, error_msg, duration_ms, started_at, completed_at, machine FROM requests WHERE completed_at > ? AND source != 'session' ORDER BY completed_at DESC LIMIT ?"`

Return dict gains `"sessions": sessions or 0`.

Add the new function:

```python
def last_activity_at() -> float | None:
    conn = _connect()
    row = conn.execute(
        "SELECT MAX(completed_at) AS m FROM requests WHERE source != 'session'"
    ).fetchone()
    conn.close()
    return row["m"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --with pytest pytest tests/test_activity.py -v`
Expected: PASS (all four new tests + Task 1's).

- [ ] **Step 7: Commit**

```bash
git add lib/activity.py tests/test_activity.py
git commit -m "feat(activity): schema v2 — machine attribution, session exclusion, last-activity"
```

### Task 3: One-time junk-row cleanup migration

**Files:**
- Modify: `lib/activity.py` (init_db, extend the version-gated block to a `user_version` 1→2 step)
- Test: `tests/test_activity.py`

**Interfaces:**
- Consumes: `init_db` migration framework from Task 2.
- Produces: rows whose `tool` matches the known test-pollution set are deleted exactly once (guarded by `user_version` 1→2). Real usage rows are untouched.

- [ ] **Step 1: Write the failing test**

```python
def test_junk_rows_pruned_once(tmp_path):
    import time
    from lib import activity
    activity.init_db()
    now = time.time()
    for tool in ("test", "task_0", "first_task", "a", "b", "c", "failing", "test_tool", "task1"):
        activity.log_request(tool=tool, model="x", backend="ollama", source="mcp",
                             status="ok", duration_ms=1, started_at=now, completed_at=now)
    activity.log_request(tool="vision", model="qwen", backend="ollama", source="mcp",
                         status="ok", duration_ms=1, started_at=now, completed_at=now)
    activity.prune_junk_rows()
    data = activity.query_activity(3600)
    tools = {r["tool"] for r in data["history"]}
    assert tools == {"vision"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest pytest tests/test_activity.py::test_junk_rows_pruned_once -v`
Expected: FAIL — `prune_junk_rows` undefined.

- [ ] **Step 3: Implement `prune_junk_rows` and wire it into the migration**

```python
_JUNK_TOOLS = ("test", "first_task", "second_task", "a", "b", "c",
               "failing", "test_tool", "task1", "task2")
_JUNK_LIKE = ("task\\_%",)  # ESCAPE '\' — matches task_0, task_1, ...


def prune_junk_rows() -> None:
    conn = _connect()
    placeholders = ",".join("?" * len(_JUNK_TOOLS))
    conn.execute(f"DELETE FROM requests WHERE tool IN ({placeholders})", _JUNK_TOOLS)
    for pattern in _JUNK_LIKE:
        conn.execute("DELETE FROM requests WHERE tool LIKE ? ESCAPE '\\'", (pattern,))
    conn.commit()
    conn.close()
```

In `init_db`, after the `user_version < 1` block, add:

```python
    if version < 2:
        conn.commit()  # ensure the machine migration is durable first
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()
        prune_junk_rows()   # opens its own connection
        conn = _connect()   # reopen for the index/prune tail below
```

(Adjust so the index creation + retention DELETE at the end still run on an open connection; simplest is to call `prune_junk_rows()` at the very end of `init_db` guarded by the version check, then set `user_version=2`.) Keep it simple: set version and call prune at the end, before the final `conn.close()`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --with pytest pytest tests/test_activity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/activity.py tests/test_activity.py
git commit -m "chore(activity): one-time prune of test-pollution rows (user_version 1->2)"
```

---

## Phase 2 — Machine attribution (X-SP-Client)

### Task 4: MCP server captures and validates `X-SP-Client`, stamps `machine`

**Files:**
- Modify: `mcp/local-models-server.py:186-223` (middleware), `:261-288` (GPUTracker.__exit__)
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `activity.log_request(..., machine=...)` from Task 2.
- Produces:
  - Module-level `_client_ctx: contextvars.ContextVar[str]` (default `""`).
  - `_CLIENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")` and helper `_validated_client(raw: str) -> str` returning the value if it matches else `""`.
  - `GPUTracker.__exit__` passes `machine=_client_ctx.get() or "unknown-client"` to `log_request` (playground/local paths set their own; MCP path uses the header).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_server.py — add near the auth tests
import re

def test_validated_client_accepts_good_rejects_bad():
    import importlib, mcp_server_mod  # see note below
    # The test module imports the server as a module object already; reuse it.
    from importlib import import_module
    srv = sys.modules.get("local_models_server")  # loaded by existing harness
    assert srv._validated_client("jerry-laptop") == "jerry-laptop"
    assert srv._validated_client("MacBook-Pro.local") == "MacBook-Pro.local"
    assert srv._validated_client("<img onerror=x>") == ""
    assert srv._validated_client("a" * 65) == ""
    assert srv._validated_client("") == ""
```

Note: match the existing test file's import mechanism for the server module (it mocks heavy deps then imports). Reuse whatever symbol the existing tests use to reference server internals; if they use a fixture/module alias, follow that exact pattern rather than the placeholder above.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with flask pytest tests/test_mcp_server.py -k validated_client -v`
Expected: FAIL — `_validated_client` undefined.

- [ ] **Step 3: Implement the contextvar + validator**

Near the top of `mcp/local-models-server.py` (after imports, with the other module state around line 186):

```python
import contextvars
_client_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("sp_client", default="")
_CLIENT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _validated_client(raw: str) -> str:
    return raw if raw and _CLIENT_RE.match(raw) else ""
```

Ensure `re` is imported (it is used elsewhere; confirm at top).

- [ ] **Step 4: Set the contextvar in the middleware**

In `BearerAuthMiddleware.dispatch`, immediately after the auth check passes (both the session-bound and bearer branches that call `call_next`), set the client from the header. Simplest: at the very start of `dispatch`, before returning any `call_next`, do:

```python
        _client_ctx.set(_validated_client(request.headers.get("x-sp-client", "")))
```

Place it as the first line inside `dispatch` so it's set for every code path (unauth requests get a value but never reach `log_request`).

- [ ] **Step 5: Stamp `machine` in GPUTracker.__exit__**

In `GPUTracker.__exit__`, change the `activity.log_request(...)` call to pass:

```python
            machine=_client_ctx.get() or "unknown-client",
```

- [ ] **Step 6: Add an end-to-end-ish test that a logged row carries the client**

```python
def test_gpu_tracker_stamps_client_machine(tmp_path):
    srv = sys.modules["local_models_server"]
    from lib import activity
    activity.init_db()
    srv._client_ctx.set("jerry-laptop")
    with srv.GPUTracker("code:model-x", "ollama"):
        pass
    row = activity.query_activity(60)["history"][0]
    assert row["machine"] == "jerry-laptop"
```

- [ ] **Step 7: Run to verify pass**

Run: `uv run --with pytest --with flask pytest tests/test_mcp_server.py -k "client or machine" -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add mcp/local-models-server.py tests/test_mcp_server.py
git commit -m "feat(mcp): attribute requests to calling machine via validated X-SP-Client"
```

### Task 5: install.sh + audit write the `X-SP-Client` header into the MCP entry

**Files:**
- Modify: `install.sh:515-524` (the `claude mcp add-json` ENTRY construction)
- Test: manual (bash) — covered by audit unit tests in Task 12; no pytest here.

**Interfaces:**
- Consumes: the MCP server reading `X-SP-Client` (Task 4).
- Produces: the registered `local-models` entry includes `"headers": {"Authorization": "Bearer ...", "X-SP-Client": "<hostname>"}`.

- [ ] **Step 1: Modify the ENTRY builder in install.sh**

Replace the header construction (around lines 517-522):

```bash
        CLIENT_HOST=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
        ENTRY='{"type":"http","url":"http://127.0.0.1:8100/mcp"'
        HEADERS='"X-SP-Client":"'"$CLIENT_HOST"'"'
        if [ -n "$MCP_TOKEN" ]; then
            HEADERS='"Authorization":"Bearer '"$MCP_TOKEN"'",'"$HEADERS"
        fi
        ENTRY="$ENTRY"',"headers":{'"$HEADERS"'}}'
```

- [ ] **Step 2: Verify the JSON is well-formed**

Run (dry check, doesn't touch real config):
```bash
CLIENT_HOST=testhost MCP_TOKEN=abc bash -c '
ENTRY="{\"type\":\"http\",\"url\":\"http://127.0.0.1:8100/mcp\""
HEADERS="\"X-SP-Client\":\"$CLIENT_HOST\""
HEADERS="\"Authorization\":\"Bearer $MCP_TOKEN\",$HEADERS"
ENTRY="$ENTRY,\"headers\":{$HEADERS}}"
echo "$ENTRY"' | python3 -m json.tool
```
Expected: pretty-printed JSON with both headers, exit 0.

- [ ] **Step 3: Commit**

```bash
git add install.sh
git commit -m "feat(install): register MCP entry with X-SP-Client attribution header"
```

---

## Phase 3 — Session denominator

### Task 6: `sp-session-ping` writes a session row via bound parameters

**Files:**
- Create: `bin/sp-session-ping`
- Modify: `bin/post-update.sh:43-48` (link it), `install.sh:59-72` (uninstall list)
- Test: `tests/test_session_ping.py` (new)

**Interfaces:**
- Consumes: schema v2 (`source='session'`).
- Produces: a `bin/sp-session-ping [agent]` executable that inserts one row `(tool='session', source='session', model=<validated agent, default 'claude-code'>, machine=<hostname>, status='ok', duration_ms=0)` into the local `activity.db` using **bound parameters** (no string interpolation), exiting 0 always.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_ping.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest pytest tests/test_session_ping.py -v`
Expected: FAIL — `bin/sp-session-ping` doesn't exist.

- [ ] **Step 3: Write the script**

```bash
#!/bin/bash
# sp-session-ping [agent-name]
# Records one agent-session start in the LOCAL Super Puppy activity DB, as a
# denominator for tool-usage rate. Fire-and-forget: never fails a session.
# Registered as a Claude Code SessionStart hook by `sp-doctor --fix`.
AGENT="${1:-claude-code}"
DB="${SP_ACTIVITY_DB:-$HOME/.config/local-models/activity.db}"
HOST="$(scutil --get LocalHostName 2>/dev/null || hostname -s 2>/dev/null || echo unknown)"
python3 - "$DB" "$AGENT" "$HOST" <<'PY' 2>/dev/null || true
import os, re, sqlite3, sys, time
db, agent, host = sys.argv[1], sys.argv[2], sys.argv[3]
if not re.match(r'^[a-z0-9-]{1,32}$', agent):
    agent = "claude-code"
if not re.match(r'^[A-Za-z0-9._-]{1,64}$', host or ""):
    host = "unknown"
os.makedirs(os.path.dirname(db), exist_ok=True)
conn = sqlite3.connect(db, timeout=3)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=3000")
conn.execute("""CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, tool TEXT NOT NULL, model TEXT NOT NULL,
    backend TEXT NOT NULL, source TEXT NOT NULL, status TEXT NOT NULL,
    error_msg TEXT, duration_ms INTEGER NOT NULL, started_at REAL NOT NULL,
    completed_at REAL NOT NULL, machine TEXT NOT NULL DEFAULT '')""")
now = time.time()
conn.execute("INSERT INTO requests (tool, model, backend, source, status, "
             "duration_ms, started_at, completed_at, machine) "
             "VALUES ('session', ?, '', 'session', 'ok', 0, ?, ?, ?)",
             (agent, now, now, host))
conn.commit(); conn.close()
PY
exit 0
```

Make it executable: `chmod +x bin/sp-session-ping`.

Note: the script creates the table defensively (it may run before the servers ever initialize the DB). The columns match schema v2 exactly (Task 2) — a test in Task 6 Step 6 asserts this coupling.

- [ ] **Step 4: Run to verify pass**

Run: `uv run --with pytest pytest tests/test_session_ping.py -v`
Expected: PASS

- [ ] **Step 5: Register the symlink and uninstall entry**

In `bin/post-update.sh`, after line 48 add:
```bash
link bin/sp-session-ping           ~/.local/bin/sp-session-ping
```
In `install.sh` uninstall list (both `~/.local/bin/` and legacy `~/bin/` blocks) add `~/.local/bin/sp-session-ping \` and `~/bin/sp-session-ping \`.

- [ ] **Step 6: Add the schema-coupling regression test**

```python
def test_ping_row_matches_library_schema(tmp_path):
    """The hand-rolled INSERT must stay compatible with lib.activity's schema."""
    import importlib, os
    os.environ["SP_ACTIVITY_DB"] = str(tmp_path / "activity.db")
    import lib.models, lib.activity as act
    importlib.reload(lib.models); importlib.reload(act)
    act.init_db()  # library creates the table
    _run("claude-code", tmp_path / "activity.db")  # script inserts into it
    data = act.query_activity(3600)
    assert data["sessions"] == 1
    del os.environ["SP_ACTIVITY_DB"]
```

Run: `uv run --with pytest pytest tests/test_session_ping.py -v` → PASS.

- [ ] **Step 7: Commit**

```bash
git add bin/sp-session-ping bin/post-update.sh install.sh tests/test_session_ping.py
git commit -m "feat(telemetry): sp-session-ping session denominator (bound-param insert)"
```

---

## Phase 4 — Fleet heartbeat

### Task 7: `activity.local_usage_summary()` + fleet tables + upsert/query

**Files:**
- Modify: `lib/activity.py` (new functions + fleet table DDL in init_db)
- Test: `tests/test_activity.py`

**Interfaces:**
- Produces:
  - `activity.local_usage_summary(days: int = 7) -> list[dict]` — rows `{day, tool, source, count, errors, avg_ms}` grouped over the local DB's last `days` days (sessions included, as their own `source`).
  - `activity.upsert_fleet_report(machine, version, mode, usage, audit_json, received_at) -> None` — idempotent upsert into `fleet_usage` (PK `machine,day,tool,source`) and `fleet_machines` (PK `machine`); `last_seen = received_at` (server clock). Prunes `fleet_usage` older than 30 days and `fleet_machines` not seen in 30 days.
  - `activity.query_fleet() -> dict` — `{machines: [{machine, version, mode, last_seen, audit}], usage: [{machine, day, tool, source, count, errors, avg_ms}]}` over the last 7 days.
  - Fleet DDL added to `init_db`.

- [ ] **Step 1: Write the failing tests**

```python
def test_local_usage_summary_groups_by_day_tool_source(tmp_path):
    import time
    from lib import activity
    activity.init_db()
    now = time.time()
    for _ in range(3):
        activity.log_request(tool="vision", model="q", backend="ollama", source="mcp",
                             status="ok", duration_ms=100, started_at=now, completed_at=now)
    activity.log_request(tool="vision", model="q", backend="ollama", source="mcp",
                         status="error", duration_ms=200, started_at=now, completed_at=now)
    summary = activity.local_usage_summary(7)
    row = next(r for r in summary if r["tool"] == "vision")
    assert row["count"] == 4 and row["errors"] == 1 and row["source"] == "mcp"


def test_fleet_upsert_is_idempotent(tmp_path):
    import time
    from lib import activity
    activity.init_db()
    usage = [{"day": "2026-07-10", "tool": "vision", "source": "mcp",
              "count": 5, "errors": 0, "avg_ms": 120}]
    activity.upsert_fleet_report("laptop", "v1.2.0", "client", usage, '{"ok":true}', time.time())
    activity.upsert_fleet_report("laptop", "v1.2.0", "client", usage, '{"ok":true}', time.time())
    fleet = activity.query_fleet()
    rows = [u for u in fleet["usage"] if u["machine"] == "laptop"]
    assert len(rows) == 1 and rows[0]["count"] == 5   # upsert replaced, not doubled
    assert fleet["machines"][0]["machine"] == "laptop"


def test_fleet_last_seen_is_server_stamped(tmp_path):
    import time
    from lib import activity
    activity.init_db()
    t = time.time()
    activity.upsert_fleet_report("laptop", "v1", "client", [], "{}", t)
    assert abs(activity.query_fleet()["machines"][0]["last_seen"] - t) < 0.01
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest pytest tests/test_activity.py -k "usage_summary or fleet" -v`
Expected: FAIL — functions/tables undefined.

- [ ] **Step 3: Add fleet DDL to `init_db`**

Inside `init_db`, before the final commit:

```python
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fleet_usage (
            machine TEXT NOT NULL, day TEXT NOT NULL, tool TEXT NOT NULL,
            source TEXT NOT NULL, count INTEGER NOT NULL, errors INTEGER NOT NULL,
            avg_ms INTEGER NOT NULL,
            PRIMARY KEY (machine, day, tool, source)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fleet_machines (
            machine TEXT PRIMARY KEY, version TEXT, mode TEXT,
            last_seen REAL NOT NULL, audit_json TEXT
        )
    """)
```

- [ ] **Step 4: Implement the three functions**

```python
_FLEET_RETAIN_DAYS = 30


def local_usage_summary(days: int = 7) -> list[dict]:
    conn = _connect()
    cutoff = time.time() - days * 86400
    rows = [dict(r) for r in conn.execute(
        "SELECT date(completed_at,'unixepoch') AS day, tool, source, "
        "COUNT(*) AS count, "
        "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors, "
        "CAST(AVG(duration_ms) AS INTEGER) AS avg_ms "
        "FROM requests WHERE completed_at > ? GROUP BY day, tool, source",
        (cutoff,),
    ).fetchall()]
    conn.close()
    return rows


def upsert_fleet_report(machine, version, mode, usage, audit_json, received_at) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO fleet_machines (machine, version, mode, last_seen, audit_json) "
        "VALUES (?,?,?,?,?) ON CONFLICT(machine) DO UPDATE SET "
        "version=excluded.version, mode=excluded.mode, "
        "last_seen=excluded.last_seen, audit_json=excluded.audit_json",
        (machine, version, mode, received_at, audit_json),
    )
    for u in usage:
        conn.execute(
            "INSERT INTO fleet_usage (machine, day, tool, source, count, errors, avg_ms) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(machine, day, tool, source) DO UPDATE SET "
            "count=excluded.count, errors=excluded.errors, avg_ms=excluded.avg_ms",
            (machine, u["day"], u["tool"], u["source"],
             int(u["count"]), int(u["errors"]), int(u["avg_ms"])),
        )
    cutoff_day = time.strftime("%Y-%m-%d", time.gmtime(received_at - _FLEET_RETAIN_DAYS * 86400))
    conn.execute("DELETE FROM fleet_usage WHERE day < ?", (cutoff_day,))
    conn.execute("DELETE FROM fleet_machines WHERE last_seen < ?",
                 (received_at - _FLEET_RETAIN_DAYS * 86400,))
    conn.commit()
    conn.close()


def query_fleet() -> dict:
    conn = _connect()
    cutoff_day = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 7 * 86400))
    machines = [dict(r) for r in conn.execute(
        "SELECT machine, version, mode, last_seen, audit_json FROM fleet_machines "
        "ORDER BY last_seen DESC")]
    for m in machines:
        m["audit"] = m.pop("audit_json")
    usage = [dict(r) for r in conn.execute(
        "SELECT machine, day, tool, source, count, errors, avg_ms FROM fleet_usage "
        "WHERE day >= ? ORDER BY day DESC", (cutoff_day,))]
    conn.close()
    return {"machines": machines, "usage": usage}
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run --with pytest pytest tests/test_activity.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add lib/activity.py tests/test_activity.py
git commit -m "feat(activity): fleet report tables, idempotent upsert, summary/query helpers"
```

### Task 8: `/api/fleet/report` ingest endpoint (auth + validation + rate limit)

**Files:**
- Modify: `app/profile-server.py` (new route + validation helpers near the other `/api` routes, ~line 2965)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Consumes: `activity.upsert_fleet_report` (Task 7), the existing `@app.before_request _check_auth`.
- Produces:
  - `POST /api/fleet/report` — bearer-auth'd (inherits `_check_auth`); validates `machine`/`version`/`mode` against `^[A-Za-z0-9._-]{1,64}$`, validates `usage` shape, rate-limits one accepted report per machine per 5 min, stamps `last_seen` server-side. Returns 200/400/429.
  - `GET /api/fleet` — returns `activity.query_fleet()` (bearer-auth'd, proxied to desktop in client mode via existing `_proxy_to_desktop`).
  - `_FLEET_FIELD_RE`, `_fleet_rate: dict[str, float]` module state.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_profile_server.py — new class
class TestFleetReport:
    def _payload(self, machine="laptop"):
        return {"machine": machine, "version": "v1.2.0", "mode": "client",
                "sent_at": 1, "audit": [{"id": "claude-mcp", "status": "pass"}],
                "usage": [{"day": "2026-07-10", "tool": "vision", "source": "mcp",
                           "count": 3, "errors": 0, "avg_ms": 100}]}

    def test_report_accepts_valid(self, client):
        r = client.post("/api/fleet/report", json=self._payload())
        assert r.status_code == 200
        got = client.get("/api/fleet").get_json()
        assert got["machines"][0]["machine"] == "laptop"

    def test_report_rejects_bad_machine(self, client):
        p = self._payload(machine="<script>")
        r = client.post("/api/fleet/report", json=p)
        assert r.status_code == 400

    def test_report_rate_limited(self, client):
        assert client.post("/api/fleet/report", json=self._payload()).status_code == 200
        assert client.post("/api/fleet/report", json=self._payload()).status_code == 429
```

(The existing `client` fixture runs with `SP_ALLOW_NO_AUTH=1`, so no bearer header is needed in tests; a separate auth test below asserts 403 without the flag is out of scope here since the fixture forces allow-no-auth. Add an explicit-token test only if the file already has that pattern.)

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest --with flask --with requests pytest tests/test_profile_server.py -k Fleet -v`
Expected: FAIL — route 404.

- [ ] **Step 3: Implement the routes**

Near the other `/api` routes (after `api_activity`, ~line 3003):

```python
import re as _re
_FLEET_FIELD_RE = _re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_FLEET_MIN_INTERVAL = 300  # one accepted report per machine per 5 min
_fleet_rate: dict[str, float] = {}
_fleet_rate_lock = threading.Lock()


def _valid_usage(usage) -> bool:
    if not isinstance(usage, list) or len(usage) > 5000:
        return False
    for u in usage:
        if not isinstance(u, dict):
            return False
        if not _FLEET_FIELD_RE.match(str(u.get("tool", ""))):
            return False
        for k in ("count", "errors", "avg_ms"):
            if not isinstance(u.get(k), int):
                return False
    return True


@app.route("/api/fleet/report", methods=["POST"])
def api_fleet_report():
    body = request.get_json(silent=True) or {}
    machine = str(body.get("machine", ""))
    version = str(body.get("version", ""))
    mode = str(body.get("mode", ""))
    if not (_FLEET_FIELD_RE.match(machine) and _FLEET_FIELD_RE.match(version)
            and _FLEET_FIELD_RE.match(mode)):
        return jsonify({"error": "invalid machine/version/mode"}), 400
    if not _valid_usage(body.get("usage", [])):
        return jsonify({"error": "invalid usage payload"}), 400
    now = time.time()
    with _fleet_rate_lock:
        last = _fleet_rate.get(machine, 0)
        if now - last < _FLEET_MIN_INTERVAL:
            return jsonify({"error": "rate limited"}), 429
        _fleet_rate[machine] = now
    audit_json = json.dumps(body.get("audit", []))[:100_000]
    activity.upsert_fleet_report(machine, version, mode, body["usage"], audit_json, now)
    return jsonify({"ok": True}), 200


@app.route("/api/fleet")
def api_fleet():
    proxied = _proxy_to_desktop("/api/fleet", method="GET")
    if proxied is not None:
        return proxied
    return jsonify(activity.query_fleet())
```

Confirm `json` and `threading` are already imported at the top of profile-server.py (they are).

- [ ] **Step 4: Run to verify pass**

Run: `uv run --with pytest --with flask --with requests pytest tests/test_profile_server.py -k Fleet -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "feat(profile-server): /api/fleet/report ingest + /api/fleet query, validated & rate-limited"
```

### Task 9: Menubar heartbeat timer

**Files:**
- Modify: `app/menubar.py` (new timer alongside `warm_timer` at ~line 1427; heartbeat method)
- Test: `tests/test_core.py` (the menubar unit-test home) — test the payload builder pure function

**Interfaces:**
- Consumes: `activity.local_usage_summary`, the audit `run_all` (Task 12 — until then, heartbeat sends `audit: []`), `resolve_desktop_tailscale`, `_PROFILE_AUTH_TOKEN` equivalent (menubar reads token from config).
- Produces:
  - A pure helper `build_heartbeat_payload(machine, version, mode, summary, audit) -> dict` (module-level in menubar.py, unit-testable without rumps).
  - `MenuBarApp._on_heartbeat_tick(self, _)` — builds payload, POSTs to the server's `/api/fleet/report` in a daemon thread, swallows all errors.
  - `self.heartbeat_timer = rumps.Timer(self._on_heartbeat_tick, 900)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core.py
def test_build_heartbeat_payload_shape():
    import importlib.util, pathlib
    # menubar imports rumps; load only the pure helper via the module's
    # existing test shim if present. If test_core already imports menubar
    # symbols, follow that. Otherwise assert against the function directly.
    from app import menubar  # follow the file's existing import approach
    payload = menubar.build_heartbeat_payload(
        machine="laptop", version="v1.2.0", mode="client",
        summary=[{"day": "2026-07-10", "tool": "vision", "source": "mcp",
                  "count": 3, "errors": 0, "avg_ms": 100}],
        audit=[{"id": "claude-mcp", "status": "pass"}])
    assert payload["machine"] == "laptop"
    assert payload["usage"][0]["tool"] == "vision"
    assert payload["audit"][0]["status"] == "pass"
    assert "sent_at" in payload
```

Note: `app/menubar.py` imports rumps/pyobjc, which may not import cleanly in CI. If `test_core.py` already imports menubar behind a guard, reuse it. If not, put `build_heartbeat_payload` in a small importable location — but prefer keeping it in menubar.py and guarding the test with `pytest.importorskip("rumps")`.

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest pytest tests/test_core.py -k heartbeat_payload -v`
Expected: FAIL — function undefined.

- [ ] **Step 3: Implement the payload builder + timer**

Module-level in `app/menubar.py`:

```python
def build_heartbeat_payload(machine, version, mode, summary, audit):
    return {
        "machine": machine,
        "version": version,
        "mode": mode,
        "sent_at": time.time(),
        "audit": audit,
        "usage": summary,
    }
```

In `MenuBarApp.__init__`, after `self.warm_timer.start()` (line 1428):

```python
        self.heartbeat_timer = rumps.Timer(self._on_heartbeat_tick, 900)
        self.heartbeat_timer.start()
        threading.Timer(30, lambda: self._on_heartbeat_tick(None)).start()  # early first beat
```

Method:

```python
    def _on_heartbeat_tick(self, _):
        threading.Thread(target=self._send_heartbeat, daemon=True).start()

    def _send_heartbeat(self):
        try:
            from lib import activity
            machine = socket.gethostname().split(".")[0]
            version = self._current_version()  # existing version accessor
            mode = "server" if self.conf.get("IS_SERVER") == "true" else "client"
            summary = activity.local_usage_summary(7)
            try:
                from lib import audit as _audit
                audit_results = _audit.run_all()  # available after Task 12
            except Exception:
                audit_results = []
            payload = build_heartbeat_payload(machine, version, mode, summary, audit_results)
            url, token = self._fleet_report_target()  # https://{fqdn}:8101 or localhost + token
            if not url:
                return
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            requests.post(f"{url}/api/fleet/report", json=payload, headers=headers, timeout=8)
        except Exception as e:
            logging.debug("heartbeat failed: %s", e)
```

Implement `_fleet_report_target(self)` using the same resolution the app already uses for the desktop profile server (server → `http://127.0.0.1:8101`; client → `https://{desktop_fqdn}:8101`). Reuse `resolve_desktop_tailscale` / `self.desktop_fqdn`. Token is read from the existing MCP-token config path. Follow whatever accessor the app already has for version (`self._current_version()` is a placeholder — use the real one).

- [ ] **Step 4: Run to verify pass**

Run: `uv run --with pytest pytest tests/test_core.py -k heartbeat_payload -v`
Expected: PASS

- [ ] **Step 5: Manual smoke (server machine)**

With the profile server running locally:
```bash
curl -s -X POST http://127.0.0.1:8101/api/fleet/report \
  -H "Authorization: Bearer $(cat ~/.config/local-models/mcp_auth_token)" \
  -H 'Content-Type: application/json' \
  -d '{"machine":"smoketest","version":"v0.0.0","mode":"server","sent_at":1,"audit":[],"usage":[]}'
curl -s http://127.0.0.1:8101/api/fleet \
  -H "Authorization: Bearer $(cat ~/.config/local-models/mcp_auth_token)" | python3 -m json.tool
```
Expected: `{"ok":true}` then a `machines` list containing `smoketest`.

- [ ] **Step 6: Commit**

```bash
git add app/menubar.py tests/test_core.py
git commit -m "feat(menubar): 15-minute fleet heartbeat pushing usage + audit to server"
```

---

## Phase 5 — Audit library + frontends

### Task 10: Managed guidance block renderer (markers, idempotent, content-versioned)

**Files:**
- Create: `lib/audit.py`
- Test: `tests/test_audit.py` (new)

**Interfaces:**
- Produces:
  - `audit.GUIDANCE_MARKERS = ("<!-- >>> super-puppy >>> -->", "<!-- <<< super-puppy <<< -->")` (and a `# >>>`/`# <<<` variant for TOML/AGENTS files where HTML comments don't apply — but CLAUDE.md/AGENTS.md/GEMINI.md are all Markdown, so HTML comment markers work for all three).
  - `audit.GUIDANCE_TEXT` — the canonical block body: the trigger→tool table and the parallelism directive from the spec/dotfiles `8bfd5b8`. No hardware/model names.
  - `audit.render_block() -> str` — markers + text.
  - `audit.upsert_guidance(text: str) -> str` — returns `text` with the managed block inserted (appended) or, if markers already exist, replaced **only if the content differs**; content outside markers is never touched. Idempotent: `upsert_guidance(upsert_guidance(x)) == upsert_guidance(x)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_audit.py
from lib import audit


def test_upsert_appends_when_absent():
    out = audit.upsert_guidance("# My CLAUDE.md\n\nSome rules.\n")
    assert audit.GUIDANCE_MARKERS[0] in out
    assert audit.GUIDANCE_MARKERS[1] in out
    assert "Some rules." in out  # user content preserved


def test_upsert_is_idempotent():
    once = audit.upsert_guidance("# doc\n")
    twice = audit.upsert_guidance(once)
    assert once == twice


def test_upsert_replaces_stale_block_only_between_markers():
    doc = "# doc\n\nkeep me\n" + audit.GUIDANCE_MARKERS[0] + "\nOLD JUNK\n" + audit.GUIDANCE_MARKERS[1] + "\n\nkeep me too\n"
    out = audit.upsert_guidance(doc)
    assert "OLD JUNK" not in out
    assert "keep me" in out and "keep me too" in out


def test_guidance_text_has_no_hardware_names():
    lowered = audit.GUIDANCE_TEXT.lower()
    for banned in ("m3 ultra", "512gb", "m5 max", "holo3", "wan2.2", "voxtral"):
        assert banned not in lowered
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest pytest tests/test_audit.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
"""Super Puppy configuration audit: verify (and optionally fix) that installed
agent tools are wired to use SP — MCP registration, guidance blocks, session hook.

Design note (spec §S2): guidance blocks are text an AI agent treats as
instructions. Fixes here are NEVER automatic — a caller (sp-doctor / menubar /
install.sh opt-in) applies them explicitly, with a diff. The managed block is
minimal and mechanical; changing GUIDANCE_TEXT is a reviewed code diff.
"""

GUIDANCE_MARKERS = ("<!-- >>> super-puppy >>> -->", "<!-- <<< super-puppy <<< -->")

GUIDANCE_TEXT = """\
## Local Models (super-puppy)

Reach for the `local-models` MCP server whenever a task needs capabilities you \
lack (vision, audio, image/video) or cheap parallel compute (bulk transforms, \
second opinions, delegated reasoning while you keep working). Call \
`local_models_status` for what's live.

**Do things in parallel on the server and with the local cluster whenever it \
makes sense.** While you work, the local GPUs should be busy too — fire off \
`local_dispatch` before you start reasoning, not after you finish.

| Need | Tool |
|------|------|
| Look at an image or screenshot | `local_vision` |
| Plan a GUI interaction | `local_computer_use` |
| Generate / edit an image | `local_image`, `local_image_edit` |
| Generate video | `local_video` |
| Translate / transcribe / speak | `local_translate`, `local_transcribe`, `local_speak` |
| Bulk boilerplate / repetitive transforms | `local_generate` |
| Second opinion on code or designs | `local_review`, `local_candidates` |
| Summarize before reading a huge file | `local_summarize` |
| Delegate reasoning while you keep working | `local_dispatch` then `local_collect` |
| Find files by concept | `local_similarity_search`, `local_embed` |

Model/hardware specifics live in each tool's runtime description and \
`local_models_status` — trust those over anything static.
"""


def render_block() -> str:
    return f"{GUIDANCE_MARKERS[0]}\n{GUIDANCE_TEXT}\n{GUIDANCE_MARKERS[1]}"


def upsert_guidance(text: str) -> str:
    block = render_block()
    start, end = GUIDANCE_MARKERS
    if start in text and end in text:
        pre = text[: text.index(start)]
        post = text[text.index(end) + len(end):]
        return f"{pre}{block}{post}"
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return f"{text}{sep}{block}\n"
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --with pytest pytest tests/test_audit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/audit.py tests/test_audit.py
git commit -m "feat(audit): managed guidance block — idempotent, content-versioned, marker-scoped"
```

### Task 11: Atomic config writers (JSON merge + Markdown block)

**Files:**
- Modify: `lib/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces:
  - `audit.atomic_write(path: Path, content: str) -> None` — temp file in the same dir + `os.replace`, writing a `.bak` of the prior content first.
  - `audit.merge_json_key(path, dotted_key, value) -> None` — read-modify-write of a JSON file, setting a nested key **without disturbing siblings**; creates the file if absent; raises loudly on parse error.
  - `audit.append_hook(settings_path, hook_entry) -> None` — merge a SessionStart hook into `hooks.SessionStart` array, preserving existing hooks; no duplicate if an identical command is already present.

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path
from lib import audit


def test_atomic_write_makes_bak(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("old")
    audit.atomic_write(p, "new")
    assert p.read_text() == "new"
    assert (tmp_path / "f.txt.bak").read_text() == "old"


def test_merge_json_key_preserves_siblings(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"mcpServers": {"other": {"x": 1}}, "keep": True}))
    audit.merge_json_key(p, "mcpServers.local-models", {"type": "http"})
    data = json.loads(p.read_text())
    assert data["mcpServers"]["other"] == {"x": 1}
    assert data["mcpServers"]["local-models"] == {"type": "http"}
    assert data["keep"] is True


def test_merge_json_raises_on_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    import pytest
    with pytest.raises(Exception):
        audit.merge_json_key(p, "a.b", 1)


def test_append_hook_no_duplicate(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{}")
    entry = {"matcher": "*", "hooks": [{"type": "command", "command": "sp-session-ping"}]}
    audit.append_hook(p, entry)
    audit.append_hook(p, entry)
    data = json.loads(p.read_text())
    assert len(data["hooks"]["SessionStart"]) == 1
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest pytest tests/test_audit.py -k "atomic or merge or hook" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
import json
import os
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    path = Path(path)
    if path.exists():
        (path.parent / (path.name + ".bak")).write_text(path.read_text())
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())  # raises loudly on malformed


def merge_json_key(path, dotted_key, value) -> None:
    path = Path(path)
    data = _load_json(path)
    node = data
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    atomic_write(path, json.dumps(data, indent=2))


def append_hook(settings_path, hook_entry) -> None:
    path = Path(settings_path)
    data = _load_json(path)
    hooks = data.setdefault("hooks", {})
    arr = hooks.setdefault("SessionStart", [])
    cmds = {h.get("command") for e in arr for h in e.get("hooks", [])}
    new_cmds = {h.get("command") for h in hook_entry.get("hooks", [])}
    if not (new_cmds & cmds):
        arr.append(hook_entry)
    atomic_write(path, json.dumps(data, indent=2))
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --with pytest pytest tests/test_audit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/audit.py tests/test_audit.py
git commit -m "feat(audit): atomic config writers — JSON key-merge, markdown block, hook append"
```

### Task 12: Check registry + fixes (Claude Code, Codex, Gemini, detect-others)

**Files:**
- Modify: `lib/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces:
  - `audit.Check` dataclass: `{id, tool, status, detail, fixable}` (`status` ∈ `pass|fail|warn|n/a`).
  - `audit.run_all(home: Path | None = None) -> list[dict]` — runs every check against `home` (default `~`), returns serializable dicts. `home` param makes it testable against a fabricated dir.
  - `audit.fix(check_id, home=None, token=None) -> str` — applies the fix for one check; returns a human diff/summary; raises loudly on unparseable target.
  - `audit.fix_all(home=None, token=None) -> list[str]`.
  - Token-safety rule (spec §S4): the `claude-mcp` fix refuses to inline the token if `~/.claude.json` is group/other-readable or inside a git work tree; warns and writes the entry without the token instead.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
from lib import audit


def _fake_home(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude.json").write_text("{}")
    (tmp_path / ".claude" / "CLAUDE.md").write_text("# rules\n")
    (tmp_path / ".claude" / "settings.json").write_text("{}")
    return tmp_path


def test_run_all_flags_missing_mcp(tmp_path):
    home = _fake_home(tmp_path)
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-mcp"]["status"] == "fail"
    assert results["claude-guidance"]["status"] == "fail"
    # codex/gemini absent → n/a, not fail
    assert results["codex-mcp"]["status"] == "n/a"


def test_fix_mcp_then_passes(tmp_path):
    home = _fake_home(tmp_path)
    audit.fix("claude-mcp", home=home, token="secret")
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-mcp"]["status"] == "pass"


def test_fix_guidance_is_idempotent_and_passes(tmp_path):
    home = _fake_home(tmp_path)
    audit.fix("claude-guidance", home=home)
    first = (home / ".claude" / "CLAUDE.md").read_text()
    audit.fix("claude-guidance", home=home)
    second = (home / ".claude" / "CLAUDE.md").read_text()
    assert first == second
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-guidance"]["status"] == "pass"


def test_mcp_fix_refuses_token_in_world_readable(tmp_path):
    import os, stat
    home = _fake_home(tmp_path)
    cj = home / ".claude.json"
    cj.chmod(cj.stat().st_mode | stat.S_IROTH)  # world-readable
    audit.fix("claude-mcp", home=home, token="secret")
    import json
    entry = json.loads(cj.read_text())["mcpServers"]["local-models"]
    # token must NOT be inlined into a world-readable file
    assert "secret" not in json.dumps(entry)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest pytest tests/test_audit.py -k "run_all or fix" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the registry, checks, and fixes**

Implement `run_all` / `fix` / `fix_all` and per-tool check functions using `merge_json_key`, `append_hook`, `upsert_guidance`, `atomic_write`. Key details:
- `claude-mcp`: pass iff `~/.claude.json` has `mcpServers.local-models` with a `url` and an `X-SP-Client` header. Fix writes the entry (token inlined only if `~/.claude.json` is not group/other-readable and not inside a git tree — check via `os.stat().st_mode & (S_IRGRP|S_IROTH)` and walking parents for `.git`).
- `claude-guidance`: pass iff `render_block()`'s current content is present in `~/.claude/CLAUDE.md`. Fix = `atomic_write(path, upsert_guidance(path.read_text()))`.
- `claude-hook`: pass iff a `SessionStart` hook invoking `sp-session-ping` exists in `~/.claude/settings.json`. Fix = `append_hook`.
- `codex-*` / `gemini-*`: `n/a` when `~/.codex` / `~/.gemini` absent; else same shape (Codex uses `~/.codex/config.toml` + `~/.codex/AGENTS.md`; Gemini `~/.gemini/settings.json` + `~/.gemini/GEMINI.md`). TOML edit for Codex: use stdlib `tomllib` to read; write via a minimal appended `[mcp_servers.local-models]` block guarded by markers (don't round-trip-rewrite the user's TOML — append a managed section).
- `token-present`: pass iff `~/.config/local-models/mcp_auth_token` exists and is non-empty.
- `other-agents`: detect `~/.cursor`, `~/.config/opencode`, `~/Library/Application Support/Windsurf`; report `warn` "SP not configured (detect-only)" with a docs link; never fixable.

Show a per-fix summary string (e.g. `"claude-mcp: wrote mcpServers.local-models (token referenced, not inlined)"`).

- [ ] **Step 4: Run to verify pass**

Run: `uv run --with pytest pytest tests/test_audit.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/audit.py tests/test_audit.py
git commit -m "feat(audit): check registry + fixes for Claude/Codex/Gemini, token-leak guard"
```

### Task 13: `bin/sp-doctor` CLI

**Files:**
- Create: `bin/sp-doctor`
- Modify: `bin/post-update.sh` (link), `install.sh` (uninstall list + run `sp-doctor --fix --yes` for opted-in tools during setup)
- Test: `tests/test_audit.py` (subprocess smoke)

**Interfaces:**
- Consumes: `lib.audit.run_all/fix_all`.
- Produces: `sp-doctor` printing a pass/fail table; `--fix` applies fixable failures (prompts per fix unless `--yes`); `--json` emits `run_all()` as JSON; exit code 0 if all pass/n-a, 1 if any fail (useful for scripting).

- [ ] **Step 1: Write the failing test**

```python
import json, os, subprocess
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
DOCTOR = REPO / "bin" / "sp-doctor"


def test_sp_doctor_json_runs(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude.json").write_text("{}")
    (tmp_path / ".claude" / "CLAUDE.md").write_text("# x\n")
    (tmp_path / ".claude" / "settings.json").write_text("{}")
    env = {**os.environ, "SP_AUDIT_HOME": str(tmp_path)}
    r = subprocess.run([str(DOCTOR), "--json"], env=env, capture_output=True, text=True)
    data = json.loads(r.stdout)
    assert any(c["id"] == "claude-mcp" for c in data)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest pytest tests/test_audit.py -k sp_doctor -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Write `bin/sp-doctor`**

```python
#!/usr/bin/env python3
"""sp-doctor — verify (and optionally fix) that this machine's AI agents are
wired to use Super Puppy. Usage:
  sp-doctor              show the audit table
  sp-doctor --fix        apply fixable findings (prompts per fix)
  sp-doctor --fix --yes  apply without prompting
  sp-doctor --json       machine-readable output
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import audit

HOME = Path(os.environ.get("SP_AUDIT_HOME", str(Path.home())))


def _token():
    p = HOME / ".config" / "local-models" / "mcp_auth_token"
    return p.read_text().strip() if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = audit.run_all(home=HOME)
    if args.json:
        print(json.dumps(results, indent=2)); return 0
    icons = {"pass": "✅", "fail": "❌", "warn": "⚠️ ", "n/a": "· "}
    for c in results:
        print(f"  {icons.get(c['status'],'?')} {c['id']:<18} {c['detail']}")
    failed = [c for c in results if c["status"] == "fail" and c["fixable"]]
    if args.fix and failed:
        for c in failed:
            if not args.yes:
                if input(f"Fix {c['id']}? [y/N] ").strip().lower() != "y":
                    continue
            print("  " + audit.fix(c["id"], home=HOME, token=_token()))
    return 1 if any(c["status"] == "fail" for c in results) else 0


if __name__ == "__main__":
    sys.exit(main())
```

`chmod +x bin/sp-doctor`. Note: `run_all`/`fix` must accept a `home` param (Task 12) and honor `SP_AUDIT_HOME` for the config-token path too.

- [ ] **Step 4: Run to verify pass**

Run: `uv run --with pytest pytest tests/test_audit.py -k sp_doctor -v`
Expected: PASS

- [ ] **Step 5: Register symlink + install.sh hook**

`bin/post-update.sh`: `link bin/sp-doctor ~/.local/bin/sp-doctor`. `install.sh` uninstall list: add both paths. In install.sh setup, after MCP registration, run `~/.local/bin/sp-doctor --fix --yes` (the user already opted into SP by running install.sh) and echo the result.

- [ ] **Step 6: Commit**

```bash
git add bin/sp-doctor bin/post-update.sh install.sh tests/test_audit.py
git commit -m "feat(audit): sp-doctor CLI (table/--fix/--json) + install.sh integration"
```

### Task 14: Menubar "Audit…" item

**Files:**
- Modify: `app/menubar.py` (menu item near the "Activity Log" item at line 1385; a handler)
- Test: manual (rumps UI not unit-tested here)

**Interfaces:**
- Consumes: `lib.audit.run_all/fix_all`.
- Produces: an "Audit…" menu item under the tools submenu that runs `run_all`, shows an `rumps.alert` summary with a "Fix All" button, and on confirm calls `fix_all` then re-runs.

- [ ] **Step 1: Add the menu item**

After line 1386 (the Activity Log item):
```python
        self.menu_tools_sub.add(rumps.MenuItem("Audit…", callback=self.open_audit))
```

- [ ] **Step 2: Implement the handler**

```python
    def open_audit(self, _):
        from lib import audit
        results = audit.run_all()
        fails = [c for c in results if c["status"] == "fail"]
        lines = "\n".join(f"{c['status'].upper():5} {c['id']}: {c['detail']}" for c in results)
        if not fails:
            rumps.alert("Super Puppy Audit", "All checks pass.\n\n" + lines)
            return
        resp = rumps.alert("Super Puppy Audit",
                           lines + "\n\nApply fixes to the failing checks?",
                           ok="Fix All", cancel="Close")
        if resp == 1:
            summaries = audit.fix_all()
            rumps.alert("Fixes applied", "\n".join(summaries) or "Nothing to fix.")
```

- [ ] **Step 3: Manual verification**

Launch the menu bar app, open Tools → Audit…, confirm the alert lists checks and "Fix All" writes config (verify on a scratch account or with a `.bak` present).

- [ ] **Step 4: Commit**

```bash
git add app/menubar.py
git commit -m "feat(menubar): Audit… item — run checks, one-click Fix All"
```

---

## Phase 6 — Dashboard: Fleet view + empty-state fix

### Task 15: Empty-state shows last-activity; `/api/activity` returns it

**Files:**
- Modify: `app/profile-server.py:2965-3003` (api_activity), `app/activity.html:360-395` (renderHistory)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Consumes: `activity.last_activity_at` (Task 2).
- Produces: `/api/activity` response includes `"last_activity_at": <float|null>`; the empty table row reads "No requests in this period — last activity <date>".

- [ ] **Step 1: Write the failing test**

```python
def test_api_activity_includes_last_activity(client):
    import time
    from lib import activity
    activity.init_db()
    now = time.time()
    activity.log_request(tool="code", model="x", backend="ollama", source="mcp",
                         status="ok", duration_ms=5, started_at=now-1, completed_at=now)
    data = client.get("/api/activity?period=1").get_json()  # 1-second window → empty history
    assert data["last_activity_at"] is not None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest --with flask --with requests pytest tests/test_profile_server.py -k last_activity -v`
Expected: FAIL — key absent.

- [ ] **Step 3: Add `last_activity_at` to the response**

In `api_activity`, add to the returned jsonify dict:
```python
        "last_activity_at": activity.last_activity_at(),
```

- [ ] **Step 4: Update the empty-state in `activity.html`**

In `renderHistory`, replace the empty branch (line 364) to use `lastData.last_activity_at` (rendered via `textContent`/`esc`, formatted with the existing date helper). Because this string includes a formatted date only (no user input), but keep it built with `textContent` per the constraint.

- [ ] **Step 5: Run to verify pass**

Run: `uv run --with pytest --with flask --with requests pytest tests/test_profile_server.py -k last_activity -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/profile-server.py app/activity.html tests/test_profile_server.py
git commit -m "feat(activity-ui): empty state shows last-activity timestamp"
```

### Task 16: Fleet view section (XSS-safe render)

**Files:**
- Modify: `app/activity.html` (new Fleet section markup + JS fetch of `/api/fleet` + render)
- Test: `tests/test_playground_coverage.py` (assert the page references `/api/fleet`) + manual VLM screenshot check

**Interfaces:**
- Consumes: `GET /api/fleet` (Task 8).
- Produces: a "Fleet" section rendering per-machine cards (hostname, version, mode, last-seen relative time, calls today/7d, sessions 7d, calls-per-session, audit badge) — **all via `textContent`/`esc()`, never raw `innerHTML` string interpolation of server data**. Responsive (mobile + desktop).

- [ ] **Step 1: Write the failing coverage test**

```python
# tests/test_playground_coverage.py
def test_activity_page_has_fleet_view():
    from pathlib import Path
    html = (Path(__file__).resolve().parent.parent / "app" / "activity.html").read_text()
    assert "/api/fleet" in html
    assert "Fleet" in html
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run --with pytest pytest tests/test_playground_coverage.py -k fleet -v`
Expected: FAIL.

- [ ] **Step 3: Add the Fleet section markup + render JS**

Add a `<section id="fleet-view">` after the existing tool-stats section. Fetch `/api/fleet` with `_authHeaders()`. Build cards with `document.createElement` + `textContent` for every field (hostname, version, mode, audit ids). Compute calls-per-session = 7d calls / max(1, 7d sessions). Use `esc()` only if inserting into innerHTML is unavoidable — prefer node creation. Style with the page's existing CSS variables; grid that collapses to one column under 600px.

Follow-the-codebase note: `renderHistory` at line 369 currently uses `innerHTML = ...map().join('')` with `esc()` on each field. For the Fleet cards, prefer the `createElement`/`textContent` pattern already used at lines 310-314 and 393-395 — it's the XSS-safe path and the spec (§S1) makes it mandatory for the new, machine-supplied fields.

- [ ] **Step 4: Add a Content-Security-Policy header to profile-server responses**

In `app/profile-server.py`, add an `@app.after_request` that sets `Content-Security-Policy: default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'` (widen only as the existing pages require — test the Playground and Profiles pages still load). This is defense-in-depth behind the render fix.

- [ ] **Step 5: Run coverage + render a screenshot**

Run: `uv run --with pytest pytest tests/test_playground_coverage.py -v` → PASS.
Then, per the global visual-changes rule: with services up, load `http://127.0.0.1:8101/activity`, POST two fake machines via the Task 9 curl (one with a benign name, one whose hostname contains `<b>x</b>` to prove escaping — the endpoint should already 400 that, so instead inject a benign-but-distinctive name and confirm no HTML executes), screenshot, and check with a VLM (`local_vision`) that the Fleet cards render, are readable, and show no raw markup.

- [ ] **Step 6: Commit**

```bash
git add app/activity.html app/profile-server.py tests/test_playground_coverage.py
git commit -m "feat(activity-ui): fleet view — per-machine usage, sessions, audit; XSS-safe render + CSP"
```

---

## Phase 7 — Wiring, docs, smoke

### Task 17: Ensure `init_db` runs the fleet migrations on both servers at startup

**Files:**
- Verify: `mcp/local-models-server.py:1834` and `app/profile-server.py:3136` both call `activity.init_db()` at startup (they already do — confirm the new tables get created).
- Test: covered by Task 7/8 (init_db is called in fixtures).

- [ ] **Step 1: Confirm both call sites**

Run: `grep -n "activity.init_db" mcp/local-models-server.py app/profile-server.py`
Expected: one call each. No code change unless missing.

- [ ] **Step 2: Manual — start both, confirm tables exist**

```bash
sqlite3 ~/.config/local-models/activity.db ".tables"
```
Expected: `requests fleet_usage fleet_machines`.

- [ ] **Step 3: Commit (if any wiring changed)**

Only if a call site needed adding.

### Task 18: Smoke test — heartbeat round-trip against live profile server

**Files:**
- Modify: `tests/test_tools_smoke_laptop.py` (add one `@pytest.mark.smoke` test)

**Interfaces:**
- Consumes: live profile server on `MCP_PORT`/`PROFILE_PORT`, the shared smoke helpers.

- [ ] **Step 1: Add the smoke test**

```python
@pytest.mark.smoke
def test_fleet_report_roundtrip(profile_base, auth_headers):
    payload = {"machine": "smoke-laptop", "version": "v0.0.0", "mode": "server",
               "sent_at": 1, "audit": [], "usage": [
                   {"day": "2026-07-10", "tool": "vision", "source": "mcp",
                    "count": 1, "errors": 0, "avg_ms": 50}]}
    r = requests.post(f"{profile_base}/api/fleet/report", json=payload,
                      headers=auth_headers, timeout=10)
    assert r.status_code in (200, 429)  # 429 if a prior smoke run beat us
    got = requests.get(f"{profile_base}/api/fleet", headers=auth_headers, timeout=10).json()
    assert any(m["machine"] == "smoke-laptop" for m in got["machines"])
```

Follow the existing smoke helpers in `tests/_smoke_helpers.py` for `profile_base`/`auth_headers` fixtures — reuse them, don't invent new ones.

- [ ] **Step 2: Run (skips cleanly if services down)**

Run: `uv run --with pytest --with requests pytest tests/test_tools_smoke_laptop.py -k fleet_report -v`
Expected: PASS if services up, SKIP otherwise.

- [ ] **Step 3: Commit**

```bash
git add tests/test_tools_smoke_laptop.py
git commit -m "test(smoke): fleet heartbeat round-trip against live profile server"
```

### Task 19: Documentation

**Files:**
- Modify: `CLAUDE.md` (project), `docs/architecture.md`, `README.md`
- Create: `docs/usage-telemetry.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Update project `CLAUDE.md`**

Add to "Key files at runtime": `fleet_usage`/`fleet_machines` tables in activity.db, `/api/fleet/report` + `/api/fleet` endpoints, `sp-session-ping` + `sp-doctor` scripts, the SessionStart hook. Add a short "Usage Telemetry & Audit" subsection under Runtime Architecture. Note the §S2 trust-boundary decision (audit fixes are user-confirmed, never auto-applied on version bump).

- [ ] **Step 2: Write `docs/usage-telemetry.md`**

Cover: what's logged and where (local `activity.db`, never leaves the fleet), the heartbeat (cadence, payload, auth), the Fleet view, `sp-doctor` usage (table/--fix/--json/exit codes), the guidance-block markers and how to opt out (delete the block; audit won't re-add without a fix run), and the security posture (§S1-S6 summary).

- [ ] **Step 3: Update `docs/architecture.md` + README**

Architecture: add the attribution → heartbeat → fleet-view data flow diagram. README: one line under features + a pointer to `docs/usage-telemetry.md`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/architecture.md docs/usage-telemetry.md README.md
git commit -m "docs: usage telemetry + fleet audit — runtime files, sp-doctor usage, security posture"
```

### Task 20: Full-suite gate + branch finish

- [ ] **Step 1: Run the whole suite**

Run: `uv run --with pytest --with flask --with pyyaml --with requests --with pillow --with "transformers==5.12.1" --with "mlx-audio[tts] @ git+https://github.com/Blaizzy/mlx-audio.git@e42e1431fcf89af313375296c46d03a0153c4aa7" pytest tests/ -v`
Expected: all pass (smoke tests skip if services down).

- [ ] **Step 2: Verify no test touched the real DB**

`ls -l ~/.config/local-models/activity.db` — mtime unchanged across the run (Task 1 fixture guarantees this; this is the confirmation).

- [ ] **Step 3: Use the finishing-a-development-branch skill** to choose merge/PR and integrate. Do NOT auto-tag — releases go through `bin/release.sh` per the repo rules, and auto-update ships tags fleet-wide in ~2 minutes.

---

## Self-Review Notes

**Spec coverage:** schema v2 (T2), test isolation (T1), junk cleanup (T3), attribution (T4-5), session denominator (T6), heartbeat (T7-9), audit lib + frontends (T10-14), fleet view + empty state (T15-16), wiring/docs/smoke (T17-19). Security §S1 (T4 validation, T16 render+CSP), §S2 (T10 content-versioning + T12/T14 user-confirmed fixes), §S3 (T6 bound params), §S4 (T8 rate-limit/validation/server-stamp + T12 token guard), §S5 (T6 busy_timeout + T2/T7 retention), §S6 (T11 atomic/merge writers). All mapped.

**Known soft spots for the implementer:** T4 and T9 reference existing symbols by placeholder (the server-module test alias; `self._current_version()`, `self.conf`). Read the actual file and use the real accessors — the surrounding code shows them. T12 is the largest task; if it feels too big when you reach it, split per-tool (Claude first, end-to-end, then Codex/Gemini by the single-piece-flow rule) — but the interface stays as specified.
