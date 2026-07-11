# Usage Telemetry & Fleet Audit — Design

**Date:** 2026-07-10
**Status:** Approved approach (Approach A: extend existing pipeline, aggregate heartbeats)

## Problem

The Activity Log looks empty even though the logging pipeline works. Diagnosis on the laptop (2026-07-10):

1. **The empty view is accurate, not broken.** The default 24h window had zero requests because Claude last called SP on July 7. Real usage is ~20–70 calls/day with multi-day gaps — the underlying worry ("Claude isn't leveraging SP enough") is validated by the data.
2. **No fleet visibility.** Each machine's `activity.db` is an island. In client mode the Activity page proxies to the server and shows *only the server's* data; offline/local usage on clients never appears anywhere else. Nothing records *which machine's* Claude made a remote call (`source` is just `mcp`/`playground`).
3. **No adoption signal.** There is no way to tell whether a machine's Claude Code (or Codex/Gemini) is even *configured* to use SP, or whether guidance exists telling the agent to reach for it.
4. **Hygiene bugs.** Unit tests write to the production DB (7,424 of 8,481 rows are junk like `test`, `task_0`). The dashboard's empty state gives no hint when the last activity was.

## Goals

- Every SP request is attributed to the machine whose agent made it.
- One fleet-wide view (on the user's own fleet server) showing per-machine usage, session counts, calls-per-session, version, and config health.
- An audit that checks — and on request fixes — each machine's agent-tool integration: MCP registration, guidance blocks, session hook.
- A denominator: count Claude Code sessions so "3 calls/day" can be judged against "40 sessions/day."
- Fix the empty-state UX and the test-pollution bug.

## Non-goals

- **No external telemetry.** Data flows only from a user's clients to *their* fleet server, over their tailnet, with their existing bearer token. Public installs report to themselves. Nothing leaves the user's fleet.
- No raw log shipping / merged row-level fleet history (Approach B, rejected — aggregates answer the actual question).
- No automatic silent enforcement of config (audit fixes are user-initiated).

## Architecture Overview

```
┌────────────── client machine ──────────────┐      ┌───────── fleet server ─────────┐
│ Claude Code ──MCP──────────────────────────┼──────▶ MCP server                     │
│   (X-SP-Client: <hostname> header)         │      │   └─ activity.db (machine col) │
│ SessionStart hook → sp-session-ping        │      │                                │
│   └─ local activity.db (source=session)    │      │ profile server                 │
│ menubar heartbeat (15 min)                 │      │   POST /api/fleet/report       │
│   └─ aggregates + audit results ───────────┼──────▶   └─ fleet_usage,              │
│ sp-doctor / menubar Audit                  │      │       fleet_machines tables    │
└────────────────────────────────────────────┘      │ activity.html + Fleet view     │
                                                    └────────────────────────────────┘
```

The server participates in its own fleet: its menubar heartbeat calls the same report path against localhost, so the Fleet view uniformly includes every machine.

## Components

### 1. Activity schema v2 (`lib/activity.py`)

- Add `machine TEXT NOT NULL DEFAULT ''` to `requests` via `ALTER TABLE` guarded by `PRAGMA user_version` (0 → 1). Inserts stamp the local hostname (or the attributed client, below).
- New tables (created on the fleet server by the same `init_db`):
  - `fleet_usage(machine, day, tool, source, count, errors, avg_ms, PRIMARY KEY(machine, day, tool, source))`
  - `fleet_machines(machine PRIMARY KEY, version, mode, last_seen REAL, audit_json TEXT)`
- New functions: `upsert_fleet_report(report)` (idempotent upsert by primary key), `query_fleet()` (machines + last-7-day usage), `local_usage_summary(days=7)` (what the heartbeat sends), and `last_activity_at()` for the empty-state UX.
- `query_activity()` excludes `source='session'` rows from `history`/`tool_stats` and returns session counts separately — session pings are a denominator, not requests.
- `ACTIVITY_DB` in `lib/models.py` becomes overridable via `SP_ACTIVITY_DB` env var (test isolation).
- One-time migration cleanup: delete rows where `tool` is in the known junk set (`test`, `task_%`, `first_task`, `second_task`, `a`, `b`, `c`, `failing`, `test_tool`, `task1`, `task2`). Residual oddities are acceptable; the fixture below stops new pollution.

### 2. Caller attribution (`X-SP-Client` header)

- `install.sh` and the audit fix write the MCP entry with an extra header: `"X-SP-Client": "<hostname>"` alongside the existing `Authorization` header. Same for Codex/Gemini entries where their config supports headers.
- The MCP server's `BearerAuthMiddleware` (which already sees every request) stashes the header value in a `contextvars.ContextVar`. The `GPUTracker.__exit__` → `activity.log_request` path reads it; absent header → local hostname (covers the playground and unupgraded clients).
- The profile server does the same for playground requests (always local hostname) and for any proxied tool calls (forwards the original client's header).

### 3. Session denominator (`bin/sp-session-ping`)

- A small shell script, symlinked to `~/.local/bin/` like the other `bin/` scripts. It INSERTs one row into the *local* `activity.db` via the stock `sqlite3` CLI: `tool='session', source='session', model=<agent name: first argument, default "claude-code">, machine=<hostname>, duration_ms=0, status='ok'`. Codex/Gemini hooks (if those tools grow hook support) would call `sp-session-ping codex` etc. Direct DB write — works offline, needs no server, no token. Exits 0 always; wraps in `busy_timeout` and swallows failure (a lost ping must never break a Claude session).
- The audit installs a `SessionStart` hook in `~/.claude/settings.json` invoking `sp-session-ping`. The script's INSERT must match the schema; `lib/activity.py` owns the schema and a unit test asserts the script's insert works against a freshly-migrated DB (coupling made explicit and tested).

### 4. Fleet heartbeat (menubar → server)

- New `rumps.Timer` in `app/menubar.py` (pattern of `warm_timer`): every 15 minutes, and once shortly after startup, POST to `https://{server_fqdn}:8101/api/fleet/report` (localhost when `IS_SERVER=true`) with the existing profile-server bearer token.
- Payload: `{machine, version, mode, sent_at, audit: [check results], usage: [{day, tool, source, count, errors, avg_ms} for last 7 days]}` — built from `activity.local_usage_summary()` plus a fresh audit run (the checks are cheap local file reads, so the heartbeat just re-runs them).
- Server handler validates auth (existing decorator), upserts `fleet_usage` (idempotent — overlapping 7-day windows and retries are harmless) and `fleet_machines`.
- Failure handling: fire-and-forget with short timeout; log at debug, never notify. A machine that can't reach the server simply shows a stale `last_seen` in the Fleet view — which is itself signal.

### 5. Audit (`lib/audit.py` + `bin/sp-doctor` + menubar)

A registry of checks; each returns `{id, tool, status: pass|fail|warn|n/a, detail, fixable}` and optionally implements `fix()`. Checks skip (`n/a`) when the target tool isn't installed.

| Check | Verifies | Fix |
|---|---|---|
| `token-present` | `~/.config/local-models/mcp_auth_token` exists, non-empty | — (points at install.sh) |
| `claude-mcp` | `~/.claude.json` has `mcpServers.local-models` with URL + auth + `X-SP-Client` headers | `claude mcp add-json` (same entry install.sh writes) |
| `claude-guidance` | Managed SP block present and current in `~/.claude/CLAUDE.md` | Insert/refresh block between markers |
| `claude-hook` | `SessionStart` hook invoking `sp-session-ping` in `~/.claude/settings.json` | Insert hook (JSON merge, preserve existing hooks) |
| `codex-mcp` / `codex-guidance` | `~/.codex/config.toml` MCP entry; block in `~/.codex/AGENTS.md` | Write entry / block |
| `gemini-mcp` / `gemini-guidance` | `~/.gemini/settings.json` MCP entry; block in `~/.gemini/GEMINI.md` | Write entry / block |
| `other-agents` | Detects Cursor / Zed / OpenCode / Windsurf configs without SP | Report-only, links docs |

- **Managed guidance blocks** use markers (`<!-- >>> super-puppy >>> v<N> -->` … `<!-- <<< super-puppy <<< -->`). The canonical guidance text lives in one template in `lib/audit.py`, rendered per tool file. A version number in the marker lets the audit flag stale blocks and refresh idempotently — user content outside the markers is never touched. Guidance content: what SP is, that `local_*` MCP tools exist for vision/audio/image/video/bulk transforms/second opinions, and to check `local_models_status` for what's live.
- **Frontends:**
  - `bin/sp-doctor` — prints a pass/fail table; `sp-doctor --fix` applies fixable failures (prompting per fix unless `--yes`); `--json` for scripting. Symlinked by install.sh.
  - Menubar "Audit…" item — runs checks, shows results, offers "Fix All".
  - `install.sh` runs `sp-doctor --fix --yes` for the tools the user opts into during setup.
  - Heartbeat embeds the latest results so the Fleet view shows config health per machine.

### 6. Dashboard (`app/activity.html` + `/api/activity`, `/api/fleet`)

- **Empty state fix:** `/api/activity` returns `last_activity_at`; an empty period renders "No requests in this period — last activity July 7, 23:13" with a one-click jump to a window containing it.
- **Fleet view:** new section on the Activity page (server data; clients see it through the existing proxy). Per-machine cards: hostname, version, mode, `last_seen`, calls today/7d, sessions 7d, **calls-per-session**, and an audit badge (green, or red listing failing check ids). Sessions and calls charted per day.
- Responsive (mobile + desktop), consistent with existing pages' styles.

### 7. Test hygiene

- `tests/conftest.py` fixture (autouse) sets `SP_ACTIVITY_DB` to a tmp path so no test can touch the real DB again.
- New unit tests: schema migration (v0→v1 with existing data), fleet upsert idempotency, `query_activity` session exclusion, audit checks against a fabricated `$HOME` (tmp dir with fake `~/.claude.json` etc.), fix idempotency (running fix twice yields identical files), marker-block refresh, `sp-session-ping` insert against a migrated DB, heartbeat endpoint auth (401 without token).
- Smoke: extend the existing smoke suites with one heartbeat round-trip against the live profile server.

## Data Flow (remote call, end to end)

1. Claude Code on laptop calls `local_vision` → MCP request to server with `Authorization` + `X-SP-Client: jerry-laptop` headers.
2. Server middleware stashes client name; tool runs; `GPUTracker` logs to server's `activity.db` with `machine='jerry-laptop'`.
3. Laptop's Claude session had already fired `sp-session-ping` → row in laptop's local DB.
4. Laptop menubar heartbeat posts its 7-day aggregates (including session counts) + audit results to the server.
5. Server's Fleet view: jerry-laptop — last seen 2 min ago, 14 calls / 6 sessions this week, audit green.

## Error Handling

- All telemetry paths are best-effort and silent on failure (existing `log_request` philosophy): a logging or heartbeat failure must never break a tool call, a Claude session, or the menubar. Failures log at debug/warning locally.
- Audit **fixes** are the opposite — fail loud: if a config file can't be parsed (malformed `~/.claude.json`), report the failure and touch nothing.
- Heartbeat endpoint validates payload shape; malformed reports get 400 and are dropped (bad client version can't corrupt fleet tables).

## Rollout

- Schema migration is automatic and backward-compatible (old rows get `machine=''`, displayed as the server's own hostname).
- Old clients that haven't updated simply don't send heartbeats or headers — attribution falls back, Fleet view shows them once upgraded.
- Docs updated in the same commits: project `CLAUDE.md` (new runtime files/endpoints), `docs/architecture.md`, `sp-doctor` usage docs, README mention.
- No `PROFILES_VERSION` bump needed (no profile changes).
