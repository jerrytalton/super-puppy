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
- The MCP server's `BearerAuthMiddleware` (which already sees every request) validates the header against `^[A-Za-z0-9._-]{1,64}$` and stashes it in a `contextvars.ContextVar`. The `GPUTracker.__exit__` → `activity.log_request` path reads it; absent/invalid header → `unknown-client` (never the server's own hostname, which would hide a misconfigured client's usage inside the server's row). Playground requests stamp the local hostname directly.
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

- **Managed guidance blocks** use markers (`<!-- >>> super-puppy >>> v<N> -->` … `<!-- <<< super-puppy <<< -->`). The canonical guidance text lives in one template in `lib/audit.py`, rendered per tool file. A version number in the marker lets the audit flag stale blocks and refresh idempotently — user content outside the markers is never touched. Guidance content follows the restored dotfiles section (dotfiles `8bfd5b8`): a **need → tool trigger table** keyed on stable tool names (no rot-prone model/hardware specifics — runtime descriptions and `local_models_status` carry those), plus the explicit directive to **work in parallel on the server and local cluster whenever it makes sense** (`local_dispatch` early, not after). Enumerated triggers, not abstract capability prose — the June 30 abstraction (dotfiles `9dfe4e6`) preceded a drop in MCP call volume.
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

## Security (red-team mitigations, 2026-07-10)

A context-free red-team pass raised issues that change the design. Folded in here; each maps to the component above.

### S1. Dashboard XSS — the highest-leverage bug (was implicit, now required)

Every field the Fleet view renders is attacker-influenceable: `X-SP-Client`/`machine`, `version`, `mode`, tool names, `error_msg`, and audit `detail` strings. On the profile-server origin — which holds the bearer token and exposes the audit-fix API — a stored `<img onerror>` would run in the owner's browser and could drive config-tampering fixes. Requirements:

- **Render with `textContent` / DOM node creation only. No `innerHTML`, no template interpolation of any server-supplied string.** This is a hard rule for `activity.html`, called out in the plan and checked in review.
- Serve the profile server with a strict `Content-Security-Policy: default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'` (adjust to the page's actual needs; no inline event handlers).
- **Server-side validation on ingest**: `machine`, `version`, `mode` must match `^[A-Za-z0-9._-]{1,64}$` or the report is rejected (400). Tool/source names validated against the known task/source vocabulary. `audit` check ids rendered from a client-side allowlist; free-text `detail`/`error_msg` always escaped as text.

### S2. The audit is an agent-instruction channel — keep it off the auto-update rails

The audit writes text into files an AI agent treats as instructions (CLAUDE.md/AGENTS.md/GEMINI.md guidance blocks). Auto-update ships signed tags fleet-wide in ~2 minutes. Those two must not compose into "a routine release silently rewrites every machine's agent guidance." Decisions:

- **Guidance-block writes are never automatic.** They happen only via an explicit `sp-doctor --fix` / menubar "Fix" / `install.sh` opt-in, each showing a **before/after diff** and requiring confirmation (`--yes` bypasses only in install.sh where the user already opted in). A version bump alone never rewrites a block.
- The managed block's **refresh trigger is content difference, not code version** — idempotent, so an auto-update or a crash-rollback tag change never thrashes the file (kills the flap in S6). The marker still carries a version for display, but equality is by rendered content.
- The block stays **minimal and mechanical**: what SP is, the trigger→tool table, the parallelism directive, `local_models_status`. No open-ended behavioral instructions. Its canonical text is one reviewed template; changing it is a visible diff in a PR, same as code.
- Docs state explicitly that the audit crosses the code-vs-prompt trust boundary and is gated behind user confirmation for exactly that reason.

### S3. `sp-session-ping` must not build SQL by string interpolation

The hook writes to the local DB with `model` from `$1` and `machine` from `hostname`. A shell heredoc interpolating those is a SQL-injection / DB-corruption vector (and a way to plant XSS rows that ride the heartbeat to S1). Requirements:

- The ping writes via a tiny Python one-liner (or a shared `lib` helper) using **bound parameters** — never `sqlite3` CLI string interpolation. `$1` is validated against `^[a-z0-9-]{1,32}$` and dropped to the default otherwise.
- Every connection, CLI or not, sets `PRAGMA busy_timeout=3000` (also mitigates S5).

### S4. Identity, tokens, and the `/api/fleet/report` write path

At personal-fleet scale (the user's own devices, shared tailnet) full multi-tenant identity is overkill, but two things are still required because SP is public software and the token leaks easily:

- **Keep the token out of a world-readable file.** `~/.claude.json` is often group/other-readable, synced, or committed. The audit's `claude-mcp` fix **refuses to inline the token** if `~/.claude.json` is group/other-readable or inside a git work tree, and warns loudly otherwise. Prefer referencing the mode-600 token file if the MCP client supports it.
- **`machine` is validated (S1) and the endpoint is rate-limited** — one accepted report per token per 5 minutes; excess → 429. Caps unbounded-cardinality growth and heartbeat storms.
- **`last_seen` is stamped server-side** from the server's clock on receipt, never taken from the client (S7). Client `day`/`avg_ms`/counts are advisory; `day` is clamped to a sane window before upsert.
- Per-machine tokens are **noted as a future hardening** (revocation, attribution-on-leak) but not required for v1; documented as an accepted limitation with the reasoning.

### S5. SQLite multi-writer & retention

- The hook adds a third writer (sqlite3/Python) to `activity.db`. `busy_timeout=3000` on all connections (S3); the MCP logger already swallows lock errors — verify it does so without dropping the tool response.
- Retention: `lib/activity.py` already prunes `requests` at 90 days. **The new `fleet_usage`/`fleet_machines` tables get the same treatment** — `fleet_usage` TTL'd past the dashboard's 7-day window (keep 30 for slack), stale `fleet_machines` rows pruned, with a periodic `wal_checkpoint(TRUNCATE)`.

### S6. Config-write races and array clobbering

The audit fixes mutate files Claude Code itself writes (`~/.claude.json`, `~/.claude/settings.json`):

- **Atomic writes**: temp file + `rename`, after re-reading immediately before write.
- **Merge, never replace**: the SessionStart hook is added to the existing `hooks` array by key; the MCP entry is added alongside existing servers. Existing user hooks/servers are preserved. A `.bak` copy is written before mutating.
- Note: editing `~/.claude/settings.json` is blocked by Jerry's backup-protection PreToolUse hook when *an agent* does it, but `sp-doctor` runs as a user tool, not via the agent's Edit/Bash — it operates outside that guard by design. The guard exists to stop agent self-elevation; a user-invoked audit is the sanctioned path. Called out so the interaction is intentional, not a surprise.

## Data Flow (remote call, end to end)

1. Claude Code on laptop calls `local_vision` → MCP request to server with `Authorization` + `X-SP-Client: jerry-laptop` headers.
2. Server middleware validates `X-SP-Client` (charset), stashes client name; tool runs; `GPUTracker` logs to server's `activity.db` with `machine='jerry-laptop'`.
3. Laptop's Claude session had already fired `sp-session-ping` (bound-param insert) → row in laptop's local DB.
4. Laptop menubar heartbeat posts its 7-day aggregates (including session counts) + audit results; server validates, stamps `last_seen`, upserts.
5. Server's Fleet view renders (via `textContent`): jerry-laptop — last seen 2 min ago, 14 calls / 6 sessions this week, audit green.

## Error Handling

- All telemetry paths are best-effort and silent on failure (existing `log_request` philosophy): a logging or heartbeat failure must never break a tool call, a Claude session, or the menubar. Failures log at debug/warning locally.
- Audit **fixes** are the opposite — fail loud: if a config file can't be parsed (malformed `~/.claude.json`), report the failure and touch nothing.
- Heartbeat endpoint validates auth, charset (S1), and payload shape; malformed or over-rate reports get 400/429 and are dropped (a bad or hostile client can't corrupt fleet tables).

## Rollout

- Schema migration is automatic and backward-compatible (old rows get `machine=''`, rendered as `unknown-client` — never silently folded into the server's own hostname, per S1/attribution-ambiguity).
- Old clients that haven't updated simply don't send heartbeats or headers — attribution falls back, Fleet view shows them once upgraded.
- Docs updated in the same commits: project `CLAUDE.md` (new runtime files/endpoints), `docs/architecture.md`, `sp-doctor` usage docs, README mention, and the S2 trust-boundary note.
- No `PROFILES_VERSION` bump needed (no profile changes).
