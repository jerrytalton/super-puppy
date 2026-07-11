# Usage Telemetry & Fleet Audit

Two related features: a per-machine activity log that rolls up into a fleet-wide view, and `sp-doctor`, an audit that checks whether your AI agent tools (Claude Code, Codex, Gemini CLI) are actually wired to use Super Puppy.

Design record: `docs/superpowers/specs/2026-07-10-usage-telemetry-fleet-audit-design.md`.

## What's logged, and where

Every MCP and Playground request is written to a local SQLite DB at `~/.config/local-models/activity.db` (override the path with `SP_ACTIVITY_DB`, mainly for tests). Each row records tool, model, backend, status, duration, and the machine that made the request.

**This data never leaves your own fleet.** It flows from a client machine to *your own* fleet server, over your tailnet, using the same bearer token you already configured. There is no external telemetry endpoint — a solo install with no fleet server simply never sends anything over the network for this feature.

`source='session'` rows (from `sp-session-ping`, below) are a session-count denominator, not requests — they're excluded from request/tool stats but counted separately, so "14 calls this week" can be read against "6 sessions this week."

### Machine attribution

The MCP server reads an `X-SP-Client` header (validated against `^[A-Za-z0-9._-]{1,64}$`) off each request and stamps it as the `machine` column via `log_request`. `install.sh` writes that header into the MCP entry it registers in `~/.claude.json`; an invalid or missing header is stamped `unknown-client` rather than silently attributed to the server's own hostname.

## Fleet heartbeat

The menu bar app pushes a heartbeat every 15 minutes (and once ~30 seconds after startup):

- **Where it goes**: `POST /api/fleet/report` on the fleet server's profile server — `http://127.0.0.1:8101` if this machine *is* the server, otherwise `https://{desktop_fqdn}:8101` over Tailscale. Always HTTPS + bearer auth off-box; `tailscale serve` rejects plain HTTP.
- **Payload**: `{machine, version, mode, sent_at, audit: [check results], usage: [{day, tool, source, count, errors, avg_ms}, ...]}` — the `usage` array is the last 7 days from the local activity DB (`local_usage_summary`), and `audit` is a fresh `sp-doctor`-equivalent check run (cheap local file reads).
- **Fire-and-forget**: an unreachable server, timeout, or any exception is swallowed and logged at debug — a heartbeat failure never affects app stability or blocks a tool call.
- **Server-side validation**: `machine`/`version`/`mode` and each usage item's `tool`/`source`/`day` must match `^[A-Za-z0-9._-]{1,64}$` (day: `YYYY-MM-DD`); `count`/`errors`/`avg_ms` must be integers. Malformed or non-dict bodies get a 400.
- **Rate limit**: one accepted report per machine per 5 minutes; excess reports get 429.
- **`last_seen` is stamped server-side** on receipt — never trusted from the client.
- Upserts are idempotent (`fleet_usage` keyed on `machine, day, tool, source`; `fleet_machines` keyed on `machine`), so retries and overlapping 7-day windows are harmless. Both tables are pruned to 30 days.

`GET /api/fleet` serves the aggregated data — every machine's last-seen/version/mode/audit plus the last 7 days of usage — to the Fleet view.

## Fleet view

A section on the Activity page (`app/activity.html`), polling `/api/fleet` every 10 seconds. One card per machine:

| Field | Meaning |
|-------|---------|
| Hostname | The `machine` name from the heartbeat |
| Version / mode | App version and `server`/`client` |
| Last seen | Time since the server received that machine's last heartbeat |
| Calls today / Calls 7d | Non-session requests, from `fleet_usage` |
| Sessions 7d | Count of `source='session'` rows — how many agent sessions started |
| Calls / session | `calls_7d / sessions_7d` — the adoption signal: are sessions actually calling SP tools? |
| Audit badge | Green ("all checks pass") or lists the failing check ids, from that machine's latest embedded audit run |

Every server-supplied string (`machine`, `version`, `mode`, tool names, audit `detail` text) is rendered with `textContent` only — no `innerHTML`, no template interpolation — because these fields are attacker-influenceable (a hostile `X-SP-Client` header, a spoofed heartbeat). The profile server also sends a `Content-Security-Policy` header as defense in depth behind that render discipline.

If a period has no requests, the page shows "No requests in this period — last activity `{time}`" instead of a bare empty state.

## `sp-doctor`

Checks (and optionally fixes) whether this machine's agent tools are wired to use Super Puppy: MCP registration, the managed guidance block, and the `SessionStart` hook, for Claude Code, Codex, and Gemini CLI. Symlinked to `~/.local/bin/sp-doctor` by `install.sh`.

### Usage

```bash
sp-doctor              # print the audit table
sp-doctor --fix        # apply fixable failures, prompting per fix
sp-doctor --fix --yes  # apply without prompting (used by install.sh)
sp-doctor --json       # machine-readable output, for scripting
```

Exit code is `1` if any check reports `fail`, `0` otherwise (all `pass`/`warn`/`n/a`) — safe to use as a CI-style gate.

### What each check verifies

| Check | Verifies | Fix |
|---|---|---|
| `token-present` | `~/.config/local-models/mcp_auth_token` exists and is non-empty | none — points you at `install.sh` |
| `claude-mcp` | `~/.claude.json` has `mcpServers.local-models` with a URL and an `X-SP-Client` header | Writes the entry (token inlined only if safe — see below) |
| `claude-guidance` | The managed guidance block is present and current in `~/.claude/CLAUDE.md` | Inserts/refreshes the block between markers |
| `claude-hook` | A `SessionStart` hook invoking `sp-session-ping` exists in `~/.claude/settings.json` | Adds the hook, preserving any other hooks already configured |
| `codex-mcp` / `codex-guidance` | Same MCP-entry / guidance-block checks for `~/.codex/config.toml` / `~/.codex/AGENTS.md` | Same, TOML-aware; `n/a` if `~/.codex` doesn't exist |
| `gemini-mcp` / `gemini-guidance` | Same for `~/.gemini/settings.json` / `~/.gemini/GEMINI.md` | Same; `n/a` if `~/.gemini` doesn't exist |
| `other-agents` | Detects Cursor / opencode / Windsurf config directories | Report-only — points at `docs/troubleshooting.md`, never writes anything |

A check reports `n/a` (not `fail`) when the corresponding tool isn't installed, so an un-configured Codex on a Claude-only machine doesn't show as broken.

### The managed guidance block

Each fix writes (or refreshes) a block delimited by HTML comment markers — `<!-- >>> super-puppy >>> -->` … `<!-- <<< super-puppy <<< -->` — into the tool's guidance file (`CLAUDE.md`, `AGENTS.md`, or `GEMINI.md`). Content: what Super Puppy is, a need→tool trigger table (e.g. "Look at an image or screenshot" → `local_vision`), and a directive to run work on the local cluster in parallel. Nothing outside the markers is ever touched, and re-running the fix is idempotent (refresh is by content, not a version bump).

**To opt out**, delete the block (the two marker lines and everything between them) from the file — `sp-doctor` will report `claude-guidance` (or the Codex/Gemini equivalent) as failing, but nothing re-adds it unless you run `--fix` again.

**Fixes are always user-confirmed.** `sp-doctor --fix` prompts per finding unless you pass `--yes`; `install.sh` uses `--yes` only because you've already opted in during setup. Guidance-block writes in particular are never applied automatically on a version bump — see Security posture, §S2, below.

### Token-leak guard

When a fix needs to write the auth token into a config file (`claude-mcp`, `codex-mcp`, `gemini-mcp`), it inlines the token only if the target file is not group/other-readable and not inside a git work tree; otherwise it writes the entry without the token and says so in its output. Whenever a token *is* inlined, the file is `chmod`'d `0600` immediately after — including the fresh-machine case, where an ambient umask would otherwise leave a newly created file world-readable.

## Security posture

From the design's red-team pass (spec §S1–S6):

- **S1 — Dashboard XSS.** Every fleet-supplied field is attacker-influenceable. The Fleet view renders exclusively via `textContent`, backed by a `Content-Security-Policy` header and server-side field validation on ingest.
- **S2 — Trust boundary.** The audit writes into files an AI agent treats as instructions. Those writes are never automatic — always an explicit `sp-doctor --fix` / menubar "Fix" / opted-in `install.sh` run — so a routine signed-tag auto-update can never silently rewrite every machine's agent guidance.
- **S3 — No SQL injection.** `sp-session-ping` writes via bound parameters, never string-interpolated SQL; its one argument is validated against `^[a-z0-9-]{1,32}$` before use.
- **S4 — Token handling.** The audit refuses to inline the auth token into a world-readable or git-tracked config file (token-leak guard, above); per-machine tokens are a documented future hardening, not required for this version.
- **S5 — Multi-writer SQLite.** Every writer (MCP server, profile server, `sp-session-ping`) sets `busy_timeout`; `fleet_usage`/`fleet_machines` get the same 30-day retention as `requests`' existing 90-day prune.
- **S6 — Config-write safety.** All audit fixes write atomically (temp file + rename, `.bak` of the prior content) and merge rather than replace — existing hooks and MCP server entries are preserved.
