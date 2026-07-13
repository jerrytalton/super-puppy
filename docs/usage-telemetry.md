# Usage Telemetry & Fleet Audit

Two related features: a per-machine activity log that rolls up into a fleet-wide view, and `sp-doctor`, an audit that checks whether your AI agent tools (Claude Code, Codex, Gemini CLI) are actually wired to use Super Puppy.

Design record: `docs/superpowers/specs/2026-07-10-usage-telemetry-fleet-audit-design.md`.

## What's logged, and where

Every MCP and Playground request is written to a local SQLite DB at `~/.config/local-models/activity.db` (override the path with `SP_ACTIVITY_DB`, mainly for tests). Each row records tool, model, backend, status, duration, and the machine that made the request.

**This data never leaves your own fleet.** It flows from a client machine to *your own* fleet server, over your tailnet, using the same bearer token you already configured. There is no external telemetry endpoint — a solo install with no fleet server simply never sends anything over the network for this feature.

`source='session'` rows (from `sp-session-ping`, below) are a session-count denominator, not requests — they're excluded from request/tool stats but counted separately, so "14 calls this week" can be read against "6 sessions this week."

### Machine attribution

The MCP server reads an `X-SP-Client` header (validated against `^[A-Za-z0-9._-]{1,64}$`) off each request and stamps it as the `machine` column via `log_request`. `install.sh` writes that header into the MCP entry it registers in `~/.claude.json`; an invalid or missing header is stamped `unknown-client` rather than silently attributed to the server's own hostname.

### The MCP token — by env var, not inlined

The MCP entry's `Authorization` header is `Bearer ${SP_MCP_TOKEN}`, not a literal token. Claude Code expands `${SP_MCP_TOKEN}` from the environment at load time, so the secret never lands in `~/.claude.json` (the file is safe to sync or commit). Provision the env var from the untracked token file — e.g. in your shell rc:

```bash
export SP_MCP_TOKEN="$(cat ~/.config/local-models/mcp_auth_token 2>/dev/null)"
```

If `SP_MCP_TOKEN` is unset or wrong, the MCP server returns 403 (tools unavailable) — a clean failure, not a silent one. GUI-launched Claude Code needs the var in the GUI environment too (`launchctl setenv SP_MCP_TOKEN …`).

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

Checks (and, only when you ask, fixes) whether this machine's agent tools are wired to use Super Puppy: MCP registration, usage guidance, and the `SessionStart` hook, for Claude Code, Codex, and Gemini CLI. Symlinked to `~/.local/bin/sp-doctor` by `install.sh`.

The audit is **diagnostic by default** — `sp-doctor` (no flag), the menu bar's **Audit** page, and `install.sh` only *report*. Nothing writes to your config until you explicitly run `sp-doctor --fix` (or click a Fix button on the Audit page) and confirm. See [Consent model](#consent-model).

### Usage

```bash
sp-doctor              # print the audit table (read-only)
sp-doctor --fix        # apply fixable failures — shows each change, asks first
sp-doctor --fix --yes  # apply without prompting (power users / scripts)
sp-doctor --json       # machine-readable output, for scripting
```

Exit code is `1` if any check reports `fail`, `0` otherwise (all `pass`/`warn`/`n/a`) — safe to use as a CI-style gate.

### The Audit page

The menu bar's **Audit** item opens a web page (`app/audit.html`, served at `/audit`) — grouped cards per tool (Claude Code / Codex / Gemini / Super Puppy / Other), a colored status dot per check (green pass, red fail, amber warn, grey n/a), inapplicable checks greyed out, and a per-group **Fix** button that applies just that tool's fixable findings (`POST /api/audit/fix` → `audit.fix_group`). Rendered XSS-safe via `textContent` behind the profile server's CSP.

### What each check verifies

| Check | Verifies | Fix |
|---|---|---|
| `token-present` | `~/.config/local-models/mcp_auth_token` exists and is non-empty | none — points you at `install.sh` |
| `claude-mcp` | `~/.claude.json` has `mcpServers.local-models` with a URL and an `X-SP-Client` header | Merges the entry (one key; other MCP servers untouched); token by `${SP_MCP_TOKEN}`, never inlined |
| `claude-guidance` | `~/.claude/CLAUDE.md` contains local-models guidance — the managed block **or** your own hand-written section | Appends the managed block only if the file has none; never duplicates or overwrites your own |
| `claude-hook` | A `SessionStart` hook invoking `sp-session-ping` exists in `~/.claude/settings.json` | Adds the hook, preserving any other hooks already configured |
| `codex-mcp` / `codex-guidance` | Same MCP-entry / guidance-block checks for `~/.codex/config.toml` / `~/.codex/AGENTS.md` | Same, TOML-aware; `n/a` if `~/.codex` doesn't exist |
| `gemini-mcp` / `gemini-guidance` | Same for `~/.gemini/settings.json` / `~/.gemini/GEMINI.md` | Same; `n/a` if `~/.gemini` doesn't exist |
| `other-agents` | Detects Cursor / opencode / Windsurf config directories | Report-only — points at `docs/troubleshooting.md`, never writes anything |

A check reports `n/a` (not `fail`) when the corresponding tool isn't installed, so an un-configured Codex on a Claude-only machine doesn't show as broken.

### The managed guidance block

If a guidance file (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`) has *no* local-models guidance, the fix **appends** a block delimited by HTML comment markers — `<!-- >>> super-puppy >>> -->` … `<!-- <<< super-puppy <<< -->`. Content: what Super Puppy is, a need→tool trigger table (e.g. "Look at an image or screenshot" → `local_vision`), and a directive to run work on the local cluster in parallel.

**If you already maintain your own guidance** — a `## Local Models` section, or any mention of the tools (`local_models_status`, `local_dispatch`, …) — the check *passes* and the fix does nothing. The audit never duplicates or overwrites guidance you wrote yourself.

Writes go **through symlinks** to the real target (a `CLAUDE.md` symlinked into a dotfiles repo keeps its link and stays in sync), atomically, `.bak` first. **To opt out** entirely, delete the block (the two markers and everything between); `sp-doctor` reports `claude-guidance` as failing, but nothing re-adds it unless you run `--fix`.

### Consent model

The audit never writes to files you hand-maintain without your explicit say-so:

- **`install.sh`** registers the MCP entry (to `~/.claude.json`, a Claude-managed config, preserving your other servers) and *reports* the audit — it does **not** auto-apply the guidance block or session hook. It prints the `sp-doctor --fix` command to opt into those.
- **`sp-doctor --fix`** states that fixes append or merge — never overwrite your own content — shows each change's target file, and asks per item (unless you deliberately pass `--yes`).
- **The Audit page's Fix buttons** are explicit clicks, per tool group.
- **Every fix only adds**: guidance appends-or-skips, the hook appends (preserving other hooks), the MCP entry merges one key. Nothing you wrote is rewritten. And no fix is ever applied automatically on a version-bump auto-update.

## Security posture

From the design's red-team pass (spec §S1–S6):

- **S1 — Dashboard XSS.** Every fleet-supplied field is attacker-influenceable. The Fleet view renders exclusively via `textContent`, backed by a `Content-Security-Policy` header and server-side field validation on ingest.
- **S2 — Trust boundary.** The audit writes into files an AI agent treats as instructions. Those writes are never automatic — only an explicit `sp-doctor --fix`, a Fix button on the Audit page, or your own re-run — and `install.sh` reports rather than fixes (see [Consent model](#consent-model)). So a routine signed-tag auto-update can never silently rewrite every machine's agent guidance.
- **S3 — No SQL injection.** `sp-session-ping` writes via bound parameters, never string-interpolated SQL; its one argument is validated against `^[a-z0-9-]{1,32}$` before use.
- **S4 — Token handling.** The auth token is referenced by env var (`${SP_MCP_TOKEN}`), never written into a config file — so there's no inlined secret to leak, and `~/.claude.json` is safe to sync or commit. (This supersedes the earlier inline-with-guard approach; per-machine tokens remain a documented future hardening.)
- **S5 — Multi-writer SQLite.** Every writer (MCP server, profile server, `sp-session-ping`) sets `busy_timeout`; `fleet_usage`/`fleet_machines` get the same 30-day retention as `requests`' existing 90-day prune.
- **S6 — Config-write safety.** All audit fixes write atomically (temp file + rename, `.bak` of the prior content) and merge rather than replace — existing hooks and MCP server entries are preserved.
