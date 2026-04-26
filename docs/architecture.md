# Architecture

Super Puppy is a local AI model server for Apple Silicon, managed from the macOS menu bar. It exposes models via standard APIs (Ollama, OpenAI-compatible) and MCP tools for Claude Code.

## Components

```
┌─────────────────────────────────────────────────────┐
│  SuperPuppy.app (menu bar)                          │
│  ├── menubar.py          rumps-based menu bar app   │
│  ├── profile-server.py   Flask: Profiles + Playground│
│  └── manages:                                       │
│      ├── Ollama          localhost:11434             │
│      ├── MLX server      localhost:8000              │
│      └── MCP server      localhost:8100              │
└─────────────────────────────────────────────────────┘
         │ tailscale serve (optional)
         ▼
  https://{fqdn}:{port}  ← remote clients
```

### Menu bar app (`app/menubar.py`)

The central coordinator. Launches via a native macOS app bundle (`SuperPuppy.app`) so it appears in Cmd-Tab. Manages the lifecycle of Ollama, MLX, MCP, and profile server processes. Handles auto-update, mode detection (server/client/offline), service health monitoring, and Tailscale remote access toggling.

### MCP server (`mcp/local-models-server.py`)

Persistent streamable-HTTP service on port 8100. Discovers models from Ollama and MLX at startup. Exposes 17 tools (generation, vision, image, audio, video, embeddings, etc.) as MCP resources. Requires bearer token auth — fails closed without a token. Session IDs are tracked per-connection.

### Profile server (`app/profile-server.py`)

Flask app serving the Profiles UI and Playground. Fixed port 8101 on desktop, random port on laptop. The Playground is a PWA installable on phones/tablets. Bearer token auth required on every request — there is no localhost shortcut, because `tailscale serve` forwards remote requests as if they came from `127.0.0.1`. The menu bar's WKWebView injects the token via `window.__SP_TOKEN__` and an `Authorization` header on the initial nav request. Refuses to start without `MCP_AUTH_TOKEN` (`SP_ALLOW_NO_AUTH=1` is the explicit dev/test escape hatch).

### Shared library (`lib/models.py`)

Single source of truth for model constants, task definitions, capability filters, and config paths. Used by all three Python components to ensure consistent behavior.

## Networking

All services bind to `127.0.0.1`. Remote access uses Tailscale exclusively — no mDNS, no LAN binding. The menu bar's "Remote Access" toggle manages `tailscale serve`, which proxies local ports with automatic TLS:

| Port | Service |
|------|---------|
| 8100 | MCP server |
| 8101 | Profile server / Playground |
| 11434 | Ollama |
| 8000 | MLX |

Both the MCP server and profile server enforce bearer token auth on **every** request. There is no localhost shortcut: `tailscale serve` proxies remote requests as if they originated from `127.0.0.1`, so trusting the loopback address would silently bypass auth for any tailnet peer. Native `<img>`/`<audio>`/`<video>` elements that can't set headers may pass `?token=` on GETs only.

## Server / Client Model

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Server** | `IS_SERVER=true` in network.conf | Runs all services locally. Tailscale exposes them when Remote Access is on. |
| **Client** | Server reachable via Tailscale | Routes MCP requests to server. |
| **Offline** | Server unreachable | Falls back to local Ollama/MLX. |

Clients discover the server by querying `tailscale status --json` for a peer matching the configured `TAILSCALE_HOSTNAME`. The MCP wrapper script (`bin/local-models-mcp-detect`) handles this transparently.

## Auto-update

The menu bar app fetches git tags every 2 minutes. When a newer tag exists:

1. Verifies the tag's GPG/SSH signature against the **existing** `allowed_signers` (current trust root must approve the next one — see Trust root rotation below)
2. Saves pre-update commit hash and service health snapshot
3. Installs any new `allowed_signers` shipped in the verified tag
4. Checks out the tag (detached HEAD)
5. Exits non-zero so launchd's KeepAlive restarts on new code

Crash rollback: if the app dies within 90 seconds of an update (`UPDATE_CRASH_WINDOW`), it checks out the previous tag.

**Trust root rotation.** A tag may carry an updated `config/git/allowed_signers`. We verify the tag against the current file *first*, then install the new file only if verification passed. Rotations therefore have to ride a tag signed by the *outgoing* key — old installs upgrade through it, then accept the new key on the next round. Verifying *after* installing the new file would let any pushed tag self-approve, defeating the entire signed-update model.

## Security Model

- **Auth**: MCP server and profile server both require a bearer token (`MCP_AUTH_TOKEN`, sourced from 1Password) on **every** request — no localhost shortcut. Both fail closed: refuse to start without a token unless `SP_ALLOW_NO_AUTH=1` is set explicitly.
- **Network**: All services localhost-only. Tailscale provides encrypted, peer-authenticated remote access. `tailscale serve` listens on https only.
- **Path validation**: MCP tools that accept file paths validate them against `$HOME` and `/tmp` — no traversal outside these roots. Roots configurable via `MCP_ALLOWED_PATHS` in `network.conf`. Per-tool extension allowlists (`_IMAGE_EXTS`, `_AUDIO_EXTS`, `_VIDEO_EXTS`, `_TEXT_EXTS`) gate the file types each tool can read, so a prompt-injected call can't feed `~/.ssh/id_rsa` or `~/.aws/credentials` to a model.
- **Playground path restriction**: The Playground web UI restricts file access to `/tmp/` only (uploaded files), with an extension allowlist, a 100 MB cap, and a randomized basename per upload.
- **Computer use**: `local_computer_use` requires the caller to supply `screenshot_path`. There is no auto-capture — that would let a tailnet peer with a leaked token harvest screenshots silently.
- **Update verification**: Auto-update refuses unsigned git tags. Verification runs against the existing `allowed_signers` *before* any new trust root from the tag is installed (see Auto-update / Trust root rotation).
- **Install script**: Hostname inputs validated against DNS-safe characters. Config writes use grep+append (no sed injection).

### Known Limitations

- **Single shared bearer token**: All clients share one auth token. No per-user accounts, quotas, or audit trail. Revoking one user's access requires rotating the token for everyone. Tailscale ACLs provide per-device access control as a partial mitigation.
- **Auto-update polling**: Fetches git tags from GitHub every 2 minutes. Disable with `AUTO_UPDATE=false` in `network.conf`.

## Task System

Profiles map task types to models. Defined in `lib/models.py`:

- **Standard tasks**: `code`, `general`, `reasoning`, `long_context`, `translation`
- **Special tasks** (matched by model capability): `vision`, `computer_use`, `image_gen`, `image_edit`, `video`, `transcription`, `tts`, `embedding`, `unfiltered`

Task filters and the `model_matches_filter()` function are shared across all three Python consumers.

## Key Files at Runtime

| What | Where |
|------|-------|
| Profiles | `~/.config/local-models/profiles.json` |
| MCP preferences | `~/.config/local-models/mcp_preferences.json` |
| Network config | `~/.config/local-models/network.conf` |
| Auth token | `~/.config/local-models/mcp_auth_token` |
| MLX server config | `~/.config/mlx-server/config.yaml` |
| Menu bar log | `/tmp/local-models-menubar.log` |
| Instance lock | `~/.config/local-models/menubar.lock` |
