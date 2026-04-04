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

Persistent streamable-HTTP service on port 8100. Discovers models from Ollama and MLX at startup. Exposes 16 tools (generation, vision, image, audio, embeddings, etc.) as MCP resources. Requires bearer token auth — fails closed without a token. Session IDs are tracked per-connection.

### Profile server (`app/profile-server.py`)

Flask app serving the Profiles UI and Playground. Fixed port 8101 on desktop, random port on laptop. The Playground is a PWA installable on phones/tablets. Requires bearer token auth for remote requests; localhost requests (menu bar webview) skip auth.

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

Both the MCP server and profile server enforce bearer token auth on remote requests. Localhost requests to the profile server skip auth (for the menu bar webview).

## Server / Client Model

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Server** | `IS_SERVER=true` in network.conf | Runs all services locally. Tailscale exposes them when Remote Access is on. |
| **Client** | Server reachable via Tailscale | Routes MCP requests to server. |
| **Offline** | Server unreachable | Falls back to local Ollama/MLX. |

Clients discover the server by querying `tailscale status --json` for a peer matching the configured `TAILSCALE_HOSTNAME`. The MCP wrapper script (`bin/local-models-mcp-detect`) handles this transparently.

## Auto-update

The menu bar app fetches git tags every 2 minutes. When a newer tag exists:

1. Verifies the tag's GPG/SSH signature (rejects unsigned tags)
2. Saves pre-update commit hash and service health snapshot
3. Checks out the tag (detached HEAD)
4. Exits non-zero so launchd's KeepAlive restarts on new code

Crash rollback: if the app dies within 30 seconds of an update, it checks out the previous tag.

## Security Model

- **Auth**: MCP server and profile server both require bearer token for remote requests (sourced from 1Password, fail-closed). Profile server allows localhost without auth.
- **Network**: All services localhost-only. Tailscale provides encrypted, peer-authenticated remote access.
- **Path validation**: All MCP tools that accept file paths validate them against `$HOME` and `/tmp` — no traversal outside these roots.
- **Update verification**: Auto-update refuses unsigned git tags.
- **Install script**: Hostname inputs validated against DNS-safe characters. Config writes use grep+append (no sed injection).

## Task System

Profiles map task types to models. Defined in `lib/models.py`:

- **Standard tasks**: `code`, `general`, `reasoning`, `long_context`, `translation`
- **Special tasks** (matched by model capability): `vision`, `computer_use`, `image_gen`, `image_edit`, `transcription`, `tts`, `embedding`, `unfiltered`

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
