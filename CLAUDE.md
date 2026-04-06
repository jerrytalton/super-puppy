# Super Puppy

Local AI model infrastructure for Apple Silicon. Menu bar app + standard APIs + MCP tools + Tailscale serving.

## Structure

- `app/` — Menu bar app (Python/rumps, PEP 723 inline deps)
- `app/SuperPuppy.app/` — macOS app bundle wrapping the menu bar app. A native C launcher (`super-puppy.c`) embeds Python via dlopen so macOS shows "Super Puppy" in Cmd-Tab and attributes screen recording permission to the app. Built by `install.sh`.
- `mcp/` — MCP server exposing local models as tools for Claude Code
- `lib/` — Shared Python library (model discovery, task filters, config paths)
- `bin/` — Shell scripts symlinked to `~/bin/`
- `config/` — Config files symlinked to `~/.config/` and `~/Library/LaunchAgents/`
- `tests/` — pytest unit tests (test_core.py, test_deployment.py) and end-to-end tests (test_e2e.py, test_error_handling.py)
- `install.sh` — Interactive setup: symlinks scripts, copies configs, walks through configuration

## Key Design Decisions

- The MCP server discovers models live from Ollama and MLX at startup (parallel `/api/show` calls). Any new `ollama pull` is immediately available as a tool.
- The menu bar app queries model capabilities live from Ollama `/api/show` and MLX `config.json` files in the HuggingFace cache. No hardcoded param tables.
- MLX models marked `on_demand: true` download on first use and unload after idle timeout.
- All remote access uses **Tailscale only** — no mDNS, no LAN binding. `tailscale serve` proxies all ports with TLS.
- The `local-models-mcp-detect` wrapper probes the desktop via Tailscale before launching. Clients use the server if reachable, fall back to localhost.

## Runtime Architecture

The menu bar app (`app/menubar.py`) launches via `app/SuperPuppy.app` and spawns:

- **Profile server** (`app/profile-server.py`) — Flask app on a fixed port (8101 on desktop, random on laptop). Serves the Model Profiles UI (`app/profiles.html`) and the Playground (`app/tools.html`). Auto-starts on desktop when Remote Access is enabled (no idle timeout).
- **Ollama** — `http://localhost:11434` (localhost-only; Tailscale serve proxies for remote access)
- **MLX-OpenAI-Server** — `http://localhost:8000`, config at `~/.config/mlx-server/config.yaml`

### Modes

| Mode | When | What happens |
|------|------|------|
| **Server** | `IS_SERVER=true` in network.conf | Runs Ollama, MLX, MCP locally. Tailscale exposes ports when Remote Access is on. |
| **Client** | Server reachable via Tailscale | Routes to server's MCP. Falls back to local if unreachable. |
| **Offline** | Laptop, desktop unreachable | Runs local Ollama/MLX as fallback. |

### Remote access (Tailscale)

All services bind to localhost. The "Remote Access" toggle in the menu bar manages `tailscale serve`, which proxies them with automatic TLS:

| Port | Service | URL pattern |
|------|---------|-------------|
| 8100 | MCP server | `https://{fqdn}:8100/mcp` |
| 8101 | Profile server / Playground | `https://{fqdn}:8101/tools` |
| 11434 | Ollama | `https://{fqdn}:11434` |
| 8000 | MLX | `https://{fqdn}:8000` |

**All remote URLs must use `https://{tailscale_fqdn}:{port}`**, not `http://{ip}`. Tailscale serve rejects plain HTTP.

### Auto-update

The app fetches tags every 2 minutes. If a newer tagged release exists:
1. Saves pre-update commit hash and service health snapshot
2. Checks out the new tag (detached HEAD)
3. Exits non-zero so launchd's KeepAlive restarts the app on new code

Users only receive tagged releases, not every push to main. To release: `git tag v1.x.x && git push --tags`.

Crash rollback: if the app dies within 30 seconds of an update, checks out the previous tag. Skipped releases aren't retried until a newer tag lands.

### MCP authentication

The MCP server requires a bearer token (`MCP_AUTH_TOKEN`). The token is stored in `~/.config/local-models/mcp_auth_token` (sourced from 1Password via the wrapper script). **The server refuses to start without a token** (fail-closed). Session IDs from authenticated `/mcp` init requests are tracked; subsequent `/messages` requests are validated against this set.

### Key files at runtime

| What | Where |
|------|-------|
| Profiles | `~/.config/local-models/profiles.json` |
| MCP preferences | `~/.config/local-models/mcp_preferences.json` |
| Network config | `~/.config/local-models/network.conf` |
| Mode override | `~/.config/local-models/mode.conf` |
| Remote access toggle | `~/.config/local-models/remote_access.conf` |
| Auth token | `~/.config/local-models/mcp_auth_token` |
| MLX server config | `~/.config/mlx-server/config.yaml` (user-writable, survives updates) |
| Claude MCP config | `~/.claude.json` |
| Menu bar log | `/tmp/local-models-menubar.log` |
| Instance lock | `~/.config/local-models/menubar.lock` |

### Task types

Profiles map these task types to models. Defined in `lib/models.py`:
- **Standard tasks:** `code`, `general`, `reasoning`, `long_context`, `translation`
- **Special tasks** (matched by model capability): `vision`, `computer_use`, `image_gen`, `image_edit`, `transcription`, `tts`, `embedding`, `unfiltered`

Task filters (`TASK_FILTERS`) and the `model_matches_filter()` function are shared across all three Python consumers via `lib/models.py`.

### Vision capability

Qwen3.5 models (served via MLX) ARE vision-capable. The MCP server must detect this correctly for `local_vision` to work. Vision detection for MLX models cannot rely on Ollama's `model_info` — it must check the model name or HuggingFace config. The `local_vision` tool must dispatch to the correct backend (Ollama `/api/chat` vs MLX `/v1/chat/completions` with OpenAI-style image content).

## Shared Library (`lib/models.py`)

Single source of truth for constants and logic used by menubar, MCP server, and profile server:
- `KNOWN_ACTIVE_PARAMS` — MoE active parameter lookup table
- `STANDARD_TASKS`, `SPECIAL_TASKS`, `TASK_FILTERS` — task definitions and model filtering
- `active_params_b()` — 4-strategy MoE active parameter computation (AXB parse → known table → FFN subtraction → ratio fallback)
- `model_matches_filter()` — check if a model qualifies for a task
- Config path constants (`PROFILES_FILE`, `MCP_PREFS_FILE`, `CLAUDE_CONFIG_FILE`, etc.)

## Local Model Tools (MCP)

The `mcp/local-models-server.py` MCP server runs as a persistent streamable-HTTP service on port 8100, managed by the menu bar app. It exposes Ollama, MLX, and local tool models (TTS via mlx-audio, image editing via mflux) as MCP tools. Claude connects via `type: "http"` to `http://127.0.0.1:8100/mcp` (local) or `https://{fqdn}:8100/mcp` (remote). Wrapper script is `bin/local-models-mcp-detect`.

Dependencies are pinned to exact versions in PEP 723 inline metadata.

## Testing

Run all tests: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -v`

- `tests/test_core.py` — 28 unit tests (mocked subprocesses, real sockets)
- `tests/test_deployment.py` — 53 tests for auto-update, rollback, tag verification, and post-update pipeline
- `tests/test_mcp_server.py` — 47 tests for model selection, GPU tracking, auth, job store, path validation
- `tests/test_profile_server.py` — 56 tests for Flask routes, profiles CRUD, model selection, config, auth
- `tests/test_playground_coverage.py` — 4 tests ensuring MCP tools have playground UI and API routes
- `tests/test_e2e.py` — 43 end-to-end tests against live services
- `tests/test_error_handling.py` — 24 tests for error handling and model validation
- `tests/test_remote_access.sh` — bash script testing Tailscale HTTPS endpoints

## Menu Bar Features

- **Remote / Local toggle** — switch between desktop and local models. "Local (override)" shown when user forced local but desktop is reachable.
- **Service status** — green/yellow/red dots for Ollama, MLX, MCP. Shows "restarting…" during auto-restart, "not shared" when MCP is unreachable in client mode.
- **Copy Diagnostics** — dumps mode, versions, service status, recent log lines to clipboard for remote debugging.
- **Version display** — from git tags (e.g. `v1.0.0`), shown in menu.
- **Notification debounce** — connectivity changes throttled to 60-second minimum interval.

## When Modifying This Repo

- After adding a new config file, add its `link` entry in `install.sh`.
- Never add secrets, tokens, or API keys.
- Menu bar app uses PEP 723 inline metadata for dependencies — no separate requirements.txt or venv.
- Pin to Python 3.12 (pyobjc-core doesn't build on 3.14+).
- Pin exact dependency versions in PEP 723 metadata (no `>=` ranges).
- Shared constants and logic go in `lib/models.py`, not duplicated across files.
- The compiled binary and .icns icon in `SuperPuppy.app` are gitignored — `install.sh` builds them. Only the C source (`app/super-puppy.c`) and Info.plist are tracked.
- If you change `app/super-puppy.c`, re-run `install.sh` or manually: `cc -o app/SuperPuppy.app/Contents/MacOS/super-puppy app/super-puppy.c && codesign --sign - --force app/SuperPuppy.app`.
- Run tests before pushing: `uv run --with pytest pytest tests/ -v`
