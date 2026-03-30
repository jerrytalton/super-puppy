# Super Puppy

Local AI model infrastructure for Claude Code. MCP tools + menu bar app + LAN serving.

## Structure

- `app/` — Menu bar app (Python/rumps, PEP 723 inline deps)
- `app/SuperPuppy.app/` — macOS app bundle wrapping the menu bar app. A native C launcher (`super-puppy.c`) embeds Python via dlopen so macOS shows "Super Puppy" in Cmd-Tab and attributes screen recording permission to the app. Built by `install.sh`.
- `mcp/` — MCP server exposing local models as tools for Claude Code
- `bin/` — Shell scripts symlinked to `~/bin/`
- `config/` — Config files symlinked to `~/.config/` and `~/Library/LaunchAgents/`
- `install.sh` — Creates symlinks, builds the app bundle, checks dependencies, detects desktop vs laptop

## Key Design Decisions

- The MCP server discovers models live from Ollama and MLX at startup. Any new `ollama pull` is immediately available as a tool.
- The menu bar app queries model capabilities live from Ollama `/api/show` and MLX `config.json` files in the HuggingFace cache. No hardcoded param tables.
- MLX models marked `on_demand: true` download on first use and unload after idle timeout.
- The desktop hostname for LAN serving is in `config/local-models/network.conf`, detected via mDNS (Bonjour).
- The `local-models-mcp-detect` wrapper probes the desktop before launching. Laptops use the desktop if reachable, fall back to localhost.

## Runtime Architecture

The menu bar app (`app/menubar.py`) launches via `app/SuperPuppy.app` and spawns:

- **Profile server** (`app/profile-server.py`) — Flask app on a random port. Serves the Model Profiles UI (`app/profiles.html`) and the Playground (`app/tools.html`). The port is assigned at startup and passed via `PROFILE_SERVER_PORT` env var. To find it at runtime: `lsof -p $(pgrep -f profile-server) -iTCP -sTCP:LISTEN` or check the menubar app's `self.profile_port`.
- **Ollama** — `http://localhost:11434` (desktop binds `0.0.0.0:11434` for LAN access)
- **MLX-OpenAI-Server** — `http://localhost:8000`, config at `~/.config/mlx-server/config.yaml`

### Key files at runtime

| What | Where |
|------|-------|
| Profiles | `~/.config/local-models/profiles.json` |
| MCP preferences | `~/.config/local-models/mcp_preferences.json` |
| Network config | `~/.config/local-models/network.conf` |
| MLX server config | `~/.config/mlx-server/config.yaml` |
| Menu bar log | `/tmp/local-models-menubar.log` |
| Instance lock | `~/.config/local-models/menubar.lock` |

### Task types

Profiles map these task types to models. Standard tasks: `code`, `general`, `reasoning`, `long_context`, `translation`. Special tasks (matched by model capability, not general-purpose): `vision`, `image_gen`, `transcription`, `tts`, `embedding`, `uncensored`.

## Local Model Tools (MCP)

The `mcp/local-models-server.py` MCP server exposes Ollama and MLX models as tools for Claude Code. See global CLAUDE.md for usage guidance. Wrapper script is `bin/local-models-mcp-detect`.

## When Modifying This Repo

- After adding a new config file, add its `link` entry in `install.sh`.
- Never add secrets, tokens, or API keys.
- Menu bar app uses PEP 723 inline metadata for dependencies — no separate requirements.txt or venv.
- Pin to Python 3.12 (pyobjc-core doesn't build on 3.14+).
- The compiled binary and .icns icon in `SuperPuppy.app` are gitignored — `install.sh` builds them. Only the C source (`app/super-puppy.c`) and Info.plist are tracked.
- If you change `app/super-puppy.c`, re-run `install.sh` or manually: `cc -o app/SuperPuppy.app/Contents/MacOS/super-puppy app/super-puppy.c && codesign --sign - --force app/SuperPuppy.app`.
