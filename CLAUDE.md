# Super Puppy

Local AI model infrastructure for Claude Code. Menu bar app + smart routing + LAN serving.

## Structure

- `app/` — Menu bar app (Python/rumps, PEP 723 inline deps, runs via `uv run --python 3.12`)
- `mcp/` — MCP server exposing local models as tools for Claude Code
- `bin/` — Shell scripts symlinked to `~/bin/`
- `config/` — Config files symlinked to `~/.config/` and `~/Library/LaunchAgents/`
- `install.sh` — Creates symlinks, checks dependencies, detects desktop vs laptop

## Key Design Decisions

- `claude-smart` dynamically builds CCR provider model lists from what's installed (not static). Any new `ollama pull` is immediately routable.
- Role filters live in `role_filters.json`, not hardcoded. Each role specifies min/max active params, context requirements, and vision capability.
- The menu bar app queries model capabilities live from Ollama `/api/show` and MLX `config.json` files in the HuggingFace cache. No hardcoded param tables.
- MLX models marked `on_demand: true` download on first use and unload after idle timeout.
- The desktop hostname for LAN serving is in `config/local-models/network.conf`, detected via mDNS (Bonjour).

## Local Model Tools (MCP)

The `mcp/local-models-server.py` MCP server exposes Ollama and MLX models as tools for Claude Code. See global CLAUDE.md for usage guidance. Wrapper script is `bin/local-models-mcp-detect`.

## When Modifying This Repo

- After adding a new config file, add its `link` entry in `install.sh`.
- Never add secrets, tokens, or API keys.
- Menu bar app uses PEP 723 inline metadata for dependencies — no separate requirements.txt or venv.
- Pin to Python 3.12 (pyobjc-core doesn't build on 3.14+).
