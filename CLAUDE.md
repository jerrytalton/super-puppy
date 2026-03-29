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

## Local Model Tools (MCP)

The `mcp/local-models-server.py` MCP server exposes Ollama and MLX models as tools for Claude Code. See global CLAUDE.md for usage guidance. Wrapper script is `bin/local-models-mcp-detect`.

## When Modifying This Repo

- After adding a new config file, add its `link` entry in `install.sh`.
- Never add secrets, tokens, or API keys.
- Menu bar app uses PEP 723 inline metadata for dependencies — no separate requirements.txt or venv.
- Pin to Python 3.12 (pyobjc-core doesn't build on 3.14+).
- The compiled binary and .icns icon in `SuperPuppy.app` are gitignored — `install.sh` builds them. Only the C source (`app/super-puppy.c`) and Info.plist are tracked.
- If you change `app/super-puppy.c`, re-run `install.sh` or manually: `cc -o app/SuperPuppy.app/Contents/MacOS/super-puppy app/super-puppy.c && codesign --sign - --force app/SuperPuppy.app`.
