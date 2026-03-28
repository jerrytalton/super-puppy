# Super Puppy — Tasks

## Last Known Good State

MCP-first architecture is working:
- `claude` with Max subscription + `local-models` MCP server (12 tools: generate, review, vision, image, transcribe, translate, candidates, summarize, status, models_status, dispatch, collect)
- `claude-local` script reads menu bar preferences, auto-detects desktop on LAN
- `local_dispatch` / `local_collect` for parallel second opinions (async background jobs)
- Menu bar app: service status, MCP task→model preferences, capability status, auto-update from GitHub
- `install.sh` handles symlinks, MCP injection into `~/.claude.json`, CLAUDE.md guidance check
- Single-instance lock on menu bar app
- Active param calculation fixed for hybrid MoE (nemotron, deepseek, known-model lookup)
- `OLLAMA_MAX_LOADED_MODELS=6` configured for desktop
- Model preferences ranked by quality with automatic fallback
- Removed: claude-smart, CCR config, pal-mcp-server, pal-mcp-detect

## Next Steps

### Model Profile Visualizer (NEW — priority)
Build a UI pane (separate window or panel from the menu bar app) that shows:
- Columns per task type (code, general, reasoning, long context, vision, image, transcription, translation)
- Available models listed in each column, ranked by quality
- Memory cost per model shown visually
- A "memory pool" bar showing total 512GB, with loaded models filling it
- Preset profiles ("everyday" ~220GB, "deep reasoning" ~450GB, "heavy code" ~350GB)
- Click a preset to switch which models are loaded
- Drag/select models to build custom profiles
- Visual warning when a combination exceeds available memory
- The MCP server reads the active profile to pick models
Technology: either a rumps window, a SwiftUI companion app, or a web UI (localhost) opened from the menu bar

### GPU Contention & Model Scheduling
When multiple users (desktop + laptops) hit the same Ollama/MLX instance, model loading/swapping causes stalls. Need:
- Timeouts with clear error messages in the MCP server
- Profile system (above) solves the "which models are loaded" problem
- Priority: `claude-local` on the desktop vs MCP tool calls from laptops
- MLX on-demand models have idle timeouts already but no load coordination

### Model Roster — Downloads Pending
- `ollama pull glm-4.7-flash` — fast all-rounder, 200K context (downloading)
- `ollama pull deepseek-r1:671b` — full reasoning model (~404GB, swap-in only)
- `ollama pull qwen3-coder:480b` — full coding model (~300GB, swap-in only)
- GLM-5 (744B/40B active) — not yet in standard Ollama, watch for support

### Testing
- MCP tools not yet fully tested end-to-end from Claude Code
- `local_dispatch` / `local_collect` untested in real usage
- `claude-local` works but is slow on first request (model load time) — consider a "warm up" option
- Laptop-away fallback path untested

### Cleanup
- Old `claude-local` alias in `~/.zshrc` should be removed (replaced by `bin/claude-local`)
- `~/.claude-code-router/` directory may have stale config from CCR experiments
- Dead CCR dependency (`ccr` npm package) can be uninstalled: `npm uninstall -g @musistudio/claude-code-router`
