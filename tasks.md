# Super Puppy — Tasks

## Last Known Good State

MCP-first architecture is working:
- `claude` with Max subscription + `local-models` MCP server (10 tools: generate, review, vision, image, transcribe, translate, candidates, summarize, status, models_status)
- `claude-local` script reads menu bar preferences, auto-detects desktop on LAN
- Menu bar app: service status, MCP task→model preferences, capability status, auto-update from GitHub
- `install.sh` handles symlinks, MCP injection into `~/.claude.json`, CLAUDE.md guidance check
- Single-instance lock on menu bar app
- Active param calculation fixed for hybrid MoE (nemotron, deepseek)
- Removed: claude-smart, CCR config, pal-mcp-server

## Next Steps

### GPU Contention & Model Scheduling
When multiple users (desktop + laptops) hit the same Ollama/MLX instance, model loading/swapping causes stalls. Need:
- Timeouts with clear error messages in the MCP server
- Configure `OLLAMA_MAX_LOADED_MODELS=6` in LaunchAgent or start-local-models
- Pin core set: qwen3.5:122b, qwen3-coder, nemotron-3-super, glm-4.7-flash, z-image-turbo, whisper-v3 (~218GB always loaded)
- On-demand heavyweights: qwen3-vl:235b, cogito-2.1 (swap in/out as needed)
- Priority: `claude-local` on the desktop vs MCP tool calls from laptops
- MLX on-demand models have idle timeouts already but no load coordination

### Parallel Second Opinion
The local cluster is free to use. Claude should be able to dispatch a local model to work in parallel as a second opinion — not just for code review after the fact, but running alongside while Claude reasons. Example: Claude plans an architecture change, simultaneously asks nemotron or cogito to independently plan the same change, then compares before committing. Needs:
- Guidance in CLAUDE.md for when to use parallel local reasoning
- Possibly a dedicated tool (`local_parallel_reason`?) or just better guidance for using `local_candidates`
- The local model runs while Claude thinks, not sequentially

### Model Roster Optimization
Researched best models per category (March 2026). Recommended additions:
- `ollama pull glm-4.7-flash` — fast all-rounder, 200K context (downloading)
- `ollama pull deepseek-r1:671b` — full reasoning model (~404GB, swap-in only)
- `ollama pull qwen3-coder:480b` — full coding model (~300GB, swap-in only)
- GLM-5 (744B/40B active) — not yet in standard Ollama, watch for support

### Menu Bar Improvements
- Show which models are currently loaded in Ollama (vs available but unloaded)
- Show GPU memory usage
- Warn when a model selection would cause contention

### Testing
- MCP tools not yet fully tested end-to-end from Claude Code
- `claude-local` works but is slow on first request (model load time) — consider a "warm up" option
- Laptop-away fallback path untested

### Cleanup
- Old `claude-local` alias in `~/.zshrc` should be removed (replaced by `bin/claude-local`)
- `~/.claude-code-router/` directory may have stale config from CCR experiments
- Dead CCR dependency (`ccr` npm package) can be uninstalled: `npm uninstall -g @musistudio/claude-code-router`
