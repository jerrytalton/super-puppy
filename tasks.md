# Super Puppy — Tasks

## Last Known Good State

MCP-first architecture is working:
- `claude` with Max subscription + `local-models` MCP server (9 tools: generate, review, vision, image, transcribe, translate, candidates, summarize, status)
- `claude-local` script reads menu bar preferences, auto-detects desktop on LAN
- Menu bar app shows service status, MCP task→model preferences with rich labels, capability status
- `install.sh` handles symlinks + MCP injection into `~/.claude.json`
- Active param calculation fixed for hybrid MoE (nemotron, deepseek)
- README rewritten for MCP architecture
- Removed: claude-smart, CCR config, pal-mcp-server

## Next Steps

### GPU Contention & Model Scheduling
When multiple users (desktop + laptops) hit the same Ollama/MLX instance, model loading/swapping causes stalls. Need:
- Timeouts with clear error messages in the MCP server
- `OLLAMA_MAX_LOADED_MODELS` tuning
- Consider pinning high-use models (qwen3.5, qwen3-coder) so they stay loaded
- Priority: `claude-local` on the desktop vs MCP tool calls from laptops
- MLX on-demand models have idle timeouts already but no load coordination

### Menu Bar Improvements
- Show which models are currently loaded in Ollama (vs available but unloaded)
- Show GPU memory usage
- Warn when a model selection would cause contention

### Testing
- MCP tools not yet tested end-to-end from Claude Code (server starts, discovery works, but no real tool calls verified)
- `claude-local` works but is slow on first request (model load time) — consider a "warm up" option
- Laptop-away fallback path untested

### Cleanup
- Old `claude-local` alias in `~/.zshrc` should be removed (replaced by `bin/claude-local`)
- `~/.claude-code-router/` directory may have stale config from CCR experiments
- Dead CCR dependency (`ccr` npm package) can be uninstalled: `npm uninstall -g @musistudio/claude-code-router`
