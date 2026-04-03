# Super Puppy — Tasks

## Last Known Good State

MCP-first architecture working with 16 tools:
- `local_generate`, `local_review`, `local_vision`, `local_computer_use`, `local_image`, `local_image_edit`
- `local_transcribe`, `local_speak`, `local_translate`, `local_candidates`, `local_summarize`
- `local_embed`, `local_similarity_search`, `local_dispatch`, `local_collect`, `local_models_status`
- Menu bar app: service status, model profiles, playground, task preferences, auto-update (tag-based)
- Interactive `install.sh`: server/client setup, auth tokens, Tailscale walkthrough, profile-based model pulling
- Model Profiles UI (`profiles.html`) and Playground UI (`tools.html`) served by profile-server
- Single-instance lock, auto-update with crash rollback, session-based MCP auth
- Active param calculation for hybrid MoE (nemotron, deepseek, known-model lookup)
- GPLv3 licensed, prepared for public release

## Next Steps

### Profile Server Authentication
The profile server (Flask) has no authentication. When exposed via Tailscale, anyone on the tailnet has full access to all tools. Add bearer token auth when `PROFILE_HOST` is `0.0.0.0`.

### MCP Server Improvements
- Add `/health` endpoint returning service status, model counts, memory pressure
- Add progress notifications for first-time model loads (30-60s waits with no feedback)
- Make `pick_model` error messages actionable (show which models were tried and why they didn't match)
- Add `computer_use` to default `mcp_preferences.json`

### Operational
- Add `--status` MCP server check to `start-local-models --status`
- Add `--help` to all bin/ scripts
- Add `install.sh --uninstall` to reverse symlinks and unload LaunchAgents
- Add disk space check before model pulls in install.sh
- Add log rotation for `/tmp/local-models-*.log` files

### Testing
- 56 tests passing (25 unit + 43 e2e + 28 error handling)
- `local_computer_use` needs e2e test coverage
- Laptop-away fallback path needs testing
