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

## Recently Fixed (2026-06-29)

- **`local_vision` routed to a blind model.** Ollama's `-mlx`/`-mlx-bf16` tags advertise `capabilities:["vision"]` but ship no vision tower and silently hallucinate. Fixes: `model_has_vision()` detects via `model_info` vision keys only (never the lying `capabilities` array); `pick_model`/`_pick_model_for_task` are capability-gated so a vision pref that resolves to a tower-less model is skipped; vision presets repointed to GGUF `qwen3.6:27b` (PROFILES_VERSION 27→28); an unresolvable model override now errors instead of silently substituting.
- **Correctness harness** (`tests/test_tools_correctness.py`, `correctness` marker): asserts each tool's chosen model actually honors its input (e.g. solid-color image → model names the color). Excluded from the default run; `bin/release.sh` runs it on version bumps. Proven to fail on the broken `-mlx-bf16` model.
- **Still open:** the mlx-openai-server (`:8000`) vision models (`qwen3.5-small`/`-fast`, used by the 32gb/512gb profiles) may have the same blind-vision issue — `qwen3.5-small` claimed it "can't access the image" in testing. The correctness harness will surface this on the next release run; verify and repoint if needed.

## Next Steps

### Auto-Update: Remaining Items
- **Heavy MCP user**: 10-minute max deferral may interrupt long inference jobs. Consider longer ceiling or smarter detection.
- **`KeepAlive` policy**: `SuccessfulExit: false` means a clean exit (0) permanently kills the app. Consider `KeepAlive: true` with an intentional-quit marker.

### MCP Server Improvements
- Add `/health` endpoint returning service status, model counts, memory pressure
- Add progress notifications for first-time model loads (30-60s waits with no feedback)
- Add `computer_use` to default `mcp_preferences.json`

### Operational
- Add `--status` MCP server check to `start-local-models --status`
- Add `--help` to all bin/ scripts
- Add disk space check before model pulls in install.sh
- Add log rotation for `/tmp/local-models-*.log` files

### Testing
- 198 unit tests passing (53 deployment + 33 core + 47 MCP + 56 profile + 4 playground + 5 remaining)
- `local_computer_use` needs e2e test coverage
- Laptop-away fallback path needs testing
