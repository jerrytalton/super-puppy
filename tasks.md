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
- **All tiers' vision → GGUF `qwen3.6:27b`** (PROFILES_VERSION 28→29). The `:8000` MLX path can't serve vision reliably (see below), so `32gb`/`512gb` were repointed off `qwen3.5-small`/`qwen3.5:122b` to the GGUF tag that's verified working end-to-end.
- **Correctness harness extended** to chat, summarize, transcription (real speech fixture), and embedding — plus xfail(run=False) trackers for the two known-broken tools below.

## Known Issues — `:8000` MLX-openai-server (as of 2026-07-02)

Root cause: **mlx 0.31.2 made compute streams thread-local**; mlx-openai-server loads a model on one thread and generates on another, so array eval crashes/hangs (`There is no Stream(gpu, 1) in current thread`; mlx-lm #1256, structural fix PR #1088 closed unmerged). No released version combo fixes it (mlx 0.31.2 is newest; deps forbid downgrade). Verified tool matrix on a clean server:

| Tool | Backend | State |
|---|---|---|
| vision, image_gen | Ollama | ✅ works |
| transcription (whisper) | :8000 | ✅ works |
| text chat (Qwen models) | :8000 | ✅ works |
| text chat (`llama-3b`, Llama arch) | :8000 | ❌ `Stream(gpu,1)` — but **unused** (health-check only) |
| **computer_use** (multimodal VLM) | :8000 | ❌ **generation hangs / RPC-timeout** — the real casualty |
| **tts** (`local_speak`) | mlx-audio | ❌ errors `'mlx-audio'` — separate dispatch/env bug, likely fixable |

Actionable follow-ups (not the upstream mlx bug):
- Fix the `local_speak` `'mlx-audio'` dispatch error (ours).
- `llama-3b` failing likely makes SP's status report MLX "down" even though real (on-demand) models work — point the health-check at a working model or stop eager-loading it.
- `computer_use`: blocked on upstream mlx/mlx-openai-server; only "fix" is a full pre-0.31.2 stack downgrade (loses qwen3.6 support). Track mlx-lm #1256.

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
