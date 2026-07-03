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
| **computer_use** (multimodal VLM) | ~~:8000~~ → **mlx_vlm subprocess** | ✅ **FIXED** — see below |
| **tts** (`local_speak`) | mlx-audio | ✅ **FIXED** — GPU-tracking dict KeyError'd on non-ollama/mlx backends; now a defaultdict |

### computer_use fix — one-shot mlx_vlm subprocess (`lib/mlx_vlm.py`)

Rather than the persistent `:8000` server (which hangs), MLX grounding
models are now dispatched as a one-shot `mlx_vlm generate` subprocess —
load + generate in a single process/thread, sidestepping the thread-local
stream bug (same pattern as mflux/mlx-audio). Shared by the MCP server
(async) and profile server (sync). Output is normalized: UI-Venus/Qwen-VL
`Click(box=(x,y))` (0-1000 space) → pixel-coord JSON click action.
`install.sh` installs a dedicated `mlx-vlm` uv tool env with torch +
torchvision. Verified end-to-end (MCP tool + profile-server + harness).

Remaining `:8000` notes (not blocking any tool):
- `llama-3b` (Llama arch) still `Stream(gpu,1)` on `:8000`, but it's the
  unused startup health-check model. SP's MLX status is liveness-only
  (`/v1/models`), so it reports "up" regardless — no false-down.
- The persistent-server LM/whisper paths that DO work (Qwen text,
  transcription) are unaffected. Track mlx-lm #1256 for the real fix.

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
