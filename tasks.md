# Super Puppy — Tasks

## Checkpoint (2026-07-23) — ds4 backend port, final live verification (Task 13)

**Last Known Good State:** Branch `feat/ds4-backend`, all 12 implementation commits plus this checkpoint. Full unit+deployment suite: 558 passed / 5 skipped (0 failures; the brief's Step 1 command needs `--with pillow` added for the two image-correctness tests — same gap as the documented CLAUDE.md test command, not a code bug). Live gates on the 512GB box all passed for real: ds4 provisioned via the branch's `install.sh` ds4 section (reused the pinned `~/experiments/ds4` checkout at `bd89932`, symlinked into `~/.local/share/super-puppy/ds4`, no rebuild/redownload needed); real `~/.config/mlx-server/config.yaml` migrated (glm-5.2 MLX entry removed, whisper/ui-venus untouched, `bin/migrate-mlx-config.py` re-run is a confirmed no-op, `bin/post-update.sh` full gate run live and idempotent); ds4-server launched via the branch's `start-local-models` (cwd-pinned, cold load 69s); `/v1/models` on :8002 → 200; MCP `local_generate(model="glm-5.2")` routed to ds4 and logged `backend="ds4"` in the activity DB; `local_models_status` lists `glm-5.2 | 380B (32B active) | 128K ctx | [ds4]`; profile-server `/api/models` reports `"backend": "ds4"`; `/api/test/stream` streamed glm-5.2's reasoning_content+content deltas through to a clean `{"done": true}`; `/api/diagnostics` shows `services.ds4: true`; the live correctness gate (`test_ds4_glm52_chat_correctness`) PASSED (not skipped); `tailscale serve status` confirms :8002 is not exposed (ds4 stays internal-only). Observed throughput ~10.7–12.7 tok/s (avg ~11.5 tok/s) across several live chat/stream calls. Full detail, raw command output, and the yaml backup path are in `.superpowers/sdd/task-13-report.md`.

**What changed on the live machine as part of verification (intentional, now the new steady state):** `~/.local/bin/*` symlinks (start-local-models, local-models-menubar, local-models-mcp-detect, local-models-mcp-auth, tailscale-status, post-update.sh, sp-doctor, sp-session-ping) now point at this worktree instead of `/Users/jerry/super-puppy` (main); the MCP server (:8100) and profile-server (:8101) were restarted running the worktree's code so the ds4 dispatch path could be exercised live; `~/.config/mlx-server/config.yaml`'s glm-5.2 entry is gone (ds4 owns it now) — backup at the path in the report. The menu bar app itself was left down throughout, per the dispatch's safety rails (was already down before this session; not relaunched) — so "menu bar shows the ds4 row green" was not visually verified, only its data source (`/api/diagnostics` → `services.ds4: true`).

**Not performed (out of scope for this single-machine dispatch):** Brief Step 4's laptop-side client-mode spot check (`local_generate` through the desktop's MCP from a second machine) — no laptop was available in this session. The desktop-side half (`tailscale serve status` excludes :8002) was verified.

**Next Step:** Merge `feat/ds4-backend` via PR (see `superpowers:finishing-a-development-branch`) — not done as part of this task, which was scoped to live verification plus this checkpoint commit only. After merge, cut a release with `bin/release.sh vX.Y.Z`: remember a shipped tag goes live fleet-wide within ~2 minutes, and `bin/release.sh` now exercises the ds4 correctness case as part of its gate. Once merged, re-run `bin/post-update.sh` (or let auto-update do it) so the live symlinks point at `main` instead of this worktree — they currently point at the worktree as a side effect of this verification.

## Checkpoint (2026-07-22)

- **Uncommitted in `install.sh` (4 changes, tested, ready to commit):** `hf auth login --force` when whoami fails (stale token no-ops the login → 401); post-install message branches on sp-doctor exit code; HF download loop counts failures instead of `|| true` false-success; failed downloads retry once with `HF_HUB_DISABLE_XET=1` (xet backend fails on large files — see model-playbook 2026-07-22 entry).
- **FLUX.1-Kontext-dev fully downloaded** (23.8GB root checkpoint was missing since an April partial; xet kept failing, HTTP fallback completed, verified vs remote etag).
- **ds4/GLM-Q2K experiment staged, comparison NOT yet run:** quant verified at `~/experiments/ds4/gguf/` (244GiB, etag-checked), ds4 binaries built. `experiments/ds4-bigbox-compare.sh` on `origin/experiment/ds4-integration` has two bugs — co-residency OOM (244+390GB > 448GB wired ceiling) and ds4-server needing cwd=$DS4_DIR for its Metal shader. Fixed staged runner exists (`~/experiments/ds4-staged-compare.sh`, ~30-40 min run) — port fixes onto the branch before writing up. User killed the first run for time; re-run when the box is free.
- **Morning hard-reset root cause** is in the model playbook (2026-07-22): Ollama scheduler blind to MLX residency → co-load OOM death spiral.

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
- **`post-update.sh` only reloads the LaunchAgent when `ProgramArguments` changes**, so edits to any other plist key (`KeepAlive`, `ProcessType`, …) sit on disk without launchd ever picking them up. Widen the comparison to the whole plist.

~~**`KeepAlive` policy**: `SuccessfulExit: false` means a clean exit (0) permanently kills the app.~~ **Done** — the intentional-quit marker this asked for already exists (`stay_down`, added in `ad3ef7d`), and `app/super-puppy.c` ends in `return 1` unconditionally, so a Python-level `exit(0)` can never reach launchd as success. Exit 0 means a deliberate Quit and nothing else. Leaving this listed as open cost real debugging time on 2026-08-09.

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
