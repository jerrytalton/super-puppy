# Super Puppy — Tasks

## Checkpoint (2026-08-17) — unfiltered → uncensored Qwen3.8-27B MLX (branch `feat/uncensored-qwen38-mlx`, PROFILES_VERSION 37)

**Last Known Good State:** One commit on `feat/uncensored-qwen38-mlx`. unfiltered task: dolphin3:8b → orcarouter's abliterated Qwen3.8-27B (gated HF repo, quants in subfolders — unloadable by repo id, so new `MLX_SUBFOLDER_REPOS` machinery downloads `--include "<quant>/*" --local-dir ~/.local/share/super-puppy/models/…` and the MLX config's `$HOME`-templated `model_path` points there; post-update.sh expands `$HOME` and bounces MLX when the merge appends entries). 6-bit on 64gb, 8-bit on 128/512gb, served `model_type: lm` (VLM path still hangs, mlx-lm #1256). Containment hardened: `uncensored` in ALWAYS_EXCLUDE, `get_eligible_tasks` caps abliterated models to unfiltered-only, MCP any-LLM fallback skips ALWAYS_EXCLUDE. Red-teamed (fresh sub-agent): caught a real blocker — profile-server discovery's HF-cache name mangling hid local-dir entries — plus the missing MLX restart; both fixed. Verified live on the 512GB box: full unit suite green (671 + new tests), post-update.sh merged + auto-restart picked up both served names on :8000, standalone load 7s (mmap), 17×23=391 sane, `test_everyday_unfiltered_stream` PASSED through the real profile-server → MLX stack. Gated-repo caveat: every machine needs an HF login that accepted the repo terms (docs/troubleshooting.md), and the current OAuth login expires 2026-08-21 — replace with a fine-grained token.

**Next Step:** Push branch + PR, merge, then release (vX.Y.Z) so the fleet picks up PROFILES_VERSION 37; laptop needs its HF login to accept the orcarouter terms before its auto-pull can fetch the weights.

## Checkpoint (2026-08-15) — image_edit → FLUX.2-edit (klein-9B), PROFILES_VERSION 35

**Last Known Good State:** Bake-off (3 edit tasks × Kontext / FLUX.2-edit klein-9B / Qwen-Image-Edit-2511-8bit, mflux 0.18.1) retired Kontext as the preset: flux2-edit matched its quality at 2.6× the speed and 13.6GB peak vs 31GB; qwen-edit re-synthesizes the whole frame through mflux (details in the model playbook, incl. "VLM judges confabulate edit-artifact grades — verify by eye"). All tiers' image_edit → `black-forest-labs/FLUX.2-klein-9B` (shares weights with 128/512gb image_gen; edit dispatch already existed via `mflux_edit_command`). mflux upgraded 0.17.4→0.18.1 (adds ERNIE, Ideogram 4, Krea-2 — gen-side bake-off NOT yet done). Unit suite 612 passed; live smoke + image-edit correctness green, output verified by eye.

**Update (2026-08-15, corrected rerun):** the first edit bake-off's "klein-9B" runs were silently the 4B — mflux `--base-model` without `--model` ignores the variant (playbook entry added; md5-identical outputs proved it). True 9B (explicit `--model`): edit 106s/24GB, quality indistinguishable from 4B on all three cases. **image_edit → FLUX.2-klein-4B on all tiers (PROFILES_VERSION 36)** — the verified quality at half the cost, weight-shared with the 64gb gen pick. Gen-side bake-off done (klein-4B/9B corrected, Qwen-Image-2512-8bit): qwen wins photorealism detail but garbles text via mflux-8bit and costs 4× — **gen picks unchanged**. Auto-pull now covers HF repos too (PR #35).

**Next Step:** Release (vX.Y.Z) — fleet picks up PROFILES_VERSION 36 + HF auto-pull; machines need mflux ≥0.17 for flux2-edit (all have it).

## Checkpoint (2026-08-14) — qwen3.8 adoption (branch `feat/qwen38-profiles`)

**Last Known Good State:** Two commits on `feat/qwen38-profiles` (off origin/main @ v1.6.2): (1) `model_has_vision()` accepts `ollama_projector_info` — Ollama ≥0.32 ships qwen3.8's vision encoder as a separate 888MB projector blob, zero `model_info` vision keys, honest signal moved to `/api/show`'s `projector_info` (`clip.has_vision_encoder`); `capabilities` still never trusted. Both Ollama discovery call sites wired. (2) PROFILES_VERSION 33→34: vision → `qwen3.8:27b` all tiers, 64gb/128gb text tasks → `qwen3.8:27b-mlx`, code picks unchanged; code TASK_FILTER unstuck (priority `coding`, include `qwen3` + `muse-glimmer`). Verified: 609 unit tests (pre-commit), live smoke + correctness 25 passed on this laptop (three vision color cases through qwen3.8:27b), `qwen3.8:27b-mlx` pulled and serving. Playbook updated (dotfiles): Ollama 0.32.12's MLX engine now passes real vision tests (0.30.10/0.32.7 "-mlx lies" entries are version-bound); split-projector detection entry added.

**Next Step:** Merge via PR and cut a release with `bin/release.sh`. The weights-don't-converge gap (red-team 2026-08-14) is CLOSED by the menu bar auto-pull path: `_maybe_autopull` (kicked after services start and on each warm tick) serially pulls the active profile's missing Ollama tags via streaming `/api/pull`, with an hourly per-model failure cooldown and progress on the Ollama menu row. Wire loop verified live (tinyllama re-pull → success; planner reports [] on this fully-pulled 128gb machine). Still noted (LOW, not blocking): `lib/models.py` trusts a `capabilities` key *inside* model_info (inert, pre-existing) three lines under the NOTE disclaiming it.

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
