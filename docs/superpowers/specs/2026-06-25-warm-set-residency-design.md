# Warm-set model residency

**Date:** 2026-06-25
**Status:** Design approved; pending spec review → implementation plan.
**Builds on:** `2026-06-24-ram-tier-profiles-design.md` (the RAM-tier profiles this refines).

## Problem

The RAM-tier profiles were sized for *on-demand, one-model-at-a-time* loading (peak resident = the single largest model). But Super Puppy's actual feature is keeping a profile **warm** for instant task-switching (the "Warm" button + Ollama `keep_alive` + MLX idle timeouts). Two consequences surfaced in use:

1. The Models memory bar sums **every** model in the profile against the RAM cap. For the 128gb tier that's ~198GB vs 128GB — a scary red "overflow" for a profile that's actually fine on-demand. The bar's mental model (everything resident at once) doesn't match how the models load.
2. There's no way to say "keep *these* models hot, stream the rest." The "Warm" button warms the whole profile, which on a real machine can't fit and just makes the models evict each other.

## Hardware reality (the design driver)

On Apple Silicon the GPU is a single, time-sliced resource: two models "running at once" take turns and roughly halve each other's throughput — **warming multiple models buys zero compute concurrency.** Its only benefit is avoiding reload latency when you switch. The real failure mode is **memory-pressure thrash**: weights live in unified memory (~75% GPU-addressable by default), and pushing resident weights + the active model's KV cache past physical RAM makes macOS swap — catastrophic for inference. `mlx-openai-server` runs each model in its own subprocess, so warming N models = N full weight copies resident.

**Principle:** keep warm only the models you rapidly alternate between, sized so the warm set **+ the active model's KV cache + OS/apps** stays comfortably inside RAM. Everything bursty or heavy (image, video, TTS, transcription, the frontier model when it's not the workhorse, the dedicated coder when it's a second big model) loads on demand. Within a profile this is naturally small, because several tasks share one model.

## Design

### 1. Schema (`lib/models.py`)

Each profile gains a `warm` list of task keys naming the hot set. Uniform across tiers: **`warm: ["general", "embedding"]`** — the text workhorse plus the embedding model. (Tasks that share the workhorse model — e.g. 128gb's general/reasoning/long_context/translation/vision are all `qwen3.6:27b-mlx-bf16` — are covered by warming `general`.)

Add `WARM_BUDGET_FRACTION = 0.65`. Bump `PROFILES_VERSION` → **27**.

`migrate_profiles` (already the shared migrator) refreshes presets, so existing installs pick up the warm lists. Custom (non-preset) profiles that lack a `warm` key default to `[]` — empty warm set = all on-demand = today's behavior. The contract test gains: every preset's `warm` keys exist in its own `tasks`.

### 2. Warm set, per tier

All tiers: `warm = ["general", "embedding"]`.

| Tier | Warm (resident) | ~Warm size / 65% budget | State | Streams on-demand |
|---|---|---|---|---|
| 32gb | `qwen3.5-small` + `embeddinggemma:300m` | ~5 / 21 GB | ✓ under | image, tts, transcription |
| 64gb | `qwen3.6:27b-mlx` + `qwen3-embedding:8b` | ~28 / 42 GB | ✓ under | code, computer_use, image, unfiltered |
| 128gb | `qwen3.6:27b-mlx-bf16` + `qwen3-embedding:8b` | ~63 / 83 GB | ✓ under | code, computer_use, image, video, edit |
| 512gb | `glm-5.2` + `qwen3-embedding:8b` | ~426 GB | ⚠ tight (dominant) | code, vision, image, video, edit |

The 512gb workhorse (`glm-5.2`, ~418GB) alone exceeds 65% of 512GB; that's an accepted **dominant-model "tight"** state, not an error.

### 3. Keep-warm ticker (`app/menubar.py`)

The single mechanism that keeps warm models resident across *both* request paths (Playground and MCP). The menubar already runs periodic work via `rumps.Timer`; add a keep-warm tick (own timer, ~240s — under the MLX 300s idle timeout):

- Read the active profile (`load_profiles()["active"]`) and its `warm` task keys → resolve to model names → only act when local services are up.
- For each warm **MLX** model: POST a 1-token completion to `MLX_URL/v1/chat/completions` (resets its idle timer so it stays loaded).
- For each warm **Ollama** model: POST `/api/generate` with empty prompt and a long `keep_alive` (e.g. `"30m"`) to refresh residency.
- Best-effort: log and continue on failure; never block the UI thread (run in a daemon thread like the other menubar background work).

Skips entirely in client/offline mode or when MLX/Ollama aren't running.

### 4. Differentiated keep-alive (no-thrash)

Non-warm models must unload promptly so they don't linger beside the warm set:

- **`app/profile-server.py`:** the five request sites currently send a blanket `OLLAMA_KEEP_ALIVE = "30m"`. Make it warm-aware: a helper `keep_alive_for(model)` returns `"30m"` if the model is in the active profile's warm set, else a short value (`"30s"`) so non-warm Ollama models evict right after use. (The `/warm` endpoint's own preload keeps `"30m"`.)
- **MLX non-warm:** already unload via the config's `on_demand_idle_timeout: 300`. No change.
- **MCP server (`mcp/local-models-server.py`):** sets no `keep_alive` today (Ollama default ~5m). Non-warm models therefore already evict in ~5m, and warm models are kept alive by the ticker. No change required in phase 1; a warm-aware `keep_alive` on the MCP path is a possible later refinement (noted, not built).

### 5. Warm endpoint scope (`app/profile-server.py`)

`/api/profiles/<name>/warm` currently preloads every distinct model in the profile. Change it to preload only the models referenced by the profile's `warm` task list. Return shape unchanged (`{ok, loaded: [...]}`).

### 6. Memory endpoint + bar (`app/profile-server.py`, `app/profiles.html`)

Move the residency math server-side so it's unit-testable; the JS only renders.

**New `GET /api/profiles/<name>/memory`** returns:
```
{
  "cap_bytes": <max_ram_gb * 1<<30>,
  "budget_bytes": <cap_bytes * 0.65>,
  "warm": [ {"name","task","bytes"} ... ],        # distinct warm-set models
  "warm_bytes": <sum of warm>,
  "on_demand": [ {"name","task","bytes"} ... ],    # distinct non-warm models
  "largest_on_demand_bytes": <max on_demand, or 0>,
  "peak_bytes": <warm_bytes + largest_on_demand_bytes>,
  "state": "ok" | "tight" | "thrash"
}
```
Model sizes reuse the existing size lookups (`_get_ollama_model_size`, `_get_hf_model_size`, MLX `vram_bytes`/weights estimate) — the same source the current bar uses. `state` is purely threshold-based (no special-casing model count):
- `thrash` — `warm_bytes > cap_bytes`: the warm set alone doesn't fit physical RAM → guaranteed swap. Always bad; an authoring error. None of the presets hit this.
- `tight` — not thrash, **and** (`warm_bytes > budget_bytes` **or** `peak_bytes > cap_bytes`): the warm set fits, but either it eats into the KV/OS reserve, or loading the largest on-demand model can't coexist with it (so an on-demand call evicts the warm set and forces a later reload). This is the accepted state for a dominant-model tier (512's `glm-5.2`).
- `ok` — `warm_bytes ≤ budget_bytes` **and** `peak_bytes ≤ cap_bytes`: comfortable headroom; warm set plus the largest on-demand model coexist with room for KV/OS.

For the presets this yields 32gb/64gb/128gb = `ok`, 512gb = `tight` (its ~426GB warm set exceeds the 65% budget and, with the 122B vision model on-demand, `peak > cap` — so vision calls evict GLM, which is the honest, intended tradeoff at that tier).

**Bar (`renderMemory` in `profiles.html`)** — single horizontal bar over `0…cap_bytes`:
- Solid stacked segments = warm models (`warm[]`).
- A hatched segment = the largest on-demand model, drawn immediately after the warm segments (the transient peak when it loads).
- Vertical markers at `budget_bytes` (65%) and `cap_bytes`.
- Color by `state`: green (ok) / amber (tight) / red (thrash).
- Legend lists warm models (solid) and on-demand models (hatched/ghost) with sizes, and a one-line peak check, e.g. `peak: warm 63 + coder 52 = 115 ≤ 128 ✓`.

The bar now reads from `/api/profiles/<name>/memory`, not from summing all task picks client-side.

## Plumbing summary

| File | Change |
|---|---|
| `lib/models.py` | `warm` list on every preset; `WARM_BUDGET_FRACTION = 0.65`; `PROFILES_VERSION = 27`; migration covers it; contract test for warm-key validity |
| `app/menubar.py` | keep-warm ticker (daemon, ~240s) pinging the active profile's warm models |
| `app/profile-server.py` | warm-aware `keep_alive_for()`; `/warm` loads only the warm set; new `/api/profiles/<name>/memory` |
| `app/profiles.html` | `renderMemory` reads the memory endpoint; single bar with hatched transient-peak overlay, budget/cap markers, state colors |
| tests | warm-key contract; `migrate` adds warm / custom defaults to `[]`; `/warm` loads only warm-set (mocked backends); `/api/.../memory` state math (ok/tight/thrash, peak) |

## Testing

- **Schema/contract:** every preset's `warm` keys ⊆ its `tasks`; warm models resolve to real backends.
- **Migration:** v26→v27 adds `warm` to presets; a custom profile without `warm` migrates to `warm: []`.
- **Warm endpoint:** mock Ollama `/api/ps` + the load calls; assert only warm-set models are preloaded, non-warm are not.
- **Memory endpoint:** feed known model sizes; assert `warm_bytes`, `peak_bytes`, and `state` for each case — warm ≤ budget and peak ≤ cap ⇒ `ok`; warm > budget but ≤ cap (512-like dominant) ⇒ `tight`; warm ≤ cap but peak > cap ⇒ `tight`; warm > cap ⇒ `thrash`.
- **Keep-warm ticker:** unit-test the pure selection (active profile → warm model names by backend); the HTTP pings are best-effort and mocked.
- The bar's rendering is thin JS over the tested endpoint; no JS unit test (consistent with the existing UI).

## Migration / compatibility

- `PROFILES_VERSION` 26→27. Existing installs migrate on next launch (menubar `seed_profiles_if_missing` and profile-server `load_profiles`, both via `migrate_profiles`). Old presets already retired in v26; this only adds `warm`.
- Custom profiles preserved; absent `warm` ⇒ `[]` ⇒ all on-demand (no behavior change for them).
- No new model downloads — warm/on-demand is a residency policy over the existing profile models.

## Out of scope

- Warm-aware `keep_alive` on the MCP request path (ticker already keeps warm models resident; MCP non-warm models evict via Ollama's ~5m default). Possible later refinement.
- Per-model KV-cache size modeling. The 65% budget is a flat reserve heuristic for KV + OS + transient load; not a precise KV calculation.
- User-editable warm sets in the UI. Warm sets are author-defined per preset in this phase.
