# RAM-tier model profiles

**Date:** 2026-06-24
**Status:** Design approved; pending spec review → implementation plan.

## Goal

Replace the four ad-hoc named profiles (`laptop`/`desktop`/`everyday`/`maximum`) with four profiles keyed to the **machine class they target**: `32gb`, `64gb`, `128gb`, `512gb`. Each profile runs the best models that the tier's **RAM and GPU** can drive *with comfortable headroom* for the user's real work — these are people's actual computers, not dedicated inference boxes.

## Design principle

RAM and GPU scale together across Apple Silicon: small-RAM machines have weaker GPUs (base M5, M1 Max), large-RAM machines have strong ones (M5 Max, M3 Ultra). So "best models that fit the RAM" is moderated by "…that the GPU can drive at usable speed." The lever is **MoE vs dense**:

- **Weak GPU tiers** favor small or low-active-param **MoE** models (e.g. a 3B-active MoE is fast even on a weak GPU) at 4-bit — keeps latency low.
- **Strong GPU tiers** can afford **dense** models at **higher precision** (bf16) — higher quality, which the GPU can actually drive.
- The top tier runs **frontier** models.

Footprint must leave headroom: size the everyday workhorse well under the cap. Big models (vision, frontier) are acceptable because they're used in bursts and unload on idle, but they should still leave the machine usable mid-call.

All model choices were re-verified against the Ollama library and `mlx-community` on 2026-06-24 (see "Research notes").

## Profiles

`pick_profile_for_ram` selects the largest profile whose `max_ram_gb` ≤ system RAM (a 256GB machine → `128gb`; a 512GB machine → `512gb`).

| Task | **32gb** (cap 32, weak GPU) | **64gb** (cap 64, mid) | **128gb** (cap 128, strong) | **512gb** (cap 512, top) |
|---|---|---|---|---|
| general / reasoning / long_context / translation | `qwen3.5-9b` 4-bit (small/fast, ~5GB) | `qwen3.6:27b-mlx` 4-bit (dense, ~20GB) | `qwen3.6:27b-mlx-bf16` (dense bf16, ~55GB) | `glm-5.2` 4-bit (frontier, ~418GB) |
| code | (reuse workhorse) | `qwen3.6:27b-coding-mxfp8` (~31GB) | `qwen3-coder-next` (~52GB) | `qwen3-coder-next` |
| vision | (reuse workhorse ⚠️) | (reuse `27b-mlx`) | (reuse `27b-mlx-bf16`) | `qwen3.5:122b` (~95GB) |
| computer_use | — | `ui-venus` (UI-Venus-1.5-8B) | `ui-venus` | `ui-venus` |
| unfiltered | — | `dolphin3:8b` | `dolphin3:8b` | `dolphin3:8b` |
| transcription | `whisper-v3-turbo` | `whisper-v3-turbo` | `whisper-v3-turbo` | `whisper-v3-turbo` |
| tts | `Kokoro-82M` | `fish-s2-pro` | `fish-s2-pro` | `fish-s2-pro` |
| embedding | `embeddinggemma:300m` | `qwen3-embedding:8b` | `qwen3-embedding:8b` | `qwen3-embedding:8b` |
| image_gen | `flux2-klein` | `flux2-klein` | `z-image-turbo:bf16` | `z-image-turbo:bf16` |
| image_edit | — | — | `FLUX.1-Kontext-dev` | `FLUX.1-Kontext-dev` |
| video | — | — | `Wan2.2-T2V` | `Wan2.2-T2V` |

Approx peak resident footprint (one model at a time; on-demand unload): 32gb ~6GB, 64gb ~31GB, 128gb ~55GB, 512gb ~418GB — each leaving headroom under its cap.

### Changes vs the current presets

- **New 128gb tier** (fills the 64→512 gap a 128GB machine fell through).
- **Computer-use → `UI-Venus-1.5-8B`** on every tier (8B, MLX, highest reported ScreenSpot-Pro grounding ≈69.6%). Replaces `holo3-35b` and `fara:7b`.
- **Transcription → `whisper-large-v3-turbo`** everywhere (~6× faster, near-identical accuracy; drop-in for the whisper path).
- **512 frontier → `GLM-5.2-4bit`** (~418GB, leads open-weight intelligence index) replaces `qwen3.5-397b-8bit`.
- **Code on 128/512 → `qwen3-coder-next`** (80B/3B-active MoE, agentic-coding-tuned, 256K ctx).
- **TTS on 64/128/512 → `fish-s2-pro`** (Fish Audio S2 Pro, highest open-weight TTS Elo, MLX-native, voice cloning + emotion tags) replaces Voxtral. Kokoro stays on 32gb.
- **Embedding on 32gb → `embeddinggemma:300m`** (Google, newer small embedder) replaces `nomic-embed-text`. `qwen3-embedding:8b` (MTEB #1) unchanged on the other tiers.

## Model → backend mapping

Backend is inferred from the string (existing convention): `:` → Ollama tag; `/` → HF repo; otherwise → MLX served-name resolved via the MLX config.

- **Ollama tags:** `qwen3.6:27b-mlx`, `qwen3.6:27b-mlx-bf16`, `qwen3.6:27b-coding-mxfp8`, `qwen3-coder-next:<tag>`, `qwen3.5:122b`, `qwen3-embedding:8b`, `embeddinggemma:300m`, `dolphin3:8b`, `x/flux2-klein:latest`, `x/z-image-turbo:bf16`.
- **MLX served-names** (defined in the MLX config, `on_demand: true`):
  - `qwen3.5-9b` → `mlx-community/Qwen3.5-9B-4bit` (existing `qwen3.5-small`)
  - `whisper-v3-turbo` → `mlx-community/whisper-large-v3-turbo` (NEW)
  - `ui-venus` → `mlx-community/UI-Venus-1.5-8B-bf16` (NEW)
  - `glm-5.2` → `mlx-community/GLM-5.2-4bit` (NEW)
- **HF repos** (downloaded, invoked by mflux / mlx-audio): `mlx-community/Kokoro-82M-bf16` (32gb tts), `mlx-community/fishaudio-s2-pro-8bit-mlx` (`fish-s2-pro`, 64/128/512 tts), `black-forest-labs/FLUX.1-Kontext-dev`, `AITRADER/Wan2.2-T2V-A14B-mlx-bf16`. (Voxtral stays available for multi-voice/9-language use, just no longer the default `tts`.)

## Plumbing

1. **`lib/models.py`** — rewrite `DEFAULT_PROFILES` to the four tiers above; bump `PROFILES_VERSION`; update the explanatory comment. `active` default = a safe middle (`64gb`); the app re-selects by RAM at runtime via `pick_profile_for_ram`, whose fallback changes `laptop` → `32gb`.
2. **MLX config** — collapse `config/mlx-server/config.yaml` + `config-laptop.yaml` into **one** config where every served model is `on_demand: true`. Add the new served-names (`whisper-v3-turbo`, `ui-venus`, `glm-5.2`). The profile, not the config, decides what's pulled/used; on-demand means a 32GB machine never loads the 397B/GLM-5.2.
3. **`install.sh`** — update the RAM→suggested-tier mapping and the profile prompt list (`32gb 64gb 128gb 512gb skip`); drop the per-profile `MLX_CONFIG` case (one config now). The per-profile served-name → model_path resolution (already added) handles the rest.
   - **Download only what's missing for the chosen profile.** Before pulling, check what's already present and skip it — never re-download or re-prompt for a model the machine already has:
     - Ollama tags: present if listed by `ollama list` (match the exact tag).
     - HF repos: present if already materialized in the HF cache (no incomplete blobs).
     Compute the chosen profile's full model set, subtract what's present, and pull only the remainder — reporting counts ("N of M already present, pulling K"). Re-running `install.sh` on a configured machine, or switching to a profile that overlaps the current one, should download nothing it already has.
4. **Migration** — on the `PROFILES_VERSION` jump, the profile-server `load_profiles` migration drops the old preset names (`laptop`/`desktop`/`everyday`/`maximum`) cleanly rather than preserving them as fake "custom" profiles, and resets `active` to a valid tier if it pointed at an old name.
5. **Tests** — rename the smoke-test model maps to the new tiers; keep the contract test (every preset resolves ≥1 pullable model); add/adjust migration tests for the drop-old-presets behavior.

## Verify during implementation

1. **`qwen3.5-9b` vision** — confirm the 9B is actually multimodal (the family is, but verify the 9B specifically). If not, give `32gb` vision a small dedicated burst model instead of reusing the workhorse.
2. **`qwen3-coder-next` tag** — confirm the exact Ollama tag/quant landing ~52GB at 4-bit.
3. **`GLM-5.2-4bit` footprint** — full 4-bit is ~418GB (leaves ~94GB on 512GB). If that's too tight alongside the user's other work, fall back to a REAP-pruned variant (~214–265GB).
4. **`UI-Venus-1.5-8B`** — unproven in this stack: confirm it loads/serves through the MLX computer_use path and grounds acceptably on a few real screenshots before trusting it as the default.
5. **MLX served-model load** — confirm `whisper-large-v3-turbo`, `UI-Venus-1.5-8B`, and `GLM-5.2-4bit` each load via `mlx-openai-server` (or the appropriate runner) as configured.
6. **`fish-s2-pro` TTS** — confirm the installed `mlx-audio` version serves `mlx-community/fishaudio-s2-pro-8bit-mlx` through the `local_speak` path, and that its language coverage is acceptable (Voxtral remains the fallback for multi-language needs).
7. **`embeddinggemma:300m` context** — 2K context window; confirm it's adequate for the embedding chunk sizes used (qwen3-embedding's 32K is unaffected on the other tiers).

## Out of scope

- **Qwen3-ASR** for transcription — higher accuracy than whisper-turbo but a different inference path (mlx-audio STT, not whisper); deferred as a future upgrade.
- **Vision model change** — keep the validated `qwen3.6`/`qwen3.5:122b` multimodal picks; do not adopt a separate Qwen3-VL line in this work.

## Research notes (2026-06-24)

- `qwen3.6` is still the newest open-weight Qwen (3.7-Max is proprietary/API-only). Dense 27B remains the best dense model.
- `z-image-turbo` / `flux2-klein` confirmed as current 2026 image models (mflux supports both).
- Verified available in `mlx-community`: `whisper-large-v3-turbo`, `GLM-5.2-4bit` (~418GB), `UI-Venus-1.5-8B-bf16`, `Qwen3-VL-30B-A3B-Instruct` (MoE). Verified on Ollama: `qwen3-coder-next` (80B/3B, 52–85GB).
- ScreenSpot-Pro grounding: UI-Venus-1.5 ≈69.6%, MAI-UI ≈67.9% (73.5% w/ zoom); Holo3-35B "top-tier" (exact unpublished). UI-Venus chosen for best grounding-per-GB.
- Kimi K2.6 (~562GB at MLX smart-quant) and DeepSeek V4 (1.6T) do not fit 512GB; GLM-5.2 does.
- TTS: Voxtral is mid-pack on TTS Arena (~Elo 1056). `Fish Audio S2 Pro` (4B, open, MLX-native via `mlx-community/fishaudio-s2-pro-8bit-mlx`, voice cloning + emotion tags) has the highest open-weight Elo (1128.7); chosen for 64/128/512. Kokoro-82M kept for 32gb (tiny/fast). Chatterbox-Turbo also strong (beats ElevenLabs in blind tests) and stays available for cloning.
- Embedding: `qwen3-embedding:8b` still #1 on MTEB multilingual (70.58); kept on 64/128/512. `embeddinggemma:300m` (Google, on Ollama, 768-dim) adopted for 32gb over `nomic-embed-text`.
