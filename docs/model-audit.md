# Model Audit — April 2026

M3 Ultra, 512GB RAM. Ollama 0.20.2 (MLX backend). mlx-openai-server for on-demand models. mflux for image gen. mlx-audio for TTS/STT.

## Current Inventory

| Store | Disk |
|-------|------|
| Ollama (`~/.ollama/models/`) | 493 GB |
| HuggingFace cache | 1.0 TB |
| **Total** | **~1.5 TB** |

22 Ollama model tags. 7 active mlx-server models, 9 unused HF cache downloads.

---

## Master Model List by Tool

For each tool: what's installed, what's best-in-class, and where to run it.

### Code

| Model | Params | Active | Quant | Size | Ollama | MLX | Status |
|-------|--------|--------|-------|------|--------|-----|--------|
| **Qwen3-Coder-Next** | 80B | 3B MoE | Q4_K_M | 51 GB | `qwen3-coder-next` | mlx-community | **Installed (Ollama). Best in class.** |
| Codestral | 22B | dense | Q4 | ~13 GB | `codestral` | mlx-community | Not installed. Strong multi-language, simpler tasks. |
| DeepSeek-Coder-V3 | 671B | 37B MoE | Q4 | ~350 GB | `deepseek-coder-v3` | — | Not installed. Massive, fits at Q4 but overkill. |

**Verdict**: Qwen3-Coder-Next is the right pick. No change needed.

---

### General

| Model | Params | Active | Quant | Size | Ollama | MLX | Status |
|-------|--------|--------|-------|------|--------|-----|--------|
| **Qwen3.5-122B-A10B** | 122B | 10B MoE | Q4_K_M | 81 GB | `qwen3.5:122b` | mlx-community/4bit | **Installed (both).** |
| **Qwen3.5-397B-A17B** | 397B | 17B MoE | 4-bit | ~220 GB | — | mlx-community | **Installed (MLX only, on-demand).** |
| Qwen3.5-35B-A3B | 36B | 3B MoE | Q4_K_M | 23 GB | `qwen3.5:35b` | mlx-community/4bit | **Installed (both). Duplicate.** |
| Gemma 4 31B Dense | 31B | dense | Q4_K_M | 19 GB | `gemma4:31b` | mlx-community | **Installed (Ollama).** |
| **Gemma 4 26B-A4B** | 26B | 3.8B MoE | Q4/Q8 | ~16 GB | `gemma-4-26b-a4b-it` | mlx-community | **Not installed. 300 tok/s at Q8 on M2 Ultra. Apache 2.0.** |
| Llama 4 Scout | 109B | 17B MoE | Q4 | ~60 GB | `llama4-scout` | — | Not installed. 10M context window. |
| Phi-4 | 14B | dense | Q4 | ~8 GB | `phi4` | mlx-community | Not installed. Punches above weight, tiny. |

**Verdict**: Current routing (qwen3.5-fast → qwen3.5-large) is solid. **Gemma 4 26B-A4B is worth adding** — 3.8B active params at 300 tok/s is absurdly efficient for a general assistant, Apache licensed. Good "fast" tier alternative.

---

### Reasoning

| Model | Params | Active | Quant | Size | Ollama | MLX | Status |
|-------|--------|--------|-------|------|--------|-----|--------|
| **Nemotron-3-Super** | 124B | 12B MoE | Q4_K_M | 86 GB | `nemotron-3-super` | MoringLabs/3.6bit | **Installed (both). Duplicate.** |
| **DeepSeek-R1-0528** | 671B | 37B MoE | TQ1_0 | ~162 GB | `deepseek-r1` | mlx-community/4bit | **Not installed. 87.5% AIME. Fits at ultra-low quant.** |
| DeepSeek-R1-Distill-Qwen3-8B | 8B | dense | Q4 | ~5 GB | `deepseek-r1:8b` | mlx-community | Not installed. Distilled R1 reasoning in 5 GB. |
| QwQ-32B | 32B | dense | Q4 | ~20 GB | `qwq` | mlx-community | Not installed. Stepwise reasoning. |
| Qwen3.5 (thinking mode) | various | — | — | — | — | — | Already installed. Toggle `/think`. |

**Verdict**: Nemotron-Super is your current primary and it's good. **DeepSeek-R1-0528** at ~162GB would be the reasoning king if you want to dedicate the RAM. The 8B distill is a cheap experiment.

---

### Vision

All Qwen3.5 models are **inherently multimodal** — vision is native, not bolted on. Qwen3.5 outperforms the older Qwen3-VL family on vision benchmarks. No need for separate vision-specific models from the Qwen3 generation.

| Model | Params | Active | Quant | Size | Ollama | MLX | Status |
|-------|--------|--------|-------|------|--------|-----|--------|
| **Qwen3.5-397B-A17B** *(MoE)* | 397B | 17B | 4-bit | ~220 GB | — | mlx-community | **Installed (MLX on-demand). Best vision quality.** |
| **Qwen3.5-122B-A10B** *(MoE)* | 122B | 10B | Q4_K_M | 81 GB | `qwen3.5:122b` | mlx-community | **Installed. Outperforms Qwen3-VL-235B.** |
| **Qwen3.5-35B-A3B** *(MoE)* | 36B | 3B | Q4_K_M | 23 GB | `qwen3.5:35b` | mlx-community | **Installed. Fast everyday vision (3B active).** |
| Qwen3.5-27B (dense) | 28B | dense | Q4_K_M | 17 GB | `qwen3.5:27b` | mlx-community | **Installed. Dense alternative.** |
| Gemma 4 31B (dense, vision) | 31B | dense | Q4_K_M | 19 GB | `gemma4:31b` | mlx-community | **Installed. Apache 2.0.** |
| **Gemma 4 26B-A4B** *(MoE)* | 26B | 3.8B | Q4/Q8 | ~16 GB | `gemma4:26b` | mlx-community | **Not installed. Fast vision + text, Apache 2.0.** |
| ~~Qwen3-VL-235B-A22B~~ | 235B | 22B | Q4_K_M | 143 GB | `qwen3-vl:235b` | mlx-vlm | **Installed but REDUNDANT. Qwen3.5-122B beats it.** |
| GLM-OCR | 1.1B | dense | F16 | 2.2 GB | `glm-ocr` | — | **Installed. OCR specialist.** |

**Verdict**: **`qwen3-vl:235b` (143 GB) can be deleted** — Qwen3.5-122B outperforms it at 81 GB. Your everyday vision should route through Qwen3.5-35B-A3B (3B active, fast) with 122B or 397B as fallbacks. All Qwen3.5 models handle vision natively.

---

### Computer Use

| Model | Params | Quant | Size | Ollama | MLX | Status |
|-------|--------|-------|------|--------|-----|--------|
| **UI-TARS-72B** | 72B | 4-bit | ~40 GB | — | mlx-community | **Installed (MLX on-demand).** |
| UI-TARS-7B | 1.8B | F16 | 3.6 GB | `avil/ui-tars` | — | **Installed (Ollama). Lightweight fallback.** |
| Fara-7B | 7.6B | Q4_K_M | 6.0 GB | `maternion/fara:7b` | — | **Installed (Ollama).** |

**Verdict**: Good setup. No changes needed.

---

### Image Generation (mflux / MLX-native)

| Model | Params | Size | Status |
|-------|--------|------|--------|
| **FLUX.2 Klein 4B** | 4B | ~5.7 GB | **Installed** (`x/flux2-klein`). Sub-second. |
| FLUX.2 Klein 9B | 9B | ~11 GB | **Installed** (`x/flux2-klein:9b`). Higher quality. |
| **Z-Image Turbo** | 10B | ~12 GB | **Installed** (`x/z-image-turbo`). Fast. |
| Z-Image Turbo BF16 | 16B | ~32 GB | **Installed** (`x/z-image-turbo:bf16`). Full precision. |
| FLUX.1 Dev | 12B | ~12 GB | Not installed via Ollama. Available in mflux directly. |
| FLUX.1 Schnell | 12B | ~12 GB | Not installed via Ollama. Available in mflux directly. |

**Verdict**: You have both small and large quants of Klein and Z-Image. The question is which quality level you actually use. The larger quants produce better images but cost 2-3x disk. Keep both if disk isn't critical.

---

### Image Editing (mflux / MLX-native)

| Model | Status |
|-------|--------|
| **FLUX.1 Kontext Dev** | **Installed.** Best-in-class for local editing. |

**Verdict**: No changes.

---

### Translation

| Model | Params | Status |
|-------|--------|--------|
| Qwen3.5 (any size) | various | **Installed.** 201 languages, routed via `qwen3.5-fast`. |
| Cogito 2.1 | 3B/14B | Not installed. 30+ languages, reasoning-enhanced translation. |

**Verdict**: Qwen3.5 handles translation well. Cogito could be added if you want a dedicated translator, but it's not a clear upgrade.

---

### TTS (mlx-audio / MLX-native)

| Model | Params | Size | Status |
|-------|--------|------|--------|
| **Voxtral 4B** | 4.1B | ~8 GB | **Installed.** 20 voices, 9 languages. |
| **Chatterbox** | 350M | ~700 MB | **Installed.** Voice cloning. |
| Kokoro | 82M | ~170 MB | Not installed. 54 voices, ultra-fast. |
| Sesame CSM-1B | 1B | ~8 GB | Not installed. Highest quality (4.7 MOS). |
| Qwen3-TTS | new | unknown | Not installed. New entrant. |

**Verdict**: Voxtral + Chatterbox is a strong combo. **Kokoro** at 82M is basically free and adds 54 voice presets — worth adding if mlx-audio supports it.

---

### Transcription (mlx-audio / MLX-native)

| Model | Params | Size | Status |
|-------|--------|------|--------|
| **Whisper Large v3** | 1.5B | ~3 GB | **Installed (MLX on-demand).** |
| Whisper Large v3 Turbo | 809M | ~1.6 GB | Not installed. 5x faster, 99% of quality. |
| **Qwen3-ASR-1.7B** | 1.7B | ~3 GB | Not installed. SOTA accuracy, MLX via mlx-audio. |

**Verdict**: **Whisper v3 Turbo** would be a speed upgrade at near-identical quality. **Qwen3-ASR** is worth evaluating if accuracy matters more than speed.

---

### Embedding

| Model | Params | Dims | Context | Size | Status |
|-------|--------|------|---------|------|--------|
| **all-minilm** | 23M | 384 | 512 | 45 MB | **Installed.** Current primary. |
| **mxbai-embed-large** | 334M | 1024 | 512 | 669 MB | **Installed.** Higher quality. |
| **nomic-embed-text** | 137M | 768 | 2048 | 274 MB | **Installed.** Longer context. |
| nomic-embed-text-v2-moe | MoE | 768 | 8192 | ~300 MB | Not installed. MoE, 100+ languages, 8K context. |

**Verdict**: Good coverage. **nomic-embed-text-v2-moe** is an upgrade if you need multilingual or longer-context embeddings.

---

### Uncensored

| Model | Params | Size | Status |
|-------|--------|------|--------|
| **Dolphin 3.0 8B** | 8B | 4.9 GB | **Installed.** |
| Dolphin 3.0 R1 Mistral 24B | 24B | ~15 GB | Not installed. Uncensored + reasoning. |

**Verdict**: Dolphin 3.0 8B is fine for basic uncensored use. The 24B R1 variant adds reasoning if you need it.

---

### Long Context

| Model | Claimed | Reliable | Status |
|-------|---------|----------|--------|
| Llama 4 Scout | 10M | ~6M | Not installed. |
| **Qwen3.5-397B** | 256K (1M hosted) | ~170K | **Installed (MLX).** |
| **Qwen3.5-122B** | 256K | ~170K | **Installed.** |
| **Nemotron-3-Super** | 1M | untested | **Installed.** |

**Verdict**: Qwen3.5 at 170K reliable context covers most needs. Llama 4 Scout's 10M is unique but would cost ~60 GB at Q4.

---

## Disk to Reclaim

### Redundant model (Qwen3-VL superseded by Qwen3.5)

| Model | Size | Why |
|-------|------|-----|
| `qwen3-vl:235b` | **143 GB** | Qwen3.5-122B outperforms it on vision. Biggest single win. |

### Duplicates (same model in Ollama + mlx-server)

| Model | Ollama | Ollama Size | MLX Name | Decision |
|-------|--------|-------------|----------|----------|
| Qwen3.5-35B-A3B | `qwen3.5:35b` | 23 GB | `qwen3.5-fast` | **Keep Ollama** (vision prefs use it). Remove from mlx-server OR vice versa. |
| Qwen3.5-9B | `qwen3.5:9b` | 6.6 GB | `qwen3.5-small` | **Remove Ollama copy** (not in any prefs). |
| Nemotron-3-Super | `nemotron-3-super` | 86 GB | `nemotron-super` | **Keep MLX** (on-demand loading matters at 86 GB). Remove Ollama, update prefs. |

### Total reclaimable: ~259 GB (143 + 116)

---

## Unused HF Cache

These MLX models are downloaded but not configured:

| Model | Action |
|-------|--------|
| mlx-community/Qwen3.5-35B-A3B-8bit | Delete (you have the 4bit) |
| mlx-community/Mistral-7B-Instruct-v0.3-4bit | Delete (superseded by Qwen3.5) |
| mlx-community/DeepSeek-R1-4bit | Keep if you want to try R1, otherwise delete |
| mlx-community/Qwen2.5-VL-3B-Instruct-4bit | Delete (superseded by Qwen3-VL/Qwen3.5) |
| mlx-community/Qwen-Image-2512-8bit | Delete (not used) |
| machiabeli/Qwen-Image-2512-4bit-MLX | Delete (not used) |
| machiabeli/Qwen-Image-2512-8bit-MLX | Delete (not used) |

**Potential savings: ~200+ GB** from HF cache cleanup.

---

## Recommended Profile Changes

### "Everyday" Profile (current active)

| Task | Current | Recommended | Why |
|------|---------|-------------|-----|
| code | qwen3-coder-next *(MoE, 3B active)* | **no change** | Best code model |
| general | qwen3.5-fast *(MoE, 3B active)* | **no change** (or add Gemma 4 26B-A4B as alt) | qwen3.5-fast is great; Gemma is faster |
| reasoning | nemotron-super *(MoE, 12B active)* | **no change** | Nemotron is strong; R1-0528 is upgrade path |
| vision | qwen3.5-large (397B MLX) | **qwen3.5:35b → qwen3.5:122b → qwen3.5-large** | 35B (3B active) for everyday, 122B/397B for hard cases |
| long_context | nemotron-super | **no change** | |
| translation | qwen3.5-fast | **no change** | |
| embedding | all-minilm | **mxbai-embed-large** | Higher quality, already installed |

### Top Additions by Impact

1. **Delete `qwen3-vl:235b`** (saves 143 GB) — Qwen3.5-122B outperforms it on vision
2. **Gemma 4 26B-A4B** *(MoE, 3.8B active)* (~16 GB, Ollama `gemma4:26b`) — 300 tok/s, vision-capable, Apache 2.0. Fastest general assistant option.
3. **Whisper v3 Turbo** (~1.6 GB, MLX) — 5x faster transcription, near-identical quality
4. **Kokoro** (82M, MLX via mlx-audio) — ultra-fast TTS, 54 voices, nearly free on disk
5. **Fix vision routing** — route through Qwen3.5 MoE models (35B→122B→397B), not Qwen3-VL

### MoE Highlights (your preferred architecture)

Every model in your everyday profile is already MoE except the specialist tools:

| Model | Total | Active | Use |
|-------|-------|--------|-----|
| Qwen3-Coder-Next | 80B | 3B | Code |
| Qwen3.5-35B-A3B | 36B | 3B | General, translation, everyday vision |
| Qwen3.5-122B-A10B | 122B | 10B | Serious vision, general fallback |
| Qwen3.5-397B-A17B | 397B | 17B | Hardest tasks |
| Nemotron-3-Super | 124B | 12B | Reasoning, long context |
| Gemma 4 26B-A4B (add) | 26B | 3.8B | Fast general alt, Apache 2.0 |

---

*Generated April 9, 2026. Models and benchmarks reflect state as of this date.*
