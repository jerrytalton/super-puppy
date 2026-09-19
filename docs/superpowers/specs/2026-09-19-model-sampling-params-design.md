# Per-model sampling parameters — design (revised after verification)

**Status:** implemented (scaled down) · **Date:** 2026-09-19

> **This spec was rewritten after a red-team.** The original premise —
> "MLX-served models run at temp 1.0" — was **false**. Verified against the
> *running* mlx-openai-server: its LM path defaults to **temp 0.7 / top_p 0.95
> / top_k 20 / min_p 0** (handler `models/mlx_lm.py`; the YAML launch leaves
> `DEFAULT_TEMPERATURE` unset, confirmed absent from the live process env). The
> earlier "1.0" was `config.py`'s dataclass default, which the YAML path never
> applies. So MLX is already at ~Qwen's recommendation. A curated cross-backend
> resolver was over-engineered for the real gap; this is the scaled-down design
> that shipped.

## The actual gap (verified)

- **MLX-served text** (`qwen3.5-small` on 32GB, uncensored MLX quants): temp
  **0.7** — already fine.
- **Ollama-served text** (`qwen3.8:27b-mlx`, `qwen3.8:27b`, `qwen3-coder-next`
  — the main-tier workhorses): Modelfile bakes **`temperature 1`** (verified
  via `ollama show --modelfile`), hotter than Qwen's documented ~0.7. top_p
  0.95 / top_k 20 / min_p 0 / presence_penalty 0 there are already correct.
- **mlx-openai-server is installed unpinned** (`install.sh`), so its good
  defaults could silently drift on an upgrade — the real reason to lock things
  down.

## What shipped

1. **Pin `mlx-openai-server==1.7.0`** in `install.sh` (install + remediation
   lines). Its defaults are already good; pinning stops drift.
2. **Light Ollama temperature override.** `ollama_sampling(model)` in
   `lib/models.py` returns request `options` for Ollama chat models:
   `OLLAMA_SAMPLING_OVERRIDES = {"qwen3": {"temperature": 0.7}}`. Ollama merges
   request `options` over the Modelfile per-key (documented in the playbook via
   the presence_penalty override), so this nudges temp 1.0 → 0.7 while leaving
   the already-correct params untouched. Applied in `chat_ollama` (MCP) and
   both profile-server Ollama chat paths (`_chat` + `_chat_stream`).

**Deliberately minimal, per the red-team:**
- **temperature only** — never inject `presence_penalty` (the playbook
  documents a 1.5 bake truncating structured output).
- **No mode split** — one value (0.7), not separate thinking/non-thinking sets.
  Qwen's thinking rec (~0.6) vs non-thinking (~0.7) differ by a hair, and MLX
  can't even honor `enable_thinking`, so a mode-keyed set would misfire there.
- **No `generation_config` reader** — it reads the wrong file (`config.json`)
  and the flagship MLX model ships no sampling config anyway.
- **Additive/safe** — a model with no override entry gets `{}` → SP sends no
  `options` → exactly current behavior. Embedders (`qwen3-embedding`) are
  excluded (they never hit the chat path).

## Not done (out of scope, low value)
- ds4/glm-5.2 sampling (its own defaults are fine; unverified it honors
  request params — revisit only if a quality issue surfaces).
- Surfacing effective params in the UI; per-profile/per-call overrides.
- laya (non-autoregressive) and the VLM path (already temp 0).

## Testing
- `ollama_sampling`: qwen3.x tags → `{"temperature": 0.7}`; non-qwen and
  embedders → `{}`; only `temperature` is ever set.
- Dispatch: `_chat` sends the override in `options` for a qwen Ollama model and
  no `options` for a non-qwen one.
