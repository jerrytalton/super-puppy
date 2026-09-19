# Per-model sampling parameters — design

**Status:** design (pre-implementation) · **Date:** 2026-09-19 · **Precedes:** implementation plan

## Motivation (the gap, verified)

SP sends **no sampling parameters** on any chat request — only `max_tokens` and the thinking
on/off toggle (`chat_ollama`/`chat_mlx`/`chat_ds4` in `mcp/local-models-server.py`, and
`_chat_stream` in `app/profile-server.py`; grep confirms no `temperature`/`top_p`/etc. anywhere in
`app`/`mcp`/`lib` except the VLM CLI's hardcoded `temp 0.0`). So effective sampling is whatever each
backend defaults to:

- **Ollama-served** (`:` tags — `qwen3.8:27b`, `qwen3.8:27b-mlx`, `qwen3-coder-next`, …): the model's
  **Modelfile `PARAMETER` bakes** apply. Usually reasonable, but author-dependent and sometimes wrong
  (the playbook's `qwen3.5:9b presence_penalty 1.5` broke JSON output).
- **mlx-openai-server-served** (bare names in `config/mlx-server/config.yaml` — `qwen3.5-small` = the
  whole 32GB tier's text tasks, `ui-venus`, the uncensored/unfiltered MLX quants): run at the server's
  **generic defaults temp 1.0 / top_p 1.0** — it does **not** read the model's `generation_config.json`
  (verified: request schema defaults `temperature=None` → falls back to `default_temperature=1.0`).
  These are effectively un-tuned and too hot for Qwen.
- **ds4** (glm-5.2): its own launch defaults.

Additional gaps: even where params apply they are **static**, but Qwen publishes **different** sampling
for thinking vs non-thinking; and there's **no single source of truth**, so the same base model behaves
differently across tiers (Ollama qwen3.8 vs MLX qwen3.5-small).

## Goals

1. A single source of truth for each model's recommended sampling, applied on **every** chat request
   across all three chat backends.
2. **Mode-aware**: thinking vs non-thinking use the model's respective recommended values, keyed off
   SP's existing think toggle.
3. Close the worst gap first (MLX temp-1.0), proving the whole apply-path end-to-end, then fan out.
4. Never regress a working model; keep it easy to correct a bad backend default.

## Non-goals

- Rewriting Ollama Modelfiles (we send request-level params instead — reversible, non-invasive).
- A full sampling UI / per-call sampling knobs (YAGNI; the think toggle stays the only user control).
- Tuning laya (non-autoregressive, no sampling) or the VLM path (already `temp 0`).

## Design decisions (settled with the user)

1. **Hybrid source of truth.** Base = the model's own config; curated overrides layered on top where we
   know better.
2. **Mode-aware** thinking vs non-thinking param sets.
3. **Rollout: MLX first**, then Ollama, then ds4, then optional playground exposure.

## Architecture

### The resolver — `lib/models.py`

```python
def model_sampling(name: str, backend: str, thinking: bool) -> dict:
    """Return the sampling params SP should SEND for this request.
    Backend-specific because the 'base' differs (see below). Empty dict =
    send nothing (let the backend apply its own)."""
```

Two inputs feed it:

- **Curated overrides** — `SAMPLING_OVERRIDES`, a table keyed by exact name or family prefix, each
  entry carrying a `think` and a `no_think` param set (temperature, top_p, top_k, min_p,
  presence_penalty, …). Seeded from published recommendations (Qwen3.x thinking: ~temp 0.6/top_p 0.95/
  top_k 20/min_p 0; non-thinking: ~temp 0.7/top_p 0.8/top_k 20/presence_penalty ~1.5 — final values
  pinned from Qwen's model card at implementation) and known-bad-bake fixes.
- **Config base** — the model's own recommended defaults, needed only where the backend won't apply
  them itself (MLX). Read from `generation_config.json` in the HF cache / local serving dir via the
  existing `lib/hf_scanner.read_newest_hf_config` (already handles absolute local-dir paths).

**Per-backend contract** (this is the load-bearing nuance):

| Backend | Base already applied? | What `model_sampling` returns |
|---------|-----------------------|-------------------------------|
| `ollama` | Yes — Modelfile params apply automatically | **curated overrides only** (Ollama merges request `options` over the Modelfile) |
| `mlx` | No — server ignores `generation_config` | **full effective set**: `generation_config` base **←** curated overrides |
| `ds4` | Yes — its launch defaults | curated overrides only (Phase 3; confirm ds4 merges request params) |

Precedence (low→high): backend/Modelfile/config base → curated override. (No per-call sampling override
is exposed today; the think toggle only selects which mode's set to use. A future per-call/per-profile
override would sit above curated.)

### Apply points

- **MCP** (`mcp/local-models-server.py`): `chat_ollama`/`chat_mlx`/`chat_ds4` merge
  `model_sampling(model, backend, think)` into the request — Ollama into `options`, MLX/ds4 into the
  top-level body.
- **Profile server** (`app/profile-server.py`): `_chat_stream` (and any non-stream chat path) does the
  same, so the Playground and client-mode traffic get identical treatment.
- laya/decision and vision paths are untouched.

## Rollout (phased; each phase ships independently)

**Phase 1 — MLX (the real gap) + the resolver.** Build `model_sampling` + `SAMPLING_OVERRIDES` +
`generation_config` base-read; wire it into `chat_mlx` and the profile-server MLX branch; seed Qwen3.x
overrides. Result: `qwen3.5-small` (32GB tier) and the uncensored MLX quants stop running at temp 1.0.
Prove end-to-end (assert the outgoing request body carries the resolved params; smoke a real call).

**Phase 2 — Ollama overrides.** Send curated overrides in `options` where we know better than the
Modelfile (incl. the `qwen3.5:9b presence_penalty 0` fix). Models without an override entry are
unchanged (Modelfile still governs).

**Phase 3 — ds4.** Confirm ds4 honors request-level sampling; apply overrides for glm-5.2 if warranted.

**Phase 4 (optional) — surface it.** Show the effective params on the Playground / model card;
consider a per-profile override. Only if wanted.

## Error handling / safety

- Unknown model or no override + no readable config → return `{}` → SP sends nothing → **exact current
  behavior** (backend default). So the change is strictly additive and can't regress a model we don't
  have an entry for.
- Malformed `generation_config` → treated as absent (fall through), never raises into a chat call.
- Values are validated to a known key set before sending, so a typo in the table can't inject an
  arbitrary field.

## Testing

- **Unit** (`model_sampling`): curated-over-config merge; mode selection (think vs no_think); exact vs
  family-prefix match; unknown model → `{}`; per-backend contract (ollama = overrides only, mlx = full
  effective set); malformed generation_config tolerated.
- **Dispatch**: each of `chat_ollama`/`chat_mlx`/`chat_ds4` and `_chat_stream` includes the resolved
  params in the outgoing body (mock the HTTP client, assert the body) — and includes **nothing** when
  the resolver returns `{}`.
- **Smoke**: a live MLX chat with a Qwen override confirms the call succeeds with the params applied
  (assert on the request, not model output).

## Risks

- **Pinning the exact published values.** The recommended numbers must be taken from each model's
  current card at implementation time, not memory — the plan's Phase 1 first step is to fetch and pin
  Qwen3.x's published thinking/non-thinking values with a source link in the table.
- **Ollama merge semantics.** Confirm request `options` truly override Modelfile params (expected, but
  verify with one probe before relying on it in Phase 2).
- **ds4 request-param support** is unverified (Phase 3 gate).
