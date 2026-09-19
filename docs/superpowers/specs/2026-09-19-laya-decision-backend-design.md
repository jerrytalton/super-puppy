# Laya decision backend — design

**Status:** design (pre-implementation) · **Date:** 2026-09-19 · **Precedes:** implementation plan

## Motivation

**Jev** (TypeSafe AI, Sept 2026) introduced a new model class — "System One" models: given
unstructured program state plus *typed questions*, they return **calibrated probabilities / typed
decisions** (choice, ordinal score, yes-no) in a single non-autoregressive forward pass. No text
generation, so no parsing errors and no hallucination; ~200× faster and ~400× cheaper than an LLM on
classification. Jev itself is proprietary/API.

**Laya** (`convaiinnovations/laya`, Apache 2.0) is the best open-weight implementation of that idea and
what this design adds to Super Puppy. Two checkpoints:
- `convaiinnovations/laya-multilingual` — mmBERT-base backbone, 322M params, 100+ languages. **Fleet default.**
- `convaiinnovations/laya` — ModernBERT-large backbone, 421M params, English, best English accuracy. Optional/override.

### API (in-process, verified by spike)

```python
import laya
agent = laya.load("convaiinnovations/laya-multilingual")
result = agent.predict(
    state,        # dict of text fields, e.g. {"subject": ..., "body": ...}
    questions,    # {name: {"type": "choice"|"score"|"noul", "instructions": str, "criteria": ...}}
)
# result["answers"][name] carries, by type:
#   choice: {"choice": <key>, "probabilities": {opt: p}, "confidence": p, "action": {"act_probability": p}}
#   score:  {"score": <float>, "legend": {i: label}, "probabilities": {i: p}, "confidence": p}
#   noul:   {"noul": <P(true)>, "confidence": p}
# result["usage"] = {"input_tokens": n, "output_tokens": 0}
```

`noul` = calibrated P(true) for a yes/no. `criteria` is a dict of option→description for `choice`, a
list of ordinal labels for `score`, absent for `noul`.

### Spike results (2026-09-19, 128gb-tier machine, this repo's stack)

- Installs and runs on Apple Silicon: torch 2.14.0, **MPS available**.
- **Cold load ~24s; warmed inference 14.5 ms/call** (below the advertised 33ms). This gap is the whole
  reason the model must stay **resident** — cold-start-per-call is a non-starter.

## The load-bearing architectural fact

Every existing Super Puppy task is either generative (Ollama/MLX/ds4 chat, mflux images, mlx-audio TTS,
mlx-video) or an encoder (`embedding`). Laya is neither: it is a **discriminative typed-decision** model
with a question-schema-in / typed-answer-out interface, served by **none** of the existing backends
(`ollama` / `mlx` / `ds4`). It therefore threads a new task type *and* a new backend value through
discovery, dispatch, the MCP tool surface, the playground, provisioning, profiles, and tests — the
service script alone is not sufficient. The cost is breadth, not novelty; each seam already has a
precedent to copy (ds4 for "a third backend on its own internal port"; mflux/mlx-audio for "an HF-repo
model provisioned via `hf download`").

## Design decisions (settled with the user)

1. **Surface:** MCP tool + playground only. No internal use of laya inside SP's own routing/pick logic
   (YAGNI; revisit as a separate project if wanted).
2. **Variants:** `laya-multilingual` is the fleet default; English `laya` supported as an override.
3. **Serving:** a **dedicated persistent service**, model resident. Not in-process (keeps torch out of
   the MCP/profile servers, which stay thin dispatchers) and not subprocess-per-call (24s cold load).
4. **Topology:** laya is available on **every install, all tiers** (~0.7–1.3GB, universally
   local-capable, unlike server-only ds4), on an **internal-only** port (never in `tailscale serve`).
   **But it is only started *resident* on the machine that actually serves decisions** — i.e. in
   **server** mode and **offline** mode — not on a client that routes to the desktop (see decision 5).
   *(Red-team fix: "always resident everywhere" + "route to server" made a client's local laya
   permanent dead weight — ~1.3GB + a torch/MPS process + a 24s cold load + a status dot for a model it
   only uses when the desktop is unreachable.)* On a client (desktop reachable), laya is **lazy-started
   on first offline fallback**, mirroring how the profile server auto-starts only where it's needed. So
   the 128gb M5 Max still runs laya locally — as a server, or offline — without paying for it while it's
   a client proxying to the desktop.
5. **Routing:** **simple** — `local_decide` follows SP's normal client→server MCP routing. In client
   mode a laptop's call executes on the desktop's MCP against the desktop's laya; the laptop's own laya
   runs only in offline mode. (A local-first exception was considered and declined — least code wins.)

## Components

### 1. New task type — `decision` (`lib/models.py`)

- Add `decision` to `SPECIAL_TASKS` with `{"label": "Decision", "prefixes": ["laya"]}` — capability/name
  matched, not an LLM filter. Add to the `SPECIAL_TASKS` task-key list used by consumers.
- **Not** added to `TASK_FILTERS` (it is not an LLM; the min_active_b/min_ctx gates don't apply).
- Add `laya` to `LLM_BACKENDS`? **No** — it is not a chat backend. Introduce it as a standalone backend
  string `"laya"` used only by the decision dispatch path, mirroring how mflux/mlx-audio backends sit
  outside `LLM_BACKENDS`.

### 2. New backend service — `app/laya-server.py`

- Small Flask app (SP already standardizes on Flask) on an **internal-only** port `8003`
  (`LAYA_PORT` in `network.conf`, following `DS4_PORT`/8002). Bind localhost only.
- PEP 723 inline deps pin `laya` (which pulls torch/transformers) — **isolated to this service's env**,
  never added to the MCP or profile server.
- Loads `laya-multilingual` resident at startup (~24s); English `laya` as a second served name, loaded
  on-demand on first request for it and kept resident (idle-unload optional, low priority).
- Endpoints:
  - `POST /decide` — body `{"model": <served-name>, "state": {...}, "questions": {...}}` → laya's
    `predict()` result verbatim. 400 on schema errors, 503 while a model is still loading.
  - `GET /v1/models` — served names, for discovery parity with the other backends.
  - `GET /health` — liveness.
- Started by `bin/start-local-models` on **every** machine (all tiers), like Ollama/MLX. Health-checked
  the same way; menu bar shows a service dot (green/yellow/red) like ds4's.

### 3. MCP tool — `local_decide` (`mcp/local-models-server.py`)

- Signature: `local_decide(state: dict, questions: dict, model: str | None = None)`.
- Validates the typed-question schema (each question has a valid `type` ∈ {choice, score, noul} and the
  matching `criteria` shape), resolves `model`, dispatches to the laya service `/decide`, returns the
  typed answers + calibrated probabilities.
- **Model resolution must use a laya-only resolver that FAILS LOUD — never the shared `pick_model`
  cascade.** *(Red-team fix, the single most likely implementation bug):* `pick_model` is invoked with
  `fallback_to_general=True` and, on a miss, falls through to "any LLM in `LLM_BACKENDS`"
  (`mcp/local-models-server.py:633-640`). Reusing it for `decision` would silently route a caller's
  `state`+`questions` to glm-5.2/qwen as a chat prompt — the exact "decision and chat models are not
  interchangeable" failure this design forbids. Implement `resolve_decision_model(model)`: exact/prefix
  match against laya-backed registry entries only; if nothing resolves, raise a clear error. No general
  fallback, ever.
- Discovery: the MCP server's model discovery adds a laya branch querying `LAYA_URL/v1/models` with a
  short per-call `timeout` (match ds4's `timeout=5`, `mcp/local-models-server.py:511`, so a cold/loading
  laya can't stall the whole discovery gather during its ~24s load). Served names get `backend="laya"`,
  `task="decision"`, and metadata from new `LAYA_*` constants in `lib/models.py` (name, params; no
  context/vision) — there is no existing table laya fits (`KNOWN_ACTIVE_PARAMS` is MoE-active-params
  only), so add constants exactly like `DS4_MODEL_NAME`/`DS4_TOTAL_PARAMS_B`/…
- Logged to the activity DB like every other request (`lib/activity.py`).

### 4. Profile-server discovery + playground (`app/profile-server.py`, `app/tools.html`)

- **Add `_fetch_laya_models(existing)` to `_fetch_all_models`**, mirroring `_fetch_ds4_models`
  (`app/profile-server.py:1214`, called at :1297). *(Red-team fix — the spec previously listed only
  `get_eligible_tasks` and missed the real seam.)* Without this discovery branch: the Profiles UI can't
  display/assign/validate the `decision` pick, `get_eligible_tasks` is never invoked on a laya model (so
  the decision-only rule is dead code), **and the pick can't resolve on-demand** — `resolve_pref_candidate`
  gates HF-repo-id resolution on `task in HF_TASK_BACKENDS` (`lib/models.py:360`) and `decision` is
  deliberately not in `HF_TASK_BACKENDS`, so a `decision → "convaiinnovations/laya-multilingual"` pref
  not already in the registry returns `None`. The discovery branch is what puts laya in the registry so
  the pref resolves by name. Decide behavior when the service is down: the pick should still validate
  against the profile (surface "decision backend not running") rather than silently vanish.
- `test_playground_coverage` requires every MCP tool to have a playground UI card **and** an `/api/test`
  route (add `local_decide → {"decide"}` to `MCP_TO_PLAYGROUND`). Add a `decide` card (state textarea +
  a small typed-question builder) and a `decide` branch in `/api/test` that dispatches to the laya
  service and renders the typed answer + probabilities.
- `get_eligible_tasks`: a laya-backed model qualifies for `decision` only (its own class), never the LLM
  pools.

### 5. Provisioning & profiles (`lib/models.py`, `install.sh`, `app/menubar.py`)

- Add `decision` → `convaiinnovations/laya-multilingual` to **all four tier presets**. Bump
  `PROFILES_VERSION`.
- The pick is an HF repo id (`/`, no `:`) → the existing `profile_hf_models` / `hf download` autopull
  category fetches it. **Risk to verify (below):** confirm `laya.load()` reads the same HF cache that
  `hf download` populates, so provisioning composes with autopull rather than laya re-downloading.
- The English `laya` override is downloaded only when a profile/override names it.

### 6. Modes / remote access / memory (`app/menubar.py`, `bin/start-local-models`)

- Port `8003` is **never** added to the `tailscale serve` tuple (`app/menubar.py:2373-2374`) — assert
  this with a test. localhost-bind + never-served is the trust boundary; **no bearer auth on 8003 is
  correct and consistent** (ds4:8002 and Ollama/MLX have none either — only the served 8100/8101 carry
  the token). Input safety for arbitrary `state`/`questions` → torch is handled by the MCP-boundary
  schema validation in §3, not by transport auth.
- **Residency is mode-gated** (decision 4): `start-local-models` starts laya resident in **server** and
  **offline** modes; in **client** mode it is not started (lazy-start on offline fallback).
- **Memory accounting** *(red-team fix)*: the contention-aware keep-warm math
  (`app/menubar.py:276-294`, gating ~:976) sizes headroom from the model registry's VRAM accounting,
  which is **blind to a separate torch process**. Where laya runs resident, subtract a fixed laya
  reserve (~1.5GB) from available-memory before the warm-budget/headroom decision, or keep-warm will
  over-commit by laya's footprint on the tighter tiers. (Mode-gating already keeps it off client
  machines that route away.)

## Data flow (a decide call)

```
Claude → MCP local_decide(state, questions, model?)
      → resolve model (default: profile 'decision' pick, laya-multilingual)
      → POST localhost:8003/decide {model, state, questions}
      → laya-server: agent.predict(state, questions)   # ~14ms warmed
      → typed answers + calibrated probabilities + confidence
      → MCP returns result; logged to activity DB
```

## Error handling

- **Model still loading** (first ~24s after service start): `/decide` returns 503; the MCP tool surfaces
  a clear "laya is still loading, retry in a moment" rather than hanging.
- **Bad question schema:** validated at the MCP tool boundary → 400 with the offending field named
  (fail loud, no silent coercion).
- **Service down:** discovery omits laya; `local_decide` returns an actionable "decision backend not
  running" error (mirrors the ds4-down path). Never falls back to an LLM — a decision model and a chat
  model are not interchangeable.

## Testing

- **Unit:** schema validation (each type + malformed), model resolution/default, dispatch shaping
  (mocked service), discovery adds the laya branch, `get_eligible_tasks` returns `decision`-only.
- **Playground coverage:** the enforced UI+route test passes for `decide`.
- **Smoke (`tests/_smoke_helpers.py`):** a real decision through the live stack — e.g. the invoice-triage
  example — asserting a well-formed typed answer with probabilities that sum to ~1. Marked `smoke`,
  skips cleanly when the service is down.
- **No mocking of the laya wire format in smoke** — same discipline as the other smoke tests.

## ds4-seam checklist (enumerate in the plan, don't leave to discovery)

A new backend on an internal port touches the same seams ds4 did. Grep `ds4`/`DS4_`/`8002` to confirm
each; the plan must cover:

- `bin/start-local-models`: start laya (mode-gated), and add it to `stop_services()` (`pkill` like
  `ds4-server`, :113-119) and `show_status` (:163-169).
- `bin/local-models-mcp-detect`: `export LAYA_URL="http://localhost:${LAYA_PORT:-8003}"` (mirrors the
  unconditional `DS4_URL` export at :72-73) so the desktop's MCP finds laya.
- `lib/models.py`: `LAYA_PORT` into `_NETWORK_DEFAULTS` **and** `_NUMERIC_KEYS` (:30-46) and the
  `config/local-models/network.conf` template; confirm against `validate_network_conf`. Plus `LAYA_*`
  metadata constants.
- `app/menubar.py`: a laya service status dot; a laya line in Copy Diagnostics (:2412-2413).
- MCP + profile-server discovery branches (both), each with a short timeout.
- Tests: `PROFILES_VERSION` bump trips the release.sh fleet cross-version compat gate and
  `test_profile_server`/`test_deployment`; the smoke harness is hand-maintained tuples
  (`tests/_smoke_helpers.py` `CHAT_CASES`/`FIXTURE_CASES`) so add a decision case tuple + a
  typed-question body builder + a "probabilities sum to ~1" assertion; add the `MCP_TO_PLAYGROUND` entry.

## Risks

1. **`laya.load()` vs `hf download` cache — RESOLVED (verified 2026-09-19).** The spike's
   `laya.load("convaiinnovations/laya-multilingual")` populated the standard HF hub cache
   (`~/.cache/huggingface/hub/models--convaiinnovations--laya-multilingual`) — the same cache
   `hf download` writes — and `laya-multilingual` is a **standalone repo** (`model.safetensors` at root,
   plus `encoder/` + `tokenizer/` subdirs), **not** an MLX-subfolder-style repo. So `profile_hf_models`
   autopull composes with `laya.load()` and no dedicated provisioning step is needed. (Confirm the same
   for the English `laya` repo before referencing it.)
2. **torch is a heavy dep** (~2GB) — acceptable because it is isolated to the laya service's env and
   pulled once; it never touches the MCP/profile servers or their startup time.
3. **Resident memory on the 32gb tier** — ~0.7–1.3GB for the multilingual model alongside the other
   warm models. Fits, but confirm against the warm-budget math; if tight, make laya idle-unload on the
   32gb tier only.
4. **Two checkpoints, one service** — English `laya` as a second served name doubles resident memory if
   both are loaded; default to multilingual-only-resident and load English on demand.

## Non-goals

- Using laya inside SP's own routing/model-pick logic (separate future project).
- Exposing laya cross-machine via Tailscale (each machine self-serves).
- Reproducing or wrapping Jev (proprietary); laya is the open substitute.
