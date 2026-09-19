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
4. **Topology:** runs **locally on every install, all tiers** (32/64/128/512) — laya is ~0.7–1.3GB
   resident and universally local-capable, unlike ds4 (server-only because glm-5.2 is 244GB). Port is
   **internal-only** (never in `tailscale serve`); each machine self-serves, so there is no cross-machine
   serving to expose.
5. **Routing:** **simple** — `local_decide` follows SP's normal client→server MCP routing. In client
   mode a laptop's call executes on the desktop's MCP against the desktop's laya; the laptop's own laya
   is used only in offline mode. (A local-first exception was considered and declined — least code wins.)

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
  matching `criteria` shape), resolves `model` (default = profile's `decision` pick), dispatches to the
  laya service `/decide`, returns the typed answers + calibrated probabilities.
- Discovery: the MCP server's model discovery adds a laya branch querying `LAYA_URL/v1/models`; the
  served names get `backend="laya"`, `task="decision"`, and hardcoded metadata (params from the known
  table; no context/vision). Without this the model is invisible to `pick_model`.
- Logged to the activity DB like every other request (`lib/activity.py`).

### 4. Playground + profile server (`app/profile-server.py`, `app/tools.html`)

- `test_playground_coverage` requires every MCP tool to have a playground UI card **and** an `/api/test`
  route. Add a `decide` card (state textarea + a small typed-question builder) and a `decide` branch in
  `/api/test` that dispatches to the laya service and renders the typed answer + probabilities.
- `get_eligible_tasks`: a laya-backed model qualifies for `decision` only (its own class), never the LLM
  pools.

### 5. Provisioning & profiles (`lib/models.py`, `install.sh`, `app/menubar.py`)

- Add `decision` → `convaiinnovations/laya-multilingual` to **all four tier presets**. Bump
  `PROFILES_VERSION`.
- The pick is an HF repo id (`/`, no `:`) → the existing `profile_hf_models` / `hf download` autopull
  category fetches it. **Risk to verify (below):** confirm `laya.load()` reads the same HF cache that
  `hf download` populates, so provisioning composes with autopull rather than laya re-downloading.
- The English `laya` override is downloaded only when a profile/override names it.

### 6. Modes / remote access (`app/menubar.py`)

- Port `8003` is **never** added to the `tailscale serve` tuple (internal-only).
- No client/server serving logic: every machine runs its own laya; client-mode `local_decide` routes
  through the desktop's MCP per the "simple" decision above.

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

## Risks

1. **`laya.load()` vs `hf download` cache** — provisioning assumes they share the HF cache. Verify in the
   first implementation step; if laya uses its own download path, adjust the autopull category (a small
   dedicated "laya provisioning" step, like the MLX-subfolder one) rather than forcing it.
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
