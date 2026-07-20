# ds4 (DwarfStar 4) as an SP backend — cost/benefit assessment

**Status:** exploration (branch `experiment/ds4-integration`) · **Date:** 2026-07-19

> Not a build. A feasibility/cost-benefit read before committing. Primary
> source: <https://github.com/antirez/ds4> (README fetched 2026-07-19). Note:
> DeepSeek V4 and ds4 both post-date the Jan-2026 knowledge cutoff — facts here
> are from the repo/README, to be re-verified before any work starts.

## Verdict

**Integration cost: LOW–MODERATE. Benefit: real but NICHE.** ds4 is worth a
time-boxed spike, not a committed integration yet. It's a *single-lane
frontier model*, not a parallel workhorse — which decides where (and whether)
it fits.

## What ds4 is (primary-source facts)

- antirez's (Redis creator) self-contained **pure-C** inference engine, **MIT**,
  hand-written **Metal** kernels (macOS is the *primary* target; CUDA/ROCm also).
  Not MLX, not Ollama, not GGML — its own engine. Runs on Apple Silicon directly.
- Purpose-built for **DeepSeek V4 Flash/PRO** (284B MoE, ~13B active, 1M context).
  "not our only target… the exact model may change."
- **Asymmetric 2-bit/8-bit quant**: only routed MoE experts → 2-bit
  (IQ2_XXS up/gate, Q2_K down); shared experts, projections, routing, attention
  stay high-precision.
- **`ds4-server`**: OpenAI-compatible `/v1/chat/completions`, `/v1/models`,
  `/v1/completions`; Anthropic `/v1/messages`; a Codex `/v1/responses`. Default
  `127.0.0.1:8000`. **Full tool-calling** (tools/tool_choice ↔ DeepSeek DSML),
  SSE streaming. Persistent daemon, disk KV cache.
- **Build**: `make` (Metal default), LLVM, minimal deps. Model download via
  `download_model.sh {q2-imatrix | q2-q4-imatrix | q4-imatrix | pro-q2-imatrix}`.
- **Concurrency (the caveat)**: *single live graph/session; concurrent requests
  serialize.* No request-level batching.
- Perf (reported): ~26.68 tok/s gen on a 128GB M3 Max; prefill degrades with
  context; slow enough that it's an **overnight/batch** tool, not interactive.

## Why it maps onto SP hardware

| SP tier | ds4 model |
|---|---|
| Laptop M5 Max (128GB) | `q2-imatrix` / `q2-q4-imatrix` (V4 Flash 2-bit) |
| Big box M3 Ultra (512GB) | `pro-q2-imatrix` (V4 PRO) or `q4-imatrix` (Flash 4-bit) |

## Integration surface & cost

The router already speaks OpenAI `/v1/chat/completions` to mlx-openai-server, so
ds4 is "another MLX-shaped backend on a different port." Touchpoints:

1. **Service mgmt** (`bin/start-local-models`, menu bar app) — spawn/monitor
   `ds4-server` on a **new port** (8000 is taken by MLX; use e.g. 8002).
   *Moderate* — mirror the existing mlx-openai-server lifecycle. **Biggest chunk.**
2. **Discovery** (`mcp/local-models-server.py::discover_models`) — query ds4
   `/v1/models`, tag `backend: "ds4"`. *Low* — parallels the MLX `/v1/models` path.
3. **Chat routing** (`chat()` + the hardcoded `("ollama","mlx")` eligibility
   tuples at ~587/591) — ds4 reuses the MLX OpenAI path pointed at DS4_URL; add
   `"ds4"` to the tuples. *Low.*
4. **Config** — a ds4 config + `DS4_URL`/`DS4_PORT` alongside `MLX_URL`. *Low.*
5. **Profiles / `lib/models.py`** — expose ds4 models to task→model mapping;
   likely a dedicated "deep/overnight" profile rather than the everyday tools. *Low.*
6. **`install.sh`** — clone+`make` ds4, run `download_model.sh` (large weights).
   *Moderate* — big download; 512GB PRO weights are not small.
7. **Scope**: ds4 is **text-only** (chat/code/reasoning/long_context). Vision/
   audio/image stay on MLX/Ollama. No change there.

Net: no new protocol work (the expensive part is avoided); the cost is
service-lifecycle + install/download plumbing, following the mlx-openai-server
template.

## Benefit

- A capable **frontier local model** (V4, 1M context, MIT) on hardware SP owns,
  at zero marginal token cost — the "long/overnight agentic run" use case that
  motivated the offload work.
- OpenAI API + **tool-calling** → works with agent loops out of the box.
- Asymmetric quant is baked in — no SP quant work.

## The caveat that shapes everything

**Serialized single-session** contradicts SP's "cheap *parallel* compute" value
prop (multiple `local_dispatch` calls, fleet clients). ds4 can host **one** long
agentic run at a time; it can't be the parallel workhorse. So it fits as a
*dedicated deep/overnight lane*, alongside — not replacing — the Ollama/MLX
parallel tools. glm-5.2 (already served) fills a similar niche today.

## Open questions (resolve in the spike, before committing)

1. **glm5.2 branch** — real/working or abandoned experiment? Only then is
   "retire `apply-mlx-glm52-patch.sh`" on the table. (README doesn't document it.)
2. **Real perf on the 512GB M3 Ultra** with PRO — tok/s, prefill, cold-load,
   KV-cache disk I/O — vs glm-5.2 today. Is V4 PRO actually *better* than what SP
   already runs, enough to justify a second big-model backend?
3. **Weight size / download** for PRO — disk budget on the big box.
4. **Stability** — the engine is weeks old and beta.

## Recommendation

Time-boxed spike, laptop first:
1. `git clone antirez/ds4 && make`; `download_model.sh q2-imatrix`; run
   `ds4-server` on :8002; `curl` a tool-calling `/v1/chat/completions` to confirm
   the shape SP expects.
2. If clean, wire a **read-only** discovery + a single "deep" profile pointing at
   ds4 on the laptop, and measure a real long agentic run vs glm-5.2.
3. Only then decide on big-box PRO + full install.sh/service integration.

Do **not** merge into main until the spike answers Q1–Q4. This branch holds the
assessment; the spike would extend it.

## Corrections to earlier (this session)
- I said ds4 was "Grace-Blackwell/CUDA-oriented, useless to SP." **Wrong** — it's
  Metal-first and faster on Mac than the article's GB10.
- I implied ds4 could readily replace the glm-5.2 mlx-lm patch. **Overstated** —
  glm5.2 is an undocumented branch; unconfirmed.
