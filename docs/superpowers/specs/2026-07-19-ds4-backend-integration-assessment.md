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

## Spike results (laptop M5 Max 128GB, 2026-07-20)

Ran the laptop-first spike. **All build/interface/run questions: PASS.**

| Question | Result |
|---|---|
| Builds on macOS? | ✅ `make` → `ds4-server` in ~30s with **Command Line Tools only** (shaders compile at runtime; no full Xcode). The earlier "needs Xcode" worry was wrong. |
| Weights obtainable? | ✅ 81 GB `q2-imatrix` from public `huggingface.co/antirez/deepseek-v4-gguf`, **no token**, resumable, ~60 MB/s. |
| Runs on the 128GB laptop? | ✅ Loaded, served in ~4s (OS file-cache warm). M5 Max, Metal 4. |
| OpenAI API + **tool-calling**? | ✅ **Proven** — a real `/v1/chat/completions` with `tools` returned a correct OpenAI `tool_call` (`get_weather{"city":"Paris"}`). This is the make-or-break integration feature. |
| Generation speed? | ✅ **~32 tok/s** full residency (300 tok, 18-tok prompt, 9.4s) — *faster* than the M3 Max's reported 26.7. Output accurate. |
| RAM vs speed | ⚠️ Measured tradeoff (below). Full residency is fast but hogs RAM; a **bounded** SSD-streaming cache coexists with SP's services at ~half speed. |
| Interface flags | `--port` (used :8002), `--metal`, `--ctx`, `--ssd-streaming` + `--ssd-streaming-cache-experts NGB`, disk KV cache (`--kv-disk-dir`). `/v1/models` returns `deepseek-v4-flash`, `supported_parameters` includes `tools`/`tool_choice`. |

**RAM/speed tradeoff (measured on the M5 Max, 300-tok gens):**

| Mode | Speed | Steady RAM free | Verdict |
|---|---|---|---|
| Full residency (~81 GB) | **~32 tok/s** | **14%** (~18 GB) | Fast, but hogs the laptop — risky alongside on-demand MLX/Ollama loads |
| `--ssd-streaming` (default auto cache) | ~15–16 tok/s | drops to ~19% & climbing | **No real RAM win** — the auto budget is 80% of total RAM; it just defers residency |
| `--ssd-streaming --ssd-streaming-cache-experts 24GB` | ~14–18 tok/s | **47%** (~60 GB) | **The coexistence sweet spot** — bounded RAM, ds4 lives beside SP's other services, at ~half speed |

Takeaway: on a shared 128 GB laptop, the usable config is **bounded SSD streaming** (explicit expert-cache cap), accepting ~half the throughput for RAM headroom. Full residency only makes sense on a box dedicated to ds4. The 512 GB box would run PRO fully resident with room to spare.

**Takeaway:** the *technical* integration is de-risked — build trivial, wire protocol is exactly SP's MLX pattern, tool-calling works, speed is fine. The **open question is now purely a product one** (below): is a *single-lane* V4 worth a second big-model backend when glm-5.2 already occupies that niche, and does the 512GB PRO variant justify it? The laptop can't answer that — needs the big box.

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

1. ~~Build + run + tool-calling spike on the laptop.~~ **DONE (2026-07-20) — all
   pass** (see Spike results). ds4 lives at `~/experiments/ds4`, 81 GB q2 model on
   disk, `ds4-server --metal --port 8002 -m ds4flash.gguf` serves it.
2. **Next, the deciding test (needs the big box):** on the 512GB M3 Ultra, run
   `pro-q2-imatrix` and compare a real long agentic run against glm-5.2 —
   quality, tok/s, cold-load. Also check the `glm5.2` branch state. Only if V4
   PRO clearly beats glm-5.2 is a second big-model backend worth it.
3. If yes: wire read-only discovery + a "deep" profile + `install.sh`/service
   integration (the low-cost plumbing above).

Do **not** merge into main until step 2 answers the product question. The
technical feasibility is settled; the value question is not.

## Corrections to earlier (this session)
- I said ds4 was "Grace-Blackwell/CUDA-oriented, useless to SP." **Wrong** — it's
  Metal-first and faster on Mac than the article's GB10.
- I implied ds4 could readily replace the glm-5.2 mlx-lm patch. **Overstated** —
  glm5.2 is an undocumented branch; unconfirmed.
