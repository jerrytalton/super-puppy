# ds4 backend port — total surface-area audit

**Status:** audit (pre-implementation) · **Date:** 2026-07-22 · **Precedes:** implementation plan
**Context:** Big-box comparison (2026-07-22, `~/experiments/ds4-compare-20260722-161803/`) decided the ship:
glm-5.2 **Q2K-routed GGUF on ds4** (244GiB, 15.5 tok/s) replaces glm-5.2 **4bit on mlx-openai-server**
(390GB resident, 18.3 tok/s) on the **512GB tier only**. Quality indistinguishable; the win is ~146GB of
memory headroom — the co-residency OOM class that hard-reset the box on 2026-07-22 goes away, and the
pinned mlx-lm patch (`bin/apply-mlx-glm52-patch.sh`) retires. MLX keeps whisper/vision/TTS/ui-venus.
Builds on `docs/superpowers/specs/2026-07-19-ds4-backend-integration-assessment.md` (laptop spike: build,
OpenAI API, tool-calling on DeepSeek V4 all pass).

## The load-bearing architectural fact

`glm-5.2` is not a service today — it is a `served_model_name` inside the shared mlx-openai-server
(`config/mlx-server/config.yaml:57-63`). Every consumer discovers it from `MLX_URL/v1/models` plus that
yaml, and every dispatcher is a **binary branch**: `backend == "ollama"` → `:11434`, else → MLX `:8000`.
Moving glm-5.2 to `ds4-server :8002` therefore threads a third backend value through **discovery,
dispatch, status, memory math, tests, and docs** — the service scripts alone are not sufficient.

ds4 is OpenAI-compatible, so each new dispatch branch is structurally a copy of the MLX branch pointed at
`DS4_URL`. The cost is breadth, not novelty.

## Design decisions to make first

1. **Discovery + metadata seam.** ds4's `/v1/models` returns one model with *no* params/context/vision
   metadata. Discovery must hardcode `total/active params`, `context` (131072), `vision=False`, and size
   (~244GiB — the GGUF is not an HF snapshot; none of the existing sizing paths can see it). Without this,
   `TASK_FILTERS` min_active_b/min_ctx gates silently drop glm-5.2 from every task list.
2. **Port exposure.** Recommend `:8002` **internal-only** (not in the `tailscale serve` tuple,
   `app/menubar.py:1861-1883`): client-mode chat is already brokered by the desktop's 8100 (MCP) and 8101
   (profile-server `_proxy_to_desktop`). Only the legacy direct-call parity path would need `8002` served.
3. **Residency model.** ds4 loads at startup (~70s) and never unloads: `on_demand`/warm/keep-alive
   concepts don't apply. Memory math must treat 244GiB as fixed overhead
   (`app/profile-server.py:1935-1995`), and the menubar's 60s stuck-loading watchdog
   (`app/menubar.py:2253-2266`) must not kill the legitimate ~70s load.
4. **Build ownership + auto-update.** Clone (pinned to `bd89932`, glm5.2 branch) + `make` replaces the
   glm52-patch block (`install.sh:642-650`), tier-gated to ≥512GB. If mirrored into `bin/post-update.sh`,
   a ds4 build failure **rolls back the whole update** (`app/menubar.py:3170-3189`) — must be
   failure-tolerant. cwd quirk: `ds4-server` resolves `metal/flash_attn.metal` relative to cwd, so launch
   must `cd` into the checkout (the MLX launch deliberately `cd "$HOME"` — don't copy it).
5. **Weights location + download.** ds4's own `download_model.sh` on the glm5.2 branch has the exact
   target (`glm-antirez-q2` → `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf`). Decide: reuse it vs install.sh's HF loop.
   Either way install.sh's served-name resolution (`install.sh:808-834`) must stop resolving glm-5.2 to
   `mlx-community/GLM-5.2-4bit` (418GB we no longer need). Add a disk-space precheck (none exists; 244GiB).
6. **Migration for existing installs.** `post-update.sh`'s MLX config merge is append-only — it will not
   remove the user's existing glm-5.2 MLX entry. Without an explicit migration, an updated 512GB box
   double-serves glm-5.2 (MLX claims it first in discovery order) and keeps 418GB of dead weights. Needs a
   one-shot migration + `PROFILES_VERSION` bump (`lib/models.py:429`).

## Ship gates (verify before merging)

- **Tool-calling on ds4+GLM-5.2: ✅ VERIFIED (2026-07-22).** Full round-trip on the big box: `tools`
  request → correct OpenAI `tool_call` (`get_weather{"city":"Paris"}`, `finish_reason: "tool_calls"`,
  reasoning in `reasoning_content`) → tool result fed back → correct synthesized final answer,
  `finish_reason: "stop"`. Gate cleared.
- **`enable_thinking` semantics: ⚠️ VERIFIED BROKEN as a toggle (2026-07-22).**
  `chat_template_kwargs.enable_thinking: false` returns HTTP 200 (no error) but does not stop thinking —
  the model's reasoning moves from `reasoning_content` into `content` (no think-block markers for the
  parser to strip). `chat_ds4()` must NOT forward the think toggle; glm-5.2 on ds4 always thinks, and
  callers should read `reasoning_content` + `content` as the MLX path already does. In the comparison,
  glm-5.2 burned its whole 512-token budget thinking on both backends.
- **Concurrency.** ds4 serializes requests (single live session). glm-5.2 is the 512gb tier's
  general/reasoning/long_context workhorse; parallel `local_dispatch` fan-outs will queue. Accepted
  tradeoff or needs queue-depth surfacing in status?
- **Stability.** Engine is weeks old. Pin the commit; keep the MLX path documented as fallback for one
  release.
- **Release gating.** The live glm-5.2 path is only exercised by `test_tools_smoke_everyday.py` (`slow`,
  excluded from `bin/release.sh`). Add a `correctness`-marked ds4 case if the ship should be release-gated.
- **⚠️ ds4 JSON encoder can emit unescaped control characters** (observed 2026-07-22 in a long
  `reasoning_content`). Python's default strict JSON parser (`requests` `.json()`, `json.loads`) raises
  on these. `chat_ds4()` must parse with `json.loads(..., strict=False)` (or sanitize) or long thinking
  responses will intermittently crash dispatch. Consider reporting upstream.

## SSD streaming on the big box: measured, not shipping (2026-07-22)

`--ssd-streaming --ssd-streaming-cache-experts 48GB` with the GLM Q2K quant on the 512GB M3 Ultra:
**RSS 244GiB → 36GB**, but throughput **15.5 → 3.5–4.9 tok/s** (cold → third 512-tok run; the GLM routed
working set thrashes a 48GB expert budget far worse than the laptop spike's DeepSeek Flash ~2× penalty).
Server startup is near-instant (demand-fill). Verdict for this ship: **full residency** — 244GiB leaves
~200GB headroom on the 512 box, which already achieves the memory goal, and 4–5 tok/s is too slow for the
tier's primary workhorse. Follow-up worth its own spike: 36GB RSS means a streamed glm-5.2 Q2K plausibly
fits the **128GB laptop** as an offline frontier option (~4–5 tok/s, laptop SSD TBD) — out of scope here.

## Touchpoint inventory

### 1. Config & ports
| Where | What |
|---|---|
| `config/local-models/network.conf:11-12` + `lib/models.py:29-41` | add `DS4_PORT=8002` to template, `_NETWORK_DEFAULTS`, `_NUMERIC_KEYS` |
| `config/mlx-server/config.yaml:57-63` | **remove** glm-5.2 entry (MLX keeps whisper/vision/tts/ui-venus) |
| `lib/models.py:505-529` (512gb preset) | task→name map unchanged (`glm-5.2` stays); bump `PROFILES_VERSION` to force refresh |
| `lib/models.py:113-116` | optional: GLM-5.2 entry in `KNOWN_ACTIVE_PARAMS` for correct active-param display |

### 2. Service lifecycle
| Where | What |
|---|---|
| `bin/start-local-models:184-205, 92-108, 110-145` | tier-gated ds4 launch (cwd-pinned, `:8002` readiness probe with >70s deadline), stop (`pkill ds4-server`), status line |
| `install.sh:642-650` | replace glm52-patch block with ds4 clone+make (pinned commit); add to `MISSING_RUNTIMES` (`:667-691`) |
| `install.sh:808-834, 868-925` | glm-5.2 download resolution → Q2K GGUF; disk-space guard |
| `bin/post-update.sh` | tier-gated, failure-tolerant ds4 rebuild; glm-5.2 MLX-entry migration |
| `bin/local-models-mcp-detect:11-12,34-60,68` | `DS4_URL` export (+ FQDN rewrite only if 8002 is ever served) |
| `install.sh:46-132` **and** `uninstall.sh:25-121` | both uninstallers: kill ds4-server, remove checkout/config, mention GGUF in cache note |
| `bin/apply-mlx-glm52-patch.sh` | retire (sole invocation is install.sh:648); update doc refs |

### 3. Dispatch & discovery (the functional core)
| Where | What |
|---|---|
| `mcp/local-models-server.py:60-63` | `DS4_URL` constant |
| `mcp/local-models-server.py:356-492` | ds4 discovery block with hardcoded metadata |
| `mcp/local-models-server.py:656-663` | `chat()` else-branch misroutes ds4→MLX today: add `chat_ds4()` + `elif` |
| `mcp/local-models-server.py:583-591` | `("ollama","mlx")` fallback/eligibility tuples → shared `_LLM_BACKENDS` incl. `"ds4"` |
| `mcp/local-models-server.py:680-684, 1876-1882, 1885-1917, 1920-1939` | status counts, `/gpu` probe block, activity backend loop |
| `app/profile-server.py:81-82, 1055-1071, 1132-1169, 1203-1216` | `DS4_URL`; `_fetch_ds4_models()` (`is_loaded=True`, `on_demand=False`, hardcoded size); aggregate |
| `app/profile-server.py:1228` | `_LLM_BACKENDS` must include `"ds4"` or glm-5.2 gets zero eligible tasks → invisible in every dropdown |
| `app/profile-server.py:2067-2071, 2133-2280, 2642` | `_chat_url` / `_chat` / `_chat_stream` ds4 branches |
| `app/profile-server.py:1838-1895, 1935-1995` | warm loop skips ds4 (fine); memory bar needs 244GiB fixed residency |
| `app/menubar.py:234-235, 2583-2586` | `DS4_LOCAL`; inject `DS4_URL` into profile-server env |
| `app/menubar.py:726-739` | warm-ping classifier (`":" in name → ollama else mlx`) must route glm-5.2 → ds4 |
| `app/menubar.py:2247-2286, 2074-2149` | `ds4_ok`/`ds4_loading`, auto-restart, `_restart_ds4()` with long poll; watchdog threshold |
| `app/menubar.py:1379-1439, 2418-2519` | menu item + status dot ("1 model" — pluralization assumes dynamic lists) |
| `app/menubar.py:1894-1927` | Copy Diagnostics ds4 line |
| `app/menubar.py:3120-3132` | pre-update health snapshot: add ds4 field if health-gated |

### 4. UI
| Where | What |
|---|---|
| `app/activity.html:131-140, 379, 385/445` | `.backend-ds4` CSS (light+dark) + color case (binary ollama-vs-accent today) |
| `app/diagnostics.html:127-136` + `api_diagnostics` (`profile-server:3143-3213`) | ds4 service row + probe |
| `app/profiles.html:563, 892-894, 1476-1488` | exclude ds4 model from pull/missing-model prompts (pre-provisioned) |
| `app/tools.html` | no direct change; depends on eligible_tasks (item 3) |
| `app/audit.html`, `super-puppy.c`, Info.plist | no change |

### 5. Tests
**Hard breakages the moment glm-5.2 leaves the MLX config:**
- `tests/test_core.py:623-642` `test_warm_models_bare_names_are_mlx_served` — generalize to ds4-served names
- `tests/test_profile_server.py:510-526, 528-541, 543-570` — glm-5.2 sizing/pull trio assumes MLX config + HF repo

**New/updated coverage:**
- `test_core.py:614-620` warm-ping classifier ds4 case; `:317-348` network.conf template/repair with `DS4_PORT`
- `test_profile_server.py:110-115` `_chat_url("ds4") == ":8002/v1/chat/completions"`
- `test_mcp_server.py:95-96, 132-133, 178-180` + `test_error_handling.py:101-102, 113-155` — `"ds4"` GPU key, resolution, error-detail
- `test_e2e.py:66` add 8002 to port skip-list; `:234-239, 312-320` backend keys/values
- `test_deployment.py:85-92, 748-787, 890-907` — ds4 health snapshot/regression if health-gated; post-update config-ref guard
- `tests/_smoke_helpers.py:49-50, 99-104, 184-198` — `DS4_URL`, `:8002` reachability (else ds4 outages hide as skips), ds4 error text in `SKIP_SUBSTRINGS`
- `test_tools_smoke_everyday.py` — exercises ds4 automatically via `DEFAULT_PROFILES`; stays `slow`
- No-ops verified: `test_playground_coverage.py`, `test_tools_smoke_laptop.py`
- Fleet compat gate (`bin/release.sh:56-57`, `tests/fleet/`): verify no port/backend assumptions; remember a shipped tag goes live fleet-wide within ~2 min

### 6. Docs
- `CLAUDE.md`: 18, 23 (glm52-patch design note — rewrite), 30-31, 37, 49-50, 91, 112, 121, 128, 143-144
- `README.md`: 55-86, 138, 149, 167-185, 226-243 (tier table + patch footnote), 282, 285
- `docs/architecture.md`: 13-14, 24, 28, 42-47, 75, 125
- `docs/troubleshooting.md`: 151-205 — the glm-5.2 mlx-patch/eviction section is obsoleted wholesale; replace with ds4 build/run/troubleshooting
- `docs/tailscale-setup.md:37-44` — only if 8002 is served (recommend not)
- `docs/usage-telemetry.md:9` — note `"ds4"` as a backend value (no schema change; `activity.db` backend column is free text)
- No change: `docs/model-prompting.md`, `docs/RELEASING.md` (additive wire-contract note optional)

### Backend-agnostic (verified no change)
`lib/activity.py` (free-text backend column), GPU-tracking internals (`defaultdict`), pure picker helpers
(`resolve_pref_candidate`, `pick_model_from_prefs`), `lib/audit.py`, `lib/mlx_vlm.py`, modes
(server/client/offline), launchd plists (servers are children of the menubar chain, no new plist needed).

## Assumptions ds4 violates (design around, don't fight)

1. Exactly two chat backends (binary else-branches everywhere).
2. Models are discovered dynamically with rich metadata (ds4: one pinned model, none).
3. Every model has an Ollama tag, an MLX served-name, or an HF snapshot (ds4's GGUF: none of the three).
4. Residency is managed (warm/keep-alive/idle-unload) — ds4 is always-resident.
5. Missing models are pull-on-demand from the UI — ds4's model is pre-provisioned by install.
6. "Not ollama/mlx" ⟹ HF-subprocess backend (`HF_TASK_BACKENDS`) — latent trap, no harmful instance found.

## Effort estimate

~15 code files (mostly mechanical "third branch" additions following the MLX template), 2 hard test
breakages + ~8 test files of new coverage, 6 docs. The novel design work is items 1, 4, 6 in Design
decisions. Realistic: 2–3 focused days including the migration path and ship-gate verification.
