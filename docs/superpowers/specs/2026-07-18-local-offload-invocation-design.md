# Local-Model Offload Invocation — Design

**Status:** proposed · **Date:** 2026-07-18

## Problem

SP *exposes* local-model abilities (vision, bulk generate, summarize, embed,
similarity search, async dispatch/collect, review/candidates). It does **not**
get *invoked* during Claude Code sessions: over months of daily use, real
`mcp`/`playground` activity is ~0 — the only recent row is a test call. Yet
there is large idle GPU capacity (especially the 512GB M3 Ultra, usually up)
during exactly those sessions.

The gap is **invocation**, not capability. This spec designs the correct,
evidence-backed way to make a Claude Code session route the *right* work to the
local cluster — saving frontier tokens and/or wall-clock, without degrading
results.

## What the research settled

Two independent research passes (Claude Code invocation mechanisms; prior art
in coding agents) converged.

### The single highest-value, lowest-risk pattern: reconnaissance/context-offload
- Codebase recon (file discovery, grep, "summarize this dir/log before I read
  it") is *what local models are genuinely good at in an agent loop* — it's
  mostly file-reading and string-matching. (OpenCode, Simon Willison, Claude
  Code's own Haiku-backed `Explore` subagent.)
- Reported **40–70% main-thread token reduction** on research-heavy tasks,
  because the subagent's raw Grep/Read output stays in *its* context and only a
  compressed digest returns.
- The biggest frontier-token cost in agentic coding is **ingestion** (pulling
  files/logs/search results into context). Offloading the *reading* is where
  the savings are real; offloading the *writing* is where they're illusory.

### The structural reason SP is invisible today: Tool Search
- Claude Code runs **Tool Search** by default: it withholds MCP tool schemas
  from context and loads them on demand via similarity search. Verified live in
  this session — the `local-models` tools arrived *deferred*; the model must
  already decide it needs a local tool and go search. This suppresses exactly
  the opportunistic use we want. It is tunable (descriptions / selective
  eager-load / `ENABLE_TOOL_SEARCH`).

### What the harness can and cannot enforce
- **Enforced:** subagent tool-allowlist + model pinning; `PreToolUse`/
  `UserPromptSubmit` hooks can *inject text that reaches the model* and *block*
  a tool; `permissions.allow` removes friction.
- **Not enforced:** whether the model *delegates* / *picks* a tool. Hooks cannot
  call an MCP tool or force a specific one. CLAUDE.md, skill auto-invocation,
  and tool descriptions are advisory.
- **Conclusion:** no config forces tool preference without removing choice. The
  achievable goal is: make the local option **visible, frictionless, and the
  default for the categories where it wins.**

### Hard "don'ts" (every deployment that tried these regressed)
- **Never route the primary edit/coding turn to a local/open model** — malformed
  diffs, hallucinated features, corrupted files on retry (CCR, Aider editors,
  Cline, icefire555).
- **Architect/editor split** helps only when *planning* dominates and is fragile
  on diff-format tasks — skip.
- **Cheap-model digests degrade quality invisibly** — a weak local model
  summarizing a codebase silently corrupts the main agent's understanding. Match
  model size to recon complexity; use a strong local coder, not a tiny model.

## Design

Phased, progressive. Ship Phase 1 end-to-end, prove the savings, then expand.

### Phase 1 — Make the local option visible & frictionless (lowest effort)
1. **Sharpen the offload tools' descriptions** for Tool-Search recall — so
   `local_summarize`, `local_similarity_search`, `local_dispatch`/`collect` are
   discovered on recon-shaped needs (keywords: "read/summarize large file",
   "search codebase by concept", "offload/parallel"). Names/descriptions are the
   only Tool-Search lever.
2. **`permissions.allow: ["mcp__local-models__*"]`** in the user settings so
   local calls never prompt — remove friction.

### Phase 2 — The reconnaissance subagent (the core, best-evidenced pattern)
A project/user subagent `recon-local` that:
- Is **tool-restricted** to `Read, Glob, Grep, Bash, mcp__local-models__*`
  (enforced — it cannot edit or hit the network), runs on **Haiku** (cheap
  driver), and its system prompt routes the heavy reading/summarization to the
  **local cluster** (`local_summarize`, `local_similarity_search`,
  `local_dispatch`) and returns a compressed digest to the main thread.
- Is **capability-aware**: first calls `local_models_status`; prefers a strong
  local coder on the 512GB box (e.g. `qwen3-coder-next` / `glm-5.2`) for the
  digest; if the big box is down, uses the laptop's best model or returns a
  "recon better done inline" signal rather than a weak digest.
- Description says "use proactively for codebase reconnaissance / understanding
  large files or dirs before reading them." (Delegation is advisory — this makes
  recon the natural path, not a guarantee.)

The main thread delegates recon to it; surgical edits stay on the frontier model
with exact bytes.

### Phase 3 — Structural nudge on recon-shaped prompts (optional)
A `UserPromptSubmit` hook that detects recon-shaped requests ("how does X work",
"where is Y handled", "understand/summarize this codebase/dir/log") and injects
a directive to delegate to `recon-local` / `local_summarize` before ingesting
raw files. Fires regardless of the model's (mis)calibration; still a suggestion,
but harness-injected so it reliably *reaches* the model.

### Phase 4 — Background-slot proxy (orthogonal, separate track, opt-in)
Front Claude Code with a proxy (Claude Code Router or LiteLLM) that routes **only
the background/housekeeping slot** (`ANTHROPIC_DEFAULT_HAIKU_MODEL` — titles,
compaction/diff summaries) to a local model. Keep `default`/`sonnet`/`opus` on
the frontier model. Near-zero quality risk, sheds invisible spend. Bigger infra
change (alters how Claude Code launches) → separate decision; **not** in the
first build. Gotchas: local ctx ≥32K; `ANTHROPIC_SMALL_FAST_MODEL` is deprecated
(silently fails); verify MCP tool-call wire format matches the client parser.

## Explicitly out of scope
- Routing the primary coding/edit turn to a local model.
- Architect/editor diff-split.
- Any "weak model arbitrates the frontier model's reasoning" second-opinion loop
  (see the separate delegation research: single weak verifier has accept-bias;
  disagreement is a *flag*, not a verdict).

## Measurement (savings must be visible)
- **Usage:** real `mcp` rows land in the activity DB (`local_summarize` /
  `local_dispatch`) with `machine` tag — proves invocation, closes the original
  "SP never gets used" complaint.
- **Token savings:** the recon subagent logs `raw_tokens_read` (in its own
  context) vs `digest_tokens_returned` — the delta is frontier tokens avoided on
  the main thread. Surface a rough per-session tally.
- **Guardrail metric:** track digest-then-correction events (main thread had to
  re-read raw after a digest) — if high, the local model is too weak / the task
  was mis-routed.

## Open decisions (for the human)
1. **How far in the first build:** Phase 1+2 only (in-repo + user config,
   reversible, highest value) vs. also Phase 3 hook vs. also Phase 4 proxy.
   Recommendation: **1+2 first**, prove savings, then decide on 3/4.
2. **Recon digest model:** pin to `qwen3-coder-next` (80B/4B-active, 256K ctx)
   when the 512GB box is up? Fallback behavior when it's off (laptop model vs
   "do it inline")?
3. **Scope of the subagent:** user-level (`~/.claude/agents/`, all projects incl.
   Blacklake/dddg) vs project-level (super-puppy only) for the first cut.
