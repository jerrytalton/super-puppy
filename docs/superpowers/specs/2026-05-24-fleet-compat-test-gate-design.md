# Fleet cross-version compatibility test gate

- **Date:** 2026-05-24
- **Status:** Approved (design); ready for implementation plan
- **Scope:** Track #1 of the fleet-update rework. Tracks #2 (runtime compat handshake + honest fallback) and #3 (supervisor split) are separate specs and explicitly out of scope here.

## Problem

Super Puppy runs as a fleet: one server plus several clients, all the same app, talking to each other over Tailscale (MCP on `:8100`, profile-server proxy on `:8101`, and the third-party Ollama/MLX APIs). Every machine independently auto-updates to the latest signed tag within ~2 minutes, with no coordination and no compatibility contract.

The pain (named by the owner): **mixed versions break.** A release that changes the cross-machine wire — exactly like the `421 Invalid Host header` incident — breaks client↔server communication until the whole fleet converges. Because machines auto-pull tags within ~2 minutes, **a bad tag is live almost instantly**, so any safety check must run *before the tag is pushed*. Post-push CI is too late.

There is no automated check today that a new release still interoperates with the version already deployed on the fleet.

## Goal

A **pre-tag gate** that refuses to ship a release which breaks interop with the previously released version, in either direction of skew:

- **Direction A — new server ↔ old client:** `server@HEAD` must satisfy the requests `client@prev-tag` makes. (This is the direction the 421 broke: a new server stopped honoring a request older clients send.)
- **Direction B — old server ↔ new client:** `server@prev-tag` must satisfy the requests `client@HEAD` makes.

Writing the gate also produces, as a side effect, an **executable definition of the cross-machine contract** — the thing that's currently implicit.

## Non-goals

- **Ollama (`:11434`) and MLX (`:8000`) APIs.** Third-party, not versioned by SP; SP only points at them. Out of scope.
- **Runtime degradation / handshake / fallback** (track #2). The gate *prevents* breaking tags; it does not make a broken pair degrade gracefully at runtime.
- **The supervisor/self-update mechanism** (track #3). Unchanged here. Note: track #3 is the riskiest change in the system (a broken updater can't update itself out of being broken), and this gate is the safety net that later lets #3 be done with proof the fleet still interoperates.
- **Coordinated/lockstep rollout.** The owner's pain is compatibility, not control; skew is tolerated, not eliminated.
- **Eliminating restart-induced session drops** (continuity). Not the stated pain.

## The contract surface (what's under test)

The SP-specific cross-machine wire, kept deliberately small:

1. **`/mcp` transport (`:8100`)** — under bearer auth, with the `Host` header set to the Tailscale FQDN (the exact 421 path):
   - `initialize` returns `200` + an `mcp-session-id`.
   - `tools/list` returns the tool set (names + input schemas). Tool definitions are static (decorator-defined), so this works without Ollama/MLX.
   - one `tools/call` to an inference-free tool (e.g. `local_models_status`) returns a well-formed JSON-RPC result (content may be empty when no backends are up — that's fine).
   - auth contract: missing/blank token → `403`; valid token + allowlisted FQDN Host → not `421`.
2. **`/api/mcp-models` (`:8100`)** — the menubar's reachability probe: `200`, JSON with a `models` list (possibly empty). Auth-exempt and Host-check-exempt, by contract.
3. **profile-server proxy (`:8101`)** — the `X-SP-Proxy-Hops` chain: a proxied request increments the hop header and is not refused. Secondary assertion; smaller blast radius than `/mcp`.

The contract is **inference-free**: every assertion is about status codes, auth/host behavior, and JSON *shape*, never model output. This keeps the gate runnable in CI with no Ollama/MLX present.

## Harness design (live worktree)

Two real versions, run sequentially to avoid port conflicts.

### Components

- **`tests/fleet/contract_probe.py`** — a dependency-light, committed script (stdlib `urllib`/`json` only; no rumps/pyobjc/Flask import). It is the *executable consumer contract*: given `--base`, `--token`, `--fqdn`, it issues the canonical client requests above and exits `0` (all green) or non-zero with a diff of what failed. Each release carries its own copy, so a tag's probe encodes what *that version's* client expects.
- **`tests/fleet/run_compat.py`** (or a function invoked by `bin/release.sh`) — the orchestrator.

### Orchestration

1. Determine `prev` = most recent existing tag (`git tag --sort=-v:refname | head -1`), and `head` = the commit about to be tagged.
2. `git worktree add <tmp> <prev>` to materialize the previous release.
3. **Direction B (always available):** start `server@prev` — `uv run <tmp>/mcp/local-models-server.py` (its own pinned deps) with a representative env (`MCP_AUTH_TOKEN=<test>`, `MCP_HOST=127.0.0.1`, `MCP_PORT=<test port>`, `MCP_ALLOWED_HOSTS=<test fqdn>:*`). Wait for readiness (poll `/api/mcp-models`). Run **`head`'s** `contract_probe.py` against it. Tear down.
4. **Direction A (when `prev` has the probe):** start `server@HEAD` the same way; run **`prev`'s** `contract_probe.py` (from `<tmp>/tests/fleet/`) against it. Tear down. If `prev` predates the probe (true for the first gated release against v1.0.21), **skip with a logged warning** — Direction A activates from the second gated release onward.
5. Repeat for the profile-server proxy contract (start the relevant servers on test ports, assert the hop behavior).
6. `git worktree remove <tmp>`; ensure teardown runs even on failure.
7. Exit non-zero if any probe failed; print the failing assertions and a before/after of the changed response.

### Representative launch

Each version's server is launched the way the menubar *intends* to launch it (correct token + FQDN allowlist), so the gate tests **wire-contract regressions assuming a correct launch**. Runtime/lifecycle bugs (e.g. adopting a stale process without `MCP_ALLOWED_HOSTS`) are out of scope here — they're covered by the existing `tests/test_mcp_lifecycle.py` unit tests.

## Gate placement: `bin/release.sh`

Releasing goes *through* a script, making the gate unskippable and retiring the manual tag/sign/verify dance done by hand previously. Steps:

1. Preconditions: clean tree, on `main`, in sync with `origin/main`.
2. Run the standard suite (`pytest tests/` minus live-only markers).
3. Run the fleet compat gate (the harness above). **Abort on failure.**
4. Create the signed tag (`git tag -s vX.Y.Z`), then **verify it against the repo's real `config/git/allowed_signers`** (`git -c gpg.ssh.allowedSignersFile=... tag -v`) — the check currently done by hand. Abort if verification fails.
5. `git push origin main` and `git push origin vX.Y.Z`.

Version number: passed as an argument (`bin/release.sh v1.0.22`); the script validates it's greater than `prev`.

## File layout

```
tests/fleet/contract_probe.py   # executable consumer contract (stdlib only)
tests/fleet/run_compat.py       # worktree orchestrator (or a pytest module)
bin/release.sh                  # the gate + sign + push
docs/RELEASING.md               # how to cut a release (replaces tribal knowledge)
```

## Testing the harness itself

The orchestrator has real logic (worktree lifecycle, server readiness polling, teardown-on-failure). A small test asserts: teardown removes the worktree even when a probe fails; a deliberately incompatible stub server makes the gate exit non-zero. The probe's own assertions are the contract; they need no separate test.

## Risks & mitigations

- **First-release bootstrap gap:** Direction A can't run until the prior tag carries the probe. Mitigation: Direction B runs immediately; Direction A activates next release. Documented, not silent.
- **Flaky server startup in CI:** readiness is condition-polled (poll `/api/mcp-models`), not a fixed sleep; a startup timeout fails the gate loudly rather than hanging.
- **`uv` cold dep resolution for the prev worktree** can be slow on first run. Acceptable for a release-time gate; `uv`'s cache makes subsequent runs fast.
- **Contract drift between probe and real client:** the probe is hand-written and could omit a request the real client makes. Mitigation: derive the probe's request list directly from the menubar's client-mode code paths (`get_mcp_models`, the reachability probe, the proxy hop) and keep them in sync; track #2's runtime handshake later provides a second, runtime check.
- **Adjacent-version check only:** the gate compares HEAD against the single most-recent tag, not every version still alive on the fleet (we deliberately don't track fleet state). This is sufficient *because* the additive-only contract rule makes compatibility transitive: if every release is compatible with its immediate predecessor, an N-versions-behind machine is also compatible. The rule is load-bearing — a non-additive change that slips through breaks the transitivity assumption. Because the fleet auto-updates every ~2 min, machines rarely sit more than one version behind anyway.

## Future (separate specs)

- **#2 Runtime compat handshake + honest fallback:** server advertises a fleet-contract version/capabilities; client falls back to local on mismatch instead of routing into a break. The probe/contract defined here becomes the basis for the advertised contract.
- **#3 Supervisor split:** move the self-update mechanism out of the app; safe to attempt *because* this gate proves interop survives the change.
