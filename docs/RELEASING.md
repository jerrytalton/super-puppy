# Releasing Super Puppy

Releases ship to the fleet via signed git tags; every machine auto-updates to
the latest tag within ~2 minutes. **A bad tag is live almost immediately**, so
releases go through `bin/release.sh`, which gates on tests + cross-version
compatibility *before* the tag is pushed.

## Cut a release

```bash
bin/release.sh v1.0.22            # full release: gate, sign, verify, push
bin/release.sh v1.0.22 --dry-run  # everything except the push
```

The script will refuse unless: the tree is clean, you're on `main`, `main` is
in sync with `origin/main`, the test suite passes, and the fleet compat gate
passes. The suite excludes live-service e2e tests (`-m "not slow and not e2e"`)
so `bin/release.sh` runs from any clean checkout; verifying a live stack is
healthy is a separate concern.

## The fleet compat gate

`tests/fleet/run_compat.py` worktrees the previous tag and checks both
directions of version skew using `tests/fleet/contract_probe.py` — the
executable definition of the cross-machine wire contract (`/mcp` auth +
Tailscale-Host, `/api/mcp-models`). The `:8101` proxy-hop loop guard is
defined in `contract_probe.py` (opt-in via `--profile-base`) but is not yet
wired into the orchestrator; exercising it is a planned follow-up.

**Compatibility rule:** the wire contract is **additive-only**. Adding fields,
endpoints, or tools is fine; changing or removing what an existing peer relies
on is a breaking change and must not ship without a deliberate contract-version
bump (this needs the planned runtime-handshake work — track #2 — which is not yet built). The gate checks only the
adjacent prior version; transitivity across older fleet members depends on this
rule holding.

## Signing

Tags are SSH-signed; the trusted key lives in `config/git/allowed_signers`.
A new key must ride a tag signed by the *outgoing* key (the running fleet
verifies the next tag against the key it already trusts).
