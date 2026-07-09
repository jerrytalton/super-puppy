#!/bin/bash
# Patch the mlx-openai-server tool env so GLM-5.2 (glm_moe_dsa) can load.
#
# Why: GLM-5.2 puts DSA indexer weights on 21 of 78 layers and shares them
# across the rest (config.indexer_types). Released mlx-lm (<= 0.31.3) builds
# an indexer on every layer, so strict load fails with "Missing 285
# parameters". The fix is upstream PR ml-explore/mlx-lm#1463, unmerged as of
# 2026-07-08. mlx-lm 0.31.3 itself needs mlx >= 0.31.2, which still carries
# the thread-local-stream hang (mlx-lm #1256) — so instead of upgrading, this
# ports the PR's two model files onto the installed mlx-lm 0.31.1.
#
# Also fixes two mlx-openai-server issues that break a big on-demand model
# in practice (both still present in 1.8.1):
#   1. The hardcoded 300s handler-readiness timeout — a 390GB checkpoint
#      takes 80-140s to load on an M3 Ultra and can exceed 300s under load.
#      Raised to 1800s.
#   2. The warm-request refcount bypass — requests to an already-loaded
#      on-demand model skip ensure_on_demand_loaded, so the idle timer never
#      re-arms and the model unloads 300s after load REGARDLESS of traffic
#      (the menu bar's 240s keep-warm pings were useless; the 390GB model
#      thrash-reloaded every ~5 minutes). Routed on-demand models through
#      the refcounting path.
#
# Usage: bin/apply-mlx-glm52-patch.sh
#   Idempotent — safe to re-run (e.g. after `uv tool upgrade mlx-openai-server`
#   wipes the patch). Exits 0 without touching anything if the installed
#   mlx-lm already supports indexer_types. Only needed on machines serving
#   glm-5.2 (the 512gb tier); install.sh runs it automatically there.
#
# Remove this patch once PR #1463 is merged AND a released mlx-lm/mlx pair
# without the #1256 stream hang reaches the tool env.

set -euo pipefail

PR_SHA="f2a8ed52bc209e921cc1d85b304b55befcf2788e"
EXPECTED_MLX_LM_VERSION="0.31.1"
TOOL_ENV="$HOME/.local/share/uv/tools/mlx-openai-server"

log() { echo "[glm52-patch] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

SITE=$(ls -d "$TOOL_ENV"/lib/python*/site-packages 2>/dev/null | head -1)
[ -n "$SITE" ] || die "mlx-openai-server tool env not found at $TOOL_ENV — install it first"
MODELS_DIR="$SITE/mlx_lm/models"
PYTHON="$TOOL_ENV/bin/python"

if grep -q "indexer_types" "$MODELS_DIR/glm_moe_dsa.py" 2>/dev/null; then
    log "installed mlx-lm already supports indexer_types — nothing to do"
    exit 0
fi

INSTALLED_VERSION=$("$PYTHON" -c "import mlx_lm; print(mlx_lm.__version__)")
[ "$INSTALLED_VERSION" = "$EXPECTED_MLX_LM_VERSION" ] \
    || die "installed mlx-lm is $INSTALLED_VERSION, patch is ported against $EXPECTED_MLX_LM_VERSION — re-validate before applying"

log "fetching mlx-lm PR #1463 at $PR_SHA..."
CLONE_DIR=$(mktemp -d)
trap 'rm -rf "$CLONE_DIR"' EXIT
git clone --quiet --depth 1 https://github.com/ml-explore/mlx-lm.git "$CLONE_DIR"
git -C "$CLONE_DIR" fetch --quiet --depth 1 origin refs/pull/1463/head
FETCHED_SHA=$(git -C "$CLONE_DIR" rev-parse FETCH_HEAD)
[ "$FETCHED_SHA" = "$PR_SHA" ] \
    || die "PR head moved ($FETCHED_SHA != pinned $PR_SHA) — re-review the PR before updating the pin"
git -C "$CLONE_DIR" checkout --quiet "$PR_SHA"

log "backing up and installing model files..."
for f in glm_moe_dsa.py deepseek_v32.py; do
    cp "$MODELS_DIR/$f" "$MODELS_DIR/$f.orig-$INSTALLED_VERSION"
    cp "$CLONE_DIR/mlx_lm/models/$f" "$MODELS_DIR/$f"
done

HANDLER="$SITE/app/core/handler_process.py"
if grep -q "timeout=300)" "$HANDLER" && grep -q "did not become ready within 300 s" "$HANDLER"; then
    log "raising handler readiness timeout 300s -> 1800s..."
    cp "$HANDLER" "$HANDLER.orig"
    sed -i '' \
        -e 's/response = await asyncio.wait_for(ready_queue.get(), timeout=300)/response = await asyncio.wait_for(ready_queue.get(), timeout=1800)/' \
        -e 's/did not become ready within 300 s/did not become ready within 1800 s/' \
        "$HANDLER"
else
    log "handler readiness timeout already patched or upstream changed — skipping"
fi

ENDPOINTS="$SITE/app/api/endpoints.py"
if grep -q "the get_handler fast path bypassed refcounting" "$ENDPOINTS"; then
    log "warm-request refcount fix already applied — skipping"
else
    log "fixing warm-request refcount bypass in endpoints.py..."
    cp "$ENDPOINTS" "$ENDPOINTS.orig"
    "$PYTHON" - "$ENDPOINTS" <<'PYEOF'
import sys
p = sys.argv[1]
src = open(p).read()
old = '''    registry = getattr(raw_request.app.state, "registry", None)
    if registry is not None and model_id is not None:
        # Try the normal (already-loaded) path first
        try:
            return registry.get_handler(model_id)
        except KeyError:
            pass

        # Check if this is an on-demand model that needs loading
        if registry.is_on_demand(model_id):'''
new = '''    registry = getattr(raw_request.app.state, "registry", None)
    if registry is not None and model_id is not None:
        # On-demand models always go through ensure_on_demand_loaded — even
        # when already resident — so the ref count and idle timer see every
        # request, not just the one that triggered the load. (Upstream bug:
        # the get_handler fast path bypassed refcounting, so warm traffic
        # never re-armed the idle timer and the model unloaded mid-use.)
        if not registry.is_on_demand(model_id):
            # Normal (statically loaded) path
            try:
                return registry.get_handler(model_id)
            except KeyError:
                pass

        if registry.is_on_demand(model_id):'''
if src.count(old) != 1:
    sys.exit("endpoints.py anchor not found exactly once — upstream changed, re-port the patch")
open(p, "w").write(src.replace(old, new))
PYEOF
fi

log "verifying..."
"$PYTHON" -c "
import ast
from mlx_lm.models import glm_moe_dsa, deepseek_v32
src = open(glm_moe_dsa.__file__).read()
assert 'indexer_types' in src, 'patched file lacks indexer_types'
ep = '$ENDPOINTS'
ast.parse(open(ep).read())
assert 'bypassed refcounting' in open(ep).read(), 'endpoints patch missing'
print('[glm52-patch] verified: indexer_types support + refcount fix in place', flush=True)
" >&2

log "done — restart the MLX server (menu bar: restart services) to pick this up"
