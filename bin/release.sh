#!/bin/bash
# Cut a signed Super Puppy release — the ONLY supported way to ship a tag.
# Runs the test suite + the fleet compat gate BEFORE creating/pushing the tag,
# because the fleet auto-pulls new tags within ~2 minutes (a bad tag is live
# almost instantly). Then signs, verifies the signature against the repo's
# allowed_signers, and pushes main + the tag.
#
# Usage: bin/release.sh vX.Y.Z [--dry-run]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0" || echo "$0")")/.." && pwd)"
cd "$REPO_DIR"

VERSION="${1:-}"
DRY_RUN="${2:-}"
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "usage: bin/release.sh vX.Y.Z [--dry-run]" >&2
    exit 2
fi
if [ -n "$DRY_RUN" ] && [ "$DRY_RUN" != "--dry-run" ]; then
    echo "unknown argument: '$DRY_RUN' (the only optional 2nd arg is --dry-run)" >&2
    exit 2
fi

PREV="$(git tag --sort=-v:refname | head -1)"
if [ -n "$PREV" ] && [ "$(printf '%s\n%s\n' "$PREV" "$VERSION" | sort -V | tail -1)" != "$VERSION" ]; then
    echo "refusing: $VERSION is not greater than latest tag $PREV" >&2
    exit 2
fi

if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then
    echo "refusing: tag $VERSION already exists locally (latest is ${PREV:-none}); if it's a stale --dry-run artifact, remove it: git tag -d $VERSION" >&2
    exit 2
fi

echo "== preconditions =="
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "not on main" >&2; exit 1; }
git fetch --quiet origin
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { echo "main not in sync with origin/main" >&2; exit 1; }

echo "== test suite =="
uv run --with pytest --with flask --with pyyaml --with requests --with mlx-audio \
    pytest tests/ -q -m "not slow and not e2e"

echo "== fleet compat gate (vs $PREV) =="
uv run tests/fleet/run_compat.py

echo "== sign tag $VERSION =="
git tag -s "$VERSION" -m "$VERSION"

echo "== verify signature against allowed_signers =="
git -c gpg.ssh.allowedSignersFile="$REPO_DIR/config/git/allowed_signers" tag -v "$VERSION"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "dry run: created+verified $VERSION locally; not pushing. Remove with: git tag -d $VERSION"
    exit 0
fi

echo "== push =="
git push origin main
git push origin "$VERSION"
echo "released $VERSION"
