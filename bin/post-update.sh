#!/bin/bash
#
# Post-update hook: idempotent, non-interactive.
# Re-links scripts, rebuilds the app bundle, and signs it.
# Safe to run unattended after git checkout of a new tag.
#
# Called by:
#   - The menu bar app's auto-updater (after checking out a new tag)
#   - install.sh (for the shared build/link steps)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_TAG="post-update"

log() { echo "[$LOG_TAG] $*"; }

# ── Symlinks ────────────────────────────────────────────────────────
link() {
    local src="$REPO_DIR/$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        mv "$dst" "${dst}.bak"
    fi
    ln -sfn "$src" "$dst"
}

link bin/start-local-models        ~/bin/start-local-models
link bin/local-models-menubar      ~/bin/local-models-menubar
link bin/local-models-mcp-detect   ~/bin/local-models-mcp-detect
link bin/local-models-mcp-auth     ~/bin/local-models-mcp-auth
link bin/tailscale-status          ~/bin/tailscale-status
link bin/post-update.sh            ~/bin/post-update.sh

# Configs (symlinked — read-only reference)
link config/mlx-server/config.yaml         ~/.config/mlx-server/config.yaml
link config/mlx-server/config-laptop.yaml  ~/.config/mlx-server/config-laptop.yaml

# LaunchAgent
link config/launchd/com.local-models.menubar.plist \
    ~/Library/LaunchAgents/com.local-models.menubar.plist

# Server-only LaunchAgent
NETWORK_CONF="$HOME/.config/local-models/network.conf"
if [ -f "$NETWORK_CONF" ]; then
    # shellcheck source=/dev/null
    source "$NETWORK_CONF"
    if [ "${IS_SERVER:-false}" = "true" ]; then
        link config/launchd/setenv.OLLAMA_HOST.plist \
            ~/Library/LaunchAgents/setenv.OLLAMA_HOST.plist
    fi
fi

log "Symlinks updated"

# ── Git tag verification (needed for auto-update) ──────────────────
ALLOWED_SIGNERS="$HOME/.config/git/allowed_signers"
REPO_SIGNERS="$REPO_DIR/config/git/allowed_signers"
if [ -f "$REPO_SIGNERS" ]; then
    mkdir -p "$(dirname "$ALLOWED_SIGNERS")"
    cp "$REPO_SIGNERS" "$ALLOWED_SIGNERS"
    git -C "$REPO_DIR" config gpg.ssh.allowedSignersFile "$ALLOWED_SIGNERS"
    log "Tag verification configured"
fi

# ── Build app bundle ────────────────────────────────────────────────
APP_MACOS="$REPO_DIR/app/SuperPuppy.app/Contents/MacOS"
APP_RES="$REPO_DIR/app/SuperPuppy.app/Contents/Resources"
APP_SRC="$REPO_DIR/app/super-puppy.c"

if [ ! -f "$APP_SRC" ]; then
    log "ERROR: $APP_SRC not found"
    exit 1
fi

mkdir -p "$APP_MACOS"
NEEDS_SIGN=false
APP_BIN="$APP_MACOS/super-puppy"

if [ ! -f "$APP_BIN" ] || [ "$APP_SRC" -nt "$APP_BIN" ]; then
    if ! cc -o "$APP_BIN" "$APP_SRC" 2>&1; then
        log "ERROR: Failed to compile $APP_SRC"
        exit 1
    fi
    log "Compiled launcher binary"
    NEEDS_SIGN=true
else
    log "Launcher binary up to date"
fi

# Generate .icns from icon.png
mkdir -p "$APP_RES"
if [ ! -f "$APP_RES/AppIcon.icns" ] || [ "$REPO_DIR/app/icon.png" -nt "$APP_RES/AppIcon.icns" ]; then
    ICONSET=$(mktemp -d)/AppIcon.iconset
    mkdir -p "$ICONSET"
    for pair in "16 16" "32 16" "32 32" "64 32" "128 128" "256 128" "256 256" "512 256" "512 512" "1024 512"; do
        px=${pair%% *}; base=${pair##* }
        if [ "$px" = "$((base * 2))" ]; then
            out="$ICONSET/icon_${base}x${base}@2x.png"
        else
            out="$ICONSET/icon_${base}x${base}.png"
        fi
        sips -z $px $px "$REPO_DIR/app/icon.png" --out "$out" > /dev/null 2>&1
    done
    iconutil -c icns "$ICONSET" -o "$APP_RES/AppIcon.icns" 2>/dev/null && log "Generated app icon" || true
    NEEDS_SIGN=true
else
    log "App icon up to date"
fi

# Generate PWA icons
PWA_DIR="$REPO_DIR/app/pwa"
mkdir -p "$PWA_DIR"
if [ ! -f "$PWA_DIR/icon-512.png" ] || [ "$REPO_DIR/app/icon.png" -nt "$PWA_DIR/icon-512.png" ]; then
    for size in 152 180 192 512; do
        sips -z $size $size "$REPO_DIR/app/icon.png" --out "$PWA_DIR/icon-${size}.png" > /dev/null 2>&1
    done
    log "Generated PWA icons"
else
    log "PWA icons up to date"
fi

# Code sign — only when binary or icon changed (re-signing invalidates TCC
# permissions like Screen Recording, forcing the user to re-authorize)
if $NEEDS_SIGN; then
    codesign --sign - --force "$REPO_DIR/app/SuperPuppy.app" > /dev/null 2>&1
    log "Signed app bundle (ad-hoc)"
elif ! codesign --verify "$REPO_DIR/app/SuperPuppy.app" > /dev/null 2>&1; then
    codesign --sign - --force "$REPO_DIR/app/SuperPuppy.app" > /dev/null 2>&1
    log "Signed app bundle (signature was invalid)"
else
    log "App signature valid"
fi

log "Done"
