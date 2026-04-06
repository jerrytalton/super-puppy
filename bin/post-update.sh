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

SCRIPT="$(readlink -f "$0" 2>/dev/null || readlink "$0" 2>/dev/null || echo "$0")"
REPO_DIR="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
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

# MLX configs (copied on first install, new models merged on update)
MLX_DIR="$HOME/.config/mlx-server"
mkdir -p "$MLX_DIR"
for conf in config.yaml config-laptop.yaml; do
    user_conf="$MLX_DIR/$conf"
    repo_conf="$REPO_DIR/config/mlx-server/$conf"
    if [ ! -e "$user_conf" ] || [ -L "$user_conf" ]; then
        # First install or upgrading from old symlink
        [ -L "$user_conf" ] && rm "$user_conf"
        cp "$repo_conf" "$user_conf"
        log "Installed default $user_conf"
    else
        # Merge: append model entries from repo that user doesn't have yet
        python3 - "$repo_conf" "$user_conf" <<'PYEOF'
import re, sys
repo_path, user_path = sys.argv[1], sys.argv[2]
with open(repo_path) as f:
    repo_text = f.read()
with open(user_path) as f:
    user_text = f.read()
user_models = set(re.findall(r'model_path:\s*(.+)', user_text))
# Split repo config into model blocks (comment + entry)
blocks = re.split(r'\n(?=  #[^\n]*\n  - model_path:)', repo_text)
new_blocks = []
for block in blocks:
    m = re.search(r'model_path:\s*(.+)', block)
    if m and m.group(1).strip() not in user_models:
        # Extract just this model block (from comment through last indented line)
        lines = block.strip().split('\n')
        entry = []
        capture = False
        for line in lines:
            if line.strip().startswith('#') and not capture:
                capture = True
                entry.append(line)
            elif capture or line.strip().startswith('- model_path:'):
                capture = True
                entry.append(line)
        if entry:
            new_blocks.append('\n'.join(entry))
if new_blocks:
    with open(user_path, 'a') as f:
        for block in new_blocks:
            f.write('\n' + block + '\n')
    for block in new_blocks:
        name = re.search(r'model_path:\s*(.+)', block).group(1).strip()
        print(f'  Added new MLX model: {name}')
PYEOF
    fi
done

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
