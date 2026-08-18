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

# Clean up old ~/bin symlinks from pre-v1.x installs
for old in ~/bin/start-local-models ~/bin/local-models-menubar \
           ~/bin/local-models-mcp-detect ~/bin/local-models-mcp-auth \
           ~/bin/tailscale-status ~/bin/post-update.sh; do
    if [ -L "$old" ]; then
        target=$(readlink "$old" 2>/dev/null || true)
        if [[ "$target" == "$REPO_DIR"* ]]; then
            rm "$old"
        fi
    fi
done
rmdir ~/bin 2>/dev/null || true

link bin/start-local-models        ~/.local/bin/start-local-models
link bin/local-models-menubar      ~/.local/bin/local-models-menubar
link bin/local-models-mcp-detect   ~/.local/bin/local-models-mcp-detect
link bin/local-models-mcp-auth     ~/.local/bin/local-models-mcp-auth
link bin/tailscale-status          ~/.local/bin/tailscale-status
link bin/post-update.sh            ~/.local/bin/post-update.sh
link bin/sp-session-ping           ~/.local/bin/sp-session-ping
link bin/sp-doctor                 ~/.local/bin/sp-doctor

# Tailscale CLI shim (see bin/tailscale). The macOS app ships its CLI only
# inside the bundle; we expose it via a wrapper — but ONLY when the app is
# present and nothing else already provides `tailscale`, so we never shadow a
# Homebrew or official-CLI install. Our own wrapper counts as "provided", so
# re-running this never flip-flops.
_ts_have="$(command -v tailscale 2>/dev/null || true)"
if [ -x "/Applications/Tailscale.app/Contents/MacOS/Tailscale" ] \
   && { [ -z "$_ts_have" ] || [ "$_ts_have" = "$HOME/.local/bin/tailscale" ]; }; then
    link bin/tailscale             ~/.local/bin/tailscale
fi

# MLX configs (copied on first install, new models merged on update)
MLX_DIR="$HOME/.config/mlx-server"
mkdir -p "$MLX_DIR"
rm -f "$MLX_DIR/config-laptop.yaml"
for conf in config.yaml; do
    user_conf="$MLX_DIR/$conf"
    repo_conf="$REPO_DIR/config/mlx-server/$conf"
    if [ ! -e "$user_conf" ] || [ -L "$user_conf" ]; then
        # First install or upgrading from old symlink
        [ -L "$user_conf" ] && rm "$user_conf"
        cp "$repo_conf" "$user_conf"
        # Local-dir model_path entries are $HOME-templated in the repo
        # config (mlx-openai-server does no ~ or env expansion itself).
        sed -i '' "s|\$HOME|$HOME|g" "$user_conf"
        log "Installed default $user_conf"
    else
        # Merge: append model entries from repo that user doesn't have yet
        merge_out=$(python3 - "$repo_conf" "$user_conf" <<'PYEOF'
import os, re, sys
repo_path, user_path = sys.argv[1], sys.argv[2]
with open(repo_path) as f:
    repo_text = f.read()
# Expand $HOME-templated local-dir model_path entries before comparing,
# so an already-merged (expanded) entry isn't re-appended every update.
repo_text = repo_text.replace('$HOME', os.path.expanduser('~'))
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
)
        [ -n "$merge_out" ] && printf '%s\n' "$merge_out"
        # mlx-openai-server reads its config at launch only — a running
        # instance can't serve names merged in above. Stop it; the app's
        # start_services relaunches it (start-local-models starts MLX
        # whenever port 8000 is dead) with the merged config.
        if printf '%s' "$merge_out" | grep -q 'Added new MLX model'; then
            if pkill -f "mlx-openai-server" 2>/dev/null; then
                log "Stopped MLX server — will relaunch with merged config"
            fi
        fi
    fi
done

# One-shot migration (2026-07 ds4 ship): glm-5.2 is served by ds4 on the
# 512GB tier now. The merge above is append-only and will never remove the
# user's old glm-5.2 MLX entry, which would double-serve the name (MLX
# claims it first in discovery order) and keep 418GB of dead weights.
# Failure-tolerant on purpose: post-update.sh failing rolls back the whole
# update (menubar._auto_update), and a cosmetic yaml migration must never
# do that.
#
# Gated on the ds4 binary actually being present, not just the RAM tier:
# auto-update never runs install.sh, so a 512GB machine that hasn't been
# provisioned with ds4 (fresh box, install skipped/failed) would otherwise
# have its working glm-5.2 MLX entry stripped with nothing to replace it —
# silent model loss with no UI signal. DS4_DIR resolved exactly like
# install.sh: network.conf override, else the default share dir.
DS4_DIR=$(grep '^DS4_DIR=' "$HOME/.config/local-models/network.conf" 2>/dev/null \
    | head -1 | cut -d= -f2- | tr -d '"' || true)
DS4_DIR="${DS4_DIR:-$HOME/.local/share/super-puppy/ds4}"
if [ "$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')" -ge 512 ] \
   && [ -f "$MLX_DIR/config.yaml" ] \
   && [ -x "$DS4_DIR/ds4-server" ]; then
    python3 "$REPO_DIR/bin/migrate-mlx-config.py" "$MLX_DIR/config.yaml" glm-5.2 \
        && log "MLX config migration checked (glm-5.2 → ds4)" \
        || log "WARNING: glm-5.2 MLX-entry migration failed (non-fatal)"
fi

# LaunchAgent — reload if ProgramArguments changed so launchd uses the new path
PLIST_DST="$HOME/Library/LaunchAgents/com.local-models.menubar.plist"
PLIST_LABEL="com.local-models.menubar"
old_args=""
if [ -f "$PLIST_DST" ]; then
    old_args=$(/usr/libexec/PlistBuddy -c "Print :ProgramArguments" "$PLIST_DST" 2>/dev/null || true)
fi
link config/launchd/com.local-models.menubar.plist "$PLIST_DST"
new_args=$(/usr/libexec/PlistBuddy -c "Print :ProgramArguments" "$PLIST_DST" 2>/dev/null || true)
if [ -n "$old_args" ] && [ "$old_args" != "$new_args" ]; then
    log "ProgramArguments changed — reloading LaunchAgent"
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    launchctl load "$PLIST_DST" 2>/dev/null || true
fi

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
