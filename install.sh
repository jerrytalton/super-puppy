#!/bin/bash
#
# Super Puppy installer.
# Symlinks configs, scripts, and LaunchAgents into place.
# Auto-detects desktop vs laptop.
#
# Run from the repo root: ./install.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

link() {
    local src="$REPO_DIR/$1"
    local dst="$2"

    mkdir -p "$(dirname "$dst")"

    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "  Backing up $dst -> ${dst}.bak"
        mv "$dst" "${dst}.bak"
    fi

    ln -sfn "$src" "$dst"
    echo "  $dst -> $src"
}

echo "Installing Super Puppy..."

# Scripts
link bin/claude-local              ~/bin/claude-local
link bin/start-local-models        ~/bin/start-local-models
link bin/local-models-menubar      ~/bin/local-models-menubar
link bin/local-models-mcp-detect   ~/bin/local-models-mcp-detect
link bin/local-models-mcp-auth     ~/bin/local-models-mcp-auth

# Configs
link config/mlx-server/config.yaml         ~/.config/mlx-server/config.yaml
link config/mlx-server/config-laptop.yaml  ~/.config/mlx-server/config-laptop.yaml
link config/local-models/network.conf      ~/.config/local-models/network.conf

# MCP preferences: copy defaults if no file exists (not symlinked — profile viewer writes to it)
MCP_PREFS="$HOME/.config/local-models/mcp_preferences.json"
if [ ! -e "$MCP_PREFS" ]; then
    cp "$REPO_DIR/config/local-models/mcp_preferences.json" "$MCP_PREFS"
    echo "  Installed default $MCP_PREFS"
else
    echo "  $MCP_PREFS already exists, keeping"
fi

# LaunchAgent: menu bar app (all machines)
link config/launchd/com.local-models.menubar.plist \
    ~/Library/LaunchAgents/com.local-models.menubar.plist

# Desktop only: Ollama listens on all interfaces
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')
if [ "$RAM_GB" -ge 256 ]; then
    link config/launchd/setenv.OLLAMA_HOST.plist \
        ~/Library/LaunchAgents/setenv.OLLAMA_HOST.plist
    echo "  (desktop detected — installed Ollama LAN LaunchAgent)"
fi

# Register local-models MCP in Claude Code config
CLAUDE_JSON="$HOME/.claude.json"
if [ -f "$CLAUDE_JSON" ]; then
    # Read auth token from 1Password (fall back to cache)
    OP_REF="op://kasngocpevbljoznv5mz2equga/Super Puppy MCP/credential"
    MCP_TOKEN=$(op read "$OP_REF" 2>/dev/null || cat "$HOME/.config/local-models/mcp_auth_token" 2>/dev/null || true)
    python3 -c "
import json, sys
token = sys.argv[1]
with open('$CLAUDE_JSON') as f:
    d = json.load(f)
entry = {'type': 'http', 'url': 'http://127.0.0.1:8100/mcp'}
if token:
    entry['headers'] = {'Authorization': f'Bearer {token}'}
d.setdefault('mcpServers', {})['local-models'] = entry
with open('$CLAUDE_JSON', 'w') as f:
    json.dump(d, f, indent=2)
" "$MCP_TOKEN"
    echo "  Registered local-models MCP (streamable-http on port 8100) in $CLAUDE_JSON"
else
    echo "  $CLAUDE_JSON not found — run claude once first, then re-run install.sh"
fi

# Install dependencies
echo ""
echo "Checking dependencies..."

if ! command -v uv > /dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v op > /dev/null; then
    if command -v brew > /dev/null; then
        echo "  Installing 1password-cli..."
        brew install 1password-cli || true
    else
        echo "  WARNING: 1password-cli not found. Install manually: brew install 1password-cli"
    fi
fi

if ! command -v ollama > /dev/null; then
    if command -v brew > /dev/null; then
        echo "  Installing ollama..."
        brew install ollama || true
    else
        echo "  ERROR: ollama not found and brew is not available. Install manually: https://ollama.com"
        exit 1
    fi
fi

if ! command -v mlx-openai-server > /dev/null; then
    echo "  Installing mlx-openai-server..."
    uv tool install --python 3.12 mlx-openai-server
fi

missing=()
command -v uv > /dev/null || missing+=("uv")
command -v ollama > /dev/null || missing+=("ollama")
command -v mlx-openai-server > /dev/null || missing+=("mlx-openai-server")

if [ ${#missing[@]} -eq 0 ]; then
    echo "  All dependencies installed."
else
    echo "  ERROR: Failed to install: ${missing[*]}"
    exit 1
fi

# Build Super Puppy app bundle
echo ""
echo "Building Super Puppy.app..."
APP_MACOS="$REPO_DIR/app/SuperPuppy.app/Contents/MacOS"
APP_RES="$REPO_DIR/app/SuperPuppy.app/Contents/Resources"
APP_SRC="$REPO_DIR/app/super-puppy.c"
if [ ! -f "$APP_SRC" ]; then
    echo "  ERROR: $APP_SRC not found"
    exit 1
fi
mkdir -p "$APP_MACOS"
NEEDS_SIGN=false
APP_BIN="$APP_MACOS/super-puppy"
if [ ! -f "$APP_BIN" ] || [ "$APP_SRC" -nt "$APP_BIN" ]; then
    if ! cc -o "$APP_BIN" "$APP_SRC" 2>&1; then
        echo "  ERROR: Failed to compile $APP_SRC"
        exit 1
    fi
    echo "  Compiled launcher binary"
    NEEDS_SIGN=true
else
    echo "  Launcher binary up to date"
fi

# Generate .icns from icon.png (1x and @2x retina variants)
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
    iconutil -c icns "$ICONSET" -o "$APP_RES/AppIcon.icns" 2>/dev/null && echo "  Generated app icon" || true
    NEEDS_SIGN=true
else
    echo "  App icon up to date"
fi

# Ad-hoc code sign — only when binary or icon changed (re-signing invalidates
# macOS TCC permissions like Screen Recording, forcing the user to re-authorize)
if $NEEDS_SIGN; then
    codesign --sign - --force "$REPO_DIR/app/SuperPuppy.app" > /dev/null 2>&1
    echo "  Signed app bundle (ad-hoc)"
    echo "  ⚠  Re-signing may require re-enabling Screen Recording permission"
elif ! codesign --verify "$REPO_DIR/app/SuperPuppy.app" > /dev/null 2>&1; then
    codesign --sign - --force "$REPO_DIR/app/SuperPuppy.app" > /dev/null 2>&1
    echo "  Signed app bundle (signature was invalid)"
else
    echo "  App signature valid, skipping re-sign"
fi

# Start the menu bar app
echo ""
echo "Starting menu bar app..."
launchctl unload ~/Library/LaunchAgents/com.local-models.menubar.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.local-models.menubar.plist

# Check for CLAUDE.md local-models guidance
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
if [ -f "$CLAUDE_MD" ]; then
    if ! grep -q "Local Model Cluster" "$CLAUDE_MD" 2>/dev/null; then
        echo ""
        echo "  ⚠  Missing local-models guidance in $CLAUDE_MD"
        echo "     Claude won't know when to use local model tools without it."
        echo "     Add the '## Local Model Cluster' section from the README."
    fi
else
    echo ""
    echo "  ⚠  No $CLAUDE_MD found. Claude won't know about local model tools."
    echo "     Create it and add the '## Local Model Cluster' section from the README."
fi

# Pull models appropriate for this machine's memory tier
# Server (512GB+) manages its own models — only pull for laptop/desktop.
echo ""
if [ "$RAM_GB" -lt 256 ]; then
    echo "Pulling models for local use..."

    # Shared across all non-server tiers
    MODELS=(
        "qwen3.5:9b"
        "glm-4.7-flash:latest"
        "all-minilm:latest"
        "dolphin3:8b"
        "x/flux2-klein:latest"
    )

    if [ "$RAM_GB" -ge 48 ]; then
        # Desktop tier (64GB+): add larger models
        MODELS+=(
            "qwen3-coder-next:latest"
            "x/z-image-turbo:latest"
        )
        echo "  Detected desktop ($RAM_GB GB) — pulling desktop + laptop models"
    else
        echo "  Detected laptop ($RAM_GB GB) — pulling laptop models"
    fi

    total=${#MODELS[@]}
    current=0
    for model in "${MODELS[@]}"; do
        current=$((current + 1))
        echo "  [$current/$total] $model"
        ollama pull "$model" 2>&1 | tail -1
    done

    # MLX models are on-demand (downloaded on first use via mlx-openai-server)
    echo "  MLX models (qwen3.5-fast, whisper-v3, etc.) download on first use."
else
    echo "Server detected ($RAM_GB GB) — skipping model pull."
    echo "Manage models via the menu bar app or ollama pull."
fi

echo ""
echo "Done! Next steps:"
echo "  1. start-local-models           # start servers"
echo "  2. claude                        # start coding (local-models MCP auto-connects)"
