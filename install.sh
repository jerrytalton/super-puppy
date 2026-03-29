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
    if ! python3 -c "import json; d=json.load(open('$CLAUDE_JSON')); exit(0 if 'local-models' in d.get('mcpServers',{}) else 1)" 2>/dev/null; then
        python3 -c "
import json
with open('$CLAUDE_JSON') as f:
    d = json.load(f)
d.setdefault('mcpServers', {})['local-models'] = {
    'type': 'stdio',
    'command': 'bash',
    'args': ['-c', '\$HOME/bin/local-models-mcp-detect']
}
with open('$CLAUDE_JSON', 'w') as f:
    json.dump(d, f, indent=2)
"
        echo "  Added local-models MCP to $CLAUDE_JSON"
    else
        echo "  local-models MCP already in $CLAUDE_JSON"
    fi
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
if ! cc -o "$APP_MACOS/super-puppy" "$APP_SRC" 2>&1; then
    echo "  ERROR: Failed to compile $APP_SRC"
    exit 1
fi
echo "  Compiled launcher binary"

# Generate .icns from icon.png
mkdir -p "$APP_RES"
ICONSET=$(mktemp -d)/AppIcon.iconset
mkdir -p "$ICONSET"
for size in 16 32 64 128 256 512; do
    sips -z $size $size "$REPO_DIR/app/icon.png" --out "$ICONSET/icon_${size}x${size}.png" > /dev/null 2>&1
done
iconutil -c icns "$ICONSET" -o "$APP_RES/AppIcon.icns" 2>/dev/null && echo "  Generated app icon" || true

# Ad-hoc code sign (required for TCC / screen recording permission)
codesign --sign - --force "$REPO_DIR/app/SuperPuppy.app" > /dev/null 2>&1
echo "  Signed app bundle (ad-hoc)"

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

echo ""
echo "Done! Next steps:"
echo "  1. ollama pull qwen3.5          # pull a model"
echo "  2. start-local-models           # start servers"
echo "  3. claude                        # start coding (local-models MCP auto-connects)"
