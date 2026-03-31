#!/bin/bash
#
# Super Puppy installer.
# Symlinks configs, scripts, and LaunchAgents into place.
# Auto-detects desktop vs laptop.
#
# Run from the repo root: ./install.sh

set -euo pipefail

FORCE_TOKEN_REFRESH=false
for arg in "$@"; do
    case "$arg" in
        --rotate-token) FORCE_TOKEN_REFRESH=true ;;
    esac
done

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
    # Read auth token from cache, only hit 1Password if no cache exists
    TOKEN_CACHE="$HOME/.config/local-models/mcp_auth_token"
    if $FORCE_TOKEN_REFRESH; then
        rm -f "$TOKEN_CACHE"
        echo "  Cleared cached token, will read from 1Password"
    fi
    if [ -f "$TOKEN_CACHE" ] && [ -s "$TOKEN_CACHE" ]; then
        MCP_TOKEN=$(cat "$TOKEN_CACHE")
    else
        OP_REF="op://kasngocpevbljoznv5mz2equga/Super Puppy MCP/credential"
        MCP_TOKEN=$(op read "$OP_REF" 2>/dev/null || true)
        if [ -n "$MCP_TOKEN" ]; then
            mkdir -p "$(dirname "$TOKEN_CACHE")"
            echo "$MCP_TOKEN" > "$TOKEN_CACHE"
            chmod 600 "$TOKEN_CACHE"
        fi
    fi
    if command -v claude > /dev/null; then
        claude mcp remove local-models -s local 2>/dev/null || true
        claude mcp remove local-models -s user 2>/dev/null || true
        ENTRY='{"type":"http","url":"http://127.0.0.1:8100/mcp"'
        if [ -n "$MCP_TOKEN" ]; then
            ENTRY="$ENTRY"',"headers":{"Authorization":"Bearer '"$MCP_TOKEN"'"}'
        fi
        ENTRY="$ENTRY"'}'
        claude mcp add-json -s user local-models "$ENTRY" 2>/dev/null
        echo "  Registered local-models MCP (streamable-http on port 8100)"
        # Register open-websearch if not already present
        if ! claude mcp get open-websearch -s user > /dev/null 2>&1; then
            claude mcp add-json -s user open-websearch \
                '{"type":"stdio","command":"npx","args":["-y","open-websearch@latest"],"env":{"MODE":"stdio"}}' 2>/dev/null
            echo "  Registered open-websearch MCP"
        else
            echo "  open-websearch MCP already registered"
        fi
    else
        echo "  claude CLI not found — install Claude Code first, then re-run install.sh"
    fi
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

# Pull models appropriate for this machine's memory tier.
# Model lists match the DEFAULT_PROFILES in app/profile-server.py.
# Server (512GB+) manages its own models — only pull for laptop/desktop.
echo ""
if [ "$RAM_GB" -lt 256 ]; then
    echo "Pulling models for local use..."

    # Ollama models for the Laptop profile (~19GB total)
    OLLAMA_MODELS=(
        "qwen3.5:9b"
        "dolphin3:8b"
        "all-minilm:latest"
    )

    # HuggingFace models (downloaded via huggingface-cli for MLX/mlx-audio)
    HF_MODELS=(
        "mlx-community/Kokoro-82M-bf16"
    )

    if [ "$RAM_GB" -ge 48 ]; then
        # Desktop profile (~63GB total): add larger models
        OLLAMA_MODELS+=(
            "glm-4.7-flash:latest"
            "x/flux2-klein:latest"
        )
        HF_MODELS=(
            "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
        )
        echo "  Detected desktop ($RAM_GB GB) — pulling Desktop profile models"
    else
        echo "  Detected laptop ($RAM_GB GB) — pulling Laptop profile models"
    fi

    total=${#OLLAMA_MODELS[@]}
    current=0
    for model in "${OLLAMA_MODELS[@]}"; do
        current=$((current + 1))
        echo "  [$current/$total] ollama: $model"
        ollama pull "$model" 2>&1 | tail -1
    done

    # Download HuggingFace models (TTS etc.)
    if command -v huggingface-cli > /dev/null; then
        for model in "${HF_MODELS[@]}"; do
            echo "  huggingface: $model"
            huggingface-cli download "$model" --quiet 2>&1 | tail -1 || true
        done
    else
        echo "  Installing huggingface-cli..."
        uv tool install --python 3.12 huggingface_hub[cli] 2>/dev/null || true
        export PATH="$(uv tool dir --bin 2>/dev/null):$PATH"
        if command -v huggingface-cli > /dev/null; then
            for model in "${HF_MODELS[@]}"; do
                echo "  huggingface: $model"
                huggingface-cli download "$model" --quiet 2>&1 | tail -1 || true
            done
        else
            echo "  WARNING: huggingface-cli install failed. TTS models will download on first use."
        fi
    fi

    # MLX LLM models are on-demand (downloaded on first use via mlx-openai-server)
    echo "  MLX models (qwen3.5-fast, qwen3.5-small, whisper-v3) download on first use."
else
    echo "Server detected ($RAM_GB GB) — skipping model pull."
    echo "Manage models via the menu bar app or ollama pull."
fi

echo ""
echo "Done! Next steps:"
echo "  1. start-local-models           # start servers"
echo "  2. claude                        # start coding (local-models MCP auto-connects)"
