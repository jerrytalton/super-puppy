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

# Check dependencies
echo ""
echo "Checking dependencies..."
missing=()
command -v ollama > /dev/null || missing+=("ollama (brew install ollama)")
command -v mlx-openai-server > /dev/null || missing+=("mlx-openai-server (uv tool install --python 3.12 mlx-openai-server)")
command -v uv > /dev/null || missing+=("uv (curl -LsSf https://astral.sh/uv/install.sh | sh)")

if [ ${#missing[@]} -eq 0 ]; then
    echo "  All dependencies installed."
else
    echo "  Missing:"
    for m in "${missing[@]}"; do
        echo "    - $m"
    done
fi

echo ""
echo "Done! Next steps:"
echo "  1. ollama pull qwen3.5          # pull a model"
echo "  2. start-local-models           # start servers"
echo "  3. claude                        # start coding (local-models MCP auto-connects)"
