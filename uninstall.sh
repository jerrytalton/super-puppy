#!/bin/bash
#
# Uninstall Super Puppy.
#
# Removes: app bundle, symlinks, LaunchAgents, config files, Claude MCP registration.
# Leaves installed: uv, ollama, mlx-openai-server, mflux, ffmpeg (documented below).
# Does NOT remove: the git repo itself, downloaded models (Ollama/HuggingFace cache).
#
# Dependencies you may want to remove manually:
#   uv tool uninstall mlx-openai-server
#   uv tool uninstall mflux
#   brew uninstall ollama ffmpeg
#   rm -rf ~/.ollama              # Ollama models
#   rm -rf ~/.cache/huggingface   # HuggingFace model cache

set -euo pipefail

echo "Super Puppy Uninstaller"
echo "======================="
echo ""

# ── Stop services ──────────────────────────────────────────────────

echo "Stopping services..."
launchctl unload ~/Library/LaunchAgents/com.local-models.menubar.plist 2>/dev/null || true
pkill -f "menubar.py" 2>/dev/null || true
pkill -f "profile-server.py" 2>/dev/null || true
pkill -f "local-models-server.py" 2>/dev/null || true
echo "  Services stopped."

# ── Remove LaunchAgents ────────────────────────────────────────────

echo "Removing LaunchAgents..."
for plist in \
    ~/Library/LaunchAgents/com.local-models.menubar.plist \
    ~/Library/LaunchAgents/setenv.OLLAMA_HOST.plist; do
    if [ -e "$plist" ] || [ -L "$plist" ]; then
        launchctl unload "$plist" 2>/dev/null || true
        rm -f "$plist"
        echo "  Removed $(basename "$plist")"
    fi
done

# ── Remove ~/bin symlinks ──────────────────────────────────────────

echo "Removing symlinks..."
for link in \
    ~/bin/start-local-models \
    ~/bin/local-models-menubar \
    ~/bin/local-models-mcp-detect \
    ~/bin/local-models-mcp-auth \
    ~/bin/tailscale-status \
    ~/bin/post-update.sh; do
    if [ -L "$link" ]; then
        rm -f "$link"
        echo "  Removed $(basename "$link")"
    fi
done

# ── Remove config symlinks ─────────────────────────────────────────

echo "Removing config symlinks..."
for link in \
    ~/.config/mlx-server/config.yaml \
    ~/.config/mlx-server/config-laptop.yaml; do
    if [ -L "$link" ]; then
        rm -f "$link"
        echo "  Removed $link"
    fi
done

# ── Remove config files ───────────────────────────────────────────

echo ""
read -p "Remove config files (~/.config/local-models/)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ~/.config/local-models
    echo "  Removed ~/.config/local-models/"
    rmdir ~/.config/mlx-server 2>/dev/null && echo "  Removed ~/.config/mlx-server/" || true
else
    echo "  Kept config files."
fi

# ── Remove Claude MCP registration ────────────────────────────────

if command -v claude > /dev/null; then
    echo "Removing Claude MCP registration..."
    claude mcp remove local-models -s user 2>/dev/null && echo "  Removed local-models MCP" || true
fi

# ── Remove lock file ──────────────────────────────────────────────

rm -f /tmp/local-models-menubar.log

# ── Summary ────────────────────────────────────────────────────────

echo ""
echo "Done. Super Puppy has been uninstalled."
echo ""
echo "The following were NOT removed:"
echo "  - This git repo ($(cd "$(dirname "$0")" && pwd))"
echo "  - Downloaded Ollama models (~/.ollama/)"
echo "  - HuggingFace model cache (~/.cache/huggingface/)"
echo "  - Installed tools: ollama, mlx-openai-server, mflux, ffmpeg"
echo ""
echo "To remove dependencies:"
echo "  uv tool uninstall mlx-openai-server"
echo "  uv tool uninstall mflux"
echo "  brew uninstall ollama ffmpeg"
echo ""
echo "To remove downloaded models:"
echo "  rm -rf ~/.ollama"
echo "  rm -rf ~/.cache/huggingface"
