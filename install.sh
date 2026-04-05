#!/bin/bash
#
# Super Puppy installer.
# Symlinks scripts, copies configs, and walks through setup interactively.
#
# One-line install (curl from GitHub):
#   bash <(curl -fsSL superpuppy.ai/install.sh)
#
# Or from a local clone:
#   ./install.sh
#   --rotate-token   Generate a new MCP auth token and update 1Password
#   --reconfigure    Re-run interactive setup even if network.conf exists
#   --uninstall      Remove symlinks, LaunchAgents, configs, and MCP registration

set -euo pipefail

# ── Bootstrap: clone the repo if running via curl pipe ─────────────
# Detect: if this script isn't inside a git repo, we're being piped.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" 2>/dev/null)" 2>/dev/null && pwd 2>/dev/null)" || SCRIPT_DIR=""
if [ -z "$SCRIPT_DIR" ] || ! git -C "$SCRIPT_DIR" rev-parse --git-dir &>/dev/null; then
    INSTALL_DIR="$HOME/super-puppy"
    echo "Cloning Super Puppy into $INSTALL_DIR..."
    if [ -d "$INSTALL_DIR/.git" ]; then
        echo "  Already cloned — pulling latest..."
        git -C "$INSTALL_DIR" pull --quiet
    else
        git clone https://github.com/jerrytalton/super-puppy "$INSTALL_DIR"
    fi
    exec "$INSTALL_DIR/install.sh" "$@"
fi

FORCE_TOKEN_REFRESH=false
RECONFIGURE=false
UNINSTALL=false
for arg in "$@"; do
    case "$arg" in
        --rotate-token) FORCE_TOKEN_REFRESH=true ;;
        --reconfigure)  RECONFIGURE=true ;;
        --uninstall)    UNINSTALL=true ;;
    esac
done

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Uninstall ──────────────────────────────────────────────────────
if $UNINSTALL; then
    echo "Uninstalling Super Puppy..."
    echo ""

    # Stop the app
    echo "Stopping menu bar app..."
    launchctl unload ~/Library/LaunchAgents/com.local-models.menubar.plist 2>/dev/null || true
    launchctl unload ~/Library/LaunchAgents/setenv.OLLAMA_HOST.plist 2>/dev/null || true
    sleep 1

    # Remove symlinks (only if they point into this repo)
    echo "Removing symlinks..."
    for link in \
        ~/bin/start-local-models \
        ~/bin/local-models-menubar \
        ~/bin/local-models-mcp-detect \
        ~/bin/local-models-mcp-auth \
        ~/bin/tailscale-status \
        ~/bin/post-update.sh \
        ~/.config/mlx-server/config.yaml \
        ~/.config/mlx-server/config-laptop.yaml \
        ~/Library/LaunchAgents/com.local-models.menubar.plist \
        ~/Library/LaunchAgents/setenv.OLLAMA_HOST.plist; do
        if [ -L "$link" ]; then
            target=$(readlink "$link" 2>/dev/null || true)
            if [[ "$target" == "$REPO_DIR"* ]]; then
                rm "$link"
                echo "  Removed $link"
            fi
        fi
    done

    # Remove compiled artifacts
    echo "Removing build artifacts..."
    rm -f "$REPO_DIR/app/SuperPuppy.app/Contents/MacOS/super-puppy"
    rm -f "$REPO_DIR/app/SuperPuppy.app/Contents/Resources/AppIcon.icns"
    rm -f "$REPO_DIR/app/pwa"/icon-*.png 2>/dev/null || true

    # Remove MCP registration from Claude
    if command -v claude > /dev/null; then
        echo "Removing MCP registration..."
        claude mcp remove local-models 2>/dev/null || true
    fi

    # Remove config files (with confirmation)
    CONFIG_DIR="$HOME/.config/local-models"
    if [ -d "$CONFIG_DIR" ]; then
        echo ""
        echo "Config directory: $CONFIG_DIR"
        echo "  Contains: profiles, preferences, auth token, network config."
        read -rp "  Delete config files? [y/N] " confirm
        if [[ "${confirm:-n}" =~ ^[Yy] ]]; then
            rm -rf "$CONFIG_DIR"
            echo "  Removed $CONFIG_DIR"
        else
            echo "  Kept $CONFIG_DIR"
        fi
    fi

    # Remove lock file
    rm -f "$HOME/.config/local-models/menubar.lock" 2>/dev/null || true

    # Clean up git config
    git -C "$REPO_DIR" config --unset gpg.ssh.allowedSignersFile 2>/dev/null || true
    rm -f "$HOME/.config/git/allowed_signers"

    echo ""
    echo "Uninstalled. The repo itself is still at $REPO_DIR — delete it manually if you want."
    exit 0
fi

# Write a key=value pair into the user's network.conf
# Uses grep + temp file instead of sed to avoid delimiter injection.
set_conf() {
    local key="$1" value="$2"
    local conf="$HOME/.config/local-models/network.conf"
    if grep -q "^${key}=" "$conf" 2>/dev/null; then
        local tmp="${conf}.tmp"
        grep -v "^${key}=" "$conf" > "$tmp"
        echo "${key}=${value}" >> "$tmp"
        mv "$tmp" "$conf"
    else
        echo "${key}=${value}" >> "$conf"
    fi
}

# Validate a hostname: only alphanumeric, hyphens, and dots.
validate_hostname() {
    local host="$1"
    if [[ ! "$host" =~ ^[a-zA-Z0-9][a-zA-Z0-9.\-]*$ ]]; then
        echo "  ERROR: Invalid hostname '$host' — only letters, numbers, hyphens, and dots allowed." >&2
        return 1
    fi
}

echo "Installing Super Puppy..."
echo ""

# Symlinks and app bundle build (shared with auto-updater)
"$REPO_DIR/bin/post-update.sh"

# User-writable configs (copied, not symlinked — installer writes values into these)
NETWORK_CONF="$HOME/.config/local-models/network.conf"
MCP_PREFS="$HOME/.config/local-models/mcp_preferences.json"
EASTER_EGGS="$HOME/.config/local-models/easter_eggs.json"

mkdir -p "$(dirname "$NETWORK_CONF")"

if [ ! -e "$NETWORK_CONF" ] || [ -L "$NETWORK_CONF" ]; then
    # First install, or upgrading from old symlinked config
    [ -L "$NETWORK_CONF" ] && rm "$NETWORK_CONF"
    cp "$REPO_DIR/config/local-models/network.conf" "$NETWORK_CONF"
    echo "  Installed default $NETWORK_CONF"
    RECONFIGURE=true
else
    echo "  $NETWORK_CONF already exists, keeping"
fi

if [ ! -e "$MCP_PREFS" ]; then
    cp "$REPO_DIR/config/local-models/mcp_preferences.json" "$MCP_PREFS"
    echo "  Installed default $MCP_PREFS"
else
    echo "  $MCP_PREFS already exists, keeping"
fi

if [ ! -e "$EASTER_EGGS" ]; then
    cp "$REPO_DIR/config/local-models/easter_eggs.json" "$EASTER_EGGS"
    echo "  Installed default $EASTER_EGGS"
fi

# ── Interactive setup ────────────────────────────────────────────────
if $RECONFIGURE; then
    echo ""
    echo "Configuring Super Puppy..."
    RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')

    # 1. Server or client?
    if [ "$RAM_GB" -ge 128 ]; then
        default_server="y"
        echo "  This machine has ${RAM_GB}GB RAM — likely a good model server."
    else
        default_server="n"
        echo "  This machine has ${RAM_GB}GB RAM."
    fi
    printf "  Is this the model server (serves models to other machines)? [%s] " \
        "$([ "$default_server" = "y" ] && echo "Y/n" || echo "y/N")"
    read -r is_server_input
    is_server_input="${is_server_input:-$default_server}"
    if [[ "$is_server_input" =~ ^[Yy] ]]; then
        set_conf "IS_SERVER" "true"
        set_conf "SERVER_RAM_GB" "$RAM_GB"
        IS_SERVER_MODE=true
        echo "  → Server mode (${RAM_GB}GB RAM)"
    else
        set_conf "IS_SERVER" "false"
        IS_SERVER_MODE=false
        echo "  → Client mode"
    fi

    # 2. Tailscale (required for remote access between machines)
    echo ""
    SETUP_TAILSCALE=false
    TS_RUNNING=false

    if $IS_SERVER_MODE; then
        printf "  Set up Tailscale? (required for remote clients to connect) [Y/n] "
        ts_default="y"
    else
        printf "  Set up Tailscale? (required to reach the model server) [Y/n] "
        ts_default="y"
    fi
    read -r setup_tailscale_input
    setup_tailscale_input="${setup_tailscale_input:-$ts_default}"
    if [[ "$setup_tailscale_input" =~ ^[Yy] ]]; then
        SETUP_TAILSCALE=true

        if ! command -v tailscale > /dev/null 2>&1; then
            echo ""
            echo "  Tailscale is not installed."
            echo "  IMPORTANT: Use the standalone build, NOT the App Store or Homebrew cask version."
            echo "  The sandboxed versions cannot run Tailscale SSH."
            echo ""
            echo "  Download from: https://tailscale.com/download/mac"
            echo ""
            printf "  Press Enter after installing Tailscale (or 's' to skip): "
            read -r ts_wait
            if [[ "$ts_wait" =~ ^[Ss] ]]; then
                SETUP_TAILSCALE=false
            fi
        fi

        if $SETUP_TAILSCALE && command -v tailscale > /dev/null 2>&1; then
            # Check for sandboxed version
            TS_PATH=$(which tailscale)
            if [[ "$TS_PATH" == *"Tailscale.app"* ]] || [[ "$TS_PATH" == *"/Applications/"* ]]; then
                echo "  WARNING: This looks like the sandboxed Tailscale (App Store or Homebrew cask)."
                echo "  Tailscale SSH won't work. Consider reinstalling the standalone build."
                echo "  Download from: https://tailscale.com/download/mac"
                echo ""
            fi

            TS_STATUS=$(tailscale status --json 2>/dev/null \
                | python3 -c "import json,sys; print(json.load(sys.stdin).get('BackendState',''))" 2>/dev/null || true)

            if [ "$TS_STATUS" != "Running" ]; then
                echo ""
                echo "  Tailscale is not running. Starting login..."
                echo "  A browser window will open. Log in with your identity provider."
                echo ""
                tailscale up 2>&1 || true
                sleep 2
                TS_STATUS=$(tailscale status --json 2>/dev/null \
                    | python3 -c "import json,sys; print(json.load(sys.stdin).get('BackendState',''))" 2>/dev/null || true)
            fi

            if [ "$TS_STATUS" = "Running" ]; then
                TS_RUNNING=true
                echo "  ✓ Tailscale is running"

                if $IS_SERVER_MODE; then
                    # Server: set this machine's Tailscale hostname
                    printf "  Tailscale hostname for this machine [super-puppy]: "
                    read -r ts_host
                    ts_host="${ts_host:-super-puppy}"
                    if ! validate_hostname "$ts_host"; then
                        ts_host="super-puppy"
                        echo "  → Using default hostname: $ts_host"
                    fi
                    set_conf "TAILSCALE_HOSTNAME" "\"$ts_host\""

                    tailscale set --hostname "$ts_host" 2>/dev/null \
                        && echo "  ✓ Hostname set to $ts_host" \
                        || echo "  WARNING: could not set hostname (try: sudo tailscale set --hostname $ts_host)"
                fi

                # Enable Tailscale SSH
                echo ""
                echo "  Enabling Tailscale SSH (allows direct SSH between your machines)..."
                TS_SSH_OUT=$(sudo tailscale set --ssh 2>&1) \
                    && echo "  ✓ Tailscale SSH enabled" \
                    || {
                        if echo "$TS_SSH_OUT" | grep -qi "sandbox"; then
                            echo "  ✗ Failed — this is the sandboxed Tailscale build."
                            echo "    Uninstall and download the standalone build from:"
                            echo "    https://tailscale.com/download/mac"
                        else
                            echo "  WARNING: could not enable Tailscale SSH (needs sudo)"
                        fi
                    }

                # Get FQDN and generate certs (server only)
                TS_FQDN=$(tailscale status --json 2>/dev/null \
                    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('Self',{}).get('DNSName','').rstrip('.'))" 2>/dev/null || true)

                if [ -n "$TS_FQDN" ]; then
                    echo ""
                    echo "  ✓ Your Tailscale FQDN: $TS_FQDN"
                fi

                if $IS_SERVER_MODE && [ -n "$TS_FQDN" ]; then
                    CERT_DIR="$HOME/.config/local-models/certs"
                    mkdir -p "$CERT_DIR"
                    echo "  Generating HTTPS certs for remote Playground access..."
                    if tailscale cert \
                        --cert-file "$CERT_DIR/${TS_FQDN}.crt" \
                        --key-file "$CERT_DIR/${TS_FQDN}.key" \
                        "$TS_FQDN" 2>/dev/null; then
                        echo "  ✓ Certs saved to $CERT_DIR/"
                    else
                        echo "  WARNING: cert generation failed. Remote Playground will use HTTP."
                    fi
                fi

                echo ""
                echo "  Tailscale setup complete."
                if $IS_SERVER_MODE; then
                    echo ""
                    echo "  To share access with others:"
                    echo "    1. Have them install Tailscale and join your tailnet"
                    echo "    2. Approve their devices in the Tailscale admin console:"
                    echo "       https://login.tailscale.com/admin/machines"
                fi
            else
                echo "  ✗ Tailscale login did not complete. Skipping Tailscale setup."
                echo "    Run 'tailscale up' manually, then re-run install.sh --reconfigure"
            fi
        fi
    else
        echo "  → Skipping Tailscale"
    fi

    # 3. Client: which server to connect to?
    if ! $IS_SERVER_MODE; then
        echo ""
        if $TS_RUNNING; then
            # List Tailscale peers to help the user pick
            echo "  Tailscale peers on your tailnet:"
            tailscale status --json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
for peer in d.get('Peer', {}).values():
    host = peer.get('HostName', '')
    fqdn = peer.get('DNSName', '').rstrip('.')
    if host:
        print(f'    {host}  ({fqdn})')
" 2>/dev/null || true
            echo ""
        fi
        printf "  Tailscale hostname of the model server (e.g. super-puppy): "
        read -r ts_server_host
        if [ -n "$ts_server_host" ] && validate_hostname "$ts_server_host"; then
            set_conf "TAILSCALE_HOSTNAME" "\"$ts_server_host\""
            echo "  → Will connect to $ts_server_host via Tailscale"

            # Try to detect server RAM via Tailscale SSH (best-effort, queried at runtime if missing)
            if $TS_RUNNING; then
                REMOTE_RAM=$(ssh -o ConnectTimeout=3 -o BatchMode=yes "$ts_server_host" \
                    "sysctl -n hw.memsize 2>/dev/null" 2>/dev/null | awk '{printf "%d", $1 / 1073741824}' || true)
                if [ -n "$REMOTE_RAM" ] && [ "$REMOTE_RAM" -gt 0 ]; then
                    set_conf "SERVER_RAM_GB" "$REMOTE_RAM"
                    echo "  → Detected ${REMOTE_RAM}GB RAM on server"
                fi
            fi
        else
            echo "  → No server configured (standalone mode)"
        fi
    fi

    # 4. Auth token
    #
    # Server: generate a token and store it in 1Password (if available).
    # Client: read the token from 1Password (if the item exists).
    # Both: cache the token locally in mcp_auth_token.
    #
    # The well-known 1Password item name is "Super Puppy MCP Token".
    # Both machines find it by name — no manual OP_REF exchange needed.
    echo ""
    echo "  The MCP server requires a bearer token for authentication."
    TOKEN_CACHE="$HOME/.config/local-models/mcp_auth_token"
    OP_ITEM_NAME="Super Puppy MCP Token"
    OP_VAULT="${OP_VAULT:-Private}"

    if command -v op > /dev/null 2>&1 && op account list &>/dev/null; then
        OP_AVAILABLE=true
        echo "  1Password CLI detected."

        # Check if the item already exists
        OP_EXISTING=$(op item get "$OP_ITEM_NAME" --vault "$OP_VAULT" --fields password 2>/dev/null || true)
    else
        OP_AVAILABLE=false
        OP_EXISTING=""
    fi

    if $IS_SERVER_MODE; then
        # Server: generate token and push to 1Password
        if [ -f "$TOKEN_CACHE" ] && [ -s "$TOKEN_CACHE" ] && ! $FORCE_TOKEN_REFRESH; then
            echo "  → Token already exists at $TOKEN_CACHE (use --rotate-token to regenerate)"
        else
            auto_token=$(openssl rand -hex 32)
            (umask 077 && echo "$auto_token" > "$TOKEN_CACHE")
            echo "  → Generated new auth token"

            if $OP_AVAILABLE; then
                if [ -n "$OP_EXISTING" ]; then
                    op item edit "$OP_ITEM_NAME" --vault "$OP_VAULT" "password=$auto_token" &>/dev/null \
                        && echo "  → Updated token in 1Password ($OP_VAULT/$OP_ITEM_NAME)" \
                        || echo "  WARNING: could not update 1Password item"
                else
                    op item create --category=password --title="$OP_ITEM_NAME" \
                        --vault "$OP_VAULT" "password=$auto_token" &>/dev/null \
                        && echo "  → Saved token to 1Password ($OP_VAULT/$OP_ITEM_NAME)" \
                        || echo "  WARNING: could not create 1Password item"
                fi
                set_conf "OP_REF" "\"op://$OP_VAULT/$OP_ITEM_NAME/password\""
            fi
        fi
    else
        # Client: read token from 1Password or prompt
        if $OP_AVAILABLE && [ -n "$OP_EXISTING" ]; then
            (umask 077 && echo "$OP_EXISTING" > "$TOKEN_CACHE")
            set_conf "OP_REF" "\"op://$OP_VAULT/$OP_ITEM_NAME/password\""
            echo "  → Read token from 1Password ($OP_VAULT/$OP_ITEM_NAME)"
        elif $OP_AVAILABLE; then
            echo "  No '$OP_ITEM_NAME' item found in 1Password vault '$OP_VAULT'."
            echo "  Run install.sh on the server first to create it, or paste the token manually."
            printf "  Paste MCP auth token (or Enter to skip): "
            read -r manual_token
            if [ -n "$manual_token" ]; then
                (umask 077 && echo "$manual_token" > "$TOKEN_CACHE")
                echo "  → Token saved to $TOKEN_CACHE"
            else
                echo "  → Skipping token setup. MCP auth will fail until a token is configured."
            fi
        else
            # No 1Password — manual token entry
            if [ ! -f "$TOKEN_CACHE" ]; then
                printf "  Paste MCP auth token (or Enter to auto-generate): "
                read -r manual_token
                if [ -n "$manual_token" ]; then
                    (umask 077 && echo "$manual_token" > "$TOKEN_CACHE")
                    echo "  → Token saved to $TOKEN_CACHE"
                else
                    auto_token=$(openssl rand -hex 32)
                    (umask 077 && echo "$auto_token" > "$TOKEN_CACHE")
                    echo "  → Generated random token and saved to $TOKEN_CACHE"
                fi
            fi
        fi
    fi

    echo ""
    echo "  Configuration saved to $NETWORK_CONF"
    echo "  Re-run with --reconfigure to change these settings."
fi

# Reload config after setup
# shellcheck source=/dev/null
source "$HOME/.config/local-models/network.conf"

# Register local-models MCP in Claude Code config
CLAUDE_JSON="$HOME/.claude.json"
if [ -f "$CLAUDE_JSON" ]; then
    TOKEN_CACHE="$HOME/.config/local-models/mcp_auth_token"
    OP_ITEM_NAME="Super Puppy MCP Token"
    OP_VAULT="${OP_VAULT:-Private}"
    MCP_TOKEN=""
    if $FORCE_TOKEN_REFRESH && [ -f "$TOKEN_CACHE" ]; then
        # Only rotate here if interactive setup didn't already handle it
        rm -f "$TOKEN_CACHE"
        echo "  Rotating MCP auth token..."
        new_token=$(openssl rand -hex 32)
        (umask 077 && echo "$new_token" > "$TOKEN_CACHE")
        # Update 1Password if available
        if command -v op > /dev/null 2>&1 && op account list &>/dev/null; then
            if op item get "$OP_ITEM_NAME" --vault "$OP_VAULT" &>/dev/null; then
                op item edit "$OP_ITEM_NAME" --vault "$OP_VAULT" "password=$new_token" &>/dev/null \
                    && echo "  → Updated token in 1Password" \
                    || echo "  WARNING: could not update 1Password item"
            else
                op item create --category=password --title="$OP_ITEM_NAME" \
                    --vault "$OP_VAULT" "password=$new_token" &>/dev/null \
                    && echo "  → Saved token to 1Password" \
                    || echo "  WARNING: could not create 1Password item"
            fi
        fi
        echo "  → New token generated. Run install.sh on clients to pick up the new token."
    fi
    if [ -f "$TOKEN_CACHE" ] && [ -s "$TOKEN_CACHE" ]; then
        MCP_TOKEN=$(cat "$TOKEN_CACHE")
    elif [ -n "${OP_REF:-}" ]; then
        MCP_TOKEN=$(op read "$OP_REF" 2>/dev/null || true)
        if [ -n "$MCP_TOKEN" ]; then
            (umask 077 && echo "$MCP_TOKEN" > "$TOKEN_CACHE")
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
        if ! grep -q '"open-websearch"' ~/.claude.json 2>/dev/null; then
            claude mcp add-json -s user open-websearch \
                '{"type":"stdio","command":"npx","args":["-y","open-websearch@2.0.2"],"env":{"MODE":"stdio"}}' 2>/dev/null || true
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

if ! command -v op > /dev/null && [ -n "${OP_REF:-}" ]; then
    echo "  1password-cli not found (needed for OP_REF in network.conf)."
    echo "  Install with: brew install 1password-cli"
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

# mflux for image generation/editing (optional, needs 32GB+ RAM)
RAM_CHECK=$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')
if [ "$RAM_CHECK" -ge 32 ] && ! command -v mflux-generate > /dev/null; then
    echo "  Installing mflux (image generation)..."
    uv tool install --python 3.12 mflux || echo "  Warning: mflux install failed (image gen/edit will be unavailable)"
fi

# ffmpeg for audio transcription (WebM conversion, format support)
if ! command -v ffmpeg > /dev/null; then
    if command -v brew > /dev/null; then
        echo "  Installing ffmpeg (audio transcription support)..."
        brew install ffmpeg || echo "  Warning: ffmpeg install failed (some audio formats may not work)"
    else
        echo "  Note: ffmpeg not found. Install with 'brew install ffmpeg' for full audio transcription support."
    fi
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

# Pull models for the best-fitting profile.
# Derives model lists from the MLX server config and profiles.json.
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')
echo ""

# Pick the profile that best fits this machine's RAM
if [ "$RAM_GB" -ge 256 ]; then
    SUGGESTED_PROFILE="everyday"
    SUGGESTED_LABEL="Everyday (best balance for 256GB+ machines)"
    MLX_CONFIG="$REPO_DIR/config/mlx-server/config.yaml"
elif [ "$RAM_GB" -ge 48 ]; then
    SUGGESTED_PROFILE="desktop"
    SUGGESTED_LABEL="Desktop (fits in 64GB)"
    MLX_CONFIG="$REPO_DIR/config/mlx-server/config.yaml"
else
    SUGGESTED_PROFILE="laptop"
    SUGGESTED_LABEL="Laptop (lightweight models)"
    MLX_CONFIG="$REPO_DIR/config/mlx-server/config-laptop.yaml"
fi

echo "This machine has ${RAM_GB}GB RAM."
echo "  Suggested profile: $SUGGESTED_LABEL"
echo ""
echo "  Available profiles: everyday, desktop, maximum, laptop, skip"
printf "  Pull models for which profile? [%s] " "$SUGGESTED_PROFILE"
read -r chosen_profile
PROFILE_NAME="${chosen_profile:-$SUGGESTED_PROFILE}"

if [ "$PROFILE_NAME" = "skip" ]; then
    echo "  Skipping model pull. Pull models later with ollama pull or the menu bar app."
else
    echo ""
    echo "Pulling models for '$PROFILE_NAME' profile..."

    # Override MLX config for high-memory profiles
    case "$PROFILE_NAME" in
        everyday|maximum) MLX_CONFIG="$REPO_DIR/config/mlx-server/config.yaml" ;;
        laptop)           MLX_CONFIG="$REPO_DIR/config/mlx-server/config-laptop.yaml" ;;
        desktop)          MLX_CONFIG="$REPO_DIR/config/mlx-server/config.yaml" ;;
    esac

    PROFILES_FILE="$HOME/.config/local-models/profiles.json"

    # MLX models from server config
    HF_MODELS=()
    while IFS= read -r path; do
        HF_MODELS+=("$path")
    done < <(grep 'model_path:' "$MLX_CONFIG" | sed 's/.*model_path: *//')

    # Parse profile tasks into Ollama models (contain ":") and HuggingFace repos
    # (contain "/" but not ":" — distinguishes "org/model" from "ollama/ns:tag").
    # Models with neither (e.g. "qwen3.5-fast") are MLX served names, already
    # covered by the MLX config parse above.
    #
    # Wait for profiles.json — the menu bar app (started above) writes it on launch.
    OLLAMA_MODELS=()
    for i in $(seq 1 30); do
        [ -f "$PROFILES_FILE" ] && break
        sleep 1
    done
    if [ -f "$PROFILES_FILE" ]; then
        while IFS= read -r model; do
            HF_MODELS+=("$model")
        done < <(python3 -c "
import json, pathlib
data = json.loads(pathlib.Path('$PROFILES_FILE').read_text())
profile = data.get('profiles', {}).get('$PROFILE_NAME', {})
seen = set()
for model in profile.get('tasks', {}).values():
    if model and '/' in model and ':' not in model and model not in seen:
        seen.add(model)
        print(model)
")
        while IFS= read -r model; do
            OLLAMA_MODELS+=("$model")
        done < <(python3 -c "
import json, pathlib
data = json.loads(pathlib.Path('$PROFILES_FILE').read_text())
profile = data.get('profiles', {}).get('$PROFILE_NAME', {})
seen = set()
for model in profile.get('tasks', {}).values():
    if model and ':' in model and model not in seen:
        seen.add(model)
        print(model)
")
    else
        echo "  WARNING: profiles.json not found after 30s — pulling MLX models only"
    fi

    # Deduplicate HF_MODELS
    HF_MODELS=($(printf '%s\n' "${HF_MODELS[@]}" | awk '!seen[$0]++'))

    total=${#OLLAMA_MODELS[@]}
    current=0
    for model in "${OLLAMA_MODELS[@]}"; do
        current=$((current + 1))
        echo "  [$current/$total] ollama: $model"
        ollama pull "$model"
    done

    # Download HuggingFace models
    if ! command -v hf > /dev/null; then
        echo "  Installing hf..."
        brew install hf 2>/dev/null || true
    fi
    if command -v hf > /dev/null; then
        total=${#HF_MODELS[@]}
        current=0
        for model in "${HF_MODELS[@]}"; do
            current=$((current + 1))
            echo "  [$current/$total] huggingface: $model"
            hf download "$model" || true
        done
    else
        echo "  WARNING: hf install failed. HuggingFace models will download on first use."
    fi
fi

# Git tag signature verification (needed for auto-update)
echo ""
echo "  Configuring git tag verification..."
ALLOWED_SIGNERS="$HOME/.config/git/allowed_signers"
mkdir -p "$(dirname "$ALLOWED_SIGNERS")"
cp "$REPO_DIR/config/git/allowed_signers" "$ALLOWED_SIGNERS"
git -C "$REPO_DIR" config gpg.ssh.allowedSignersFile "$ALLOWED_SIGNERS"
echo "  ✓ Tag signature verification configured"

echo ""
echo "Done! Next steps:"
echo "  1. start-local-models           # start servers"
echo "  2. claude                        # start coding (local-models MCP auto-connects)"
