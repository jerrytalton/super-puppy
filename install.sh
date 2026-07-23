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
    pkill -f "ds4-server" 2>/dev/null || true
    sleep 1

    # Remove symlinks (only if they point into this repo)
    echo "Removing symlinks..."
    for link in \
        ~/.local/bin/start-local-models \
        ~/.local/bin/local-models-menubar \
        ~/.local/bin/local-models-mcp-detect \
        ~/.local/bin/local-models-mcp-auth \
        ~/.local/bin/tailscale-status \
        ~/.local/bin/tailscale \
        ~/.local/bin/post-update.sh \
        ~/.local/bin/sp-session-ping \
        ~/.local/bin/sp-doctor \
        ~/bin/start-local-models \
        ~/bin/local-models-menubar \
        ~/bin/local-models-mcp-detect \
        ~/bin/local-models-mcp-auth \
        ~/bin/tailscale-status \
        ~/bin/tailscale \
        ~/bin/post-update.sh \
        ~/bin/sp-session-ping \
        ~/bin/sp-doctor \
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

    # Remove MLX configs (may be real files, not symlinks)
    MLX_CONF_DIR="$HOME/.config/mlx-server"
    if [ -d "$MLX_CONF_DIR" ]; then
        rm -rf "$MLX_CONF_DIR"
        echo "  Removed $MLX_CONF_DIR"
    fi

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

    # ds4 checkout + 244GiB GGUF
    DS4_DIR=$(grep '^DS4_DIR=' "$HOME/.config/local-models/network.conf" 2>/dev/null \
        | head -1 | cut -d= -f2- | tr -d '"' || true)
    DS4_DIR="${DS4_DIR:-$HOME/.local/share/super-puppy/ds4}"
    if [ -e "$DS4_DIR" ]; then
        echo ""
        echo "ds4 directory: $DS4_DIR (checkout + glm-5.2 GGUF, ~244GiB)"
        echo "  Not removed automatically. Delete it manually if you want:"
        echo "  rm -rf \"$DS4_DIR\""
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

# Expose the macOS app's Tailscale CLI when it's bundle-only (see bin/tailscale).
# The standalone (macsys) build ships no `tailscale` on PATH, so `command -v`
# reports "not installed" even though the app is present. Link our shim — but
# only when nothing else already provides `tailscale`, so we never shadow a
# Homebrew or official-CLI install. Our own shim counts as "provided".
ensure_tailscale_wrapper() {
    local have; have="$(command -v tailscale 2>/dev/null || true)"
    if [ -x "/Applications/Tailscale.app/Contents/MacOS/Tailscale" ] \
       && { [ -z "$have" ] || [ "$have" = "$HOME/.local/bin/tailscale" ]; }; then
        mkdir -p "$HOME/.local/bin"
        ln -sfn "$REPO_DIR/bin/tailscale" "$HOME/.local/bin/tailscale"
        hash -r 2>/dev/null || true   # drop cached "not found" so command -v re-resolves
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

        # The app may be installed with its CLI only inside the bundle; link our
        # shim so `command -v tailscale` reflects reality, not a missing symlink.
        ensure_tailscale_wrapper

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
            else
                ensure_tailscale_wrapper   # they may have just installed the app
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
        CLIENT_HOST=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
        # The token is referenced by env var, NOT inlined: Claude Code expands
        # ${SP_MCP_TOKEN} from the environment at load time, so the literal
        # secret never lands in ~/.claude.json (safe to sync/commit). The
        # ${SP_MCP_TOKEN} below is single-quoted so the shell leaves it literal.
        ENTRY='{"type":"http","url":"http://127.0.0.1:8100/mcp","headers":{"Authorization":"Bearer ${SP_MCP_TOKEN}","X-SP-Client":"'"$CLIENT_HOST"'"}}'
        claude mcp add-json -s user local-models "$ENTRY" 2>/dev/null
        echo "  Registered local-models MCP (streamable-http on port 8100)"
        echo "  → The MCP entry reads the token from \$SP_MCP_TOKEN. Make it available"
        echo "    to your shell (so 'claude' inherits it). Add to your shell rc:"
        echo "        export SP_MCP_TOKEN=\"\$(cat ~/.config/local-models/mcp_auth_token)\""
        echo "    (or from the keychain, if you keep it there). GUI-launched Claude"
        echo "    needs it in the GUI environment too — 'launchctl setenv SP_MCP_TOKEN ...'."
        # Register open-websearch if not already present
        if ! grep -q '"open-websearch"' ~/.claude.json 2>/dev/null; then
            claude mcp add-json -s user open-websearch \
                '{"type":"stdio","command":"npx","args":["-y","open-websearch@2.0.2"],"env":{"MODE":"stdio"}}' 2>/dev/null || true
            echo "  Registered open-websearch MCP"
        else
            echo "  open-websearch MCP already registered"
        fi
        # Report the wiring status, but do NOT auto-write to files you
        # hand-maintain (CLAUDE.md / settings.json). The guidance block and
        # session-start hook are opt-in: run `sp-doctor --fix`, which shows
        # each change and asks before writing. (MCP registration above went
        # to ~/.claude.json, a Claude-managed config, preserving your other
        # servers.) sp-doctor exits nonzero when any check fails — branch on
        # that so the closing message matches what the audit just showed.
        echo ""
        if ~/.local/bin/sp-doctor 2>/dev/null; then
            echo ""
            echo "  MCP registered; no failing checks. Agents are wired to Super Puppy."
        else
            echo ""
            echo "  MCP registered. The ❌ items above are optional wiring — agent"
            echo "  guidance, the session-start hook, and any extra Claude accounts"
            echo "  (shown as @account) — which SP never writes without your OK."
            echo ""
            echo "  To finish wiring them up now, run:"
            echo ""
            echo "      sp-doctor --fix"
            echo ""
            echo "  It walks the ❌ items one by one, shows each change, and asks"
            echo "  before writing. It only appends or merges — it never overwrites"
            echo "  content you maintain yourself."
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
    uv tool install --python 3.12 mlx-openai-server \
        || echo "  Warning: mlx-openai-server install failed (MLX models will be unavailable)"
fi

# mflux for image generation/editing (optional, needs 32GB+ RAM)
RAM_CHECK=$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')
if [ "$RAM_CHECK" -ge 32 ] && ! command -v mflux-generate > /dev/null; then
    echo "  Installing mflux (image generation)..."
    uv tool install --python 3.12 mflux || echo "  Warning: mflux install failed (image gen/edit will be unavailable)"
fi

# mlx-vlm for computer_use / MLX vision grounding. Dispatched as a one-shot
# subprocess (lib/mlx_vlm.py) rather than through mlx-openai-server, whose
# persistent server hangs on multimodal generation (mlx 0.31.2 thread-local
# stream bug, mlx-lm #1256). Needs torch for the Qwen3-VL processors, so pull
# it into the tool env. lib.mlx_vlm.command() looks for this env first.
if [ "$RAM_CHECK" -ge 32 ] && [ ! -x "$HOME/.local/share/uv/tools/mlx-vlm/bin/python" ]; then
    echo "  Installing mlx-vlm (computer_use)..."
    uv tool install --python 3.12 "mlx-vlm==0.4.4" --with torch --with torchvision \
        || echo "  Warning: mlx-vlm install failed (computer_use on MLX models will be unavailable)"
fi

# ds4 (512gb tier): serves glm-5.2 from a Q2K GGUF — 244GiB resident vs the
# retired 390GB mlx-openai-server path, and no more pinned mlx-lm patch.
# Pinned commit on the glm5.2 branch: the engine is weeks old, so we ship
# exactly what was verified (tool-calling round-trip, 15.5 tok/s, strict
# JSON quirk documented in docs/troubleshooting.md).
DS4_COMMIT="bd89932"
DS4_GGUF_REPO="antirez/GLM-5.2-GGUF"
DS4_GGUF_FILE="GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"
if [ "$RAM_CHECK" -ge 512 ]; then
    DS4_DIR=$(grep '^DS4_DIR=' "$HOME/.config/local-models/network.conf" 2>/dev/null \
        | head -1 | cut -d= -f2- | tr -d '"' || true)
    DS4_DIR="${DS4_DIR:-$HOME/.local/share/super-puppy/ds4}"

    # Reuse a dev checkout: if ~/experiments/ds4 already has the binary and
    # the 244GiB GGUF, symlink it — never re-download 244GiB.
    if [ ! -e "$DS4_DIR" ] && [ -x "$HOME/experiments/ds4/ds4-server" ] \
       && [ -f "$HOME/experiments/ds4/gguf/$DS4_GGUF_FILE" ]; then
        mkdir -p "$(dirname "$DS4_DIR")"
        ln -sfn "$HOME/experiments/ds4" "$DS4_DIR"
        echo "  ds4: reusing existing checkout at ~/experiments/ds4"
    fi

    if [ ! -x "$DS4_DIR/ds4-server" ]; then
        echo "  Building ds4 (glm-5.2 engine, pinned $DS4_COMMIT)..."
        if [ ! -d "$DS4_DIR/.git" ]; then
            mkdir -p "$DS4_DIR"
            git clone --branch glm5.2 https://github.com/antirez/ds4 "$DS4_DIR" \
                || echo "  Warning: ds4 clone failed (glm-5.2 will be unavailable)"
        fi
        if [ -d "$DS4_DIR/.git" ]; then
            git -C "$DS4_DIR" fetch --quiet origin glm5.2 || true
            git -C "$DS4_DIR" checkout --quiet "$DS4_COMMIT" \
                || echo "  Warning: pinned ds4 commit $DS4_COMMIT not found"
            (cd "$DS4_DIR" && make ds4-server) \
                || echo "  Warning: ds4 build failed (glm-5.2 will be unavailable)"
        fi
    fi

    # Weights: 244GiB GGUF, with a disk-space precheck (none of the existing
    # pull paths can see this file — it's not an HF snapshot layout we scan).
    if [ -x "$DS4_DIR/ds4-server" ] && [ ! -f "$DS4_DIR/gguf/$DS4_GGUF_FILE" ]; then
        if ! command -v hf > /dev/null; then
            brew install hf 2>/dev/null || true
        fi
        FREE_GB=$(df -g "$HOME" | awk 'NR==2 {print $4}')
        if [ "${FREE_GB:-0}" -lt 260 ]; then
            echo "  Warning: only ${FREE_GB}GB free — the glm-5.2 GGUF needs ~250GB."
            echo "           Free space and re-run install.sh to download it."
        elif command -v hf > /dev/null; then
            echo "  Downloading glm-5.2 GGUF (~244GiB — this takes a while)..."
            hf download "$DS4_GGUF_REPO" "$DS4_GGUF_FILE" --local-dir "$DS4_DIR/gguf" \
                || echo "  Warning: glm-5.2 GGUF download failed — re-run install.sh to retry."
        else
            echo "  Warning: hf CLI unavailable — cannot download the glm-5.2 GGUF."
        fi
    fi

    if [ -f "$DS4_DIR/gguf/$DS4_GGUF_FILE" ]; then
        ln -sfn "gguf/$DS4_GGUF_FILE" "$DS4_DIR/ds4flash.gguf"
    fi
    set_conf "DS4_DIR" "\"$DS4_DIR\""
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

# Record any runtimes that failed to install. We do NOT abort here: one
# failed runtime (e.g. a flaky brew download) must not leave the install
# half-finished with no menu bar app and no signal. Instead we report loudly,
# remember what's missing, and finish — the user gets a running app plus an
# actionable punch-list, and re-running install.sh is idempotent.
MISSING_RUNTIMES=()
command -v uv > /dev/null || MISSING_RUNTIMES+=("uv")
command -v ollama > /dev/null || MISSING_RUNTIMES+=("ollama")
command -v mlx-openai-server > /dev/null || MISSING_RUNTIMES+=("mlx-openai-server")
if [ "$RAM_CHECK" -ge 512 ]; then
    DS4_DIR_CHECK=$(grep '^DS4_DIR=' "$HOME/.config/local-models/network.conf" 2>/dev/null \
        | head -1 | cut -d= -f2- | tr -d '"' || true)
    [ -x "${DS4_DIR_CHECK:-$HOME/.local/share/super-puppy/ds4}/ds4-server" ] \
        || MISSING_RUNTIMES+=("ds4-server")
fi

remediation_for() {
    case "$1" in
        ollama)            echo "brew install ollama" ;;
        mlx-openai-server) echo "uv tool install --python 3.12 mlx-openai-server" ;;
        uv)                echo "curl -LsSf https://astral.sh/uv/install.sh | sh" ;;
        ds4-server)        echo "re-run install.sh (clones antirez/ds4@bd89932 and runs make ds4-server)" ;;
        *)                 echo "(see https://github.com/jerrytalton/super-puppy)" ;;
    esac
}

if [ ${#MISSING_RUNTIMES[@]} -eq 0 ]; then
    echo "  All dependencies installed."
else
    echo ""
    echo "  ⚠  WARNING: some runtimes did not install: ${MISSING_RUNTIMES[*]}"
    echo "     Super Puppy will still start, but tools needing these won't work"
    echo "     until you install them and re-run install.sh:"
    for dep in "${MISSING_RUNTIMES[@]}"; do
        printf '       %-18s → %s\n' "$dep" "$(remediation_for "$dep")"
    done
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
# Use the user's MLX config (post-update.sh already copied it from the repo default)
MLX_CONF_DIR="$HOME/.config/mlx-server"
MLX_CONFIG="$MLX_CONF_DIR/config.yaml"
if   [ "$RAM_GB" -ge 512 ]; then SUGGESTED_PROFILE="512gb"
elif [ "$RAM_GB" -ge 128 ]; then SUGGESTED_PROFILE="128gb"
elif [ "$RAM_GB" -ge 64  ]; then SUGGESTED_PROFILE="64gb"
else                              SUGGESTED_PROFILE="32gb"
fi
SUGGESTED_LABEL="${SUGGESTED_PROFILE/gb/ GB}"

echo "This machine has ${RAM_GB}GB RAM."
echo "  Suggested profile: $SUGGESTED_LABEL"
echo ""
echo "  Available profiles: 32gb, 64gb, 128gb, 512gb, skip"
printf "  Pull models for which profile? [%s] " "$SUGGESTED_PROFILE"
read -r chosen_profile
PROFILE_NAME="${chosen_profile:-$SUGGESTED_PROFILE}"

if [ "$PROFILE_NAME" = "skip" ]; then
    echo "  Skipping model pull. Pull models later with ollama pull or the menu bar app."
else
    echo ""
    echo "Pulling models for '$PROFILE_NAME' profile..."

    PROFILES_FILE="$HOME/.config/local-models/profiles.json"

    # MLX served-model repos are resolved per-profile below, NOT harvested
    # wholesale from the config — the config lists the entire server catalog
    # (e.g. the 397B, ~400GB), so a blanket pull would download hundreds of GB
    # a profile never uses. We map only the profile's served-names to their
    # model_path repos.
    HF_MODELS=()

    # Parse profile tasks into Ollama models (contain ":") and HuggingFace repos
    # (contain "/" but not ":" — distinguishes "org/model" from "ollama/ns:tag").
    # Models with neither (e.g. "qwen3.5-fast") are MLX served names, already
    # covered by the MLX config parse above.
    #
    # Ensure profiles.json exists. Seed the presets directly from lib.models so
    # the model list never depends on the menu bar app having started: the
    # profile server normally owns this file, but it only runs when remote
    # access is enabled, so a fresh install would otherwise have no profiles.
    OLLAMA_MODELS=()
    if [ ! -f "$PROFILES_FILE" ]; then
        echo "  Seeding default model profiles..."
        python3 -c '
import json, sys
sys.path.insert(0, sys.argv[1])
from lib.models import DEFAULT_PROFILES, PROFILES_FILE
PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
PROFILES_FILE.write_text(json.dumps(DEFAULT_PROFILES, indent=2))
' "$REPO_DIR" 2>/dev/null || true
    fi
    # Fallback: if seeding failed, give the menu bar app a chance to write it.
    if [ ! -f "$PROFILES_FILE" ]; then
        printf "  Waiting up to 30s for the menu bar app to publish model profiles"
        for i in $(seq 1 30); do
            [ -f "$PROFILES_FILE" ] && break
            printf "."
            sleep 1
        done
        printf "\n"
    fi
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
        # Resolve the profile's MLX served-names (no ':' and no '/') to their
        # model_path repos via the MLX server config, and pull only those.
        while IFS= read -r path; do
            HF_MODELS+=("$path")
        done < <(python3 -c "
import json, pathlib
data = json.loads(pathlib.Path('$PROFILES_FILE').read_text())
profile = data.get('profiles', {}).get('$PROFILE_NAME', {})
served = {m for m in profile.get('tasks', {}).values()
          if m and ':' not in m and '/' not in m}
served.discard('glm-5.2')  # ds4-served: provisioned by the ds4 install step, not the MLX config
name_to_path, cur_path = {}, None
for line in pathlib.Path('$MLX_CONFIG').read_text().splitlines():
    s = line.strip()
    if 'model_path:' in s:
        cur_path = s.split('model_path:', 1)[1].strip()
    elif 'served_model_name:' in s and cur_path:
        name_to_path[s.split('served_model_name:', 1)[1].strip()] = cur_path
import sys
seen = set()
for nm in served:
    path = name_to_path.get(nm)
    if not path:
        print(f'  NOTE: served-name {nm!r} has no model_path in the MLX config — not pre-pulled', file=sys.stderr)
    elif path not in seen:
        seen.add(path)
        print(path)
")
    else
        echo "  WARNING: profiles.json not found — no models resolved, skipping pull"
    fi

    # Deduplicate HF_MODELS. Guard the expansion: on bash 3.2 (macOS default),
    # "${arr[@]}" on an empty array aborts under `set -u`.
    if [ ${#HF_MODELS[@]} -gt 0 ]; then
        HF_MODELS=($(printf '%s\n' "${HF_MODELS[@]}" | awk '!seen[$0]++'))
    fi

    # Build the set of already-present Ollama tags once.
    PRESENT_OLLAMA=""
    command -v ollama > /dev/null && PRESENT_OLLAMA=$(ollama list 2>/dev/null | awk 'NR>1{print $1}')

    if [ ${#OLLAMA_MODELS[@]} -eq 0 ]; then
        echo "  No Ollama models for the '$PROFILE_NAME' profile."
    elif ! command -v ollama > /dev/null; then
        echo "  Skipping Ollama model pulls — ollama is not installed."
    else
        total=${#OLLAMA_MODELS[@]}; current=0; pulled=0
        for model in "${OLLAMA_MODELS[@]}"; do
            current=$((current + 1))
            if printf '%s\n' "$PRESENT_OLLAMA" | grep -qx "$model"; then
                echo "  [$current/$total] ollama: $model — already present, skipping"
                continue
            fi
            echo "  [$current/$total] ollama: $model — pulling"
            ollama pull "$model" || echo "    WARNING: failed to pull $model"
            pulled=$((pulled + 1))
        done
        echo "  Ollama: $pulled pulled, $((total - pulled)) already present."
    fi

    # Download HuggingFace models
    if ! command -v hf > /dev/null; then
        echo "  Installing hf..."
        brew install hf 2>/dev/null || true
    fi
    if [ ${#HF_MODELS[@]} -eq 0 ]; then
        echo "  No HuggingFace/MLX models to download for the '$PROFILE_NAME' profile."
    elif command -v hf > /dev/null; then
        # Authenticate to HuggingFace before downloading: some repos are gated
        # (e.g. FLUX) and auth also avoids anonymous rate limits. Use HF's own
        # conventions — an existing login, or the standard HF_TOKEN env var.
        if hf auth whoami > /dev/null 2>&1; then
            echo "  HuggingFace: already logged in."
        elif [ -n "${HF_TOKEN:-}" ]; then
            # Non-interactive path (CI, curl pipe): use the standard env var.
            if hf auth login --token "$HF_TOKEN" > /dev/null 2>&1; then
                echo "  HuggingFace: logged in via HF_TOKEN."
            else
                echo "  WARNING: HF_TOKEN was set but login failed — gated repos may 401."
            fi
        elif [ -t 0 ]; then
            # Interactive terminal: prompt for a one-time login so downloads
            # aren't silently anonymous/rate-limited. --force because whoami
            # just failed: any stored token is stale, and without it hf
            # no-ops ("already logged in") and the install proceeds to 401.
            echo "  Not logged in to HuggingFace — starting login"
            echo "  (create a token at https://huggingface.co/settings/tokens):"
            hf auth login --force || echo "  WARNING: login skipped/failed — gated repos will 401."
        else
            echo "  WARNING: not logged in to HuggingFace and no HF_TOKEN set — downloads"
            echo "           will be anonymous (gated repos like FLUX will 401). Run"
            echo "           'hf auth login' or set HF_TOKEN to enable them." >&2
        fi
        HF_CACHE="$HOME/.cache/huggingface/hub"
        total=${#HF_MODELS[@]}; current=0; pulled=0; hf_failed=0
        for model in "${HF_MODELS[@]}"; do
            current=$((current + 1))
            cache_name="models--${model//\//--}"
            if [ -d "$HF_CACHE/$cache_name/snapshots" ] && \
               [ -z "$(find "$HF_CACHE/$cache_name/blobs" -name '*.incomplete' 2>/dev/null)" ]; then
                echo "  [$current/$total] huggingface: $model — already present, skipping"
                continue
            fi
            echo "  [$current/$total] huggingface: $model — downloading"
            # Retry with xet disabled: the xet backend can fail on large files
            # ("Unable to parse string as hex hash value"); plain HTTP works.
            if hf download "$model" || { echo "  download failed — retrying without xet..."
                                         HF_HUB_DISABLE_XET=1 hf download "$model"; }; then
                pulled=$((pulled + 1))
            else
                hf_failed=$((hf_failed + 1))
                echo "  WARNING: download failed for $model — re-run install.sh to retry."
            fi
        done
        echo "  HuggingFace: $pulled downloaded, $hf_failed failed, $((total - pulled - hf_failed)) already present."
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

if [ ${#MISSING_RUNTIMES[@]} -ne 0 ]; then
    echo ""
    echo "  ⚠  Install incomplete — these runtimes are still missing:"
    for dep in "${MISSING_RUNTIMES[@]}"; do
        printf '       %-18s → %s\n' "$dep" "$(remediation_for "$dep")"
    done
    echo "     Install them, then re-run ./install.sh to finish setup."
fi
