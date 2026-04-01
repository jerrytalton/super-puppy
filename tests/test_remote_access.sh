#!/bin/bash
#
# Test remote access to the desktop's profile server and MCP server
# via Tailscale, simulating what an iOS device or remote laptop sees.
#
# Usage: bash tests/test_remote_access.sh
#
# Requires: tailscale running, desktop reachable, Remote Access enabled.

set -euo pipefail

CONF="$HOME/.config/local-models/network.conf"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve desktop Tailscale FQDN and IP
TS_JSON=$(tailscale status --json 2>/dev/null)
TS_HOSTNAME=$(grep 'TAILSCALE_HOSTNAME' "$CONF" | cut -d'"' -f2)

DESKTOP_IP=$(echo "$TS_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for p in d.get('Peer', {}).values():
    if p.get('HostName') == '$TS_HOSTNAME':
        for ip in p.get('TailscaleIPs', []):
            if '.' in ip:
                print(ip)
                break
        break
" 2>/dev/null)

DESKTOP_FQDN=$(echo "$TS_JSON" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for p in d.get('Peer', {}).values():
    if p.get('HostName') == '$TS_HOSTNAME':
        print(p.get('DNSName', '').rstrip('.'))
        break
" 2>/dev/null)

if [ -z "$DESKTOP_IP" ]; then
    echo "FAIL: Desktop '$TS_HOSTNAME' not found in Tailscale peers"
    exit 1
fi

echo "Desktop: $TS_HOSTNAME ($DESKTOP_IP)"
echo "FQDN:    $DESKTOP_FQDN"
echo ""

PASS=0
FAIL=0
SKIP=0

check() {
    local name="$1"
    local url="$2"
    local expect="${3:-200}"

    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" 2>/dev/null || echo "000")

    if [ "$code" = "$expect" ]; then
        echo "  PASS  $name ($code)"
        PASS=$((PASS + 1))
    elif [ "$code" = "000" ]; then
        echo "  FAIL  $name (connection failed)"
        FAIL=$((FAIL + 1))
    else
        echo "  FAIL  $name (expected $expect, got $code)"
        FAIL=$((FAIL + 1))
    fi
}

# --- MCP server (port 8100) ---
echo "== MCP Server (port 8100) =="

echo "  Direct TCP (Tailscale IP):"
check "TCP connect" "http://$DESKTOP_IP:8100/api/mcp-models"

echo "  Via tailscale serve (HTTPS):"
if [ -n "$DESKTOP_FQDN" ]; then
    check "HTTPS connect" "https://$DESKTOP_FQDN:8100/api/mcp-models"
else
    echo "  SKIP  No FQDN available"
    SKIP=$((SKIP + 1))
fi
echo ""

# --- Profile server / Playground (port 8101) ---
PROFILE_PORT=$(grep '^PROFILE_PORT' "$CONF" | sed 's/.*=//; s/"//g; s/^ *//' || echo "8101")
echo "== Profile Server / Playground (port $PROFILE_PORT) =="

echo "  Direct TCP (Tailscale IP):"
check "TCP connect" "http://$DESKTOP_IP:$PROFILE_PORT/"

echo "  Via tailscale serve (HTTPS):"
if [ -n "$DESKTOP_FQDN" ]; then
    # These are the endpoints the Playground (iOS PWA) fetches on load
    check "Playground HTML"  "https://$DESKTOP_FQDN:$PROFILE_PORT/tools"
    check "/api/models"      "https://$DESKTOP_FQDN:$PROFILE_PORT/api/models"
    check "/api/tasks"       "https://$DESKTOP_FQDN:$PROFILE_PORT/api/tasks"
    check "/api/profiles"    "https://$DESKTOP_FQDN:$PROFILE_PORT/api/profiles"
    check "/api/system"      "https://$DESKTOP_FQDN:$PROFILE_PORT/api/system"
    check "/api/gpu"         "https://$DESKTOP_FQDN:$PROFILE_PORT/api/gpu"
    check "/manifest.json"   "https://$DESKTOP_FQDN:$PROFILE_PORT/manifest.json"
    check "/sw.js"           "https://$DESKTOP_FQDN:$PROFILE_PORT/sw.js"
else
    echo "  SKIP  No FQDN available"
    SKIP=$((SKIP + 8))
fi
echo ""

# --- Summary ---
TOTAL=$((PASS + FAIL + SKIP))
echo "== Results: $PASS passed, $FAIL failed, $SKIP skipped (of $TOTAL) =="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Troubleshooting:"
    echo "  - Is Remote Access enabled on the desktop menu bar?"
    echo "  - Run 'tailscale serve status' on the desktop to check proxies"
    echo "  - Check /tmp/local-models-profile-server.log on the desktop"
    exit 1
fi
