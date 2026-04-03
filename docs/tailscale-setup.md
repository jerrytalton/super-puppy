# Tailscale Setup for Remote Access

Super Puppy uses Tailscale for all remote access. No LAN binding, no firewall exceptions, no manual cert management.

## First-time setup

### 1. Install Tailscale (standalone build)

**Do NOT use `brew install --cask tailscale`** — that installs the sandboxed App Store version which can't run Tailscale SSH.

Download the standalone `.pkg` from: https://tailscale.com/download/mac

### 2. Set the hostname

```bash
tailscale set --hostname super-puppy
```

### 3. Start Tailscale

```bash
tailscale up
```

Log in with your identity provider. Enable 2FA on that account.

### 4. Enable Tailscale SSH (optional)

```bash
sudo tailscale set --ssh
```

Allows SSH between tailnet machines without sshd. `install.sh` attempts this automatically.

### 5. Enable Remote Access

In the Super Puppy menu bar on the desktop: click **Remote Access** to toggle it on. This runs `tailscale serve` to proxy all service ports over HTTPS:

| Port | Service |
|------|---------|
| 8100 | MCP server |
| 8101 | Profile server / Playground |
| 11434 | Ollama |
| 8000 | MLX |

No manual cert management needed — Tailscale handles TLS automatically.

## How it works

All services bind to `127.0.0.1` (localhost only). `tailscale serve` intercepts incoming Tailscale traffic and proxies it to localhost with TLS termination. Remote clients use `https://{fqdn}:{port}`.

When Remote Access is toggled off, `tailscale serve reset` removes all proxies.

## Accessing from iPhone/iPad

1. Install Tailscale on the iOS device and join the tailnet
2. On the server machine, click **Copy Playground URL** in the Super Puppy menu
2. On the server machine, click **Copy Playground URL** in the Super Puppy menu
3. On the phone, open the URL in Safari
4. Tap Share → Add to Home Screen (PWA — works offline for cached pages)

## Accessing from a laptop

The laptop's Super Puppy app automatically detects the desktop via Tailscale and switches to Client mode. No manual configuration needed. The menu shows **Remote** (checked) when connected to the desktop.

To force local mode: click **Local** in the menu. The desktop stays available — click **Remote** to switch back anytime.

## Recommended ACLs

In the [Tailscale admin console](https://login.tailscale.com/admin/acls):

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["autogroup:owner"],
      "dst": ["super-puppy:*"],
      "comment": "Owner — full access to all Super Puppy ports"
    },
    {
      "action": "accept",
      "src": ["group:family"],
      "dst": ["super-puppy:8101"],
      "comment": "Playground — family"
    }
  ],
  "groups": {
    "group:family": ["user1@example.com", "user2@example.com"]
  }
}
```

## Troubleshooting

**"Connection failed" in Playground or Profiles:**
- Is Remote Access enabled on the desktop? (Check for checkmark in menu)
- Run `tailscale serve status` on the desktop to verify proxies are active
- Verify the device is on the same tailnet: `tailscale status`

**Copy Diagnostics** (desktop menu bar) dumps mode, versions, service status, and recent logs to clipboard — useful for remote debugging.
