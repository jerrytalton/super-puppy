# Tailscale Setup for Remote Access

Super Puppy uses Tailscale to securely access local models from outside the LAN.

## First-time setup

### 1. Install Tailscale

`install.sh` installs it automatically, or:

```bash
brew install --cask tailscale
```

### 2. Set the hostname

```bash
tailscale set --hostname super-puppy
```

This keeps your real machine name private. The Tailscale hostname is only visible within your tailnet.

### 3. Start Tailscale

Open the Tailscale app or:

```bash
tailscale up
```

Log in with your identity provider (Google, GitHub, etc.). Enable 2FA on that account.

### 4. Generate HTTPS certs

`install.sh` does this automatically when Tailscale is running on the desktop. To do it manually:

```bash
mkdir -p ~/.config/local-models/certs
tailscale cert \
  --cert-file ~/.config/local-models/certs/super-puppy.<tailnet>.ts.net.crt \
  --key-file ~/.config/local-models/certs/super-puppy.<tailnet>.ts.net.key \
  super-puppy.<tailnet>.ts.net
```

Replace `<tailnet>` with your tailnet name (visible in Tailscale admin console).

### 5. Enable Remote Access

In the Super Puppy menu bar: click **Remote Access** to toggle it on. The profile server restarts on port 8101 with HTTPS.

### 6. Add family devices

Each person installs Tailscale on their phone/laptop and joins your tailnet. You approve them in the Tailscale admin console.

## Accessing from iPhone/iPad

1. Install Tailscale on the iOS device and join the tailnet
2. On the Mac Studio, click **Copy Playground URL** in the Super Puppy menu
3. On the phone, open the URL in Safari
4. Tap Share → Add to Home Screen

## Recommended ACLs

In the [Tailscale admin console](https://login.tailscale.com/admin/acls), restrict access:

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["autogroup:owner"],
      "dst": ["super-puppy:8100"],
      "comment": "MCP server — owner only"
    },
    {
      "action": "accept",
      "src": ["group:family"],
      "dst": ["super-puppy:8101"],
      "comment": "Playground — family"
    },
    {
      "action": "accept",
      "src": ["group:family"],
      "dst": ["super-puppy:11434", "super-puppy:8000"],
      "comment": "Model backends — family (needed for Playground)"
    }
  ],
  "groups": {
    "group:family": ["user1@example.com", "user2@example.com"]
  }
}
```

## Cert renewal

Tailscale certs last 90 days and auto-renew. `install.sh --rotate-token` or restarting with Remote Access enabled will refresh them. If certs expire, the Playground falls back to HTTP on localhost (LAN still works).
