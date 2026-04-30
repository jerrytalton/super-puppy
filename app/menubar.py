# /// script
# requires-python = ">=3.12"
# dependencies = ["rumps==0.4.0", "pyyaml==6.0.3", "pyobjc-framework-WebKit==12.1"]
# ///
"""
Local Models — macOS menu bar app.

Shows the status of the local model infrastructure (Ollama + MLX-OpenAI-Server).
Auto-detects whether this machine is the desktop (server mode) or a laptop
(client mode), and whether the desktop is reachable on the LAN.

Run with:  uv run app/menubar.py
Or via:    open app/SuperPuppy.app
"""

import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

import objc
import rumps
from AppKit import NSCommandKeyMask, NSObject, NSWindow
import WebKit  # must be imported before _WebViewUIDelegate for block metadata


class _ProfileWindow(NSWindow):
    """NSWindow subclass that handles keyboard shortcuts in a menu-bar app."""

    def performKeyEquivalent_(self, event):
        if event.modifierFlags() & NSCommandKeyMask:
            key = event.charactersIgnoringModifiers()
            if key == "w":
                self.performClose_(None)
                return True
            # Standard edit shortcuts — forward to the first responder
            from AppKit import NSApp
            actions = {"c": "copy:", "v": "paste:", "x": "cut:", "a": "selectAll:", "z": "undo:"}
            if key in actions:
                NSApp.sendAction_to_from_(actions[key], None, self)
                return True
        return NSWindow.performKeyEquivalent_(self, event)


class _ProfileWindowDelegate(NSObject):
    """Clears the app's window reference on close."""
    callback = None

    def windowWillClose_(self, notification):
        try:
            if self.callback:
                self.callback()
        except Exception as e:  # PyObjC → NSException → abort(), so swallow.
            _raw_log(f"windowWillClose_ raised: {type(e).__name__}: {e}")


class _WebViewMessageHandler(NSObject):
    """Receives postMessage calls from WKWebView JavaScript."""
    on_message = None  # callable(body_dict)

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        try:
            if self.on_message:
                self.on_message(message.body())
        except Exception as e:
            _raw_log(f"didReceiveScriptMessage raised: {type(e).__name__}: {e}")


class _WebViewUIDelegate(NSObject):
    """WKUIDelegate that auto-grants media capture (microphone) permission.

    The WebKit import above must happen before this class is defined so
    pyobjc-framework-WebKit's block metadata is registered when the ObjC
    method trampoline is created.
    """

    @objc.typedSelector(b"v@:@@@q@?")
    def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
        self, webView, origin, frame, mediaType, decisionHandler
    ):
        try:
            # WKPermissionDecision.grant = 1
            decisionHandler(1)
        except Exception as e:
            _raw_log(f"requestMediaCapturePermission raised: {type(e).__name__}: {e}")
            try:
                decisionHandler(0)  # deny — WebKit must get a reply or it hangs
            except Exception:
                pass

    @objc.typedSelector(b"v@:@@@@?")
    def webView_runOpenPanelWithParameters_initiatedByFrame_completionHandler_(
        self, webView, parameters, frame, completionHandler
    ):
        try:
            from AppKit import NSOpenPanel
            panel = NSOpenPanel.openPanel()
            panel.setAllowsMultipleSelection_(parameters.allowsMultipleSelection())
            if panel.runModal() == 1:  # NSModalResponseOK
                completionHandler(panel.URLs())
            else:
                completionHandler(None)
        except Exception as e:
            _raw_log(f"runOpenPanel raised: {type(e).__name__}: {e}")
            try:
                completionHandler(None)
            except Exception:
                pass


def _raw_log(msg: str) -> None:
    """Write a line directly to fd 2 (stderr, redirected to the menubar
    log file by launchd). Bypasses the Python `logging` module so the
    breadcrumb survives when an ObjC/libc-level exit has already torn
    down the StreamHandler or flushed Python's stdio buffers.

    Used exclusively in exit-defense hooks. Never raises — if the write
    fails, the process is about to die anyway."""
    try:
        os.write(2, f"{time.strftime('%H:%M:%S')} RAW {msg}\n".encode())
    except Exception:
        pass


class _WebViewNavigationDelegate(NSObject):
    """Recover from WebContent-process termination by reloading the page.

    Why: Without this, a WebContent crash (OOM, JIT abort, streaming bug)
    leaves a dead WKWebView. If it was the only visible window, AppKit's
    default "terminate after last window closed" behavior fires while the
    app is in Regular activation policy — producing a clean NSApplication
    exit 0 that bypasses launchd's restart-on-failure. Reloading keeps the
    window alive, so the termination cascade never starts."""

    def webViewWebContentProcessDidTerminate_(self, webView):
        try:
            logging.warning("WKWebView WebContent process terminated — reloading")
            webView.reload()
        except Exception as e:
            _raw_log(f"webContentProcessDidTerminate raised: {type(e).__name__}: {e}")


class _NonTerminatingDelegateProxy(NSObject):
    """Wraps rumps's NSApplicationDelegate to force
    applicationShouldTerminateAfterLastWindowClosed: to NO.

    Why: When a WKWebView window is open, we're in
    NSApplicationActivationPolicyRegular (required for dock/cmd-tab/keyboard
    input), and the default last-window-closed behavior for Regular apps is
    to terminate the process. For a menubar app that should never be true —
    the menu bar item is our "last window" and it isn't an NSWindow at all.
    Returning NO here makes the app survive any window close, crash-induced
    or otherwise, so launchd never sees a clean exit 0 from this path."""

    _real_delegate = None

    def initWithDelegate_(self, delegate):
        self = objc.super(_NonTerminatingDelegateProxy, self).init()
        if self is None:
            return None
        self._real_delegate = delegate
        return self

    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        _raw_log("delegate: applicationShouldTerminateAfterLastWindowClosed → NO")
        return False

    def applicationShouldTerminate_(self, sender):
        """Refuse termination. A menu-bar app has no notion of a
        "normal quit" — any exit is either an intentional auto-update
        restart (which goes through os._exit(1) and bypasses this
        delegate entirely) or an accident we want to stop.

        Without this we get clean-exit-0 deaths whenever Cmd-Q is pressed
        while the Profiles / Playground window is key, or when macOS
        sends a quit-all on sleep/logout. NSApp.terminate: then calls
        exit(0) and launchd's SuccessfulExit=false policy treats it as
        success → no restart → SP silently dead. Returning
        NSTerminateCancel keeps the process alive; a real restart still
        works because os._exit(1) sidesteps AppKit entirely.

        Returns NSTerminateNow (1) / NSTerminateCancel (0) / NSTerminateLater (2)."""
        _raw_log("delegate: applicationShouldTerminate: called → NSTerminateCancel")
        return 0  # NSTerminateCancel

    def applicationWillTerminate_(self, notification):
        """Last-chance hook before NSApplication kills the process. Raw-log
        so we know the termination is actually happening, then forward."""
        _raw_log("delegate: applicationWillTerminate: — process is dying")
        if self._real_delegate.respondsToSelector_(b"applicationWillTerminate:"):
            self._real_delegate.applicationWillTerminate_(notification)

    def respondsToSelector_(self, selector):
        # NSObject's default respondsToSelector_ returns YES for methods
        # defined on this class (including applicationShouldTerminate...),
        # so we check that first and fall through to the wrapped delegate
        # for everything rumps needs (applicationWillTerminate:, menu
        # callbacks, etc). This avoids any SEL-string comparison, which is
        # fragile under PyObjC's selector marshalling.
        if objc.super(_NonTerminatingDelegateProxy, self).respondsToSelector_(selector):
            return True
        return self._real_delegate.respondsToSelector_(selector)

    def forwardingTargetForSelector_(self, selector):
        if self._real_delegate.respondsToSelector_(selector):
            return self._real_delegate
        return None




# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from lib.models import CLAUDE_CONFIG_FILE  # early import for MCP_TOOLS_FILE
ICON_PATH = os.path.join(SCRIPT_DIR, "icon-menubar.png")
ICONS_DIR = os.path.join(SCRIPT_DIR, "icons")
NETWORK_CONF = os.path.expanduser("~/.config/local-models/network.conf")
MCP_TOOLS_FILE = str(CLAUDE_CONFIG_FILE)
OLLAMA_LOCAL = "http://localhost:11434"
MLX_LOCAL = "http://localhost:8000"
MODE_CONF = os.path.expanduser("~/.config/local-models/mode.conf")
POLL_INTERVAL = 8           # seconds between status refreshes
UPDATE_CHECK_INTERVAL = 120  # seconds between git update checks (2 min)
UPDATE_IDLE_SECONDS = 60     # don't auto-update if MCP active within 60s
MAX_UPDATE_DEFERRALS = 5     # max consecutive MCP idle deferrals before forcing update
UPDATE_CRASH_WINDOW = 90     # seconds — if app dies within this after update, roll back
UPDATE_SKIP_EXPIRY = 86400   # seconds (24h) — skipped releases become retryable
UPDATE_STARTED_FILE = os.path.expanduser("~/.config/local-models/update_started")
UPDATE_SKIPPED_FILE = os.path.expanduser("~/.config/local-models/update_skipped")
UPDATE_PRE_HASH_FILE = os.path.expanduser("~/.config/local-models/update_pre_hash")
LAUNCH_ATTEMPTED_FILE = os.path.expanduser("~/.config/local-models/launch_attempted")
PRE_UPDATE_HEALTH_FILE = os.path.expanduser("~/.config/local-models/pre_update_health.json")
MCP_LOG_FILE = "/tmp/local-models-mcp.log"

MODEL_PREFS_FILE = os.path.expanduser("~/.config/local-models/model_preferences.json")
AUTH_TOKEN_CACHE = os.path.expanduser("~/.config/local-models/mcp_auth_token")


def _load_auth_token() -> str:
    """Read MCP_AUTH_TOKEN from the env, falling back to the on-disk cache.

    Pure read — does NOT mutate os.environ.  When the menubar spawns
    profile-server it injects the token into the child env explicitly;
    leaking it into the parent process's environ would pollute pytest
    runs (test_core imports this module, which would then leak the real
    token into every other test's idea of os.environ)."""
    token = os.environ.get("MCP_AUTH_TOKEN", "").strip()
    if token:
        return token
    try:
        with open(AUTH_TOKEN_CACHE) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return ""


_AUTH_TOKEN = _load_auth_token()


def load_force_local():
    """Read the FORCE_LOCAL flag from mode.conf."""
    if os.path.exists(MODE_CONF):
        with open(MODE_CONF) as f:
            for line in f:
                line = line.strip()
                if line.startswith("FORCE_LOCAL="):
                    return line.split("=", 1)[1].strip('"').strip("'") == "true"
    return False


def save_force_local(force: bool):
    """Write the FORCE_LOCAL flag to mode.conf."""
    os.makedirs(os.path.dirname(MODE_CONF), exist_ok=True)
    with open(MODE_CONF, "w") as f:
        f.write(f'FORCE_LOCAL={"true" if force else "false"}\n')


def load_network_conf():
    """Parse the shell-style network.conf into a dict."""
    conf = {
        "MODEL_SERVER_HOST": "",
        "OLLAMA_PORT": "11434",
        "MLX_PORT": "8000",
        "PROBE_TIMEOUT": "2",
    }
    if os.path.exists(NETWORK_CONF):
        with open(NETWORK_CONF) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    conf[key.strip()] = val.strip().strip('"').strip("'")
    return conf


def get_ram_gb():
    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
    return int(out.strip()) // (1024 ** 3)


def is_desktop():
    return get_ram_gb() >= 256


def http_get_json(url: str, timeout: int = 3) -> dict | list | None:
    """Fetch JSON from a URL, return None on failure."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "LocalModelsMenubar/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# Cache for resolve_desktop_tailscale (avoids repeated slow `tailscale status`)
_ts_cache = {"ip": "", "ts": 0}
_TS_CACHE_TTL = 30


def resolve_desktop_tailscale(hostname: str) -> tuple[str, str]:
    """Resolve the desktop's Tailscale IP and FQDN from the peer list.

    Returns (ip, fqdn) tuple. Results are cached for 30 seconds.
    """
    if not hostname:
        return "", ""
    now = time.time()
    if now - _ts_cache["ts"] < _TS_CACHE_TTL and _ts_cache.get("host") == hostname:
        return _ts_cache["ip"], _ts_cache.get("fqdn", "")
    ip = ""
    fqdn = ""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, encoding='utf-8', timeout=5)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data.get("BackendState") == "Running":
                for peer in data.get("Peer", {}).values():
                    if peer.get("HostName") == hostname:
                        fqdn = peer.get("DNSName", "").rstrip(".")
                        for addr in peer.get("TailscaleIPs", []):
                            if "." in addr:
                                ip = addr
                                break
                        break
    except Exception as e:
        logging.warning("resolve_desktop_tailscale exception: %s", e)
    _ts_cache["ip"] = ip
    _ts_cache["fqdn"] = fqdn
    _ts_cache["host"] = hostname
    _ts_cache["ts"] = now
    return ip, fqdn


def probe_service(base_url: str, timeout: int = 2) -> bool:
    """Check if a service is responding."""
    try:
        req = urllib.request.Request(f"{base_url}/api/version"
                                     if "11434" in base_url
                                     else f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def probe_port(port: int, host: str = "127.0.0.1", timeout: int = 1) -> bool:
    """Check if something is listening on a TCP port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def process_is_running(name: str) -> bool:
    """Check if a process with the given name fragment is running."""
    try:
        result = subprocess.run(["pgrep", "-f", name],
                                capture_output=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


def get_ollama_models(base_url: str, timeout: int = 3) -> list[str]:
    """Get list of model names from Ollama."""
    data = http_get_json(f"{base_url}/api/tags", timeout=timeout)
    if data and "models" in data:
        return [m["name"] for m in data["models"]]
    return []


def get_mlx_models(base_url: str, timeout: int = 3) -> list[str]:
    """Get list of model names from MLX-OpenAI-Server."""
    data = http_get_json(f"{base_url}/v1/models", timeout=timeout)
    if data and "data" in data:
        return [m["id"] for m in data["data"]]
    return []


def get_mcp_models(mcp_url: str = "http://127.0.0.1:8100", timeout: int = 3) -> list[str]:
    """Get HF-backed model names from the MCP server."""
    data = http_get_json(f"{mcp_url}/api/mcp-models", timeout=timeout)
    if data and "models" in data:
        return data["models"]
    return []


# ---------------------------------------------------------------------------
# MCP tool preferences
# ---------------------------------------------------------------------------

from lib.models import (
    STANDARD_TASKS, SPECIAL_TASKS, TASK_FILTERS, KNOWN_ACTIVE_PARAMS,
    ALWAYS_EXCLUDE, active_params_b, model_matches_filter,
    MCP_PREFS_FILE as _MCP_PREFS_PATH, CLAUDE_CONFIG_FILE,
    validate_network_conf,
)
MCP_PREFS_FILE = str(_MCP_PREFS_PATH)

MCP_TASK_LABELS = {k: v for k, v in STANDARD_TASKS.items()}
MCP_TASK_FILTERS = TASK_FILTERS

MCP_DEFAULT_PREFS = {
    "code": ["qwen3-coder:480b", "qwen3-coder", "qwen2.5-coder:32b", "glm-4.7-flash", "qwen3.5"],
    "general": ["qwen3.5", "glm-4.7-flash", "nemotron-3-super", "qwen3.5-fast"],
    "translation": ["cogito-2.1", "qwen3.5", "glm-4.7-flash"],
    "reasoning": ["deepseek-r1:671b", "cogito-2.1", "nemotron-3-super", "qwen3.5-397b-8bit", "qwen3.5", "glm-4.7-flash"],
    "long_context": ["qwen3.5", "nemotron-3-super", "glm-4.7-flash", "deepseek-r1:671b"],
}


def _model_matches_filter(model_name: str, raw_info: dict, task_filter: dict) -> bool:
    return model_matches_filter(
        model_name,
        raw_info.get("active", 0),
        raw_info.get("ctx", 0),
        task_filter,
    )

MCP_SPECIAL_TASKS = SPECIAL_TASKS


# ---------------------------------------------------------------------------
# Update detection (git tags)
# ---------------------------------------------------------------------------

def get_version(ref="HEAD"):
    """Get the version tag describing *ref*.

    Returns the most recent reachable tag (e.g. "v1.0.0"), or the tag with a
    short distance suffix (e.g. "v1.0.0+3") if HEAD is ahead of the tag.
    Returns "dev" if no tags exist.
    """
    try:
        desc = subprocess.check_output(
            ["git", "-C", REPO_DIR, "describe", "--tags", "--always", ref],
            text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        # git describe returns "v1.0.0" if exactly on tag, or
        # "v1.0.0-3-gabcdef0" if 3 commits past. Normalize the latter.
        if "-" in desc and desc.startswith("v"):
            parts = desc.rsplit("-", 2)
            if len(parts) == 3:
                return f"{parts[0]}+{parts[1]}"
        return desc
    except Exception:
        return "dev"


def get_latest_remote_tag():
    """Return (tag_name, tag_hash) for the latest semver tag on origin.

    Tags are sorted by version (v1.0.0 < v1.1.0 < v2.0.0). Returns ("", "")
    if no version tags exist.
    """
    try:
        result = subprocess.run(
            ["git", "-C", REPO_DIR, "tag", "--list", "v*", "--sort=-version:refname"],
            capture_output=True, text=True, encoding='utf-8', timeout=5)
        tags = result.stdout.strip().splitlines()
        if not tags:
            return "", ""
        latest = tags[0]
        tag_hash = subprocess.check_output(
            ["git", "-C", REPO_DIR, "rev-parse", f"{latest}^{{commit}}"],
            text=True, timeout=5).strip()
        return latest, tag_hash
    except Exception:
        return "", ""


def check_repo_update_available():
    """Check if a newer tagged release exists on origin.

    Returns (behind_count, remote_version, remote_tag_hash).
    behind_count is 1 if a newer tag exists, 0 otherwise (exact commit count
    isn't meaningful for tag-based updates).
    """
    try:
        fetch = subprocess.run(
            ["git", "-C", REPO_DIR, "fetch", "--quiet", "--tags", "--force", "--prune", "--prune-tags"],
            capture_output=True, text=True, encoding='utf-8', timeout=15)
        if fetch.returncode != 0:
            logging.warning("git fetch failed: %s", fetch.stderr.strip())
            return 0, "", ""

        latest_tag, tag_hash = get_latest_remote_tag()
        if not latest_tag:
            return 0, "", ""

        # Check if HEAD is already at or past this tag
        current_hash = subprocess.check_output(
            ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
            text=True, timeout=5).strip()
        if current_hash == tag_hash:
            return 0, "", ""

        # Check if the tag is an ancestor of HEAD (already past it)
        merge_base = subprocess.run(
            ["git", "-C", REPO_DIR, "merge-base", "--is-ancestor",
             tag_hash, "HEAD"],
            capture_output=True, timeout=5)
        if merge_base.returncode == 0:
            return 0, "", ""  # HEAD is ahead of or at the latest tag

        return 1, latest_tag, tag_hash
    except Exception as e:
        logging.warning("Update check failed: %s", e)
    return 0, "", ""


def verify_tag_signature(tag):
    """Verify a git tag has a valid GPG/SSH signature. Returns (ok, detail)."""
    result = subprocess.run(
        ["git", "-C", REPO_DIR, "tag", "-v", tag],
        capture_output=True, text=True, encoding='utf-8', timeout=10)
    if result.returncode == 0:
        return True, "signature verified"
    # Check if gpg/ssh-keygen is missing vs. bad signature
    stderr = result.stderr.lower()
    if "no public key" in stderr or "could not verify" in stderr:
        return False, f"no trusted public key for {tag}"
    if "not signed" in stderr or "no signature" in stderr:
        return False, f"tag {tag} is not signed"
    return False, result.stderr.strip()[:200]


def _update_allowed_signers(target_ref):
    """Install allowed_signers from a verified tag/ref.

    MUST be called AFTER verify_tag_signature succeeds against the existing
    trust root. Calling this beforehand inverts the trust model: any pushed
    tag could ship its own allowed_signers and self-approve.

    Key rotations should ship in a tag signed by the outgoing key. Old
    installs upgrade through that tag, then the new key is installed.
    """
    try:
        result = subprocess.run(
            ["git", "-C", REPO_DIR, "show",
             f"{target_ref}:config/git/allowed_signers"],
            capture_output=True, text=True, encoding='utf-8', timeout=5)
        if result.returncode != 0 or not result.stdout.strip():
            return
        signers_path = os.path.expanduser("~/.config/git/allowed_signers")
        os.makedirs(os.path.dirname(signers_path), exist_ok=True)
        with open(signers_path, "w") as f:
            f.write(result.stdout)
        subprocess.run(
            ["git", "-C", REPO_DIR, "config",
             "gpg.ssh.allowedSignersFile", signers_path],
            capture_output=True, timeout=5)
    except Exception as e:
        logging.debug("Could not update allowed_signers from %s: %s", target_ref, e)


def apply_repo_update(target_tag):
    """Check out a tagged release. Returns (success, output).

    Verifies the tag signature against the *existing* allowed_signers,
    then installs any new allowed_signers from the verified tag, then
    force-checks out the tag (detached HEAD) and cleans untracked files.
    End users don't have local changes to preserve — the repo IS the
    installed app.
    """
    try:
        sig_ok, sig_detail = verify_tag_signature(target_tag)
        if not sig_ok:
            logging.warning("Refusing unsigned update %s: %s", target_tag, sig_detail)
            return False, f"Tag signature verification failed: {sig_detail}"

        # Tag is trusted under the current allowed_signers. Now it's safe
        # to roll forward any signing-key rotation it carries.
        _update_allowed_signers(target_tag)

        checkout = subprocess.run(
            ["git", "-C", REPO_DIR, "checkout", "--force", target_tag],
            capture_output=True, text=True, encoding='utf-8', timeout=30)
        if checkout.returncode != 0:
            logging.error("git checkout --force %s failed: %s",
                          target_tag, checkout.stderr.strip())
            return False, checkout.stderr.strip()
        # Remove untracked files that might conflict with the new version
        subprocess.run(
            ["git", "-C", REPO_DIR, "clean", "-fd", "--exclude=*.log"],
            capture_output=True, timeout=10)
        logging.info("Checked out %s (force)", target_tag)
        return True, f"Checked out {target_tag}"
    except Exception as e:
        logging.error("apply_repo_update exception: %s", e)
        return False, str(e)


_mcp_configured_cache = {"val": None, "ts": 0}


def is_mcp_configured():
    """Check if local-models MCP is registered in Claude config. Cached 60s."""
    now = time.time()
    if _mcp_configured_cache["val"] is not None and now - _mcp_configured_cache["ts"] < 60:
        return _mcp_configured_cache["val"]
    result = False
    if os.path.exists(MCP_TOOLS_FILE):
        try:
            with open(MCP_TOOLS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            result = "local-models" in data.get("mcpServers", {})
        except Exception:
            pass
    _mcp_configured_cache["val"] = result
    _mcp_configured_cache["ts"] = now
    return result


def load_mcp_prefs():
    """Load {task: model_name} overrides."""
    if os.path.exists(MCP_PREFS_FILE):
        try:
            with open(MCP_PREFS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


PROFILES_FILE = os.path.expanduser("~/.config/local-models/profiles.json")


def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"active": None, "profiles": {}}


def save_profiles(data):
    os.makedirs(os.path.dirname(PROFILES_FILE), exist_ok=True)
    with open(PROFILES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def pick_profile_for_ram(ram_gb, profiles):
    """Pick the best profile for the given RAM.

    Uses max_ram_gb from each profile and picks the largest that fits.
    Falls back to 'laptop' or the first profile.
    """
    candidates = []
    for name, prof in profiles.items():
        max_ram = prof.get("max_ram_gb", 0)
        if max_ram and max_ram <= ram_gb:
            candidates.append((max_ram, name))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return "laptop" if "laptop" in profiles else next(iter(profiles), None)


def save_mcp_prefs(prefs):
    """Save MCP task→model preferences."""
    os.makedirs(os.path.dirname(MCP_PREFS_FILE), exist_ok=True)
    with open(MCP_PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)


# Provider icon file paths (set after ICONS_DIR is defined)
PROVIDER_ICON_PATH = {
    "ollama": os.path.join(ICONS_DIR, "ollama.png"),
    "mlx": os.path.join(ICONS_DIR, "mlx.png"),
    "anthropic": os.path.join(ICONS_DIR, "claude.png"),
}

# Task type descriptions and filtering rules for the menu
ROLE_LABELS = {
    "default": "Routine Tasks",
    "think": "Complex Reasoning",
    "background": "Background",
    "longContext": "Long Context",
    "webSearch": "Web Search",
    "image": "Vision",
}

def load_role_filters():
    """Load role filter config (unused, kept for compatibility)."""
    return {}


def model_fits_role(role: str, provider: str, total: float, active: float, ctx: int, has_vision: bool, filters: dict) -> bool:
    """Check if a model is appropriate for a given role based on filter config."""
    if role not in filters:
        return True  # unknown role — show everything

    f = filters[role]

    # Provider filter
    allowed_providers = f.get("providers")
    if allowed_providers and provider not in allowed_providers:
        return False

    # Min active params
    min_active = f.get("min_active_params_b", 0) or 0
    if active > 0 and active < min_active:
        return False

    # Max active params
    max_active = f.get("max_active_params_b")
    if max_active is not None and active > max_active:
        return False

    # Min context
    min_ctx = f.get("min_context", 0) or 0
    if ctx > 0 and ctx < min_ctx:
        return False

    # Vision requirement
    if f.get("requires_vision") and not has_vision:
        return False

    return True


def query_ollama_all_models(base_url, timeout=5):
    """Query Ollama /api/tags for all installed models with basic details.

    Returns {model_name: {"params": str, "family": str}} for all installed models.
    """
    data = http_get_json(f"{base_url}/api/tags", timeout=timeout)
    if not data:
        return {}
    result = {}
    for m in data.get("models", []):
        name = m.get("name", "")
        details = m.get("details", {})
        result[name] = {
            "params": details.get("parameter_size", ""),
            "family": details.get("family", ""),
        }
    return result




def query_ollama_model_detail(base_url, model_name, timeout=5):
    """Query Ollama /api/show for full model architecture info."""
    try:
        data = json.dumps({"name": model_name}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/show",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            info = json.loads(resp.read())

        model_info = info.get("model_info", {})
        details = info.get("details", {})
        family = details.get("family", "")

        def _get(suffix, default=None):
            for k, v in model_info.items():
                if k.endswith(suffix) and ".vision." not in k:
                    return v
            return default

        # Total params
        total_raw = model_info.get("general.parameter_count", 0)
        total_b = total_raw / 1e9 if total_raw else 0
        if not total_b:
            ps = details.get("parameter_size", "")
            try:
                total_b = float(ps.rstrip("B"))
            except (ValueError, AttributeError):
                pass

        # Context length
        ctx = 0
        for k, v in model_info.items():
            if "context_length" in k:
                ctx = int(v)
                break

        # MoE detection
        expert_count = _get(".expert_count")
        expert_used = _get(".expert_used_count")
        if expert_count:
            expert_count = int(expert_count)
        if expert_used:
            expert_used = int(expert_used)

        # Active params — delegate MoE computation to shared library
        expert_ffn = _get(".expert_feed_forward_length", 0)
        embed_len = _get(".embedding_length", 0)
        block_count = _get(".block_count", 0)
        active_b = active_params_b(
            model_name, total_b, family,
            expert_count, expert_used,
            expert_ffn=expert_ffn or 0,
            embed_len=embed_len or 0,
            block_count=block_count or 0,
        )

        has_vision = any("vision" in k for k in model_info)

        return {
            "total_params": round(total_b),
            "active_params": round(active_b),
            "context": ctx,
            "expert_count": expert_count,
            "expert_used": expert_used,
            "has_vision": has_vision,
        }
    except Exception:
        return None


def match_ollama_model(pref_name, installed_models):
    """Match a preference model name (e.g. 'qwen3.5') to an installed Ollama model.

    Tries exact match, then :latest tag, then exact base name match with
    any tag. Only falls back to prefix match as a last resort, preferring
    the variant whose tag best matches the preference name.
    """
    # Exact match (includes tag)
    if pref_name in installed_models:
        return pref_name
    # With :latest tag
    if f"{pref_name}:latest" in installed_models:
        return f"{pref_name}:latest"

    # If preference name has a tag (e.g. "qwen3.5:35b-a3b"), try matching base:tag
    if ":" in pref_name:
        # The exact name wasn't found — no good match
        return None

    # preference name has no tag (e.g. "qwen3.5") — find installed models with same base
    matches = [n for n in installed_models
               if n.split(":")[0] == pref_name]
    if matches:
        # Prefer :latest, then pick the largest
        for m in matches:
            if m.endswith(":latest"):
                return m
        def param_num(name):
            p = installed_models[name].get("params", "0")
            try:
                return float(p.rstrip("B"))
            except ValueError:
                return 0
        return max(matches, key=param_num)
    return None


def query_mlx_model_info_from_config():
    """Read the MLX server YAML config AND each model's HuggingFace config.json.

    Returns {served_name: {total_params, active_params, context}} with
    ground-truth values from the model architecture configs.
    """
    import yaml
    info = {}
    config_path = os.path.expanduser("~/.config/mlx-server/config.yaml")
    if not os.path.exists(config_path):
        return info

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception:
        return info

    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")

    for m in config.get("models", []):
        name = m.get("served_model_name", "")
        model_path = m.get("model_path", "")
        yaml_ctx = m.get("context_length", 0)

        total_b = 0.0
        active_b = 0.0
        ctx = yaml_ctx

        # Try to read the model's config.json from HuggingFace cache
        cache_dir_name = f"models--{model_path.replace('/', '--')}"
        cache_dir = os.path.join(hf_cache, cache_dir_name)

        hf_config = None
        if os.path.exists(cache_dir):
            # Find config.json in the latest snapshot
            snapshots_dir = os.path.join(cache_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                for snap in sorted(os.listdir(snapshots_dir), reverse=True):
                    cfg_path = os.path.join(snapshots_dir, snap, "config.json")
                    if os.path.exists(cfg_path):
                        try:
                            with open(cfg_path) as f:
                                hf_config = json.load(f)
                        except Exception:
                            pass
                        break

        if hf_config:
            # config.json may have top-level or nested text_config
            tc = hf_config.get("text_config", hf_config)

            num_experts = tc.get("num_experts", tc.get("num_local_experts"))
            num_experts_per_tok = tc.get("num_experts_per_tok",
                                         tc.get("num_experts_per_token"))
            hidden_size = tc.get("hidden_size", 0)
            num_layers = tc.get("num_hidden_layers", 0)
            intermediate_size = tc.get("intermediate_size",
                                       tc.get("moe_intermediate_size", 0))
            shared_expert_size = tc.get("shared_expert_intermediate_size", 0)
            vocab_size = tc.get("vocab_size", 0)
            num_heads = tc.get("num_attention_heads", 0)
            num_kv_heads = tc.get("num_key_value_heads", num_heads)
            head_dim = tc.get("head_dim", hidden_size // num_heads if num_heads else 0)

            # Context from config (override YAML if available)
            hf_ctx = tc.get("max_position_embeddings", 0)
            if hf_ctx and (not ctx or hf_ctx > ctx):
                ctx = hf_ctx

            # Compute total params estimate from architecture
            # Embedding: vocab_size * hidden_size
            embed_params = vocab_size * hidden_size

            # Per-layer attention params
            attn_params_per_layer = (
                hidden_size * num_heads * head_dim +           # Q
                hidden_size * num_kv_heads * head_dim +        # K
                hidden_size * num_kv_heads * head_dim +        # V
                num_heads * head_dim * hidden_size             # O
            )

            if num_experts and num_experts > 1:
                # MoE: each expert has its own FFN
                expert_ffn_params = num_experts * 3 * hidden_size * intermediate_size
                shared_ffn_params = 3 * hidden_size * shared_expert_size if shared_expert_size else 0
                router_params = hidden_size * num_experts
                ffn_per_layer = expert_ffn_params + shared_ffn_params + router_params

                total_params = embed_params + num_layers * (attn_params_per_layer + ffn_per_layer)
                total_b = total_params / 1e9

                # Active params: only num_experts_per_tok experts fire
                if num_experts_per_tok:
                    active_expert_ffn = num_experts_per_tok * 3 * hidden_size * intermediate_size
                    active_ffn_per_layer = active_expert_ffn + shared_ffn_params + router_params
                    active_params = embed_params + num_layers * (attn_params_per_layer + active_ffn_per_layer)
                    active_b = active_params / 1e9
                else:
                    active_b = total_b
            else:
                # Dense model
                ffn_per_layer = 3 * hidden_size * intermediate_size
                total_params = embed_params + num_layers * (attn_params_per_layer + ffn_per_layer)
                total_b = total_params / 1e9
                active_b = total_b

        # Fallback: if config.json wasn't cached, parse from model path
        if not total_b and model_path:
            path_lower = model_path.lower()
            moe = re.search(r"(\d+)b[_-]a(\d+)b", path_lower)
            if moe:
                total_b = float(moe.group(1))
                active_b = float(moe.group(2))
            else:
                m = re.search(r"(\d+)b", path_lower)
                if m:
                    total_b = float(m.group(1))
                    active_b = total_b

        # Detect vision capability
        has_vision = False
        if hf_config:
            has_vision = "vision_config" in hf_config or "vision_config" in hf_config.get("text_config", {})

        info[name] = {
            "total_params": round(total_b),
            "active_params": round(active_b),
            "context": ctx,
            "model_path": model_path,
            "has_vision": has_vision,
        }

    return info


def format_context(ctx):
    """Format context length nicely: 262144 → '256K'."""
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    elif ctx >= 1024:
        return f"{ctx // 1024}K"
    elif ctx > 0:
        return str(ctx)
    return ""


class ModelInfoCache:
    """Caches model metadata queried from providers.

    Stores (total_params_B, active_params_B, context, label) per model.
    """

    def __init__(self):
        self._cache = {}        # "provider:model" -> (name, detail) tuple
        self._sort_vals = {}    # "provider:model" -> total_params (float)
        self._raw = {}          # "provider:model" -> {total, active, ctx, has_vision}
        self._available = set() # set of "provider:model" keys that are actually usable
        self._ollama_models = None  # lazily fetched
        self._ollama_url = None     # URL used to fetch _ollama_models
        self._ollama_vision = set() # ollama models with vision capability
        self._role_filters = load_role_filters()

    def populate(self, ccr_models, ollama_url, mlx_config_info, mlx_live_models):
        """Bulk-populate cache for all preference models.

        mlx_live_models: list of model IDs currently served by MLX-OpenAI-Server.
        """
        # Re-fetch if the Ollama URL changed (e.g. switched from local to remote)
        if self._ollama_models is None or self._ollama_url != ollama_url:
            self._ollama_models = query_ollama_all_models(ollama_url)
            self._ollama_url = ollama_url
            self._available.clear()

        for provider, model in ccr_models:
            key = f"{provider}:{model}"
            if key in self._cache:
                continue

            icon = ""  # icons applied via set_icon on the MenuItem
            total = 0.0
            active = 0.0
            ctx = 0

            if provider == "ollama":
                matched = match_ollama_model(model, self._ollama_models)
                if matched:
                    self._available.add(key)
                    detail = query_ollama_model_detail(ollama_url, matched)
                    if detail:
                        total = detail["total_params"]
                        active = detail["active_params"]
                        ctx = detail["context"]
                        if detail.get("has_vision"):
                            if not hasattr(self, '_ollama_vision'):
                                self._ollama_vision = set()
                            self._ollama_vision.add(model)

            elif provider == "mlx":
                # Only show if the MLX server is actually serving this model
                if mlx_live_models and model in mlx_live_models:
                    self._available.add(key)
                if mlx_config_info and model in mlx_config_info:
                    minfo = mlx_config_info[model]
                    total = minfo.get("total_params", 0)
                    active = minfo.get("active_params", 0)
                    ctx = minfo.get("context", 0)

            elif provider == "anthropic":
                self._available.add(key)
                ctx = 1_000_000
                if "opus" in model:
                    total = 2000
                elif "sonnet" in model:
                    total = 800
                active = total

            # Format: "Total/Active • Ctx" for MoE, "Total • Ctx" for dense
            parts = []
            if total > 0 and active > 0 and active != total:
                parts.append(f"{total:.0f}B/{active:.0f}B")
            elif total > 0:
                parts.append(f"{total:.0f}B")
            ctx_str = format_context(ctx)
            if ctx_str:
                parts.append(ctx_str)

            detail = f"{' • '.join(parts)}" if parts else ""
            # Detect vision capability
            has_vision = False
            if provider == "anthropic":
                has_vision = True  # Claude always supports vision
            elif provider == "mlx" and mlx_config_info and model in mlx_config_info:
                has_vision = mlx_config_info[model].get("has_vision", False)
            elif provider == "ollama" and hasattr(self, '_ollama_vision'):
                has_vision = model in self._ollama_vision

            self._cache[key] = (model, detail)
            self._sort_vals[key] = total
            self._raw[key] = {"total": total, "active": active, "ctx": ctx,
                              "has_vision": has_vision}

    def get_label(self, provider, model):
        """Returns (name, detail) tuple."""
        key = f"{provider}:{model}"
        return self._cache.get(key, (model, ""))

    def is_available(self, provider, model):
        """Returns True if this model is actually installed/serving."""
        return f"{provider}:{model}" in self._available

    def fits_role(self, provider, model, role):
        """Returns True if this model is appropriate for the given task role."""
        key = f"{provider}:{model}"
        raw = self._raw.get(key, {})
        return model_fits_role(
            role, provider,
            raw.get("total", 0), raw.get("active", 0),
            raw.get("ctx", 0), raw.get("has_vision", False),
            self._role_filters,
        )

    def sort_key(self, provider_model):
        """Sort by total params descending."""
        provider, model = provider_model
        key = f"{provider}:{model}"
        return -self._sort_vals.get(key, 0)


# ---------------------------------------------------------------------------
# Routing preferences (which model handles each task type)
# ---------------------------------------------------------------------------

def load_routing_prefs():
    """Load {role: "provider,model"} overrides. Falls back to preference defaults."""
    if os.path.exists(MODEL_PREFS_FILE):
        try:
            with open(MODEL_PREFS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_routing_prefs(prefs):
    """Save routing preferences to disk."""
    os.makedirs(os.path.dirname(MODEL_PREFS_FILE), exist_ok=True)
    with open(MODEL_PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def _ensure_ollama_library_path():
    """Ollama's MLX image-gen runner dlopens libmlxc.dylib via OLLAMA_LIBRARY_PATH.
    The Electron launcher is supposed to set it, but has been observed to drop
    it — breaking every z-image / flux request with 'libmlxc.dylib not found'
    until Ollama restarts. Push the value into launchd's session env so the
    next Ollama launch (auto-update, restart, reboot) inherits it."""
    resources = "/Applications/Ollama.app/Contents/Resources"
    if not os.path.isdir(resources):
        return
    subprocess.run(
        ["launchctl", "setenv", "OLLAMA_LIBRARY_PATH", resources],
        check=False, timeout=5,
    )


class LocalModelsApp(rumps.App):
    def __init__(self):
        icon = ICON_PATH if os.path.exists(ICON_PATH) else None
        super().__init__("Local Models", icon=icon, template=True,
                         quit_button=None)

        _ensure_ollama_library_path()
        validate_network_conf(logger=logging.getLogger())
        self.conf = load_network_conf()
        self.desktop = is_desktop()
        self.ram_gb = get_ram_gb()
        self.ollama_port = self.conf["OLLAMA_PORT"]
        self.mlx_port = self.conf["MLX_PORT"]
        self.probe_timeout = int(self.conf["PROBE_TIMEOUT"])
        self._profile_fixed_port = int(self.conf.get("PROFILE_PORT", "8101"))
        self.ts_hostname = self.conf.get("TAILSCALE_HOSTNAME", "")

        # Desktop IP/FQDN resolved via Tailscale on first probe (laptops only)
        self.desktop_ip = ""
        self.desktop_fqdn = ""
        self.ollama_remote = OLLAMA_LOCAL
        self.mlx_remote = MLX_LOCAL

        # Mode preference (laptops only): user can force local even when
        # the desktop is reachable.  On startup, if not forced, prefer remote.
        self.force_local = load_force_local() if not self.desktop else False
        self.remote_reachable = False  # tracked every poll for greying out

        # State (protected by _lock for cross-thread access)
        self._lock = threading.Lock()
        self.mode = "unknown"          # server, client, offline, stopped
        self.ollama_ok = False
        self.mlx_ok = False
        self.ollama_loading = False    # process exists but not responding
        self.mlx_loading = False
        self.ollama_models = []
        self.mlx_models = []
        self.servers_started = False
        self.mcp_configured = is_mcp_configured()
        self.mcp_prefs = load_mcp_prefs()
        self.model_info_cache = ModelInfoCache()
        self.mlx_config_info = query_mlx_model_info_from_config()
        self.last_update_check = 0
        self.update_available = 0      # commits behind
        self._update_defer_count = 0   # consecutive MCP idle deferrals
        self.app_version = get_version()
        self._launch_hash = self._get_head_hash()
        self.app_ready = False         # set True once run loop starts
        self._health_checked = False   # post-update health comparison done
        self._rolled_back = False      # set True if startup rollback fired

        # Profile viewer / tool tester state
        self.profile_server = None
        self.profile_server_mode = None
        self.profile_port = None
        self.profile_window = None
        self.tools_window = None
        self.activity_window = None
        self._win_delegate = None
        self._tools_delegate = None

        # Menu items
        self.menu_status = rumps.MenuItem("Starting…")
        self.menu_mode_remote = rumps.MenuItem(
            "Remote", callback=self._select_remote)
        self.menu_mode_local = rumps.MenuItem(
            "Local", callback=self._select_local)
        self.menu_ollama = rumps.MenuItem("Ollama …")
        self.menu_ollama_restart = rumps.MenuItem(
            "Restart Ollama", callback=self._restart_ollama)
        self.menu_ollama.add(self.menu_ollama_restart)
        self.menu_mlx = rumps.MenuItem("MLX …")
        self.menu_mlx_restart = rumps.MenuItem(
            "Restart MLX", callback=self._restart_mlx)
        self.menu_mlx.add(self.menu_mlx_restart)
        self.menu_mcp = rumps.MenuItem("MCP …")
        self.menu_mcp_restart = rumps.MenuItem(
            "Restart MCP", callback=self._restart_mcp)
        self.menu_mcp.add(self.menu_mcp_restart)
        self.mcp_models = []  # populated on first refresh from MCP server
        self.menu_profiles = rumps.MenuItem("Models",
                                           callback=self.open_profiles)
        self.menu_playground = rumps.MenuItem("Playground",
                                             callback=self.open_tools)
        self.menu_remote_access = rumps.MenuItem("Remote Access",
                                                callback=self._toggle_remote_access)
        self.menu_tools_sub = rumps.MenuItem("Tools")
        self.menu_tools_sub.add(rumps.MenuItem("Activity Log",
                                              callback=self.open_activity))
        self.menu_tools_sub.add(rumps.MenuItem("Diagnostics",
                                              callback=self.open_diagnostics))
        self.menu_tools_sub.add(None)
        self.menu_tools_sub.add(rumps.MenuItem("Restart",
                                              callback=self.restart_app))
        self.menu_version = rumps.MenuItem(self.app_version)
        self.menu_quit = rumps.MenuItem("Quit", callback=self.quit_app)

        self.remote_access_enabled = self._load_remote_access_pref()

        menu_items = []
        if self.desktop:
            menu_items.append(self.menu_status)
            menu_items.append(self.menu_remote_access)
        else:
            menu_items += [self.menu_mode_remote, self.menu_mode_local]
        menu_items += [
            None,
            self.menu_ollama,
            self.menu_mlx,
            self.menu_mcp,
            None,
            self.menu_profiles,
            self.menu_playground,
            None,
            self.menu_tools_sub,
            None,
            self.menu_version,
            None,
            self.menu_quit,
        ]
        self.menu = menu_items

        # Easter egg: periodic cute notifications (opt-in via config)
        self._next_woof = 0
        self._schedule_woof()

        # Defer startup to first timer tick (NSMenu isn't ready during __init__)
        self.timer = rumps.Timer(self._on_tick, POLL_INTERVAL)
        self.timer.start()

    def _on_tick(self, _):
        """Timer callback. Handles first-run initialization and periodic refresh."""
        if not self.app_ready:
            self.app_ready = True
            self._startup_rollback_check()
            threading.Thread(target=self._start_services_bg, daemon=True).start()
            self._schedule_update_check()
            # Schedule clearing the update_started marker after the crash window
            threading.Timer(
                UPDATE_CRASH_WINDOW, self._mark_startup_healthy).start()
            return
        self.refresh(None)

    def _start_services_bg(self):
        """Background thread: start services, then do first poll inline.

        The post-update health check used to live here, but services
        (especially MLX) can take 30–60s to warm up — comparing right
        after spawn would always look like a regression even when the
        update was fine. The check now runs from `_finish_refresh` once
        services have settled.
        """
        self.start_services()
        with self._lock:
            if self.desktop:
                self._refresh_server_mode()
            else:
                self._refresh_client_mode()
        self._main_thread_update()

    # -------------------------------------------------------------------
    # Service management
    # -------------------------------------------------------------------

    def _resolve_desktop(self):
        """Check if the desktop's MCP server is reachable via Tailscale.

        Returns True if reachable, False otherwise.
        """
        ts_ip, ts_fqdn = resolve_desktop_tailscale(self.ts_hostname)
        if not ts_ip:
            return False
        reachable = probe_port(8100, host=ts_ip, timeout=self.probe_timeout)
        if reachable:
            self.desktop_ip = ts_ip
            self.desktop_fqdn = ts_fqdn
            # Use HTTPS via Tailscale FQDN (tailscale serve proxies these ports)
            fqdn = self.desktop_fqdn or ts_ip
            self.ollama_remote = f"https://{fqdn}:{self.ollama_port}"
            self.mlx_remote = f"https://{fqdn}:{self.mlx_port}"
            return True
        return False

    def start_services(self):
        """Start local servers (or detect desktop)."""
        if self.desktop:
            self._start_local_servers()
        else:
            found = (not self.force_local
                     and self.ts_hostname
                     and self._resolve_desktop())
            if found:
                self.mode = "client"
                self.remote_reachable = True
            else:
                self._start_local_servers()
        self._start_mcp_server()
        self._ensure_active_profile()
        # Auto-start Tailscale serve and profile server for remote clients
        if self.desktop and self.remote_access_enabled:
            self._ensure_profile_server()
            self._start_tailscale_serve()

    def _ensure_active_profile(self):
        """Make sure a valid profile is active on startup."""
        data = load_profiles()
        active = data.get("active")
        profiles = data.get("profiles", {})
        if active and active in profiles:
            self._activate_profile(active)
            return
        best = pick_profile_for_ram(self.ram_gb, profiles)
        if best:
            self._activate_profile(best)

    def _start_local_servers(self):
        """Launch Ollama and MLX-OpenAI-Server via start-local-models."""
        try:
            env = os.environ.copy()
            # Ollama always binds localhost; tailscale serve handles remote access
            # Ensure Homebrew is on PATH — launchd gives a minimal PATH
            if "/opt/homebrew/bin" not in env.get("PATH", ""):
                env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"
            self._startup_log = open("/tmp/local-models-startup.log", "w")
            subprocess.Popen(
                [os.path.expanduser("~/.local/bin/start-local-models")],
                env=env,
                stdout=self._startup_log,
                stderr=self._startup_log,
            )
            self.servers_started = True
            self._last_restart_attempt = time.time()
            self.mode = "server" if self.desktop else "offline"
            if self.desktop:
                self._prevent_sleep()
        except Exception as e:
            rumps.notification("Local Models", "Failed to start services", str(e))

    def _prevent_sleep(self):
        """Prevent system sleep while serving models (display may still sleep).

        Spawns caffeinate -s, which holds a power assertion until killed.
        """
        if getattr(self, '_caffeinate', None) is not None:
            return
        self._caffeinate = subprocess.Popen(
            ["caffeinate", "-s"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _allow_sleep(self):
        """Release the sleep prevention assertion."""
        proc = getattr(self, '_caffeinate', None)
        if proc is not None:
            proc.terminate()
            self._caffeinate = None

    def _start_mcp_server(self):
        """Launch the local MCP server, or point Claude Code at the desktop's."""
        if not self.desktop and not self.force_local and self.remote_reachable:
            # Remote mode: don't run a local server, point at the desktop
            self._stop_mcp_server()
            mcp_base = (f"https://{self.desktop_fqdn}:8100"
                        if self.desktop_fqdn
                        else f"http://{self.desktop_ip}:8100")
            self._configure_claude_mcp(f"{mcp_base}/mcp")
            return
        if getattr(self, '_mcp_proc', None) is not None:
            if self._mcp_proc.poll() is None:
                return
        # Look for an existing process on 8100. If it's our MCP and it's
        # serving requests, adopt it instead of restarting — every restart
        # drops every active SSE stream and Claude Code (every connected
        # session, on every machine) exits when its transport sees the
        # abrupt EOF. The most common trigger is auto-update: the menubar
        # exits, launchd respawns it, the new instance used to come up and
        # SIGTERM the still-healthy MCP its predecessor spawned. With
        # adoption the MCP keeps running across menubar restarts and the
        # connected sessions survive.
        orphan_pid = None
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", "tcp:8100"], text=True, stderr=subprocess.DEVNULL)
            for pid_str in out.strip().split():
                pid = int(pid_str)
                if pid != os.getpid():
                    orphan_pid = pid
                    break
        except (subprocess.CalledProcessError, ValueError):
            pass
        if orphan_pid is not None:
            if get_mcp_models("http://127.0.0.1:8100", timeout=2):
                self._mcp_proc_pid = orphan_pid
                self._configure_claude_mcp("http://127.0.0.1:8100/mcp")
                logging.info(
                    "Adopted existing local MCP server pid=%d on port 8100",
                    orphan_pid)
                return
            try:
                os.kill(orphan_pid, signal.SIGTERM)
                time.sleep(0.5)
            except OSError:
                pass
        env = os.environ.copy()
        if "/opt/homebrew/bin" not in env.get("PATH", ""):
            env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"
        # MCP always binds localhost; Tailscale serve handles remote access.
        # Allow Tailscale FQDN in Host header for proxied requests.
        if self.ts_hostname:
            ts_fqdn = getattr(self, 'desktop_fqdn', '') or self._get_own_fqdn()
            if ts_fqdn:
                env["MCP_ALLOWED_HOSTS"] = f"{ts_fqdn}:*"
        self._mcp_log = open("/tmp/local-models-mcp.log", "w")
        self._mcp_proc = subprocess.Popen(
            [os.path.expanduser("~/.local/bin/local-models-mcp-detect")],
            env=env,
            stdout=self._mcp_log,
            stderr=self._mcp_log,
            start_new_session=True,
        )
        self._configure_claude_mcp("http://127.0.0.1:8100/mcp")

    def _configure_claude_mcp(self, url):
        """Update ~/.claude.json to point local-models MCP at the given URL."""
        try:
            with open(MCP_TOOLS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            servers = data.setdefault("mcpServers", {})
            entry = servers.get("local-models", {})
            if entry.get("url") == url:
                return
            entry["type"] = "http"
            entry["url"] = url
            # Preserve existing auth headers
            servers["local-models"] = entry
            tmp = MCP_TOOLS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, MCP_TOOLS_FILE)
            logging.info("MCP config updated: %s", url)
        except Exception as e:
            logging.warning("Failed to update Claude MCP config: %s", e)

    _last_connection_notify = 0

    def _notify_connection(self, title, body):
        """Send a connectivity notification, debounced to 60s minimum interval."""
        now = time.time()
        if now - self._last_connection_notify < 60:
            return
        self._last_connection_notify = now
        try:
            rumps.notification("Super Puppy", title, body)
        except RuntimeError:
            pass

    def _get_own_fqdn(self):
        """Get this machine's Tailscale FQDN."""
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, encoding='utf-8', timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get("Self", {}).get("DNSName", "").rstrip(".")
        except Exception:
            pass
        return ""

    def _load_remote_access_pref(self):
        conf = os.path.expanduser("~/.config/local-models/remote_access.conf")
        if os.path.exists(conf):
            try:
                with open(conf) as f:
                    for line in f:
                        if line.strip() == "REMOTE_ACCESS=true":
                            return True
            except Exception:
                pass
        return False

    def _save_remote_access_pref(self, enabled):
        conf = os.path.expanduser("~/.config/local-models/remote_access.conf")
        os.makedirs(os.path.dirname(conf), exist_ok=True)
        with open(conf, "w") as f:
            f.write(f"REMOTE_ACCESS={'true' if enabled else 'false'}\n")

    def _toggle_remote_access(self, _):
        self.remote_access_enabled = not self.remote_access_enabled
        self._save_remote_access_pref(self.remote_access_enabled)
        if self.remote_access_enabled:
            self._ensure_profile_server()
            self._start_tailscale_serve()
        else:
            self._stop_tailscale_serve()
        status = "enabled" if self.remote_access_enabled else "disabled"
        try:
            rumps.notification("Super Puppy", f"Remote access {status}", "")
        except RuntimeError:
            pass
        self._update_menu()

    def _start_tailscale_serve(self):
        """Expose MCP and profile server ports via Tailscale serve."""
        # Reset any stale proxies first
        try:
            subprocess.run(
                ["tailscale", "serve", "reset"],
                capture_output=True, timeout=10)
        except Exception as e:
            logging.warning("tailscale serve reset failed: %s", e)
        for port in (8100, self._profile_fixed_port,
                     int(self.ollama_port), int(self.mlx_port)):
            try:
                result = subprocess.run(
                    ["tailscale", "serve", "--bg", "--https",
                     str(port), f"http://127.0.0.1:{port}"],
                    capture_output=True, text=True, encoding='utf-8', timeout=10)
                if result.returncode != 0:
                    logging.warning("tailscale serve %d failed: %s",
                                    port, result.stderr.strip())
                else:
                    logging.info("tailscale serve %d: ok", port)
            except Exception as e:
                logging.warning("tailscale serve %d failed: %s", port, e)

    def _stop_tailscale_serve(self):
        """Remove Tailscale serve proxies."""
        try:
            subprocess.run(
                ["tailscale", "serve", "reset"],
                capture_output=True, timeout=10)
        except Exception as e:
            logging.warning("tailscale serve reset failed: %s", e)

    def _copy_diagnostics(self, _):
        """Copy diagnostic info to clipboard for remote debugging."""
        mcp_proc = getattr(self, '_mcp_proc', None)
        mcp_alive = mcp_proc is not None and mcp_proc.poll() is None
        lines = [
            f"Super Puppy {self.app_version}",
            f"Mode: {self.mode}",
            f"Desktop: {self.desktop}",
            f"Force local: {self.force_local}",
            f"Remote reachable: {self.remote_reachable}",
            f"Desktop IP: {self.desktop_ip}",
            f"Desktop FQDN: {getattr(self, 'desktop_fqdn', '')}",
            f"Ollama: {'up' if self.ollama_ok else 'down'}",
            f"MLX: {'up' if self.mlx_ok else 'down'}",
            f"MCP process: {'alive' if mcp_alive else 'dead'}",
            f"MCP models: {len(self.mcp_models)}",
            f"Ollama models: {len(self.ollama_models)}",
            f"MLX models: {len(self.mlx_models)}",
            f"RAM: {self.ram_gb} GB",
            f"TS hostname: {self.ts_hostname}",
        ]
        if self.desktop:
            lines.append(f"Remote access: {self.remote_access_enabled}")
        # Last 5 log lines
        try:
            with open("/tmp/local-models-menubar.log", encoding="utf-8", errors="replace") as f:
                log_lines = f.readlines()[-5:]
            lines.append("\nRecent log:")
            lines.extend(l.rstrip() for l in log_lines)
        except Exception:
            pass
        text = "\n".join(lines)
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        rumps.notification("Super Puppy", "Diagnostics copied", "")

    def _copy_playground_url(self, _):
        """Copy the Playground URL to clipboard for phone/tablet setup."""
        if not self.profile_port:
            rumps.notification("Super Puppy", "No URL", "Profile server not running")
            return
        # Tailscale serve provides HTTPS on the fixed port
        try:
            result = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=3)
            data = json.loads(result.stdout)
            fqdn = data.get("Self", {}).get("DNSName", "").rstrip(".")
            if fqdn:
                url = f"https://{fqdn}:{self.profile_port}/tools"
            else:
                url = f"http://127.0.0.1:{self.profile_port}/tools"
        except Exception:
            url = f"http://127.0.0.1:{self.profile_port}/tools"

        subprocess.run(["pbcopy"], input=url.encode(), check=True)
        rumps.notification("Super Puppy", "URL copied", url)

    def _stop_mcp_server(self):
        """Stop the MCP server (whether spawned by us or adopted from a
        previous menubar instance)."""
        proc = getattr(self, '_mcp_proc', None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        elif getattr(self, '_mcp_proc_pid', None):
            # Adopted MCP — no Popen handle, manage by PID
            pid = self._mcp_proc_pid
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(50):  # up to 5s grace
                    time.sleep(0.1)
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        break
                else:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        pass
            except OSError:
                pass
        self._mcp_proc = None
        self._mcp_proc_pid = None
        log_fh = getattr(self, '_mcp_log', None)
        if log_fh and not log_fh.closed:
            log_fh.close()
        self._mcp_log = None

    def _restart_mcp(self, _):
        """Restart the MCP server (background thread)."""
        def _do():
            self._stop_mcp_server()
            time.sleep(1)
            self._start_mcp_server()
            self.refresh(None)
        threading.Thread(target=_do, daemon=True).start()

    def _select_remote(self, _):
        """User selected Remote mode."""
        if not self.remote_reachable:
            return
        self.force_local = False
        save_force_local(False)
        self._activate_profile("everyday")
        # _start_mcp_server handles the Remote-mode branch (stop local +
        # repoint Claude). The previous _restart_mcp wrapper added a
        # gratuitous stop+sleep+start cycle that just widened the
        # SSE-disconnect window for any active session.
        def _do():
            self._start_mcp_server()
            self.refresh(None)
        threading.Thread(target=_do, daemon=True).start()

    def _select_local(self, _):
        """User selected Local mode."""
        self.force_local = True
        save_force_local(True)
        data = load_profiles()
        best = pick_profile_for_ram(self.ram_gb, data.get("profiles", {}))
        if best:
            self._activate_profile(best)
        if not self.servers_started:
            self._start_local_servers()
        # _start_mcp_server adopts a healthy local MCP if one is already
        # running, or spawns a fresh one. No need for the stop+sleep+start
        # _restart_mcp wrapper here either.
        def _do():
            self._start_mcp_server()
            self.refresh(None)
        threading.Thread(target=_do, daemon=True).start()

    def _activate_profile(self, name):
        """Set the active profile and update MCP preferences to match."""
        data = load_profiles()
        profiles = data.get("profiles", {})
        if name not in profiles:
            return
        data["active"] = name
        save_profiles(data)
        # Sync task→model preferences from the profile
        profile = profiles[name]
        tasks = profile.get("tasks", {})
        prefs = load_mcp_prefs()
        for task, model in tasks.items():
            existing = prefs.get(task, [])
            if isinstance(existing, list):
                prefs[task] = [model] + [m for m in existing if m != model]
            else:
                prefs[task] = [model]
        save_mcp_prefs(prefs)

    def stop_services(self):
        """Stop local servers."""
        try:
            subprocess.run(
                [os.path.expanduser("~/.local/bin/start-local-models"), "--stop"],
                capture_output=True, timeout=10,
            )
            self.servers_started = False
            self.mode = "stopped"
            self._allow_sleep()
            self._stop_mcp_server()
            self.refresh(None)
        except Exception:
            pass

    def toggle_services(self, sender):
        if self.mode == "stopped":
            self.start_services()
        else:
            self.stop_services()

    def _restart_ollama(self, _):
        """Restart just Ollama."""
        self.ollama_ok = False
        self.ollama_loading = True
        self._update_menu()
        def _do():
            try:
                # Kill Ollama.app first — it auto-respawns `ollama serve`
                # with default (localhost) binding, racing our restart.
                subprocess.run(["pkill", "-f", "Ollama.app"],
                               capture_output=True, timeout=5)
                subprocess.run(["pkill", "-x", "ollama"],
                               capture_output=True, timeout=5)
                time.sleep(2)
                env = os.environ.copy()
                subprocess.Popen(
                    ["ollama", "serve"], env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True)
                for _ in range(10):
                    time.sleep(1)
                    if probe_service(OLLAMA_LOCAL, 2):
                        break
            except Exception as e:
                rumps.notification("Local Models", "Ollama restart failed", str(e))
            self.refresh(None)
        threading.Thread(target=_do, daemon=True).start()

    def _restart_mlx(self, _):
        """Restart just MLX-OpenAI-Server (kills entire process tree)."""
        self.mlx_ok = False
        self.mlx_loading = True
        self._update_menu()
        def _do():
            try:
                # MLX spawns child processes; kill the whole tree via pgid
                import signal
                pids = subprocess.run(
                    ["pgrep", "-f", "mlx-openai-server"],
                    capture_output=True, text=True, encoding='utf-8', timeout=5)
                for pid in pids.stdout.strip().splitlines():
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                time.sleep(2)
                # Force-kill any survivors
                subprocess.run(["pkill", "-9", "-f", "mlx-openai-server"],
                               capture_output=True, timeout=3)
                time.sleep(1)

                mlx_config = os.path.expanduser("~/.config/mlx-server/config.yaml")
                if self.ram_gb < 48:
                    mlx_config = os.path.expanduser(
                        "~/.config/mlx-server/config-laptop.yaml")
                if hasattr(self, '_mlx_log') and self._mlx_log and not self._mlx_log.closed:
                    self._mlx_log.close()
                self._mlx_log = open("/tmp/local-models-mlx-restart.log", "w")
                mlx_log = self._mlx_log
                env = os.environ.copy()
                # Ensure Homebrew is on PATH for tools like ffmpeg
                if "/opt/homebrew/bin" not in env.get("PATH", ""):
                    env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"
                subprocess.Popen(
                    ["mlx-openai-server", "launch", "--config", mlx_config,
                     "--no-log-file"],
                    stdout=mlx_log, stderr=mlx_log,
                    env=env,
                    cwd=os.path.expanduser("~"),
                    start_new_session=True)
                # Wait for it to come up
                for _ in range(15):
                    time.sleep(1)
                    if probe_service(MLX_LOCAL, 2):
                        break
            except Exception as e:
                rumps.notification("Local Models", "MLX restart failed", str(e))
            self.refresh(None)
        threading.Thread(target=_do, daemon=True).start()

    def _open_mcp_config(self, _):
        """Open the Claude config file so the user can check MCP setup."""
        subprocess.Popen(["open", MCP_TOOLS_FILE])

    # -------------------------------------------------------------------
    # Status refresh
    # -------------------------------------------------------------------

    def refresh(self, _):
        """Poll services in a background thread, then update the menu."""
        if not self._lock.acquire(blocking=False):
            return

        def _poll():
            try:
                if self.desktop:
                    self._refresh_server_mode()
                else:
                    self._refresh_client_mode()
            finally:
                self._lock.release()
                self._main_thread_update()

        threading.Thread(target=_poll, daemon=True).start()

    def _main_thread_update(self):
        """Schedule menu update on the main thread."""
        from PyObjCTools import AppHelper
        AppHelper.callAfter(self._finish_refresh)

    def _finish_refresh(self):
        """Main-thread callback after background poll completes."""
        if time.time() - self.last_update_check > UPDATE_CHECK_INTERVAL:
            self._schedule_update_check()
        if self._next_woof and time.time() >= self._next_woof:
            self._woof()

        # Run the post-update health check exactly once, on the first
        # refresh after services have stopped warming up. Comparing while
        # MLX is still loading would falsely flag every successful update
        # as a regression.
        if (not self._health_checked
                and not getattr(self, "ollama_loading", False)
                and not getattr(self, "mlx_loading", False)):
            self._post_update_health_check()

        if self.app_ready:
            self._update_menu()
            if (self.profile_server_mode is not None
                    and self.profile_server_mode != self.mode
                    and self.profile_window is not None):
                self._restart_profile_server_and_reload()

    def _on_webview_message(self, body):
        """Handle messages from the profiles/tools webview."""
        try:
            action = body.get("action") if hasattr(body, "get") else None
        except Exception:
            action = None
        if action == "download":
            self._save_file_from_url(
                str(body.get("url", "")), str(body.get("filename", "download")))
            return
        self._update_menu()

    def _save_file_from_url(self, url, filename):
        """Download a URL and present a save dialog."""
        import urllib.request
        from AppKit import NSSavePanel
        logging.info("Download requested: %s -> %s", url, filename)
        try:
            data = urllib.request.urlopen(url, timeout=10).read()
        except Exception as e:
            logging.warning("Download fetch failed for %s: %s", url, e)
            return
        panel = NSSavePanel.savePanel()
        panel.setNameFieldStringValue_(filename)
        if panel.runModal() == 1:  # NSModalResponseOK
            try:
                panel.URL().path().encode()  # validate path
                with open(panel.URL().path(), "wb") as f:
                    f.write(data)
            except Exception as e:
                logging.warning("Save failed: %s", e)

    def _restart_profile_server_and_reload(self):
        """Kill profile server, restart for new mode, reload webview."""
        self._ensure_profile_server()
        if self.profile_window is not None:
            from Foundation import NSURL, NSURLRequest
            url = NSURL.URLWithString_(
                f"http://127.0.0.1:{self.profile_port}/")
            req = NSURLRequest.requestWithURL_(url)
            wv = self.profile_window.contentView().subviews()[0]
            wv.loadRequest_(req)

    def _refresh_server_mode(self):
        self.ollama_ok = probe_service(OLLAMA_LOCAL, 2)
        self.mlx_ok = probe_service(MLX_LOCAL, 2)
        self.ollama_loading = not self.ollama_ok and process_is_running("ollama")
        self.mlx_loading = not self.mlx_ok and process_is_running("mlx-openai-server")

        # Track how long MLX has been in "loading" state — if the process is
        # alive but not responding for >60s, it's stuck. Kill it so the
        # auto-restart logic below can relaunch it.
        if self.mlx_loading:
            if not hasattr(self, '_mlx_loading_since'):
                self._mlx_loading_since = time.time()
            elif time.time() - self._mlx_loading_since > 60:
                subprocess.run(["pkill", "-9", "-f", "mlx-openai-server"],
                               capture_output=True, timeout=3)
                self.mlx_loading = False
                del self._mlx_loading_since
        else:
            if hasattr(self, '_mlx_loading_since'):
                del self._mlx_loading_since

        # Auto-restart downed services on desktop (at most once per 2 minutes)
        if (self.servers_started
                and ((not self.ollama_ok and not self.ollama_loading)
                     or (not self.mlx_ok and not self.mlx_loading))):
            now = time.time()
            if now - getattr(self, '_last_restart_attempt', 0) > 120:
                self._last_restart_attempt = now
                self._start_local_servers()

        self.ollama_models = get_ollama_models(OLLAMA_LOCAL) if self.ollama_ok else []
        self.mlx_models = get_mlx_models(MLX_LOCAL) if self.mlx_ok else []
        self.mcp_models = get_mcp_models()

        if self.ollama_ok or self.mlx_ok:
            self.mode = "server"
        elif self.ollama_loading or self.mlx_loading:
            self.mode = "server"
        else:
            self.mode = "stopped"

    def _refresh_client_mode(self):
        # Always probe the desktop so the menu shows accurate availability.
        # The TCP probe is necessary but NOT sufficient: tailscale serve
        # listens on 8100/8101 even when the backend MCP / profile server
        # is dead, so a successful connect tells us nothing about whether
        # the desktop is actually serving. We confirm by fetching the
        # model list before declaring remote reachable.
        #
        # force_local controls whether we *use* the desktop, not whether we
        # know it's there: the Remote toggle is the only UI path that clears
        # FORCE_LOCAL=true, and it greys out when remote_reachable is False.
        # Skipping the probe under force_local would trap the user in local
        # override with no escape.
        desktop_up = self.ts_hostname and self._resolve_desktop()
        was_remote = self.remote_reachable
        mcp_models = []

        if desktop_up:
            desktop_host = self.desktop_fqdn or self.desktop_ip
            desktop_ps = f"https://{desktop_host}:8101"
            all_models = http_get_json(f"{desktop_ps}/api/models", timeout=5)
            if all_models and isinstance(all_models, list):
                mcp_models = [m["name"] for m in all_models]
            else:
                mcp_models = get_mcp_models(
                    f"https://{desktop_host}:8100" if self.desktop_fqdn
                    else f"http://{self.desktop_ip}:8100")

        self.remote_reachable = bool(mcp_models)

        if self.remote_reachable and not self.force_local:
            self.mode = "client"
            self.ollama_ok = False
            self.mlx_ok = False
            self.ollama_models = []
            self.mlx_models = []
            self.mcp_models = mcp_models
            if not was_remote:
                self._notify_connection("Connected to desktop",
                                        f"via Tailscale ({self.desktop_ip})")
            return

        if was_remote and not self.force_local:
            if desktop_up and not mcp_models:
                self._notify_connection(
                    "Desktop MCP unavailable",
                    "Backend not responding — using local models")
            elif not desktop_up:
                self._notify_connection("Desktop unreachable",
                                        "Using local models")

        self.ollama_ok = probe_service(OLLAMA_LOCAL, 2)
        self.mlx_ok = probe_service(MLX_LOCAL, 2)

        if self.ollama_ok or self.mlx_ok:
            self.mode = "offline"
            self.ollama_models = get_ollama_models(OLLAMA_LOCAL) if self.ollama_ok else []
            self.mlx_models = get_mlx_models(MLX_LOCAL) if self.mlx_ok else []
        elif not self.servers_started:
            rumps.notification(
                "Local Models", "Desktop unreachable",
                "Starting local models for offline use")
            self._start_local_servers()
            time.sleep(3)
            self.ollama_ok = probe_service(OLLAMA_LOCAL, 2)
            self.mlx_ok = probe_service(MLX_LOCAL, 2)
            self.mode = "offline"
            self.ollama_models = (
                get_ollama_models(OLLAMA_LOCAL) if self.ollama_ok else [])
            self.mlx_models = (
                get_mlx_models(MLX_LOCAL) if self.mlx_ok else [])
        else:
            self.mode = "offline"
            self.ollama_models = []
            self.mlx_models = []

        self.mcp_models = get_mcp_models()

    @staticmethod
    def _styled_menu(item, dot, label, detail="", bold=False):
        """Set an NSAttributedString title with dot, label, and dim detail."""
        from AppKit import (NSFont, NSForegroundColorAttributeName,
                            NSFontAttributeName, NSColor,
                            NSMutableAttributedString,
                            NSParagraphStyleAttributeName,
                            NSMutableParagraphStyle,
                            NSFontManager)
        from Foundation import NSRange, NSString

        font = NSFont.menuFontOfSize_(13)
        if bold:
            font = NSFontManager.sharedFontManager().convertFont_toHaveTrait_(
                font, 2)  # 2 = NSBoldFontMask
        detail_font = NSFont.menuFontOfSize_(12)

        para = NSMutableParagraphStyle.alloc().init()
        tab_stop_cls = __import__(
            'AppKit', fromlist=['NSTextTab']).NSTextTab
        tab = tab_stop_cls.alloc().initWithType_location_(0, 170)
        para.setTabStops_([tab])

        main_text = f"{dot} {label}" if dot else label
        full_text = f"{main_text}\t{detail}" if detail else main_text

        # Use NSString length (UTF-16) for correct attributed string ranges
        ns_main = NSString.stringWithString_(main_text)
        ns_full = NSString.stringWithString_(full_text)
        ns_detail = NSString.stringWithString_(detail) if detail else None

        s = NSMutableAttributedString.alloc().initWithString_(full_text)
        s.addAttribute_value_range_(
            NSFontAttributeName, font, NSRange(0, ns_full.length()))
        s.addAttribute_value_range_(
            NSParagraphStyleAttributeName, para,
            NSRange(0, ns_full.length()))

        if detail:
            detail_start = ns_main.length() + 1  # +1 for tab
            s.addAttribute_value_range_(
                NSFontAttributeName, detail_font,
                NSRange(detail_start, ns_detail.length()))
            s.addAttribute_value_range_(
                NSForegroundColorAttributeName,
                NSColor.secondaryLabelColor(),
                NSRange(detail_start, ns_detail.length()))

        item._menuitem.setAttributedTitle_(s)

    def _update_menu(self):
        """Rebuild the menu to reflect current state."""

        self.title = None

        # ── Top line ──
        ollama_n = len(self.ollama_models)
        mlx_n = len(self.mlx_models)
        profiles_data = load_profiles()
        active = profiles_data.get("active")
        if active and active in profiles_data.get("profiles", {}):
            profile = profiles_data["profiles"][active].get("label", active)
        else:
            profile = "No Profile"

        GRN, YEL, RED = "\U0001f7e2", "\U0001f7e1", "\U0001f534"

        if self.desktop:
            mode_label = {"server": "Server", "stopped": "Stopped"
                          }.get(self.mode, "…")
            self._styled_menu(self.menu_status, "", mode_label)
        else:
            is_remote = self.mode == "client"
            self.menu_mode_remote.state = is_remote
            self.menu_mode_local.state = not is_remote
            if self.remote_reachable:
                self._styled_menu(self.menu_mode_remote, "", "Remote")
                self.menu_mode_remote.set_callback(self._select_remote)
            else:
                self._styled_menu(self.menu_mode_remote, "", "Remote",
                                  "unavailable")
                self.menu_mode_remote.set_callback(None)
            local_detail = "override" if self.force_local and self.remote_reachable else ""
            self._styled_menu(self.menu_mode_local, "", "Local", local_detail)

        # ── Remote Access toggle (desktop only) ──
        if self.desktop:
            if self.remote_access_enabled:
                self.menu_remote_access.title = "\u2705 Remote Access"
            else:
                self.menu_remote_access.title = "\u274c Remote Access"

        self._styled_menu(self.menu_profiles, "", "Models", profile)

        # ── Per-service status lines ──
        ollama_loading = getattr(self, 'ollama_loading', False)
        mlx_loading = getattr(self, 'mlx_loading', False)
        is_local = self.mode in ("server", "offline")

        if self.mode == "client":
            # Remote: hide local service details — they're the desktop's concern
            self.menu_ollama.hide()
            self.menu_mlx.hide()
            self.menu_mcp_restart.set_callback(None)
        else:
            self.menu_ollama.show()
            self.menu_mlx.show()
            self.menu_mcp_restart.set_callback(self._restart_mcp)
            restart_pending = (
                self.servers_started
                and time.time() - getattr(self, '_last_restart_attempt', 0) < 120)
            down_detail = "restarting…" if restart_pending else "down"

            if self.ollama_ok:
                self._styled_menu(self.menu_ollama, GRN, "Ollama",
                                  f"{ollama_n} models")
            elif ollama_loading:
                self._styled_menu(self.menu_ollama, YEL, "Ollama", "starting…")
            else:
                self._styled_menu(self.menu_ollama, RED, "Ollama", down_detail)
            self.menu_ollama_restart.set_callback(self._restart_ollama)

            if self.mlx_ok:
                self._styled_menu(self.menu_mlx, GRN, "MLX",
                                  f"{mlx_n} models")
            elif mlx_loading:
                self._styled_menu(self.menu_mlx, YEL, "MLX", "starting…")
            else:
                self._styled_menu(self.menu_mlx, RED, "MLX", down_detail)
            self.menu_mlx_restart.set_callback(self._restart_mlx)

        mcp_proc = getattr(self, '_mcp_proc', None)
        mcp_proc_alive = mcp_proc is not None and mcp_proc.poll() is None
        if self.mode == "client":
            mcp_alive = probe_port(8100, host=self.desktop_ip or "127.0.0.1")
        else:
            mcp_port_alive = probe_port(8100)
            mcp_alive = mcp_proc_alive or mcp_port_alive
        mcp_n = len(self.mcp_models)
        if mcp_alive:
            self._styled_menu(self.menu_mcp, GRN, "MCP",
                              f"{mcp_n} model{'s' if mcp_n != 1 else ''}")
        elif self.mode == "client":
            self._styled_menu(self.menu_mcp, RED, "MCP", "not shared")
        else:
            self._styled_menu(self.menu_mcp, RED, "MCP", "down")
            if self.servers_started:
                self._start_mcp_server()

        # ── Version ──
        self.menu_version.title = self.app_version
        self.menu_version.set_callback(None)

    # -------------------------------------------------------------------
    # Profile viewer (native WKWebView window)
    # -------------------------------------------------------------------

    def _ensure_profile_server(self):
        """Start (or restart) the Flask profile server.

        Handles the restart race where a prior profile-server process was
        orphaned to init (via app crash, force-quit, or anything that
        skipped our clean quit path).  In that case a naive Popen would
        silently fail to bind our port and the readiness probe would get a
        200 back from the orphan — leaving the UI pointed at stale code for
        the rest of the session.  Reconcile against the pidfile + port
        before spawning, and verify the readiness probe is answered by the
        process *we* launched via an identity token.
        """
        alive = (self.profile_server is not None
                 and self.profile_server.poll() is None
                 and self.profile_port is not None)
        if alive and self.profile_server_mode == self.mode:
            return
        if alive:
            self.profile_server.terminate()
            try:
                self.profile_server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.profile_server.kill()
                self.profile_server.wait()
            self.profile_server = None
            if hasattr(self, '_profile_log') and self._profile_log:
                self._profile_log.close()

        # Desktop: fixed port (Tailscale serve proxies it remotely)
        # Laptop: random port (local use only)
        if self.desktop:
            self.profile_port = int(getattr(self, '_profile_fixed_port', 8101))
        else:
            import socket
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            self.profile_port = s.getsockname()[1]
            s.close()
        profile_host = "127.0.0.1"

        # Evict any orphaned profile-server from a previous session before
        # trying to bind our port.  If we skip this the new Popen loses the
        # bind race, crashes, and the readiness probe below happily hits
        # the orphan and returns as if everything was fine.
        self._kill_orphan_profile_servers(self.profile_port)

        # Identity token so the readiness probe can prove the responder on
        # our port is the child we just launched, not a stale orphan that
        # happens to be bound to the same address.
        import secrets
        expected_token = secrets.token_hex(16)

        env = os.environ.copy()
        env["PROFILE_SERVER_PORT"] = str(self.profile_port)
        env["PROFILE_HOST"] = profile_host
        env["PROFILE_SERVER_TOKEN"] = expected_token
        if _AUTH_TOKEN:
            env["MCP_AUTH_TOKEN"] = _AUTH_TOKEN
        env["OLLAMA_URL"] = (
            self.ollama_remote if self.mode == "client" else OLLAMA_LOCAL)
        env["MLX_URL"] = (
            self.mlx_remote if self.mode == "client" else MLX_LOCAL)
        # Keep profile server alive when serving remote clients
        if self.desktop and getattr(self, 'remote_access_enabled', False):
            env["PROFILE_IDLE_TIMEOUT"] = "0"

        # Profile server always runs plain HTTP on localhost;
        # Tailscale serve handles TLS for remote access.
        self._profile_scheme = "http"

        log_path = "/tmp/local-models-profile-server.log"
        self._profile_log = open(log_path, "a")
        self.profile_server = subprocess.Popen(
            ["uv", "run", "--python", "3.12",
             os.path.join(SCRIPT_DIR, "profile-server.py")],
            env=env, stdout=subprocess.DEVNULL, stderr=self._profile_log)
        self.profile_server_mode = self.mode

        # Wait for OUR process to answer on /api/identity with the matching
        # token.  A 200 from a different token means we're talking to an
        # orphan — try to evict it again and keep polling.
        import urllib.request, json as _json
        base = f"{self._profile_scheme}://127.0.0.1:{self.profile_port}"
        deadline = time.time() + 8.0
        orphan_retries = 2
        while time.time() < deadline:
            if self.profile_server.poll() is not None:
                # Child exited (probably because bind failed).  Evict and
                # respawn once — covers the case where the orphan was born
                # faster than our kill round-trip.
                if orphan_retries <= 0:
                    break
                orphan_retries -= 1
                self._kill_orphan_profile_servers(self.profile_port)
                time.sleep(0.3)
                self.profile_server = subprocess.Popen(
                    ["uv", "run", "--python", "3.12",
                     os.path.join(SCRIPT_DIR, "profile-server.py")],
                    env=env, stdout=subprocess.DEVNULL,
                    stderr=self._profile_log)
                continue
            time.sleep(0.25)
            try:
                with urllib.request.urlopen(f"{base}/api/identity", timeout=1) as r:
                    data = _json.loads(r.read())
                if data.get("token") == expected_token:
                    break
                # Wrong token → someone else is bound.  Nuke and try again.
                logging.warning("profile-server identity mismatch on :%d (got token %s); "
                                "evicting and retrying", self.profile_port,
                                (data.get("token") or "")[:8])
                self._kill_orphan_profile_servers(self.profile_port)
            except Exception:
                continue
        else:
            logging.warning("profile-server readiness probe timed out on :%d",
                            self.profile_port)

    def _kill_orphan_profile_servers(self, port):
        """SIGTERM any profile-server.py processes not owned by this menubar
        instance — preferring pidfile lookup, falling back to a port sweep.

        Safe to call unconditionally: it only kills processes matching
        profile-server.py in their argv, so unrelated listeners on the same
        port are left alone (we'll fail to bind and log it instead)."""
        pidfile = os.path.expanduser("~/.config/local-models/profile-server.pid")
        victims = set()
        try:
            with open(pidfile) as f:
                data = json.loads(f.read())
            pid = int(data.get("pid") or 0)
            if pid and self._pid_is_profile_server(pid):
                victims.add(pid)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
            pass
        # Belt: walk ps and match any profile-server.py that's alive.  Also
        # check lsof for anything holding our TCP port, in case it's under a
        # different cmdline (e.g. a test harness).
        try:
            out = subprocess.check_output(
                ["ps", "-axo", "pid=,command="], text=True, timeout=5)
            for line in out.splitlines():
                pid_str, _, cmd = line.strip().partition(" ")
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                if pid == os.getpid():
                    continue
                if "profile-server.py" in cmd and "--pull-worker" not in cmd:
                    victims.add(pid)
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                text=True, timeout=3)
            for line in out.splitlines():
                try:
                    pid = int(line.strip())
                except ValueError:
                    continue
                if pid != os.getpid() and self._pid_is_profile_server(pid):
                    victims.add(pid)
        except subprocess.CalledProcessError:
            pass  # lsof exits 1 when nothing matches
        except Exception:
            pass

        if not victims:
            return
        for pid in victims:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                continue
        # Give them up to 3s to exit and release the port.
        for _ in range(30):
            time.sleep(0.1)
            if not any(self._pid_is_alive(p) for p in victims):
                break
        # Anything still alive gets SIGKILL'd.
        for pid in victims:
            if self._pid_is_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        # Clean up a stale pidfile if we just killed the process it pointed at.
        try:
            with open(pidfile) as f:
                data = json.loads(f.read())
            if int(data.get("pid") or 0) in victims:
                os.unlink(pidfile)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
            pass

    @staticmethod
    def _pid_is_alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _pid_is_profile_server(pid):
        try:
            out = subprocess.check_output(
                ["ps", "-o", "command=", "-p", str(pid)],
                text=True, timeout=2).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return False
        return "profile-server.py" in out and "--pull-worker" not in out

    def _install_termination_guard(self):
        """Wrap NSApp's delegate with a proxy that forces NO for
        applicationShouldTerminateAfterLastWindowClosed:. Idempotent — only
        installs the guard on first webview open."""
        if getattr(self, "_termination_guard", None) is not None:
            return
        from AppKit import NSApp
        real = NSApp.delegate()
        if real is None:
            logging.warning("termination guard: NSApp has no delegate yet, skipping")
            return
        guard = _NonTerminatingDelegateProxy.alloc().initWithDelegate_(real)
        NSApp.setDelegate_(guard)
        self._termination_guard = guard
        logging.info("termination guard installed (wrapped %s)",
                     type(real).__name__)

    def _open_webview(self, title, path, size=(960, 700)):
        """Open a native WKWebView window at the given server path."""
        from AppKit import (NSRect, NSBackingStoreBuffered,
                            NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
                            NSWindowStyleMaskMiniaturizable,
                            NSWindowStyleMaskResizable,
                            NSApplicationActivationPolicyRegular,
                            NSApp)
        from WebKit import (WKWebView, WKWebViewConfiguration,
                            WKUserScript)
        from Foundation import NSURL, NSMutableURLRequest

        self._ensure_profile_server()

        # Belt-and-suspenders to the per-webview setters below: stamp the
        # process's NSUserDefaults with the inline-prediction and
        # autocorrect flags off before WebKit/NSSpellChecker cache them.
        try:
            from Foundation import NSUserDefaults
            _d = NSUserDefaults.standardUserDefaults()
            for key in (
                "NSAllowsInlinePredictions",
                "NSAutomaticSpellingCorrectionEnabled",
                "NSAutomaticTextCompletionEnabled",
                "NSAutomaticTextReplacementEnabled",
                "NSAutomaticQuoteSubstitutionEnabled",
                "NSAutomaticDashSubstitutionEnabled",
                "WebAutomaticSpellingCorrectionEnabled",
                "WebContinuousSpellCheckingEnabled",
            ):
                _d.setBool_forKey_(False, key)
        except Exception:
            pass

        frame = NSRect((200, 200), size)
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                 | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        window = _ProfileWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False)
        window.setTitle_(title)
        window.center()
        window.setReleasedWhenClosed_(False)

        config = WKWebViewConfiguration.alloc().init()
        # allowsInlinePredictions is on the *configuration*, not the webview.
        # Setting it on the WKWebView (as we used to) silently no-ops because
        # respondsToSelector_ returns false — which is how 2026-04-17's
        # SIGABRT got through: NSSpellChecker was still called, its
        # correction panel pumped the run loop, a PyObjC callback raised,
        # and AppKit turned it into abort(). Must be set before
        # initWithFrame_configuration_ for WebKit to respect it.
        if config.respondsToSelector_(b"setAllowsInlinePredictions:"):
            try:
                config.setAllowsInlinePredictions_(False)
            except Exception as e:
                _raw_log(f"config.setAllowsInlinePredictions_ raised: {e}")
        prefs = config.preferences()
        try:
            prefs.setValue_forKey_(True, "mediaDevicesEnabled")
            prefs.setValue_forKey_(False, "mediaCaptureRequiresSecureConnection")
        except Exception:
            pass
        msg_handler = _WebViewMessageHandler.alloc().init()
        msg_handler.on_message = self._on_webview_message
        config.userContentController().addScriptMessageHandler_name_(
            msg_handler, "app")
        # Inject the bearer token before the page's first JS runs.  The page
        # picks it up via window.__SP_TOKEN__ and uses it for every fetch.
        # Native <img>/<audio>/<video> elements that can't set headers fall
        # back to ?token= query param.
        if _AUTH_TOKEN:
            token_js = json.dumps(_AUTH_TOKEN)
            user_script = WKUserScript.alloc().initWithSource_injectionTime_forMainFrameOnly_(
                f"window.__SP_TOKEN__ = {token_js};",
                0,  # WKUserScriptInjectionTimeAtDocumentStart
                True,
            )
            config.userContentController().addUserScript_(user_script)
        # Keep a strong Python reference so the handler (and its on_message
        # callback) survives even when another webview is opened later.
        if not hasattr(self, "_msg_handlers"):
            self._msg_handlers = []
        self._msg_handlers.append(msg_handler)
        webview = WKWebView.alloc().initWithFrame_configuration_(
            window.contentView().bounds(), config)
        ui_delegate = _WebViewUIDelegate.alloc().init()
        webview.setUIDelegate_(ui_delegate)
        window._ui_delegate = ui_delegate
        nav_delegate = _WebViewNavigationDelegate.alloc().init()
        webview.setNavigationDelegate_(nav_delegate)
        window._nav_delegate = nav_delegate
        self._install_termination_guard()
        webview.setAutoresizingMask_(0x12)
        # Disable WebKit's inline prediction + autocorrect UI.  On macOS 26
        # these route through NSSpellChecker → NSCorrectionPanel, which
        # re-enters nextEventMatchingMask and raises an NSException that
        # pyobjc cannot marshal back into Python — the whole app aborts with
        # SIGABRT on the first keystroke inside an editable field.  None of
        # this UI makes sense for our settings/playground surfaces anyway.
        _disable_sels = (
            b"setAllowsInlinePredictions:",
            b"setAutomaticSpellingCorrectionEnabled:",
            b"setAutomaticTextCompletionEnabled:",
            b"setAutomaticTextReplacementEnabled:",
            b"setAutomaticQuoteSubstitutionEnabled:",
            b"setAutomaticDashSubstitutionEnabled:",
            b"setContinuousSpellCheckingEnabled:",
            b"setGrammarCheckingEnabled:",
        )
        for sel_bytes in _disable_sels:
            if webview.respondsToSelector_(sel_bytes):
                try:
                    getattr(webview, sel_bytes.rstrip(b":").decode() + "_")(False)
                except Exception:
                    pass
        full_url = f"http://127.0.0.1:{self.profile_port}{path}"
        url = NSURL.URLWithString_(full_url)
        req = NSMutableURLRequest.requestWithURL_cachePolicy_timeoutInterval_(
            url, 1, 30)  # 1 = NSURLRequestReloadIgnoringLocalCacheData
        if _AUTH_TOKEN:
            req.setValue_forHTTPHeaderField_(
                f"Bearer {_AUTH_TOKEN}", "Authorization")
        webview.loadRequest_(req)
        window.contentView().addSubview_(webview)
        window._webview = webview

        self._set_rounded_dock_icon()
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        NSApp.activateIgnoringOtherApps_(True)
        window.makeKeyAndOrderFront_(None)
        NSApp.dockTile().display()
        return window

    def _set_rounded_dock_icon(self):
        """Render the menubar icon onto a rounded-rect tile and install it
        as the Dock icon for any open playground window. The menu bar icon
        itself stays a template (B/W) image; only the Dock representation
        gets the white background."""
        icon_path = os.path.join(SCRIPT_DIR, "icon.png")
        if not os.path.exists(icon_path):
            return
        from AppKit import (NSApp, NSBezierPath, NSColor,
                            NSCompositingOperationSourceOver, NSImage)
        from Foundation import NSMakeRect
        src = NSImage.alloc().initWithContentsOfFile_(icon_path)
        sz = 128
        radius = sz * 0.22  # macOS-style rounded rect
        dock_icon = NSImage.alloc().initWithSize_((sz, sz))
        dock_icon.lockFocus()
        rrect = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0, 0, sz, sz), radius, radius)
        rrect.addClip()
        NSColor.whiteColor().setFill()
        rrect.fill()
        src.drawInRect_fromRect_operation_fraction_(
            NSMakeRect(0, 0, sz, sz), ((0, 0), src.size()),
            NSCompositingOperationSourceOver, 1.0)
        dock_icon.unlockFocus()
        NSApp.setApplicationIconImage_(dock_icon)

    # Spec table for the playground-style windows. Adding a new pane is
    # a single entry here plus a one-line `open_*` rumps callback.
    _WINDOW_SPECS = {
        "profile":     {"attr": "profile_window",
                        "delegate_attr": "_win_delegate",
                        "title": "Models",
                        "path": "/",
                        "size": (960, 700)},
        "tools":       {"attr": "tools_window",
                        "delegate_attr": "_tools_delegate",
                        "title": "Playground",
                        "path": "/tools",
                        "size": (720, 600)},
        "activity":    {"attr": "activity_window",
                        "delegate_attr": "_activity_delegate",
                        "title": "Activity Log",
                        "path": "/activity",
                        "size": (720, 600)},
        "diagnostics": {"attr": "diagnostics_window",
                        "delegate_attr": "_diagnostics_delegate",
                        "title": "Diagnostics",
                        "path": "/diagnostics",
                        "size": (640, 520)},
    }

    def _any_other_window_open(self, key: str) -> bool:
        """True if any window other than `key` is still open. Used by the
        close-callback to decide whether to drop activation policy back to
        Accessory (no Dock icon) when the last playground window closes."""
        for k, spec in self._WINDOW_SPECS.items():
            if k == key:
                continue
            if getattr(self, spec["attr"], None) is not None:
                return True
        return False

    def _open_or_focus(self, key: str):
        """Open the window for `key`, or focus + reload it if already open.

        Replaces four near-identical `open_*` methods. The per-window
        differences (attr name, title, path, size) live in _WINDOW_SPECS.
        """
        spec = self._WINDOW_SPECS[key]
        existing = getattr(self, spec["attr"], None)
        if existing is not None:
            existing.makeKeyAndOrderFront_(None)
            if hasattr(existing, "_webview"):
                existing._webview.reload_(None)
            from AppKit import NSApp
            NSApp.activateIgnoringOtherApps_(True)
            return

        from AppKit import NSApp, NSApplicationActivationPolicyAccessory
        window = self._open_webview(
            spec["title"], spec["path"], size=spec["size"])
        delegate = _ProfileWindowDelegate.alloc().init()
        delegate.callback = lambda: (
            setattr(self, spec["attr"], None),
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            if not self._any_other_window_open(key) else None,
        )
        setattr(self, spec["delegate_attr"], delegate)
        window.setDelegate_(delegate)
        setattr(self, spec["attr"], window)

    def open_profiles(self, _):
        """Open the model profiles pane."""
        self._open_or_focus("profile")

    def open_tools(self, _):
        """Open the tool tester pane."""
        self._open_or_focus("tools")

    def open_activity(self, _):
        """Open the activity dashboard."""
        self._open_or_focus("activity")

    def open_diagnostics(self, _):
        """Open the diagnostics pane."""
        self._open_or_focus("diagnostics")

    # -------------------------------------------------------------------
    # App update (git)
    # -------------------------------------------------------------------

    def _get_head_hash(self):
        try:
            return subprocess.check_output(
                ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except Exception:
            return ""

    def _schedule_update_check(self):
        if self.conf.get("AUTO_UPDATE", "true").lower() == "false":
            return
        self.last_update_check = time.time()
        thread = threading.Thread(target=self._check_for_updates, daemon=True)
        thread.start()

    def _check_for_updates(self):
        # Don't start a new update while the previous one is still inside
        # its crash-rollback window. _auto_update overwrites
        # UPDATE_STARTED_FILE / UPDATE_PRE_HASH_FILE / PRE_UPDATE_HEALTH_FILE,
        # so a back-to-back update would lose the rollback target for the
        # *first* update and walk past tags without ever validating any of
        # them survived UPDATE_CRASH_WINDOW seconds.
        last = getattr(self, "_last_auto_update_at", 0)
        if last and time.time() - last < UPDATE_CRASH_WINDOW:
            logging.debug(
                "Skipping update check — previous update is still inside "
                "the %ss crash-rollback window (%.0fs elapsed)",
                UPDATE_CRASH_WINDOW, time.time() - last)
            return

        behind, remote_tag, remote_hash = check_repo_update_available()
        self.update_available = behind

        if behind <= 0:
            # No newer tag on origin — but the code on disk may have
            # changed since launch (commit, pull, or tag on this machine).
            # Compare the commit hash we launched from against current HEAD.
            current_hash = self._get_head_hash()
            if current_hash and current_hash != self._launch_hash:
                current_ver = get_version()
                logging.info("Code on disk changed since launch (%s → %s) — restarting",
                             self.app_version, current_ver)
                remote_tag = current_ver
            else:
                return

        # Skip if this exact release was already rolled back (within 24h)
        try:
            with open(UPDATE_SKIPPED_FILE) as f:
                skip_data = f.read().strip()
            # Format: "hash\ntimestamp" (timestamp added for expiry)
            skip_parts = skip_data.split("\n", 1)
            skip_hash = skip_parts[0]
            skip_ts = float(skip_parts[1]) if len(skip_parts) > 1 else 0
            if skip_hash == remote_hash:
                if time.time() - skip_ts < UPDATE_SKIP_EXPIRY:
                    logging.info("Skipping update to %s (rolled back %.0fh ago)",
                                 remote_tag, (time.time() - skip_ts) / 3600)
                    return
                logging.info("Skip for %s expired after 24h — retrying", remote_tag)
                os.unlink(UPDATE_SKIPPED_FILE)
        except FileNotFoundError:
            pass
        except (ValueError, OSError) as e:
            logging.warning("Bad skip file, removing: %s", e)
            try:
                os.unlink(UPDATE_SKIPPED_FILE)
            except FileNotFoundError:
                pass

        # Idle gate: don't interrupt active MCP sessions (max 5 deferrals)
        if self._mcp_recently_active():
            self._update_defer_count += 1
            if self._update_defer_count < MAX_UPDATE_DEFERRALS:
                logging.info("Deferring update — MCP active (attempt %d/%d)",
                             self._update_defer_count, MAX_UPDATE_DEFERRALS)
                return
            logging.info("MCP active but hit max deferrals (%d) — forcing update",
                         MAX_UPDATE_DEFERRALS)
        self._update_defer_count = 0

        logging.info("Auto-updating to %s", remote_tag)
        self._auto_update(remote_tag)

    def _mcp_recently_active(self):
        """Check if the MCP server has active GPU requests (real inference)."""
        try:
            resp = urllib.request.urlopen(
                "http://127.0.0.1:8100/gpu", timeout=2)
            data = json.loads(resp.read())
            return data.get("ollama", {}).get("active", 0) > 0 \
                or data.get("mlx", {}).get("active", 0) > 0
        except Exception:
            return False

    def _auto_update(self, target_tag):
        # Mark the start of this update so back-to-back update checks know
        # to skip until the crash-rollback window has elapsed (see
        # _check_for_updates).
        self._last_auto_update_at = time.time()

        # Save pre-update tag/hash for precise rollback
        pre_hash = self._launch_hash
        try:
            with open(UPDATE_PRE_HASH_FILE, "w") as f:
                f.write(pre_hash)
        except Exception:
            pass

        # Save pre-update health snapshot
        health = {
            "ollama": self.ollama_ok,
            "mlx": self.mlx_ok,
            "mcp": bool(self.mcp_models),
            "version": self.app_version,
        }
        try:
            os.makedirs(os.path.dirname(PRE_UPDATE_HEALTH_FILE), exist_ok=True)
            with open(PRE_UPDATE_HEALTH_FILE, "w") as f:
                json.dump(health, f)
        except Exception as e:
            logging.warning("Failed to save pre-update health: %s", e)

        try:
            rumps.notification(
                "Super Puppy", f"Updating to {target_tag}", "Restarting…")
        except RuntimeError:
            pass

        # Check if MCP server code changed (to decide whether to restart it).
        # For tag updates, diff HEAD (still old code) against the target tag.
        # For drift, HEAD already moved, so diff against the launch hash.
        current_hash = self._get_head_hash()
        diff_target = target_tag if current_hash == self._launch_hash else self._launch_hash
        mcp_changed = self._mcp_code_changed(diff_target)

        # If HEAD already moved past our launch hash (commit/pull on this
        # machine), the new code is on disk — skip checkout, just restart.
        # Otherwise a remote tag needs to be checked out.
        if current_hash == self._launch_hash:
            # HEAD hasn't moved locally — need to check out the remote tag
            success, output = apply_repo_update(target_tag)
            if not success:
                logging.error("Auto-update checkout failed: %s", output)
                if pre_hash:
                    logging.info("Rolling back to %s", pre_hash[:8])
                    subprocess.run(
                        ["git", "-C", REPO_DIR, "checkout", "--force", pre_hash],
                        capture_output=True, timeout=10)
                try:
                    rumps.notification("Super Puppy", "Update failed — rolled back",
                                       output[:100])
                except RuntimeError:
                    pass
                self._cleanup_update_files()
                return
        else:
            logging.info("Code already on disk — skipping checkout, restarting")

        # Run post-update hook (rebuild binary, update symlinks).
        # If this fails, the binary may be broken — abort instead of
        # restarting into a crash→rollback→skip-forever loop.
        post_update = os.path.join(REPO_DIR, "bin", "post-update.sh")
        if os.path.isfile(post_update):
            try:
                result = subprocess.run(
                    [post_update], capture_output=True, text=True, encoding='utf-8', timeout=120)
                if result.returncode != 0:
                    logging.error("post-update.sh failed (rc=%d): %s",
                                  result.returncode, result.stderr.strip())
                    # Roll back — don't restart on a broken build
                    if pre_hash:
                        logging.info("Rolling back checkout to %s", pre_hash[:8])
                        subprocess.run(
                            ["git", "-C", REPO_DIR, "checkout", "--force", pre_hash],
                            capture_output=True, timeout=10)
                    try:
                        rumps.notification("Super Puppy", "Update failed",
                                           "post-update.sh failed — rolled back")
                    except RuntimeError:
                        pass
                    self._cleanup_update_files()
                    return
                logging.info("post-update.sh: %s",
                             result.stdout.strip().split("\n")[-1])
            except Exception as e:
                logging.error("post-update.sh exception: %s — rolling back", e)
                if pre_hash:
                    subprocess.run(
                        ["git", "-C", REPO_DIR, "checkout", "--force", pre_hash],
                        capture_output=True, timeout=10)
                self._cleanup_update_files()
                return

        # Write update_started marker for crash rollback detection
        try:
            with open(UPDATE_STARTED_FILE, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

        # Restart — skip MCP kill if its code didn't change
        logging.info("Stopping services before restart (mcp_changed=%s)", mcp_changed)
        if self.profile_server and self.profile_server.poll() is None:
            self.profile_server.terminate()
        if mcp_changed:
            self._stop_mcp_server()
        lock_file = os.path.expanduser("~/.config/local-models/menubar.lock")
        try:
            os.unlink(lock_file)
        except FileNotFoundError:
            pass
        # Exit non-zero so launchd's KeepAlive restarts us on the new code.
        # os._exit bypasses atexit handlers to ensure a clean, fast exit.
        logging.info("Exiting for restart (os._exit(1)) — launchd will relaunch")
        os._exit(1)

    def _mcp_code_changed(self, target_ref="origin/main"):
        """Check if mcp/ files differ between HEAD and the target ref."""
        try:
            result = subprocess.run(
                ["git", "-C", REPO_DIR, "diff", "--name-only",
                 "HEAD", target_ref, "--", "mcp/"],
                capture_output=True, text=True, encoding='utf-8', timeout=5)
            return bool(result.stdout.strip())
        except Exception:
            return True  # assume changed on failure

    # -------------------------------------------------------------------
    # Startup: crash rollback + post-update health check
    # -------------------------------------------------------------------

    def _startup_rollback_check(self):
        """If the previous launch crashed right after an update, roll back."""
        try:
            with open(UPDATE_STARTED_FILE) as f:
                started = float(f.read().strip())
        except (FileNotFoundError, ValueError):
            return
        elapsed = time.time() - started
        if elapsed > UPDATE_CRASH_WINDOW:
            # Previous launch survived long enough — all good
            try:
                os.unlink(UPDATE_STARTED_FILE)
            except FileNotFoundError:
                pass
            return
        # If launch_attempted exists, the shell wrapper is managing the
        # rollback lifecycle (two-phase: first launch proceeds, second
        # launch triggers rollback). Rewrite update_started to NOW so
        # _mark_startup_healthy measures from this launch, not the exit.
        if os.path.exists(LAUNCH_ATTEMPTED_FILE):
            try:
                with open(UPDATE_STARTED_FILE, "w") as f:
                    f.write(str(time.time()))
            except Exception:
                pass
            return
        # Previous launch died within the crash window after an update
        logging.warning("Crash detected within %ds of update — rolling back", int(elapsed))
        try:
            current_hash = subprocess.check_output(
                ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                text=True, timeout=5).strip()
            # Roll back to the pre-update hash (which should be a tagged release)
            rollback_target = None
            try:
                with open(UPDATE_PRE_HASH_FILE) as f:
                    saved_hash = f.read().strip()
                if saved_hash:
                    rollback_target = saved_hash
            except FileNotFoundError:
                pass
            if not rollback_target:
                logging.error("No pre-update hash found, cannot roll back")
                return
            subprocess.run(
                ["git", "-C", REPO_DIR, "checkout", "--force", rollback_target],
                capture_output=True, timeout=10)
            # Rebuild binary/symlinks to match the rolled-back code
            post_update = os.path.join(REPO_DIR, "bin", "post-update.sh")
            if os.path.isfile(post_update):
                try:
                    subprocess.run([post_update], capture_output=True,
                                   text=True, timeout=120)
                except Exception as e:
                    logging.warning("post-update.sh after rollback failed: %s", e)
            # Record skipped release hash + timestamp so we don't retry
            # for 24h (but the skip expires so a fixed env can retry)
            os.makedirs(os.path.dirname(UPDATE_SKIPPED_FILE), exist_ok=True)
            with open(UPDATE_SKIPPED_FILE, "w") as f:
                f.write(f"{current_hash}\n{time.time()}")
            self.app_version = get_version()
            self._rolled_back = True
            logging.warning("Rolled back to %s (skipped %s)",
                            self.app_version, current_hash[:8])
            try:
                rumps.notification(
                    "Super Puppy", "Rolled back bad update",
                    f"Now on {self.app_version}")
            except RuntimeError:
                pass
        except Exception as e:
            logging.error("Rollback failed: %s", e)
        finally:
            try:
                os.unlink(UPDATE_STARTED_FILE)
            except FileNotFoundError:
                pass

    def _cleanup_update_files(self):
        for f in (UPDATE_STARTED_FILE, UPDATE_PRE_HASH_FILE, LAUNCH_ATTEMPTED_FILE):
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    def _mark_startup_healthy(self):
        """Clear update markers after surviving the crash window."""
        if self._rolled_back:
            return  # rollback already handled cleanup; don't clear skip file
        self._cleanup_update_files()
        # If we survived an update past the previously-skipped commit, unblock.
        try:
            with open(UPDATE_SKIPPED_FILE) as f:
                skip_data = f.read().strip()
            skip_hash = skip_data.split("\n", 1)[0]
            head = subprocess.check_output(
                ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                text=True, timeout=5).strip()
            if head != skip_hash:
                os.unlink(UPDATE_SKIPPED_FILE)
        except Exception:
            pass

    def _post_update_health_check(self):
        """Compare current health against pre-update snapshot. Notify on regressions."""
        if self._health_checked:
            return
        self._health_checked = True
        try:
            with open(PRE_UPDATE_HEALTH_FILE) as f:
                prev = json.load(f)
            os.unlink(PRE_UPDATE_HEALTH_FILE)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        regressions = []
        if prev.get("ollama") and not self.ollama_ok:
            regressions.append("Ollama")
        if prev.get("mlx") and not self.mlx_ok:
            regressions.append("MLX")
        if prev.get("mcp") and not self.mcp_models:
            regressions.append("MCP")
        if regressions:
            names = ", ".join(regressions)
            logging.warning("Post-update regression: %s", names)
            try:
                rumps.notification(
                    "Super Puppy", "Post-update issue",
                    f"{names} was healthy before update, now down")
            except RuntimeError:
                pass

    # -------------------------------------------------------------------
    # Easter egg (opt-in via ~/.config/local-models/easter_eggs.json)
    # -------------------------------------------------------------------

    def _load_easter_eggs(self):
        path = os.path.expanduser("~/.config/local-models/easter_eggs.json")
        try:
            with open(path) as f:
                data = json.load(f)
            if not data.get("enabled"):
                return None
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _schedule_woof(self):
        import random
        eggs = self._load_easter_eggs()
        if not eggs:
            self._next_woof = 0
            return
        interval = eggs.get("interval_hours", [48, 72])
        lo = int(interval[0] * 3600)
        hi = int(interval[1] * 3600) if len(interval) > 1 else lo
        self._next_woof = time.time() + random.randint(lo, hi)

    def _woof(self):
        import random
        eggs = self._load_easter_eggs()
        if not eggs:
            self._next_woof = 0
            return
        messages = eggs.get("messages", [])
        if not messages:
            self._next_woof = 0
            return
        try:
            rumps.notification("Super Puppy", "", random.choice(messages))
        except RuntimeError:
            pass
        self._schedule_woof()
    # -------------------------------------------------------------------
    # Quit — rumps' built-in Quit button calls this before exiting
    # -------------------------------------------------------------------

    def restart_app(self, _):
        """Restart the entire app (re-exec the app bundle)."""
        app_path = os.path.join(os.path.dirname(__file__), "SuperPuppy.app")
        # Clean up, then relaunch
        if self.profile_server and self.profile_server.poll() is None:
            self.profile_server.terminate()
        self._stop_mcp_server()
        # Remove lock so the new instance can start
        lock_file = os.path.expanduser("~/.config/local-models/menubar.lock")
        try:
            os.unlink(lock_file)
        except FileNotFoundError:
            pass
        subprocess.Popen(
            ["bash", "-c", f"sleep 2 && open '{app_path}'"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from PyObjCTools import AppHelper
        AppHelper.callAfter(rumps.quit_application)

    def quit_app(self, _):
        if self.profile_server and self.profile_server.poll() is None:
            self.profile_server.terminate()
            try:
                self.profile_server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.profile_server.kill()
                self.profile_server.wait()
        if self.servers_started and self.mode != "client":
            self.stop_services()
        # Tell the wrapper to stay down on next restart.  The C launcher
        # always exits non-zero so launchd will restart us — the wrapper
        # checks for this file BEFORE launching and exits 0 to stop the cycle.
        stay_down = os.path.expanduser("~/.config/local-models/stay_down")
        try:
            with open(stay_down, "w") as f:
                f.write(str(os.getpid()))
        except Exception:
            pass
        rumps.quit_application()


LOCK_FILE = os.path.expanduser("~/.config/local-models/menubar.lock")
_lock_fd = None


def acquire_lock():
    """Ensure only one instance runs. Uses flock + PID validation as fallback."""
    import fcntl
    import signal
    global _lock_fd
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    _lock_fd = open(LOCK_FILE, "a+")
    fcntl.fcntl(_lock_fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        # flock failed — check if the PID in the file is actually alive
        _lock_fd.seek(0)
        old_pid = _lock_fd.read().strip()
        if old_pid and old_pid.isdigit():
            try:
                os.kill(int(old_pid), 0)
                print("Already running (pid %s). Exiting." % old_pid, file=sys.stderr)
                sys.exit(0)
            except OSError:
                pass  # stale PID — process is dead, take over
        else:
            # No PID in file but flock held — genuinely locked
            print("Already running. Exiting.", file=sys.stderr)
            sys.exit(0)
        # Stale lock: close, recreate, and acquire
        _lock_fd.close()
        _lock_fd = open(LOCK_FILE, "w")
        fcntl.fcntl(_lock_fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Write our PID so future instances can validate
    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()


if __name__ == "__main__":
    if "--python-info" in sys.argv:
        import site
        import sysconfig
        libdir = sysconfig.get_config_var("LIBDIR")
        ldver = sysconfig.get_config_var("LDVERSION")
        print(sys.base_prefix)
        print(f"{libdir}/libpython{ldver}.dylib")
        print(site.getsitepackages()[0])
        sys.stdout.flush()
        os._exit(0)  # Phase 1 child — exit immediately, don't go through C launcher
    # Rotate log on startup (keep last 1000 lines)
    log_path = "/tmp/local-models-menubar.log"
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > 1000:
            with open(log_path, "w", encoding="utf-8") as f:
                f.writelines(lines[-500:])
    except FileNotFoundError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")
    # ── Ensure non-zero exit for launchd ──────────────────────────────
    # The C launcher returns 1, but NSApplication terminate: calls libc
    # exit(0) directly from Objective-C, bypassing both Python exceptions
    # and the C launcher's return.  Three defenses:
    #
    # 1. SIGTERM handler: intercept before NSApplication's handler fires
    # 2. atexit: last-resort hook for any libc exit(0) from ObjC
    # 3. try/except SystemExit: catch Python-level sys.exit(0)
    #
    # os._exit() in _auto_update bypasses all of these (intentional).
    import atexit

    def _exit_defense_atexit():
        _raw_log("exit-defense: atexit fired → os._exit(1)")
        try:
            logging.warning("exit-defense: atexit fired → os._exit(1)")
        except Exception:
            pass
        os._exit(1)

    def _exit_defense_sigterm(*_):
        _raw_log("exit-defense: SIGTERM handler fired → os._exit(1)")
        try:
            logging.warning("exit-defense: SIGTERM handler fired → os._exit(1)")
        except Exception:
            pass
        os._exit(1)

    atexit.register(_exit_defense_atexit)
    signal.signal(signal.SIGTERM, _exit_defense_sigterm)

    # Breadcrumb for uncaught NSException. We can't prevent the abort(),
    # but we can at least log what blew up — the default AppKit path
    # demangles through __cxa_rethrow and the log has nothing to show for
    # it (see 2026-04-17 11:05 SIGABRT, NSSpellChecker / Inline Predictions
    # → PyObjC → NSException → libc abort).
    try:
        from Foundation import NSSetUncaughtExceptionHandler
        import objc as _objc_mod

        @_objc_mod.callbackFor(NSSetUncaughtExceptionHandler)
        def _nsexc_handler(exc):
            try:
                name = str(exc.name()) if exc else "<nil>"
                reason = str(exc.reason()) if exc else ""
                _raw_log(f"uncaught NSException: {name} — {reason[:400]}")
            except Exception:
                pass

        NSSetUncaughtExceptionHandler(_nsexc_handler)
    except Exception as _e:
        _raw_log(f"NSSetUncaughtExceptionHandler install failed: {_e}")
    # Catch the other obvious signals too — any of them firing will give us
    # a breadcrumb before the atexit hook runs.
    for _sig_num, _sig_name in (
        (signal.SIGHUP, "SIGHUP"),
        (signal.SIGINT, "SIGINT"),
        (signal.SIGQUIT, "SIGQUIT"),
    ):
        def _make_handler(name):
            def _h(*_):
                _raw_log(f"exit-defense: {name} received → os._exit(1)")
                os._exit(1)
            return _h
        try:
            signal.signal(_sig_num, _make_handler(_sig_name))
        except (ValueError, OSError):
            pass

    _raw_log("exit-defense: hooks installed")

    try:
        acquire_lock()
        LocalModelsApp().run()
    except SystemExit as e:
        _raw_log(f"exit-defense: SystemExit caught (code={e.code!r})")
        try:
            logging.warning("exit-defense: SystemExit caught (code=%r) → falling through to atexit", e.code)
        except Exception:
            pass
    except BaseException as e:
        _raw_log(f"exit-defense: unhandled {type(e).__name__}: {e}")
        raise
