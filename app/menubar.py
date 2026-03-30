# /// script
# requires-python = ">=3.12"
# dependencies = ["rumps", "pyyaml", "pyobjc-framework-WebKit"]
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
        if self.callback:
            self.callback()


class _WebViewMessageHandler(NSObject):
    """Receives postMessage calls from WKWebView JavaScript."""
    on_message = None  # callable(body_dict)

    def userContentController_didReceiveScriptMessage_(self, controller, message):
        if self.on_message:
            self.on_message(message.body())


class _WebViewUIDelegate(NSObject):
    """WKUIDelegate that auto-grants media capture (microphone) permission.

    The WebKit import above must happen before this class is defined so
    pyobjc-framework-WebKit's block metadata is registered when the ObjC
    method trampoline is created.
    """

    def webView_requestMediaCapturePermissionForOrigin_initiatedByFrame_type_decisionHandler_(
        self, webView, origin, frame, mediaType, decisionHandler
    ):
        # WKPermissionDecision.grant = 1
        decisionHandler(1)




# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
ICON_PATH = os.path.join(SCRIPT_DIR, "icon-menubar.png")
ICONS_DIR = os.path.join(SCRIPT_DIR, "icons")
NETWORK_CONF = os.path.expanduser("~/.config/local-models/network.conf")
MCP_TOOLS_FILE = os.path.expanduser("~/.claude.json")
OLLAMA_LOCAL = "http://localhost:11434"
MLX_LOCAL = "http://localhost:8000"
POLL_INTERVAL = 8           # seconds between status refreshes
UPDATE_CHECK_INTERVAL = 3600 # seconds between git update checks (1 hour)

MODEL_PREFS_FILE = os.path.expanduser("~/.config/local-models/model_preferences.json")


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


def http_get_json(url, timeout=3):
    """Fetch JSON from a URL, return None on failure."""
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "LocalModelsMenubar/1.0")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def resolve_mdns(hostname, timeout=5):
    """Resolve an mDNS (.local) hostname to an IP address.

    Uses socket.getaddrinfo with a generous timeout to handle cold mDNS
    caches. Returns the IP string, or empty string on failure.
    """
    if not hostname:
        return ""
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_INET,
                                     socket.SOCK_STREAM)
        if results:
            return results[0][4][0]
    except (socket.gaierror, OSError):
        pass
    return ""


def probe_service(base_url, timeout=2):
    """Check if a service is responding."""
    try:
        req = urllib.request.Request(f"{base_url}/api/version"
                                     if "11434" in base_url
                                     else f"{base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def probe_port(port, host="127.0.0.1", timeout=1):
    """Check if something is listening on a TCP port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def process_is_running(name):
    """Check if a process with the given name fragment is running."""
    try:
        result = subprocess.run(["pgrep", "-f", name],
                                capture_output=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


def get_ollama_models(base_url, timeout=3):
    """Get list of model names from Ollama."""
    data = http_get_json(f"{base_url}/api/tags", timeout=timeout)
    if data and "models" in data:
        return [m["name"] for m in data["models"]]
    return []


def get_mlx_models(base_url, timeout=3):
    """Get list of model names from MLX-OpenAI-Server."""
    data = http_get_json(f"{base_url}/v1/models", timeout=timeout)
    if data and "data" in data:
        return [m["id"] for m in data["data"]]
    return []


def get_tts_models():
    """Get list of downloaded TTS models from HuggingFace cache."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
    from hf_scanner import scan_hf_cache
    return [m["name"].split("/")[-1] for m in scan_hf_cache({"tts"})]


# ---------------------------------------------------------------------------
# MCP tool preferences
# ---------------------------------------------------------------------------

MCP_PREFS_FILE = os.path.expanduser("~/.config/local-models/mcp_preferences.json")

# Task types that users can configure a preferred model for.
# Keys match the MCP server's pick_model() task parameter.
MCP_TASK_LABELS = {
    "code": "Code Generation",
    "general": "General Text",
    "translation": "Translation",
    "reasoning": "Reasoning & Review",
    "long_context": "Long Context",
}

MCP_DEFAULT_PREFS = {
    "code": ["qwen3-coder:480b", "qwen3-coder", "qwen2.5-coder:32b", "glm-4.7-flash", "qwen3.5"],
    "general": ["qwen3.5", "glm-4.7-flash", "nemotron-3-super", "qwen3.5-fast"],
    "translation": ["cogito-2.1", "qwen3.5", "glm-4.7-flash"],
    "reasoning": ["deepseek-r1:671b", "cogito-2.1", "nemotron-3-super", "qwen3.5-large", "qwen3.5", "glm-4.7-flash"],
    "long_context": ["qwen3.5", "nemotron-3-super", "glm-4.7-flash", "deepseek-r1:671b"],
}

# Filters: include/exclude patterns and numeric thresholds.
# "include_names" — model must match at least one prefix (if set).
# "exclude_names" — model is hidden if it matches any prefix.
# "min_active_b" / "min_ctx" — numeric minimums.
MCP_TASK_FILTERS = {
    "code": {
        "priority_names": ["coder"],
        "include_names": ["qwen3.5", "deepseek", "cogito", "nemotron",
                          "gpt-oss", "llama3.3"],
        "exclude_names": ["vl", "flux", "z-image", "whisper", "ocr", "tinyllama",
                          "goonsai", "nsfw"],
        "min_active_b": 3,
    },
    "general": {
        "exclude_names": ["coder", "vl", "flux", "z-image", "whisper", "ocr",
                          "tinyllama", "goonsai", "nsfw"],
        "min_active_b": 3,
    },
    "translation": {
        "exclude_names": ["coder", "vl", "flux", "z-image", "whisper", "ocr",
                          "tinyllama", "goonsai", "nsfw"],
        "min_active_b": 3,
    },
    "reasoning": {
        "exclude_names": ["coder", "vl", "flux", "z-image", "whisper", "ocr",
                          "tinyllama", "goonsai", "nsfw"],
        "min_active_b": 10,
    },
    "long_context": {
        "exclude_names": ["vl", "flux", "z-image", "whisper", "ocr",
                          "tinyllama", "goonsai", "nsfw"],
        "min_ctx": 64000,
    },
}


def _model_matches_filter(model_name, raw_info, task_filter):
    """Check if a model passes the task filter."""
    name_lower = model_name.lower()

    # Always exclude models matching exclude patterns
    excludes = task_filter.get("exclude_names", [])
    if any(p.lower() in name_lower for p in excludes):
        return False

    # Models matching "priority_names" always pass (e.g. coder models for code tasks)
    priority = task_filter.get("priority_names", [])
    if any(p.lower() in name_lower for p in priority):
        return True

    # Must match at least one include pattern (if set)
    includes = task_filter.get("include_names")
    if includes and not any(p.lower() in name_lower for p in includes):
        return False

    # Numeric thresholds
    active = raw_info.get("active", 0)
    min_active = task_filter.get("min_active_b", 0)
    if min_active and active > 0 and active < min_active:
        return False

    ctx = raw_info.get("ctx", 0)
    min_ctx = task_filter.get("min_ctx", 0)
    if min_ctx and ctx > 0 and ctx < min_ctx:
        return False

    return True

# Specialized task types matched by model name prefix.
MCP_SPECIAL_TASKS = {
    "vision": {
        "label": "Vision",
        "prefixes": ["qwen3-vl", "llava", "moondream"],
    },
    "image_gen": {
        "label": "Image Generation",
        "prefixes": ["x/flux2", "x/z-image", "flux", "stable-diffusion"],
    },
    "transcription": {
        "label": "Transcription",
        "prefixes": ["whisper"],
    },
}


# ---------------------------------------------------------------------------
# Update detection (git)
# ---------------------------------------------------------------------------

def check_repo_update_available():
    """Check if the local repo is behind origin/main. Returns (behind, summary)."""
    try:
        fetch = subprocess.run(["git", "-C", REPO_DIR, "fetch", "--quiet"],
                               capture_output=True, text=True, timeout=15)
        if fetch.returncode != 0:
            logging.warning("git fetch failed: %s", fetch.stderr.strip())
            return 0, ""
        result = subprocess.run(
            ["git", "-C", REPO_DIR, "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True, text=True, timeout=5)
        behind = int(result.stdout.strip())
        if behind > 0:
            log_result = subprocess.run(
                ["git", "-C", REPO_DIR, "log", "--oneline", "HEAD..origin/main"],
                capture_output=True, text=True, timeout=5)
            summary = log_result.stdout.strip()
            return behind, summary
    except Exception as e:
        logging.warning("Update check failed: %s", e)
    return 0, ""


def apply_repo_update():
    """Pull latest and re-run install.sh. Returns (success, output)."""
    try:
        pull = subprocess.run(
            ["git", "-C", REPO_DIR, "pull", "--rebase", "--autostash"],
            capture_output=True, text=True, timeout=30)
        if pull.returncode != 0:
            return False, pull.stderr.strip() or pull.stdout.strip()
        install = subprocess.run(
            ["bash", os.path.join(REPO_DIR, "install.sh")],
            capture_output=True, text=True, timeout=60)
        if install.returncode != 0:
            return False, f"install.sh failed: {install.stderr.strip()}"
        return True, pull.stdout.strip()
    except Exception as e:
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
            with open(MCP_TOOLS_FILE) as f:
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

ROLE_FILTERS_FILE = os.path.expanduser("~/.config/claude-code-router/role_filters.json")


def load_role_filters():
    """Load role filter config from JSON file."""
    if os.path.exists(ROLE_FILTERS_FILE):
        try:
            with open(ROLE_FILTERS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def model_fits_role(role, provider, total, active, ctx, has_vision, filters):
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


# Known active params for hybrid architectures where auto-detection fails.
# Keyed by Ollama family name → {total_b: active_b}.
_KNOWN_ACTIVE_PARAMS_B = {
    "nemotron_h_moe": {124: 12},
    "deepseek2": {671: 37},
}


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

        # Active params — multi-strategy for MoE models
        active_b = total_b
        if expert_count and expert_used and expert_count > 1:
            total_b_rounded = round(total_b)

            # Strategy 1: Parse "AXB" from model name
            match = re.search(r'[_-]A(\d+(?:\.\d+)?)B', model_name, re.IGNORECASE)
            if match:
                active_b = float(match.group(1))

            # Strategy 2: Known hybrid model lookup
            elif family in _KNOWN_ACTIVE_PARAMS_B:
                known = _KNOWN_ACTIVE_PARAMS_B[family]
                if total_b_rounded in known:
                    active_b = known[total_b_rounded]

            # Strategy 3: FFN subtraction (works for pure MoE like Qwen)
            else:
                expert_ffn = _get(".expert_feed_forward_length", 0)
                embed_len = _get(".embedding_length", 0)
                block_count = _get(".block_count", 0)
                if expert_ffn and embed_len and block_count:
                    total_moe = block_count * expert_count * expert_ffn * embed_len * 3
                    active_moe = block_count * expert_used * expert_ffn * embed_len * 3
                    computed = total_raw - total_moe + active_moe
                    if 0 < computed < total_raw:
                        active_b = computed / 1e9

                # Strategy 4: Simple ratio (last resort)
                if active_b == total_b:
                    active_b = total_b * expert_used / expert_count

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


def match_ollama_model(ccr_name, installed_models):
    """Match a CCR model name (e.g. 'qwen3.5') to an installed Ollama model.

    Tries exact match, then :latest tag, then exact base name match with
    any tag. Only falls back to prefix match as a last resort, preferring
    the variant whose tag best matches the CCR name.
    """
    # Exact match (includes tag)
    if ccr_name in installed_models:
        return ccr_name
    # With :latest tag
    if f"{ccr_name}:latest" in installed_models:
        return f"{ccr_name}:latest"

    # If CCR name has a tag (e.g. "qwen3.5:35b-a3b"), try matching base:tag
    if ":" in ccr_name:
        # The exact name wasn't found — no good match
        return None

    # CCR name has no tag (e.g. "qwen3.5") — find installed models with same base
    matches = [n for n in installed_models
               if n.split(":")[0] == ccr_name]
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
        """Bulk-populate cache for all CCR models.

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
    """Load {role: "provider,model"} overrides. Falls back to CCR defaults."""
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

class LocalModelsApp(rumps.App):
    def __init__(self):
        icon = ICON_PATH if os.path.exists(ICON_PATH) else None
        super().__init__("Local Models", icon=icon, template=True,
                         quit_button=None)

        self.conf = load_network_conf()
        self.desktop = is_desktop()
        self.ram_gb = get_ram_gb()
        self.desktop_host = self.conf["MODEL_SERVER_HOST"]
        self.ollama_port = self.conf["OLLAMA_PORT"]
        self.mlx_port = self.conf["MLX_PORT"]
        self.probe_timeout = int(self.conf["PROBE_TIMEOUT"])

        # Resolve mDNS hostname to IP once (avoids repeated cold-cache
        # lookups that eat into the probe timeout window)
        self.desktop_ip = resolve_mdns(self.desktop_host)
        remote_addr = self.desktop_ip or self.desktop_host
        self.ollama_remote = f"http://{remote_addr}:{self.ollama_port}"
        self.mlx_remote = f"http://{remote_addr}:{self.mlx_port}"

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
        self.update_summary = ""
        self.app_ready = False         # set True once run loop starts

        # Profile viewer / tool tester state
        self.profile_server = None
        self.profile_server_mode = None
        self.profile_port = None
        self.profile_window = None
        self.tools_window = None
        self._win_delegate = None
        self._tools_delegate = None

        # Menu items
        self.menu_status = rumps.MenuItem("Starting…")
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
        self.mcp_models = get_tts_models()  # models served directly by MCP
        self.menu_profiles = rumps.MenuItem("Model Profiles",
                                           callback=self.open_profiles)
        self.menu_tools = rumps.MenuItem("Playground",
                                        callback=self.open_tools)
        self.menu_update = rumps.MenuItem("Update Available")
        self.menu_restart = rumps.MenuItem("Restart", callback=self.restart_app)
        self.menu_quit = rumps.MenuItem("Quit", callback=self.quit_app)

        self.menu = [
            self.menu_status,
            None,
            self.menu_ollama,
            self.menu_mlx,
            self.menu_mcp,
            None,
            self.menu_profiles,
            self.menu_tools,
            None,
            self.menu_update,
            None,
            self.menu_restart,
            self.menu_quit,
        ]

        # Easter egg: periodic cute notifications (only for non-Jerry machines)
        self._next_woof = 0
        self._schedule_woof()

        # Defer startup to first timer tick (NSMenu isn't ready during __init__)
        self.timer = rumps.Timer(self._on_tick, POLL_INTERVAL)
        self.timer.start()

    def _on_tick(self, _):
        """Timer callback. Handles first-run initialization and periodic refresh."""
        if not self.app_ready:
            self.app_ready = True
            threading.Thread(target=self._start_services_bg, daemon=True).start()
            self._schedule_update_check()
            return
        self.refresh(None)

    def _start_services_bg(self):
        """Background thread: start services, then do first poll inline."""
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

    def start_services(self):
        """Start local servers (or detect desktop)."""
        if self.desktop:
            self._start_local_servers()
        else:
            if self.desktop_host and (
                probe_service(self.ollama_remote, self.probe_timeout)
                or probe_service(self.mlx_remote, self.probe_timeout)
            ):
                self.mode = "client"
            else:
                self._start_local_servers()
        self._start_mcp_server()

    def _start_local_servers(self):
        """Launch Ollama and MLX-OpenAI-Server via start-local-models."""
        try:
            env = os.environ.copy()
            if self.desktop:
                env["OLLAMA_HOST"] = f"0.0.0.0:{self.ollama_port}"
            # Ensure Homebrew is on PATH — launchd gives a minimal PATH
            if "/opt/homebrew/bin" not in env.get("PATH", ""):
                env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"
            self._startup_log = open("/tmp/local-models-startup.log", "w")
            subprocess.Popen(
                [os.path.expanduser("~/bin/start-local-models")],
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
        """Launch the MCP server (SSE on port 8100)."""
        if getattr(self, '_mcp_proc', None) is not None:
            if self._mcp_proc.poll() is None:
                return
        # Kill any orphaned process holding port 8100
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", "tcp:8100"], text=True, stderr=subprocess.DEVNULL)
            for pid_str in out.strip().split():
                pid = int(pid_str)
                if pid != os.getpid():
                    os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
        except (subprocess.CalledProcessError, ValueError):
            pass
        env = os.environ.copy()
        if "/opt/homebrew/bin" not in env.get("PATH", ""):
            env["PATH"] = f"/opt/homebrew/bin:{env.get('PATH', '')}"
        self._mcp_log = open("/tmp/local-models-mcp.log", "w")
        self._mcp_proc = subprocess.Popen(
            [os.path.expanduser("~/bin/local-models-mcp-detect")],
            env=env,
            stdout=self._mcp_log,
            stderr=self._mcp_log,
            start_new_session=True,
        )

    def _stop_mcp_server(self):
        """Stop the MCP server."""
        proc = getattr(self, '_mcp_proc', None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._mcp_proc = None

    def _restart_mcp(self, _):
        """Restart the MCP server (background thread)."""
        def _do():
            self._stop_mcp_server()
            time.sleep(1)
            self._start_mcp_server()
            self.refresh(None)
        threading.Thread(target=_do, daemon=True).start()

    def stop_services(self):
        """Stop local servers."""
        try:
            subprocess.run(
                [os.path.expanduser("~/bin/start-local-models"), "--stop"],
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
                if self.desktop:
                    env["OLLAMA_HOST"] = f"0.0.0.0:{self.ollama_port}"
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
                    capture_output=True, text=True, timeout=5)
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
                mlx_log = open("/tmp/local-models-mlx-restart.log", "w")
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

        if self.app_ready:
            self._update_menu()
            if (self.profile_server_mode is not None
                    and self.profile_server_mode != self.mode
                    and self.profile_window is not None):
                self._restart_profile_server_and_reload()

    def _on_webview_message(self, body):
        """Handle messages from the profiles/tools webview."""
        self._update_menu()

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
                and (not self.ollama_ok and not self.ollama_loading
                     or not self.mlx_ok and not self.mlx_loading)):
            now = time.time()
            if now - getattr(self, '_last_restart_attempt', 0) > 120:
                self._last_restart_attempt = now
                self._start_local_servers()

        self.ollama_models = get_ollama_models(OLLAMA_LOCAL) if self.ollama_ok else []
        self.mlx_models = get_mlx_models(MLX_LOCAL) if self.mlx_ok else []
        self.mcp_models = get_tts_models()

        if self.ollama_ok or self.mlx_ok:
            self.mode = "server"
        elif self.ollama_loading or self.mlx_loading:
            self.mode = "server"
        else:
            self.mode = "stopped"

    def _refresh_client_mode(self):
        desktop_ollama = False
        desktop_mlx = False

        if self.desktop_host:
            # Re-resolve mDNS if we don't have an IP yet (server may have
            # been down when we booted)
            if not self.desktop_ip:
                self.desktop_ip = resolve_mdns(self.desktop_host)
                if self.desktop_ip:
                    self.ollama_remote = (
                        f"http://{self.desktop_ip}:{self.ollama_port}")
                    self.mlx_remote = (
                        f"http://{self.desktop_ip}:{self.mlx_port}")
            desktop_ollama = probe_service(self.ollama_remote, self.probe_timeout)
            desktop_mlx = probe_service(self.mlx_remote, self.probe_timeout)

        if desktop_ollama or desktop_mlx:
            self.mode = "client"
            self.ollama_ok = desktop_ollama
            self.mlx_ok = desktop_mlx
            self.ollama_models = (
                get_ollama_models(self.ollama_remote) if desktop_ollama else [])
            self.mlx_models = (
                get_mlx_models(self.mlx_remote) if desktop_mlx else [])
        else:
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

        self.mcp_models = get_tts_models()

    @staticmethod
    def _styled_menu(item, dot, label, detail=""):
        """Set an NSAttributedString title with dot, label, and dim detail."""
        from AppKit import (NSFont, NSForegroundColorAttributeName,
                            NSFontAttributeName, NSColor,
                            NSMutableAttributedString,
                            NSParagraphStyleAttributeName,
                            NSMutableParagraphStyle)
        from Foundation import NSRange, NSString

        font = NSFont.menuFontOfSize_(13)
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

        # ── Top line: mode — profile ──
        ollama_n = len(self.ollama_models)
        mlx_n = len(self.mlx_models)
        profiles_data = load_profiles()
        active = profiles_data.get("active")
        if active and active in profiles_data.get("profiles", {}):
            profile = profiles_data["profiles"][active].get("label", active)
        else:
            profile = "No Profile"

        mode_label = {"server": "Server", "client": "Remote",
                      "offline": "Local", "stopped": "Stopped"
                      }.get(self.mode, "…")
        self._styled_menu(self.menu_status, "", mode_label)
        self._styled_menu(self.menu_profiles, "", "Model Profiles", profile)

        # ── Per-service status lines ──
        ollama_loading = getattr(self, 'ollama_loading', False)
        mlx_loading = getattr(self, 'mlx_loading', False)
        is_local = self.mode in ("server", "offline")

        GRN, YEL, RED = "\U0001f7e2", "\U0001f7e1", "\U0001f534"

        if self.ollama_ok:
            self._styled_menu(self.menu_ollama, GRN, "Ollama",
                              f"{ollama_n} models")
        elif ollama_loading:
            self._styled_menu(self.menu_ollama, YEL, "Ollama", "starting…")
        else:
            self._styled_menu(self.menu_ollama, RED, "Ollama", "down")
        self.menu_ollama_restart.title = (
            "Restart Ollama" if is_local else "Remote — restart from server")
        self.menu_ollama_restart.set_callback(
            self._restart_ollama if is_local else None)

        if self.mlx_ok:
            self._styled_menu(self.menu_mlx, GRN, "MLX",
                              f"{mlx_n} models")
        elif mlx_loading:
            self._styled_menu(self.menu_mlx, YEL, "MLX", "starting…")
        else:
            self._styled_menu(self.menu_mlx, RED, "MLX", "down")
        self.menu_mlx_restart.title = (
            "Restart MLX" if is_local else "Remote — restart from server")
        self.menu_mlx_restart.set_callback(
            self._restart_mlx if is_local else None)

        mcp_proc = getattr(self, '_mcp_proc', None)
        mcp_proc_alive = mcp_proc is not None and mcp_proc.poll() is None
        mcp_port_alive = probe_port(8100)
        mcp_alive = mcp_proc_alive or mcp_port_alive
        mcp_n = len(self.mcp_models)
        if mcp_alive:
            self._styled_menu(self.menu_mcp, GRN, "MCP",
                              f"{mcp_n} model{'s' if mcp_n != 1 else ''}")
        else:
            self._styled_menu(self.menu_mcp, RED, "MCP", "down")
            if self.servers_started:
                self._start_mcp_server()

        # ── Update (only actionable when available) ──
        if self.update_available > 0:
            n = self.update_available
            self.menu_update.title = (
                f"Update Available ({n} commit{'s' if n != 1 else ''})")
            self.menu_update.set_callback(self._update_now)
        else:
            self.menu_update.title = "Up to date"
            self.menu_update.set_callback(None)

        # ── Restart (always available) ──
        self.menu_restart.set_callback(self.restart_app)

    # -------------------------------------------------------------------
    # Profile viewer (native WKWebView window)
    # -------------------------------------------------------------------

    def _ensure_profile_server(self):
        """Start (or restart) the Flask profile server."""
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

        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.profile_port = s.getsockname()[1]
        s.close()

        env = os.environ.copy()
        env["PROFILE_SERVER_PORT"] = str(self.profile_port)
        env["OLLAMA_URL"] = (
            self.ollama_remote if self.mode == "client" else OLLAMA_LOCAL)
        env["MLX_URL"] = (
            self.mlx_remote if self.mode == "client" else MLX_LOCAL)

        log_path = "/tmp/local-models-profile-server.log"
        self._profile_log = open(log_path, "a")
        self.profile_server = subprocess.Popen(
            ["uv", "run", "--python", "3.12",
             os.path.join(SCRIPT_DIR, "profile-server.py")],
            env=env, stdout=subprocess.DEVNULL, stderr=self._profile_log)
        self.profile_server_mode = self.mode

        # Brief wait for server to become ready (runs on main thread,
        # so keep it short — the webview will retry on load failure)
        import urllib.request
        for _ in range(6):
            time.sleep(0.3)
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{self.profile_port}/api/system",
                    timeout=1)
                break
            except Exception:
                continue

    def _open_webview(self, title, path, size=(960, 700)):
        """Open a native WKWebView window at the given server path."""
        from AppKit import (NSRect, NSBackingStoreBuffered,
                            NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
                            NSApplicationActivationPolicyRegular,
                            NSApplicationActivationPolicyAccessory,
                            NSApp, NSImage)
        from WebKit import WKWebView, WKWebViewConfiguration, WKPreferences
        from Foundation import NSURL, NSURLRequest

        self._ensure_profile_server()

        frame = NSRect((200, 200), size)
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        window = _ProfileWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, NSBackingStoreBuffered, False)
        window.setTitle_(title)
        window.center()
        window.setReleasedWhenClosed_(False)

        config = WKWebViewConfiguration.alloc().init()
        prefs = config.preferences()
        try:
            prefs.setValue_forKey_(True, "mediaDevicesEnabled")
            prefs.setValue_forKey_(False, "mediaCaptureRequiresSecureConnection")
        except Exception:
            pass
        self._msg_handler = _WebViewMessageHandler.alloc().init()
        self._msg_handler.on_message = self._on_webview_message
        config.userContentController().addScriptMessageHandler_name_(
            self._msg_handler, "app")
        webview = WKWebView.alloc().initWithFrame_configuration_(
            window.contentView().bounds(), config)
        self._ui_delegate = _WebViewUIDelegate.alloc().init()
        webview.setUIDelegate_(self._ui_delegate)
        webview.setAutoresizingMask_(0x12)
        full_url = f"http://127.0.0.1:{self.profile_port}{path}"
        url = NSURL.URLWithString_(full_url)
        req = NSURLRequest.requestWithURL_cachePolicy_timeoutInterval_(
            url, 1, 30)  # 1 = NSURLRequestReloadIgnoringLocalCacheData
        webview.loadRequest_(req)
        window.contentView().addSubview_(webview)

        # Set dock icon with white background (menu bar icon stays template)
        icon_path = os.path.join(SCRIPT_DIR, "icon.png")
        if os.path.exists(icon_path):
            from AppKit import (NSColor, NSCompositingOperationSourceOver,
                                NSBezierPath)
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
        NSApp.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        NSApp.activateIgnoringOtherApps_(True)
        window.makeKeyAndOrderFront_(None)
        NSApp.dockTile().display()
        return window

    def open_profiles(self, _):
        """Open the model profiles pane."""
        if self.profile_window is not None:
            self.profile_window.makeKeyAndOrderFront_(None)
            from AppKit import NSApp
            NSApp.activateIgnoringOtherApps_(True)
            return

        from AppKit import NSApp, NSApplicationActivationPolicyAccessory
        window = self._open_webview("Model Profiles", "/")
        delegate = _ProfileWindowDelegate.alloc().init()
        delegate.callback = lambda: (
            setattr(self, 'profile_window', None),
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            if not self.tools_window else None)
        self._win_delegate = delegate
        window.setDelegate_(delegate)
        self.profile_window = window

    def open_tools(self, _):
        """Open the tool tester pane."""
        if self.tools_window is not None:
            self.tools_window.makeKeyAndOrderFront_(None)
            from AppKit import NSApp
            NSApp.activateIgnoringOtherApps_(True)
            return

        from AppKit import NSApp, NSApplicationActivationPolicyAccessory
        window = self._open_webview("Playground", "/tools", size=(720, 600))
        delegate = _ProfileWindowDelegate.alloc().init()
        delegate.callback = lambda: (
            setattr(self, 'tools_window', None),
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
            if not self.profile_window else None)
        self._tools_delegate = delegate
        window.setDelegate_(delegate)
        self.tools_window = window

    # -------------------------------------------------------------------
    # App update (git)
    # -------------------------------------------------------------------

    def _schedule_update_check(self):
        self.last_update_check = time.time()
        thread = threading.Thread(target=self._check_for_updates, daemon=True)
        thread.start()

    def _check_for_updates(self):
        behind, summary = check_repo_update_available()
        self.update_available = behind
        self.update_summary = summary
        if behind > 0:
            self.menu_update.title = f"Update Available ({behind} commit{'s' if behind != 1 else ''})"
            self.menu_update.set_callback(self._update_now)
            try:
                rumps.notification(
                    "Super Puppy",
                    "Update available",
                    f"{behind} new commit{'s' if behind != 1 else ''}",
                )
            except RuntimeError:
                pass
        else:
            self.menu_update.title = "Up to date"
            self.menu_update.set_callback(None)

    def _update_now(self, _):
        self.menu_update.title = "Updating..."
        thread = threading.Thread(target=self._apply_update, daemon=True)
        thread.start()

    def _apply_update(self):
        success, output = apply_repo_update()
        if success:
            try:
                rumps.notification("Super Puppy", "Updated successfully", "Restarting...")
            except RuntimeError:
                pass
            bundle = os.path.join(SCRIPT_DIR, "SuperPuppy.app")
            if os.path.isdir(bundle):
                # Detached relaunch: sleep past our quit, then open the app.
                # Must be fully detached so it survives our exit.
                subprocess.Popen(
                    ["bash", "-c",
                     f"sleep 2 && open '{bundle}'"],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            from PyObjCTools import AppHelper
            AppHelper.callAfter(rumps.quit_application)
        else:
            logging.error("Update failed: %s", output)
            short = output[:80] if output else "unknown error"
            self.menu_update.title = f"Update Failed: {short}"
            self.menu_update.set_callback(self._update_now)
            try:
                rumps.notification("Super Puppy", "Update failed", output[:100])
            except RuntimeError:
                pass

    # -------------------------------------------------------------------
    # Easter egg
    # -------------------------------------------------------------------

    _W = [
        "s61&3@p611:@-07&4@:06N",
        "s61&3@p611:@4\":4@w00'N",
        "s61&3@p611:@3&.*/%4@:06@50@4501@'03@\"@s/6#j6#@#3&\",N",
        "s61&3@p611:@806-%@-*,&@50@506$)@:063@#655N",
        "s61&3@p611:@5)*/,4@:063@4/065@*4@really@$65&N",
        "s61&3@p611:@8\"/54@50@,/08@8)\"5@you@8\"/5@'03@%*//&3_",
        "s61&3@p611:@8\"/54@50@45\"35@\"@)08-N",
        "s61&3@p611:@*4@\"4,*/(@ '03@1&3.*44*0/@50@)6(_",
        "s61&3@p611:@4\":4@0/-:@you@$\"/@13&7&/5@'03&45@'*3&4N@w*5)@:063@#655N",
        "s61&3@p611:@5)*/,4@1\"/%\"4@\"3&@5)&@best@\"/*.\"-4N@a'5&3@1611*&4L@0'@$0634&N",
        "s61&3@p611:@5)*/,4@)&@+645@4\"8@\"@.064&N",
        "s61&3@p611:@806-%@/&7&3@&\"5@\"@16''*/N",
        "s61&3@p611:@*4@\"-8\":4@8\"5$)*/(N",
    ]

    @staticmethod
    def _d(s):
        return "".join(
            chr(32 + (ord(c) - 32 - 32) % 95) if 32 <= ord(c) <= 126 else c
            for c in s)

    def _schedule_woof(self):
        import random
        user = os.environ.get("USER", "")
        if user in ("jerry", "jerrytalton"):
            self._next_woof = 0
            return
        delay = 48 * 3600 + random.randint(0, 24 * 3600)
        self._next_woof = time.time() + delay

    def _woof(self):
        import random
        try:
            rumps.notification(self._d("s61&3@p611:"), "",
                               self._d(random.choice(self._W)))
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
        subprocess.Popen(["open", app_path], start_new_session=True)
        rumps.quit_application()

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
        rumps.quit_application()


LOCK_FILE = os.path.expanduser("~/.config/local-models/menubar.lock")
_lock_fd = None


def acquire_lock():
    """Ensure only one instance runs using flock. Exits if another holds the lock."""
    import fcntl
    global _lock_fd
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    _lock_fd = open(LOCK_FILE, "w")
    fcntl.fcntl(_lock_fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Already running. Exiting.", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    if "--python-info" in sys.argv:
        import site
        import sysconfig
        libdir = sysconfig.get_config_var("LIBDIR")
        ldver = sysconfig.get_config_var("LDVERSION")
        print(sys.base_prefix)
        print(f"{libdir}/libpython{ldver}.dylib")
        print(site.getsitepackages()[0])
        sys.exit(0)
    acquire_lock()
    LocalModelsApp().run()
