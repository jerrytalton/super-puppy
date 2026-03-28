# /// script
# requires-python = ">=3.12"
# dependencies = ["rumps", "pyyaml"]
# ///
"""
Local Models — macOS menu bar app.

Shows the status of the local model infrastructure (Ollama + MLX-OpenAI-Server).
Auto-detects whether this machine is the desktop (server mode) or a laptop
(client mode), and whether the desktop is reachable on the LAN.

Periodically checks HuggingFace for trending new MLX models that fit this
machine's RAM, and offers to install them via notification.

Run with:  uv run local-models/menubar.py
"""

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error

import rumps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(SCRIPT_DIR, "icon.png")
ICONS_DIR = os.path.join(SCRIPT_DIR, "icons")
NETWORK_CONF = os.path.expanduser("~/.config/local-models/network.conf")
MCP_TOOLS_FILE = os.path.expanduser("~/.claude.json")
OLLAMA_LOCAL = "http://localhost:11434"
MLX_LOCAL = "http://localhost:8000"
POLL_INTERVAL = 15          # seconds between status refreshes
MODEL_CHECK_INTERVAL = 3600 # seconds between new-model checks (1 hour)

# HuggingFace API for discovering trending MLX models
HF_API_URL = "https://huggingface.co/api/models"
HF_SEARCH_PARAMS = {
    "library": "mlx",
    "sort": "trending",
    "direction": "-1",
    "limit": "30",
}

# Minimum downloads to consider a model "notable"
MIN_DOWNLOADS = 1000

# Models we already know about (won't re-suggest)
DISMISSED_MODELS_FILE = os.path.expanduser("~/.config/local-models/dismissed_models.json")

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


# ---------------------------------------------------------------------------
# MCP tool status
# ---------------------------------------------------------------------------

MCP_TOOL_LABELS = {
    "local_generate": "Code & Text Generation",
    "local_review": "Code Review",
    "local_vision": "Vision (Images)",
    "local_transcribe": "Transcription (Whisper)",
    "local_candidates": "Multi-Model Consensus",
    "local_summarize": "File Summarization",
    "local_models_status": "Model Status",
}


def is_mcp_configured():
    """Check if local-models MCP is registered in Claude config."""
    if os.path.exists(MCP_TOOLS_FILE):
        try:
            with open(MCP_TOOLS_FILE) as f:
                data = json.load(f)
            return "local-models" in data.get("mcpServers", {})
        except Exception:
            pass
    return False


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


def query_ollama_model_detail(base_url, model_name, timeout=5):
    """Query Ollama /api/show for full model architecture info.

    Returns {
        "total_params": float (billions),
        "active_params": float (billions) — same as total for dense models,
        "context": int,
        "expert_count": int or None,
        "expert_used": int or None,
    }
    """
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

        # Total params from general.parameter_count
        total_raw = model_info.get("general.parameter_count", 0)
        total_b = total_raw / 1e9 if total_raw else 0

        # Fallback to details.parameter_size
        if not total_b:
            ps = details.get("parameter_size", "")
            try:
                total_b = float(ps.rstrip("B"))
            except (ValueError, AttributeError):
                pass

        # Context length (key varies by model family)
        ctx = 0
        for k, v in model_info.items():
            if "context_length" in k:
                ctx = int(v)
                break

        # MoE: expert_count and expert_used_count
        expert_count = None
        expert_used = None
        for k, v in model_info.items():
            if k.endswith(".expert_count"):
                expert_count = int(v)
            elif k.endswith(".expert_used_count"):
                expert_used = int(v)

        # Compute active params for MoE
        if expert_count and expert_used and expert_count > 1:
            # For MoE: active = total * (active_experts / total_experts)
            # This is approximate but matches advertised numbers well
            # (e.g., 125B with 8/256 experts → ~3.9B active)
            active_b = round(total_b * expert_used / expert_count, 1)
        else:
            active_b = total_b

        has_vision = any("vision" in k for k in model_info)

        return {
            "total_params": round(total_b, 1),
            "active_params": round(active_b, 1),
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
            "total_params": round(total_b, 1),
            "active_params": round(active_b, 1),
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
        self._ollama_vision = set() # ollama models with vision capability
        self._role_filters = load_role_filters()

    def populate(self, ccr_models, ollama_url, mlx_config_info, mlx_live_models):
        """Bulk-populate cache for all CCR models.

        mlx_live_models: list of model IDs currently served by MLX-OpenAI-Server.
        """
        # Fetch all Ollama models in one call
        if self._ollama_models is None:
            self._ollama_models = query_ollama_all_models(ollama_url)

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
                parts.append(f"{total:g}B/{active:g}B")
            elif total > 0:
                parts.append(f"{total:g}B")
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

def estimate_size_gb(model_id):
    """Estimate download size in GB from model name heuristics.

    Looks for patterns like '70B', '32B-4bit', '8B-Instruct-4bit', etc.
    Returns estimated disk size for 4-bit quantized version.
    """
    # Find parameter count (e.g., "70B", "32B", "397B")
    match = re.search(r"(\d+)B", model_id, re.IGNORECASE)
    if not match:
        return None

    params_b = int(match.group(1))

    # Check if it's a MoE with active params noted (e.g., "397B-A17B")
    moe_match = re.search(r"(\d+)B-A(\d+)B", model_id, re.IGNORECASE)
    if moe_match:
        params_b = int(moe_match.group(1))  # total params for storage

    # 4-bit quant: ~0.5 GB per billion params; 8-bit: ~1 GB/B
    if "8bit" in model_id.lower() or "8-bit" in model_id.lower():
        return params_b * 1.0
    else:
        return params_b * 0.5  # assume 4-bit


def load_dismissed_models():
    """Load the set of model IDs the user has dismissed."""
    if os.path.exists(DISMISSED_MODELS_FILE):
        try:
            with open(DISMISSED_MODELS_FILE) as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_dismissed_models(dismissed):
    """Save the set of dismissed model IDs."""
    os.makedirs(os.path.dirname(DISMISSED_MODELS_FILE), exist_ok=True)
    with open(DISMISSED_MODELS_FILE, "w") as f:
        json.dump(list(dismissed), f)


def fetch_trending_mlx_models():
    """Fetch trending MLX models from HuggingFace."""
    params = "&".join(f"{k}={v}" for k, v in HF_SEARCH_PARAMS.items())
    url = f"{HF_API_URL}?{params}"
    return http_get_json(url, timeout=15) or []


def discover_new_models(ram_gb, installed_names, dismissed):
    """Find notable new MLX models that fit this machine and aren't installed.

    Returns a list of dicts: {id, downloads, size_gb, description}
    """
    max_size_gb = ram_gb * 0.8  # leave 20% headroom
    trending = fetch_trending_mlx_models()
    results = []

    # Normalize installed names for matching
    installed_lower = {n.lower().replace("/", "-").replace(":", "-")
                       for n in installed_names}

    for model in trending:
        model_id = model.get("id", "")  # e.g. "mlx-community/Qwen3.5-35B-A3B-4bit"
        downloads = model.get("downloads", 0)

        if downloads < MIN_DOWNLOADS:
            continue

        if model_id in dismissed:
            continue

        # Check if already installed (fuzzy match on name fragments)
        base_name = model_id.split("/")[-1].lower().replace("-", "")
        already_have = any(base_name[:15] in inst.replace("-", "").replace(":", "")
                          for inst in installed_lower)
        if already_have:
            continue

        # Size check
        size_gb = estimate_size_gb(model_id)
        if size_gb and size_gb > max_size_gb:
            continue

        # Must be a text generation model
        tags = model.get("tags", [])
        pipeline = model.get("pipeline_tag", "")
        if pipeline and pipeline not in ("text-generation", "text2text-generation", ""):
            continue

        results.append({
            "id": model_id,
            "downloads": downloads,
            "size_gb": size_gb,
            "likes": model.get("likes", 0),
            "last_modified": model.get("lastModified", ""),
        })

    # Sort by downloads (most popular first)
    results.sort(key=lambda m: m["downloads"], reverse=True)
    return results[:10]


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

        # Remote URLs (desktop on LAN)
        self.ollama_remote = f"http://{self.desktop_host}:{self.ollama_port}"
        self.mlx_remote = f"http://{self.desktop_host}:{self.mlx_port}"

        # State
        self.mode = "unknown"          # server, client, offline, stopped
        self.ollama_ok = False
        self.mlx_ok = False
        self.ollama_loading = False    # process exists but not responding
        self.mlx_loading = False
        self.ollama_models = []
        self.mlx_models = []
        self.servers_started = False
        self.new_models = []           # discovered models not yet installed
        self.dismissed = load_dismissed_models()
        self.mcp_configured = is_mcp_configured()
        self.model_info_cache = ModelInfoCache()
        self.mlx_config_info = query_mlx_model_info_from_config()
        self.last_model_check = 0
        self.app_ready = False         # set True once run loop starts

        # Menu items
        self.menu_mode = rumps.MenuItem("Mode: detecting...")
        self.menu_separator1 = rumps.separator
        self.menu_ollama = rumps.MenuItem("Ollama: ...")
        self.menu_mlx = rumps.MenuItem("MLX Server: ...")
        self.menu_separator2 = rumps.separator
        self.menu_services = rumps.MenuItem("Services")
        self.menu_models_header = rumps.MenuItem("MCP Tools")
        self.menu_separator3 = rumps.separator
        self.menu_new_models_header = rumps.MenuItem("New Models Available")
        self.menu_check_now = rumps.MenuItem("Check for New Models", callback=self.check_models_now)
        self.menu_separator4 = rumps.separator
        self.menu_action = rumps.MenuItem("Start MLX Server", callback=self.toggle_services)
        self.menu_quit = rumps.MenuItem("Quit", callback=self.quit_app)

        self.menu = [
            self.menu_mode,
            self.menu_separator1,
            self.menu_ollama,
            self.menu_mlx,
            self.menu_separator2,
            self.menu_services,
            self.menu_models_header,
            self.menu_separator3,
            self.menu_new_models_header,
            self.menu_check_now,
            self.menu_separator4,
            self.menu_action,
            None,
            self.menu_quit,
        ]

        # Defer startup to first timer tick (NSMenu isn't ready during __init__)
        self.timer = rumps.Timer(self._on_tick, POLL_INTERVAL)
        self.timer.start()

    def _on_tick(self, _):
        """Timer callback. Handles first-run initialization and periodic refresh."""
        if not self.app_ready:
            self.app_ready = True
            self.start_services()
            self._schedule_model_check()
            return
        self.refresh(None)

    # -------------------------------------------------------------------
    # Service management
    # -------------------------------------------------------------------

    def start_services(self):
        """Start local servers (or detect desktop)."""
        if self.desktop:
            self._start_local_servers()
        else:
            if self.desktop_host and probe_service(self.ollama_remote, self.probe_timeout):
                self.mode = "client"
            else:
                self._start_local_servers()

        self.servers_started = True
        self.refresh(None)

    def _start_local_servers(self):
        """Launch Ollama and MLX-OpenAI-Server via start-local-models."""
        try:
            env = os.environ.copy()
            if self.desktop:
                env["OLLAMA_HOST"] = f"0.0.0.0:{self.ollama_port}"
            subprocess.Popen(
                [os.path.expanduser("~/bin/start-local-models")],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.mode = "server" if self.desktop else "offline"
        except Exception as e:
            rumps.notification("Local Models", "Failed to start services", str(e))

    def stop_services(self):
        """Stop local servers."""
        try:
            subprocess.run(
                [os.path.expanduser("~/bin/start-local-models"), "--stop"],
                capture_output=True,
            )
            self.servers_started = False
            self.mode = "stopped"
            self.refresh(None)
        except Exception:
            pass

    def toggle_services(self, sender):
        if self.mode == "stopped":
            self.start_services()
        else:
            self.stop_services()

    # -------------------------------------------------------------------
    # Status refresh
    # -------------------------------------------------------------------

    def refresh(self, _):
        """Poll services and update the menu."""
        if self.desktop:
            self._refresh_server_mode()
        else:
            self._refresh_client_mode()

        # Periodic model discovery check
        if time.time() - self.last_model_check > MODEL_CHECK_INTERVAL:
            self._schedule_model_check()

        if self.app_ready:
            self._update_menu()

    def _refresh_server_mode(self):
        self.ollama_ok = probe_service(OLLAMA_LOCAL, 2)
        self.mlx_ok = probe_service(MLX_LOCAL, 2)
        self.ollama_loading = not self.ollama_ok and process_is_running("ollama")
        self.mlx_loading = not self.mlx_ok and process_is_running("mlx-openai-server")
        self.ollama_models = get_ollama_models(OLLAMA_LOCAL) if self.ollama_ok else []
        self.mlx_models = get_mlx_models(MLX_LOCAL) if self.mlx_ok else []

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
            desktop_ollama = probe_service(self.ollama_remote, self.probe_timeout)
            desktop_mlx = probe_service(self.mlx_remote, self.probe_timeout)

        if desktop_ollama:
            self.mode = "client"
            self.ollama_ok = True
            self.mlx_ok = desktop_mlx
            self.ollama_models = get_ollama_models(self.ollama_remote)
            self.mlx_models = get_mlx_models(self.mlx_remote) if desktop_mlx else []
        else:
            self.ollama_ok = probe_service(OLLAMA_LOCAL, 2)
            self.mlx_ok = probe_service(MLX_LOCAL, 2)

            if self.ollama_ok or self.mlx_ok:
                self.mode = "offline"
                self.ollama_models = get_ollama_models(OLLAMA_LOCAL) if self.ollama_ok else []
                self.mlx_models = get_mlx_models(MLX_LOCAL) if self.mlx_ok else []
            elif self.servers_started:
                self.mode = "offline"
                self.ollama_models = []
                self.mlx_models = []
            else:
                self.mode = "stopped"
                self.ollama_models = []
                self.mlx_models = []

    def _update_menu(self):
        """Rebuild the menu to reflect current state."""

        self.title = None

        # Mode label
        mode_labels = {
            "server": "Server Mode (LAN)",
            "client": f"Client → {self.desktop_host}",
            "offline": "Offline (local fallback)",
            "stopped": "Stopped",
        }
        self.menu_mode.title = mode_labels.get(self.mode, "Unknown")

        # Service status
        def _status(ok, loading):
            if ok:
                return "Running"
            elif loading:
                return "Loading..."
            else:
                return "Down"

        ollama_status = _status(self.ollama_ok, self.ollama_loading)
        mlx_status = _status(self.mlx_ok, self.mlx_loading)

        if self.mode == "client":
            ollama_status = "Connected" if self.ollama_ok else "Down"
            mlx_status = "Connected" if self.mlx_ok else "Down"
            self.menu_ollama.title = f"Ollama: {ollama_status} ({self.desktop_host})"
            self.menu_mlx.title = f"MLX Server: {mlx_status} ({self.desktop_host})"
        elif self.mode == "server":
            self.menu_ollama.title = f"Ollama: {ollama_status} (0.0.0.0:{self.ollama_port})"
            self.menu_mlx.title = f"MLX Server: {mlx_status} (0.0.0.0:{self.mlx_port})"
        else:
            self.menu_ollama.title = f"Ollama: {ollama_status}"
            self.menu_mlx.title = f"MLX Server: {mlx_status}"

        # MCP Tools submenu — show available tools and their status
        try:
            self.menu_models_header.clear()
        except AttributeError:
            pass

        self.mcp_configured = is_mcp_configured()
        if self.mcp_configured:
            status = "Configured" if (self.ollama_ok or self.mlx_ok) else "No backends"
            self.menu_models_header.add(rumps.MenuItem(f"  local-models: {status}"))
            self.menu_models_header.add(rumps.separator)
            for tool_name, label in MCP_TOOL_LABELS.items():
                item = rumps.MenuItem(f"  {label}")
                item.state = 1
                self.menu_models_header.add(item)
        else:
            self.menu_models_header.add(rumps.MenuItem("  (not configured in Claude)"))
            self.menu_models_header.add(rumps.MenuItem("  Add local-models to ~/.claude.json"))

        # Services submenu — each service is a sub-menu with selectable implementations
        try:
            self.menu_services.clear()
        except AttributeError:
            pass

        # Speech-to-Text
        stt_menu = rumps.MenuItem("Speech-to-Text")
        stt_whisper = rumps.MenuItem("  Whisper v3 (MLX, local)")
        stt_whisper.state = 1 if "whisper-v3" in set(self.mlx_models) else 0
        stt_menu.add(stt_whisper)
        self.menu_services.add(stt_menu)

        # Web Search
        ws_menu = rumps.MenuItem("Web Search")
        ws_open = rumps.MenuItem("  open-webSearch (Bing, DuckDuckGo, Brave)")
        ws_open.state = 1
        ws_menu.add(ws_open)
        self.menu_services.add(ws_menu)

        # Summary
        self.menu_services.add(rumps.separator)
        self.menu_services.add(rumps.MenuItem(f"  {len(self.ollama_models)} Ollama models"))
        self.menu_services.add(rumps.MenuItem(f"  {len(self.mlx_models)} MLX models"))

        # New models submenu
        try:
            self.menu_new_models_header.clear()
        except AttributeError:
            pass
        if self.new_models:
            count = len(self.new_models)
            self.menu_new_models_header.title = f"New Models Available ({count})"
            for m in self.new_models:
                size_str = f" ~{m['size_gb']:.0f}GB" if m["size_gb"] else ""
                downloads = m["downloads"]
                if downloads >= 1_000_000:
                    dl_str = f"{downloads / 1_000_000:.1f}M"
                elif downloads >= 1_000:
                    dl_str = f"{downloads / 1_000:.0f}K"
                else:
                    dl_str = str(downloads)
                label = f"  {m['id'].split('/')[-1]}{size_str} ({dl_str} downloads)"
                item = rumps.MenuItem(label, callback=self._make_install_callback(m))
                self.menu_new_models_header.add(item)

            # Dismiss all option
            self.menu_new_models_header.add(rumps.separator)
            self.menu_new_models_header.add(
                rumps.MenuItem("  Dismiss All", callback=self.dismiss_all_new_models)
            )
        else:
            self.menu_new_models_header.title = "New Models Available"
            self.menu_new_models_header.add(rumps.MenuItem("  (none found)"))

        # Action button
        if self.mode == "stopped":
            self.menu_action.title = "Start MLX Server"
        else:
            self.menu_action.title = "Stop MLX Server"

    # -------------------------------------------------------------------
    # Model discovery
    # -------------------------------------------------------------------

    def _schedule_model_check(self):
        """Run model discovery in a background thread."""
        self.last_model_check = time.time()
        thread = threading.Thread(target=self._check_for_new_models, daemon=True)
        thread.start()

    def _check_for_new_models(self):
        """Background: query HuggingFace for trending models we don't have."""
        try:
            installed = set(self.ollama_models + self.mlx_models)
            new = discover_new_models(self.ram_gb, installed, self.dismissed)

            if new and new != self.new_models:
                self.new_models = new
                # Send macOS notification
                names = [m["id"].split("/")[-1] for m in new[:3]]
                summary = ", ".join(names)
                if len(new) > 3:
                    summary += f" +{len(new) - 3} more"
                rumps.notification(
                    "Local Models",
                    f"{len(new)} new model{'s' if len(new) != 1 else ''} available",
                    summary,
                )
            elif not new:
                self.new_models = []
        except Exception:
            pass  # fail silently, we'll try again next cycle

    def check_models_now(self, _):
        """Manual trigger for model check."""
        self.menu_check_now.title = "Checking..."
        self._schedule_model_check()
        # Reset title after a delay
        threading.Timer(5.0, self._reset_check_title).start()

    def _reset_check_title(self):
        self.menu_check_now.title = "Check for New Models"

    def _make_install_callback(self, model_info):
        """Create a callback for installing a specific model."""
        def callback(_):
            self._install_model(model_info)
        return callback

    def _install_model(self, model_info):
        """Prompt to install a model, then pull it."""
        model_id = model_info["id"]
        short_name = model_id.split("/")[-1]
        size_str = f" (~{model_info['size_gb']:.0f}GB)" if model_info["size_gb"] else ""

        response = rumps.alert(
            title=f"Install {short_name}?",
            message=(
                f"Download {model_id}{size_str} from HuggingFace?\n\n"
                f"Downloads: {model_info['downloads']:,}\n"
                f"Likes: {model_info['likes']:,}\n\n"
                f"The model will be available in MLX-OpenAI-Server after restart."
            ),
            ok="Install",
            cancel="Cancel",
        )

        if response == 1:  # OK clicked
            rumps.notification("Local Models", "Downloading...", short_name)
            thread = threading.Thread(
                target=self._pull_model_background,
                args=(model_id, short_name),
                daemon=True,
            )
            thread.start()
        else:
            # Dismiss this model
            self.dismissed.add(model_id)
            save_dismissed_models(self.dismissed)
            self.new_models = [m for m in self.new_models if m["id"] != model_id]
            self._update_menu()

    def _pull_model_background(self, model_id, short_name):
        """Download a model in the background via huggingface-cli."""
        try:
            subprocess.run(
                ["huggingface-cli", "download", model_id],
                capture_output=True,
                timeout=7200,  # 2 hour max
            )
            rumps.notification(
                "Local Models",
                "Download complete",
                f"{short_name} is ready. Restart MLX-OpenAI-Server to load it.",
            )
            # Remove from new models list
            self.new_models = [m for m in self.new_models if m["id"] != model_id]
        except subprocess.TimeoutExpired:
            rumps.notification("Local Models", "Download timed out", short_name)
        except Exception as e:
            rumps.notification("Local Models", "Download failed", str(e))

    def dismiss_all_new_models(self, _):
        """Dismiss all currently listed new models."""
        for m in self.new_models:
            self.dismissed.add(m["id"])
        save_dismissed_models(self.dismissed)
        self.new_models = []
        self._update_menu()

    # -------------------------------------------------------------------
    # Quit — rumps' built-in Quit button calls this before exiting
    # -------------------------------------------------------------------

    def quit_app(self, _):
        if self.servers_started and self.mode != "client":
            self.stop_services()
        rumps.quit_application()


if __name__ == "__main__":
    LocalModelsApp().run()
