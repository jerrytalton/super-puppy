# /// script
# requires-python = ">=3.12"
# dependencies = ["flask==3.1.3", "pyyaml==6.0.3", "requests==2.33.1", "mlx-audio[tts] @ git+https://github.com/Blaizzy/mlx-audio.git"]
# ///
"""
Model Profile Server for Super Puppy.

Web-based preference pane for managing which models are loaded
and which back each MCP tool task. Launched from the menu bar app.
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import yaml
from flask import (Flask, Response, after_this_request, jsonify, request,
                     send_file, send_from_directory)

from lib.models import (
    ALWAYS_EXCLUDE,
    KNOWN_ACTIVE_PARAMS,
    MCP_PREFS_FILE,
    MLX_SERVER_CONFIG,
    NETWORK_CONF,
    PROFILES_FILE,
    SPECIAL_TASKS,
    STANDARD_TASKS,
    TASK_FILTERS,
    active_params_b,
    model_matches_filter as _model_matches_filter,
    validate_network_conf,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

# ── Config ───────────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
HTML_FILE = Path(__file__).parent / "profiles.html"
TOOLS_HTML = Path(__file__).parent / "tools.html"

IDLE_TIMEOUT = int(os.environ.get("PROFILE_IDLE_TIMEOUT", "600"))

# Playground request tracking — keyed by thread ID so overlapping requests don't clobber
_playground_lock = threading.Lock()
_playground_active: dict[int, dict] = {}  # thread_id → {tool, model, backend, started}
PORT = int(os.environ.get("PROFILE_SERVER_PORT", "0"))  # 0 = random
HOST = os.environ.get("PROFILE_HOST", "127.0.0.1")

def load_default_prefs() -> dict[str, list[str]]:
    """Load ranked model preferences from config file."""
    if MCP_PREFS_FILE.exists():
        try:
            prefs = json.loads(MCP_PREFS_FILE.read_text())
            return {k: (v if isinstance(v, (list, dict)) else [v]) for k, v in prefs.items()}
        except Exception:
            pass
    return {}

# ── Idle shutdown ────────────────────────────────────────────────────

_last_request = time.time()


def _idle_watcher():
    if IDLE_TIMEOUT <= 0:
        return  # disabled — keep running forever
    while True:
        time.sleep(30)
        if time.time() - _last_request > IDLE_TIMEOUT:
            logging.info("Idle timeout — shutting down.")
            sys.exit(0)


# ── Ollama queries ───────────────────────────────────────────────────

def ollama_get(path, timeout=10):
    try:
        return requests.get(f"{OLLAMA_URL}{path}", timeout=timeout).json()
    except Exception:
        return None


def ollama_post(path, body, timeout=10):
    try:
        return requests.post(f"{OLLAMA_URL}{path}", json=body, timeout=timeout).json()
    except Exception:
        return None


def _is_remote_ollama():
    from urllib.parse import urlparse
    host = urlparse(OLLAMA_URL).hostname or ""
    return host not in ("localhost", "127.0.0.1", "::1")


def _read_server_ram_gb():
    """Read SERVER_RAM_GB from network.conf (set for the remote desktop)."""
    if NETWORK_CONF.exists():
        for line in NETWORK_CONF.read_text().splitlines():
            line = line.strip()
            if line.startswith("SERVER_RAM_GB="):
                val = line.partition("=")[2].strip().strip('"').strip("'")
                digits = "".join(c for c in val if c.isdigit())
                if digits:
                    v = int(digits)
                    if v > 0:
                        return v
    return None


def _query_server_ram_gb():
    """Query the remote server's RAM via Tailscale SSH (best-effort, cached)."""
    if not hasattr(_query_server_ram_gb, "_cache"):
        _query_server_ram_gb._cache = None
    if _query_server_ram_gb._cache is not None:
        return _query_server_ram_gb._cache
    ts_hostname = ""
    if NETWORK_CONF.exists():
        for line in NETWORK_CONF.read_text().splitlines():
            line = line.strip()
            if line.startswith("TAILSCALE_HOSTNAME="):
                ts_hostname = line.partition("=")[2].strip().strip('"').strip("'")
    if not ts_hostname:
        return None
    try:
        raw = subprocess.check_output(
            ["ssh", "-o", "ConnectTimeout=3", "-o", "BatchMode=yes",
             ts_hostname, "sysctl -n hw.memsize"],
            text=True, timeout=5).strip()
        gb = int(raw) >> 30
        if gb > 0:
            _query_server_ram_gb._cache = gb
            return gb
    except Exception:
        pass
    return None


def get_system_info():
    try:
        if _is_remote_ollama():
            try:
                url = f"{_desktop_profile_server_url()}/api/system"
                data = requests.get(url, timeout=5).json()
                data["mode"] = "client"
                return data
            except Exception:
                pass
            gb = _read_server_ram_gb() or _query_server_ram_gb()
            if gb:
                return {"total_ram_bytes": gb << 30, "total_ram_gb": gb,
                        "mode": "client"}
        raw = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        gb = raw >> 30
        mode = "server" if gb >= 256 else "offline"
        return {"total_ram_bytes": raw, "total_ram_gb": gb, "mode": mode}
    except Exception:
        return {"total_ram_bytes": 0, "total_ram_gb": 0, "mode": "unknown"}


def compute_active_params(model_name, total_b, mi, family):
    """Multi-strategy active param calculation for MoE models.

    Extracts architecture fields from Ollama model_info and delegates
    to lib.models.active_params_b.
    """
    def _get(suffix, default=None):
        for k, v in mi.items():
            if k.endswith(suffix) and ".vision." not in k:
                return v
        return default

    expert_count = _get(".expert_count")
    expert_used = _get(".expert_used_count")
    if expert_count:
        expert_count = int(expert_count)
    if expert_used:
        expert_used = int(expert_used)

    return active_params_b(
        model_name=model_name,
        total_b=total_b,
        family=family,
        expert_count=expert_count,
        expert_used=expert_used,
        expert_ffn=int(_get(".expert_feed_forward_length", 0) or 0),
        embed_len=int(_get(".embedding_length", 0) or 0),
        block_count=int(_get(".block_count", 0) or 0),
    )


_model_cache = {"data": None, "ts": 0}
_MODEL_CACHE_TTL = 60  # seconds


def get_all_models(force_refresh: bool = False):
    """Aggregate all Ollama + MLX models with metadata. Cached for 60s."""
    now = time.time()
    if (not force_refresh
            and _model_cache["data"] is not None
            and now - _model_cache["ts"] < _MODEL_CACHE_TTL):
        return _model_cache["data"]
    models = _fetch_all_models()
    _model_cache["data"] = models
    _model_cache["ts"] = now
    return models


def _desktop_profile_server_url():
    """Derive the desktop's profile server URL from OLLAMA_URL."""
    from urllib.parse import urlparse
    parsed = urlparse(OLLAMA_URL)
    return f"{parsed.scheme}://{parsed.hostname}:8101"


def _fetch_all_models():
    """Uncached model aggregation."""
    models = {}

    # Client mode: get models from the desktop's profile server, not raw backends
    if _is_remote_ollama():
        try:
            url = f"{_desktop_profile_server_url()}/api/models"
            resp = requests.get(url, timeout=10)
            remote_models = resp.json()
            return {m["name"]: m for m in remote_models}
        except Exception as e:
            logging.warning("Failed to fetch models from desktop profile server: %s", e)

    # Server/offline mode: discover locally
    tags = ollama_get("/api/tags") or {}
    for m in tags.get("models", []):
        name = m["name"]
        details = m.get("details", {})
        disk_bytes = m.get("size", 0)
        total_b = 0.0
        try:
            ps = details.get("parameter_size", "0")
            if ps.upper().endswith("M"):
                total_b = float(ps[:-1]) / 1000
            else:
                total_b = float(ps.rstrip("B"))
        except (ValueError, AttributeError):
            pass

        # Get architecture details
        show = ollama_post("/api/show", {"name": name}, timeout=5) or {}
        mi = show.get("model_info", {})
        family = show.get("details", {}).get("family", details.get("family", ""))
        ctx = 0
        for k, v in mi.items():
            if k.endswith(".context_length"):
                ctx = int(v)
                break
        has_vision = any("vision" in k for k in mi)

        # For models where Ollama doesn't report params (image gen), estimate from disk
        if not total_b and disk_bytes:
            total_b = disk_bytes / 2e9  # rough: ~0.5 bytes per param at 4-bit
        active_b = compute_active_params(name, total_b, mi, family) if total_b else 0

        models[name] = {
            "name": name,
            "backend": "ollama",
            "disk_bytes": disk_bytes,
            "vram_bytes": int(disk_bytes * 1.2),  # estimate; overridden if loaded
            "total_params_b": round(total_b, 1),
            "active_params_b": round(active_b, 1),
            "context": ctx,
            "has_vision": has_vision,
            "family": family,
            "quant": details.get("quantization_level", ""),
            "is_loaded": False,
            "expires_at": None,
        }

    # Mark loaded models with actual VRAM
    ps = ollama_get("/api/ps") or {}
    for m in ps.get("models", []):
        name = m["name"]
        if name in models:
            models[name]["is_loaded"] = True
            models[name]["vram_bytes"] = m.get("size_vram", m.get("size", 0))
            models[name]["expires_at"] = m.get("expires_at")

    # MLX models
    try:
        resp = requests.get(f"{MLX_URL}/v1/models", timeout=5)
        mlx_models = resp.json().get("data", [])
    except Exception:
        mlx_models = []

    mlx_config = {}
    if MLX_SERVER_CONFIG.exists():
        try:
            cfg = yaml.safe_load(MLX_SERVER_CONFIG.read_text())
            for entry in cfg.get("models", []):
                served_name = entry.get("served_model_name", "")
                mlx_config[served_name] = entry
        except Exception:
            pass

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

    for m in mlx_models:
        mid = m["id"]
        cfg = mlx_config.get(mid, {})
        on_demand = cfg.get("on_demand", False)
        model_path = cfg.get("model_path", "")

        # Parse params from model path (e.g. "Qwen3.5-397B-A17B-4bit")
        # Known models whose names don't contain a param count
        _KNOWN_MLX_PARAMS = {
            "whisper": 1.5,  # whisper-large-v3 is ~1.5B
        }
        total_b, active_b = 0, 0
        total_match = re.search(r'(\d+)B', model_path)
        if total_match:
            total_b = int(total_match.group(1))
        else:
            for prefix, params in _KNOWN_MLX_PARAMS.items():
                if prefix in model_path.lower() or prefix in mid.lower():
                    total_b = params
                    break
        active_match = re.search(r'A(\d+)B', model_path)
        if active_match:
            active_b = int(active_match.group(1))
        if not active_b:
            active_b = total_b

        # Estimate VRAM: ~0.5 bytes per param at 4-bit quant
        est_bytes = int(total_b * 1e9 * 0.5) if total_b else 0

        # Detect vision: check HuggingFace config.json for vision_config
        has_vision = "vision" in model_path.lower() or "vl" in model_path.lower()
        if not has_vision and model_path:
            cache_dir = hf_cache / f"models--{model_path.replace('/', '--')}" / "snapshots"
            if cache_dir.exists():
                for snap in sorted(cache_dir.iterdir(), reverse=True):
                    hf_cfg = snap / "config.json"
                    if hf_cfg.exists():
                        try:
                            hf = json.loads(hf_cfg.read_text())
                            has_vision = "vision_config" in hf or "vision_config" in hf.get("text_config", {})
                        except Exception:
                            pass
                        break

        models[mid] = {
            "name": mid,
            "backend": "mlx",
            "disk_bytes": est_bytes,
            "vram_bytes": est_bytes,
            "total_params_b": total_b,
            "active_params_b": active_b,
            "context": cfg.get("context_length", 0),
            "has_vision": has_vision,
            "family": "mlx",
            "quant": "4bit" if "4bit" in model_path else "",
            "is_loaded": not on_demand,  # always-on models are loaded
            "expires_at": None,
            "on_demand": on_demand,
        }

    # HuggingFace cache: TTS, transcription, image_edit, image_gen models
    _TASK_BACKENDS = {
        "tts": "mlx-audio",
        "transcription": "mlx",
        "image_edit": "mflux",
        "image_gen": "mflux",
    }
    from lib.hf_scanner import scan_hf_cache
    for hf_model in scan_hf_cache(_TASK_BACKENDS.keys()):
        name = hf_model["name"]
        if name in models:
            continue
        quant_str = f"{hf_model['quant_bits']}bit" if hf_model["quant_bits"] else ""
        if not quant_str and hf_model["dtypes"]:
            quant_str = hf_model["dtypes"][0].lower()
        models[name] = {
            "name": name,
            "backend": _TASK_BACKENDS[hf_model["task"]],
            "disk_bytes": hf_model["disk_bytes"],
            "vram_bytes": hf_model["vram_bytes"],
            "total_params_b": hf_model["total_params_b"],
            "active_params_b": hf_model["total_params_b"],
            "context": 0,
            "has_vision": False,
            "family": hf_model["task"],
            "quant": quant_str,
            "is_loaded": False,
            "on_demand": True,
            "expires_at": None,
        }

    return models


def model_matches_filter(name, model_info, task_filter):
    return _model_matches_filter(
        name,
        model_info.get("active_params_b", 0),
        model_info.get("context", 0),
        task_filter,
    )


_LLM_BACKENDS = {"ollama", "mlx"}


def get_eligible_tasks(name, model_info):
    """Return list of task keys this model qualifies for."""
    tasks = []
    backend = model_info.get("backend", "")

    # TASK_FILTERS (code, general, reasoning, etc.) only apply to LLM backends
    if backend in _LLM_BACKENDS:
        for task, filt in TASK_FILTERS.items():
            if model_matches_filter(name, model_info, filt):
                tasks.append(task)

    # SPECIAL_TASKS match by name prefix (vision, image_gen, tts, etc.)
    for task, spec in SPECIAL_TASKS.items():
        name_lower = name.lower()
        if any(name.startswith(p) or p.lower() in name_lower for p in spec["prefixes"]):
            tasks.append(task)

    # HF-scanned models carry a task from the scanner — ensure it's included
    family = model_info.get("family", "")
    if family in SPECIAL_TASKS and family not in tasks:
        tasks.append(family)

    if model_info.get("has_vision") and "vision" not in tasks:
        tasks.append("vision")
    return tasks


# ── Profiles ─────────────────────────────────────────────────────────

PROFILES_VERSION = 16  # bump to force-refresh preset profiles on all machines

DEFAULT_PROFILES = {
    "version": PROFILES_VERSION,
    "active": "everyday",
    "profiles": {
        "everyday": {
            "label": "Everyday",
            "description": "Best balance for high-memory machines (256GB+)",
            "max_ram_gb": 512,
            "tasks": {
                "code": "qwen3-coder-next:latest",
                "general": "qwen3.5-fast",
                "reasoning": "nemotron-super",
                "long_context": "nemotron-super",
                "translation": "qwen3.5-fast",
                "vision": "qwen3.5-large",
                "image_gen": "x/z-image-turbo:latest",
                "image_edit": "black-forest-labs/FLUX.1-Kontext-dev",
                "transcription": "whisper-v3",
                "tts": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
                "embedding": "all-minilm:latest",
                "unfiltered": "dolphin3:8b",
                "computer_use": "avil/ui-tars:latest",

            },
        },
        "desktop": {
            "label": "Desktop",
            "description": "Fits in 64GB",
            "max_ram_gb": 64,
            "tasks": {
                "code": "qwen3.5-fast",
                "general": "qwen3.5-fast",
                "reasoning": "qwen3.5-fast",
                "long_context": "glm-4.7-flash:latest",
                "translation": "qwen3.5-fast",
                "vision": "qwen3.5:9b",
                "image_gen": "x/flux2-klein:latest",
                "transcription": "whisper-v3",
                "tts": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
                "embedding": "all-minilm:latest",
                "unfiltered": "dolphin3:8b",
                "computer_use": "maternion/fara:7b",

            },
        },
        "maximum": {
            "label": "Heavyweight",
            "description": "Biggest models for everything, damn the RAM",
            "max_ram_gb": 512,
            "tasks": {
                "code": "qwen3-coder-next:latest",
                "general": "qwen3.5-large",
                "reasoning": "qwen3.5-large",
                "long_context": "qwen3.5-large",
                "translation": "qwen3.5-large",
                "vision": "qwen3.5-large",
                "image_gen": "x/z-image-turbo:bf16",
                "image_edit": "black-forest-labs/FLUX.1-Kontext-dev",
                "transcription": "whisper-v3",
                "tts": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
                "embedding": "mxbai-embed-large:latest",
                "unfiltered": "dolphin3:8b",
                "computer_use": "avil/ui-tars:latest",

            },
        },
        "laptop": {
            "label": "Laptop",
            "description": "Fits in 32GB",
            "max_ram_gb": 32,
            "tasks": {
                "code": "qwen3.5-small",
                "general": "qwen3.5-small",
                "reasoning": "qwen3.5-small",
                "long_context": "qwen3.5-small",
                "translation": "qwen3.5-small",
                "vision": "qwen3.5:9b",
                "image_gen": "x/flux2-klein:latest",
                "transcription": "whisper-v3",
                "tts": "mlx-community/Kokoro-82M-bf16",
                "embedding": "all-minilm:latest",
                "unfiltered": "dolphin3:8b",
                "computer_use": "maternion/fara:7b",

            },
        },
    },
}


def load_profiles():
    if PROFILES_FILE.exists():
        try:
            data = json.loads(PROFILES_FILE.read_text())
            if data.get("version", 0) == PROFILES_VERSION:
                return data
            # Version bump: refresh presets, keep user's selection if it's a preset
            active = data.get("active", DEFAULT_PROFILES["active"])
            if active not in DEFAULT_PROFILES["profiles"]:
                active = DEFAULT_PROFILES["active"]
            refreshed = {**DEFAULT_PROFILES, "active": active}
            save_profiles(refreshed)
            return refreshed
        except Exception:
            pass
    save_profiles(DEFAULT_PROFILES)
    return {**DEFAULT_PROFILES}


def save_profiles(data):
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(json.dumps(data, indent=2))


def save_mcp_prefs(prefs):
    MCP_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MCP_PREFS_FILE.write_text(json.dumps(prefs, indent=2))


# ── Flask app ────────────────────────────────────────────────────────

app = Flask(__name__)

# ── Bearer token auth for remote access ─────────────────────────────
# Local requests (from menu bar webview) skip auth. Remote requests
# (via Tailscale) must provide the same MCP_AUTH_TOKEN bearer token.

_PROFILE_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")


@app.before_request
def _check_auth():
    global _last_request
    _last_request = time.time()

    if not _PROFILE_AUTH_TOKEN:
        return  # no token configured — allow all (dev/local-only mode)

    # Localhost requests skip auth (menu bar webview, local browser)
    remote_addr = request.remote_addr or ""
    if remote_addr in ("127.0.0.1", "::1"):
        return

    # Static HTML pages don't require auth (they bootstrap the token)
    if request.path in ("/", "/profiles", "/tools", "/activity", "/manifest.json", "/sw.js") \
            or request.path.startswith("/pwa/"):
        return

    # API and file-serving routes require bearer token
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {_PROFILE_AUTH_TOKEN}":
        return

    # Also accept token as query param (for EventSource/SSE which can't set headers)
    if request.args.get("token") == _PROFILE_AUTH_TOKEN:
        return

    return jsonify({"error": "unauthorized"}), 403


@app.route("/")
def index():
    return send_file(str(HTML_FILE))


@app.route("/profiles")
def profiles_page():
    return send_file(str(HTML_FILE))


@app.route("/api/auth-token")
def api_auth_token():
    """Return the auth token — only from localhost (for HTML bootstrapping)."""
    remote_addr = request.remote_addr or ""
    if remote_addr not in ("127.0.0.1", "::1"):
        return jsonify({"error": "localhost only"}), 403
    return jsonify({"token": _PROFILE_AUTH_TOKEN})


@app.route("/api/system")
def api_system():
    return jsonify(get_system_info())


@app.route("/api/models")
def api_models():
    force = request.args.get("refresh") == "1"
    models = get_all_models(force_refresh=force)
    for name, info in models.items():
        info["eligible_tasks"] = get_eligible_tasks(name, info)
    return jsonify(list(models.values()))


@app.route("/api/tasks")
def api_tasks():
    prefs = load_default_prefs()
    thinking = prefs.get("thinking", {})
    all_tasks = {}
    for key, label in STANDARD_TASKS.items():
        all_tasks[key] = {"label": label, "defaults": prefs.get(key, []),
                          "thinking": thinking.get(key, True)}
    for key, spec in SPECIAL_TASKS.items():
        all_tasks[key] = {"label": spec["label"], "defaults": prefs.get(key, []),
                          "prefixes": spec["prefixes"],
                          "thinking": thinking.get(key, False)}
    return jsonify(all_tasks)


@app.route("/api/profiles", methods=["GET"])
def api_profiles_get():
    return jsonify(load_profiles())


@app.route("/api/profiles", methods=["POST"])
def api_profiles_save():
    data = load_profiles()
    body = request.json
    name = body.get("name", "")
    if not name:
        return jsonify({"error": "name required"}), 400
    data["profiles"][name] = {
        "label": body.get("label", name),
        "description": body.get("description", ""),
        "keep_loaded": body.get("keep_loaded", []),
        "tasks": body.get("tasks", {}),
    }
    save_profiles(data)
    return jsonify({"ok": True})


@app.route("/api/profiles/<name>", methods=["DELETE"])
def api_profiles_delete(name):
    data = load_profiles()
    data["profiles"].pop(name, None)
    if data["active"] == name:
        data["active"] = None
    save_profiles(data)
    return jsonify({"ok": True})


@app.route("/api/profiles/<name>/activate", methods=["POST"])
def api_profiles_activate(name):
    """Save preferences only. Does not touch running models."""
    data = load_profiles()
    profile = data["profiles"].get(name)
    if not profile:
        return jsonify({"error": f"Profile '{name}' not found"}), 404

    current = load_default_prefs()
    if profile.get("tasks"):
        for task, pick in profile["tasks"].items():
            existing = current.get(task, [])
            current[task] = [pick] + [m for m in existing if m != pick]
    if profile.get("thinking"):
        current.setdefault("thinking", {}).update(profile["thinking"])

    # Prune model references that don't exist in any backend
    # Skip pruning if no models discovered — backends probably aren't up yet
    models = get_all_models()
    stale_warnings = []
    if models:
        def _model_exists(name):
            return name in models or any(n.startswith(name + ":") for n in models)

        for task, candidates in list(current.items()):
            if task == "thinking" or not isinstance(candidates, list):
                continue
            alive = [c for c in candidates if _model_exists(c)]
            pruned = [c for c in candidates if not _model_exists(c)]
            if pruned:
                stale_warnings.append(f"{task}: {', '.join(pruned)}")
            current[task] = alive

    save_mcp_prefs(current)

    data["active"] = name
    save_profiles(data)

    return jsonify({"ok": True, "warnings": stale_warnings})


@app.route("/api/profiles/<name>/warm", methods=["POST"])
def api_profiles_warm(name):
    """Pre-load the preferred models into server memory."""
    proxied = _proxy_to_desktop(f"/api/profiles/{name}/warm")
    if proxied is not None:
        return proxied
    data = load_profiles()
    profile = data["profiles"].get(name)
    if not profile:
        return jsonify({"error": f"Profile '{name}' not found"}), 404

    tasks = profile.get("tasks", {})
    if not tasks:
        return jsonify({"ok": True, "loaded": []})

    models = get_all_models()
    candidates = list(dict.fromkeys(tasks.values()))

    ollama_to_load = []
    mlx_to_load = []
    for name in candidates:
        if name not in models:
            continue
        backend = models[name]["backend"]
        if backend == "ollama":
            ollama_to_load.append(name)
        elif backend == "mlx":
            mlx_to_load.append(name)

    # Skip Ollama models already in memory
    ps = ollama_get("/api/ps") or {}
    already = {m["name"] for m in ps.get("models", [])}
    ollama_to_load = [m for m in ollama_to_load if m not in already]

    loaded = []

    for model in ollama_to_load:
        try:
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": model, "prompt": "", "keep_alive": "5m"},
                          timeout=300)
            loaded.append(model)
        except Exception:
            pass

    # MLX on-demand models: a tiny completion request triggers loading.
    # They persist in memory for their idle_timeout (120-300s).
    for model in mlx_to_load:
        try:
            requests.post(f"{MLX_URL}/v1/chat/completions",
                          json={"model": model, "max_tokens": 1,
                                "messages": [{"role": "user", "content": "hi"}]},
                          timeout=120)
            loaded.append(model)
        except Exception:
            pass

    _model_cache["data"] = None
    return jsonify({"ok": True, "loaded": loaded})


# ── Tool tester ──────────────────────────────────────────────────────


@app.route("/tools")
def tools_page():
    return send_file(str(TOOLS_HTML))


@contextmanager
def _track_playground(tool, model, backend):
    """Track what the playground is currently doing so /api/gpu can report it."""
    tid = threading.get_ident()
    with _playground_lock:
        _playground_active[tid] = {"tool": tool, "model": model, "backend": backend,
                                   "started": time.time()}
    try:
        yield
    finally:
        with _playground_lock:
            _playground_active.pop(tid, None)


def _pick_model_for_task(task):
    """Resolve preferred model for a task. Returns (model_name, backend, warning)."""
    prefs = load_default_prefs()
    models = get_all_models()
    candidates = prefs.get(task, [])
    for candidate in candidates:
        if candidate in models:
            return candidate, models[candidate]["backend"], None
        for name in models:
            if name.startswith(candidate + ":"):
                return name, models[name]["backend"], None
    warning = None
    if candidates:
        warning = (f"Profile models for '{task}' not available: {', '.join(candidates)} "
                   f"— using fallback")
    return None, None, warning


def _chat_url(backend):
    """Return the chat endpoint URL for a backend."""
    if backend == "mlx":
        return f"{MLX_URL}/v1/chat/completions"
    return f"{OLLAMA_URL}/api/chat"


_MISSING_TOOL_HELP = {
    "mflux-generate": "mflux is not installed. Install with: uv tool install --python 3.12 mflux",
    "mflux-generate-kontext": "mflux is not installed. Install with: uv tool install --python 3.12 mflux",
    "ffmpeg": "ffmpeg is not installed. Install with: brew install ffmpeg",
}


def _friendly_error(e, tool_name: str = "") -> str:
    """Turn common exceptions into actionable messages."""
    if isinstance(e, FileNotFoundError):
        cmd = str(e).split("'")[-2] if "'" in str(e) else ""
        hint = _MISSING_TOOL_HELP.get(cmd, "")
        return hint or f"{tool_name}: {e}"
    if isinstance(e, requests.RequestException):
        return f"{tool_name}: {_requests_error_detail(e)}"
    return f"{tool_name}: {e}"


def _requests_error_detail(e):
    """Extract a useful error message from a requests exception."""
    if isinstance(e, requests.HTTPError) and e.response is not None:
        body = e.response.text[:500] if e.response.text else ""
        if e.response.status_code == 404 and "not found" in body.lower():
            import re
            model_match = re.search(r"model '([^']+)'", body)
            model_name = model_match.group(1) if model_match else "the model"
            return (f"Model {model_name} is not downloaded. "
                    f"Pull it first: ollama pull {model_name}")
        return f"HTTP {e.response.status_code} from {e.response.url} — {body or '(empty body)'}"
    if isinstance(e, requests.ConnectionError):
        return f"Cannot connect to backend — is it running? ({e})"
    if isinstance(e, requests.Timeout):
        return f"Request timed out ({e})"
    return str(e)


def _chat(model, backend, messages, timeout=120, tool="chat"):
    """Send a chat request to the appropriate backend."""
    with _track_playground(tool, model, backend):
        try:
            if backend == "mlx":
                resp = requests.post(f"{MLX_URL}/v1/chat/completions", json={
                    "model": model, "messages": messages,
                }, timeout=300)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            else:
                resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
                    "model": model, "messages": messages, "stream": False,
                }, timeout=timeout)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except requests.RequestException as e:
            raise RuntimeError(f"Chat ({model} via {backend}): {_requests_error_detail(e)}") from e


def _chat_stream(model, backend, messages, think=True, tool="chat"):
    """Stream chat tokens as SSE events. Yields 'data: {...}\\n\\n' strings."""
    _stream_tid = threading.get_ident()
    with _playground_lock:
        _playground_active[_stream_tid] = {"tool": tool, "model": model, "backend": backend,
                                           "started": time.time()}
    try:
        if backend == "mlx":
            resp = requests.post(f"{MLX_URL}/v1/chat/completions", json={
                "model": model, "messages": messages, "stream": True,
            }, stream=True, timeout=300)
            resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Stream ({model} via {backend}): {_requests_error_detail(e)}") from e
    if backend == "mlx":
        yield f"data: {json.dumps({'model': model})}\n\n"
        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8", errors="replace")
            if text.startswith("data: "):
                text = text[6:]
            if text.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(text)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
    else:
        body = {"model": model, "messages": messages, "stream": True}
        if not think:
            body["think"] = False
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body,
                                 stream=True, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Stream ({model} via {backend}): {_requests_error_detail(e)}") from e
        yield f"data: {json.dumps({'model': model})}\n\n"
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                token = msg.get("content", "")
                thinking = msg.get("thinking", "")
                if thinking:
                    yield f"data: {json.dumps({'thinking': True})}\n\n"
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                pass
    yield "data: {\"done\": true}\n\n"
    with _playground_lock:
        _playground_active.pop(_stream_tid, None)


STREAM_TOOLS = {"code", "general", "review", "translate", "summarize"}


_MAX_PROXY_HOPS = 3


def _proxy_to_desktop(path: str, method: str = "POST"):
    """In client mode, forward requests to the desktop's profile server.

    Returns a Flask Response if proxied, or None if running locally.
    """
    if not _is_remote_ollama():
        return None
    hops = int(request.headers.get("X-SP-Proxy-Hops", "0"))
    if hops >= _MAX_PROXY_HOPS:
        return jsonify({"error": "Proxy loop detected — too many hops between servers"}), 502
    try:
        url = f"{_desktop_profile_server_url()}{path}"
        proxy_headers = {"X-SP-Proxy-Hops": str(hops + 1)}
        if method == "POST":
            resp = requests.post(url, json=request.json, headers=proxy_headers,
                                 timeout=300, stream=True)
        else:
            resp = requests.get(url, params=request.args, headers=proxy_headers,
                                timeout=30, stream=True)
        excluded = {"transfer-encoding", "content-encoding", "connection"}
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
        return Response(resp.iter_content(chunk_size=4096),
                        status=resp.status_code, headers=headers,
                        content_type=resp.headers.get("content-type"))
    except Exception as e:
        return jsonify({"error": f"Desktop unreachable: {e}"}), 502


@app.route("/api/test/stream", methods=["POST"])
def api_test_stream():
    proxied = _proxy_to_desktop("/api/test/stream")
    if proxied is not None:
        return proxied
    body = request.json
    tool = body.get("tool")
    override = body.get("model")
    think = body.get("think", True)

    _override_warning = []

    def _pick(task):
        if override:
            models = get_all_models()
            if override in models:
                return override, models[override]["backend"]
            _override_warning.append(f"Model '{override}' not found in available models — fell back to profile default for '{task}'")
        model, backend, stale_warning = _pick_model_for_task(task)
        if stale_warning:
            _override_warning.append(stale_warning)
        if not model:
            raise ValueError(f"No model available for task '{task}' — check that Ollama/MLX are running and models are loaded")
        return model, backend

    try:
        if tool == "code":
            try:
                model, backend = _pick("code")
            except ValueError:
                model, backend = _pick("general")
            messages = [{"role": "user", "content": body["prompt"]}]
        elif tool == "general":
            try:
                model, backend = _pick("general")
            except ValueError:
                model, backend = _pick("code")
            messages = [{"role": "user", "content": body["prompt"]}]
        elif tool == "review":
            model, backend = _pick("reasoning")
            messages = [
                {"role": "system", "content": "Review this code. Be concise."},
                {"role": "user", "content": body["code"]},
            ]
        elif tool == "translate":
            model, backend = _pick("translation")
            messages = [
                {"role": "system",
                 "content": f"Translate to {body['target']}. Output only the translation."},
                {"role": "user", "content": body["text"]},
            ]
        elif tool == "summarize":
            model, backend = _pick("long_context")
            fp = body["file_path"]
            if not _is_safe_test_path(fp):
                return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
            text = Path(fp).read_text(errors="replace")[:50000]
            messages = [
                {"role": "system", "content": "Summarize this content concisely."},
                {"role": "user", "content": text},
            ]
        elif tool == "unfiltered":
            model, backend = _pick("unfiltered")
            messages = [{"role": "user", "content": body["prompt"]}]
        else:
            return jsonify({"error": "Not a streaming tool"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    def _safe_stream():
        if _override_warning:
            yield f"data: {json.dumps({'warning': '; '.join(_override_warning)})}\n\n"
        try:
            yield from _chat_stream(model, backend, messages, think=think, tool=tool)
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            with _playground_lock:
                _playground_active.pop(threading.get_ident(), None)

    return Response(_safe_stream(), mimetype="text/event-stream")


@app.route("/api/test", methods=["POST"])
def api_test():
    proxied = _proxy_to_desktop("/api/test")
    if proxied is not None:
        return proxied
    body = request.json
    tool = body.get("tool")
    override = body.get("model")

    _override_warning = []

    def _pick(task):
        if override:
            models = get_all_models()
            if override in models:
                return override, models[override]["backend"]
            _override_warning.append(f"Model '{override}' not found in available models — fell back to profile default for '{task}'")
        model, backend, stale_warning = _pick_model_for_task(task)
        if stale_warning:
            _override_warning.append(stale_warning)
        if not model:
            raise ValueError(f"No model available for task '{task}' — check that Ollama/MLX are running and models are loaded")
        return model, backend

    @after_this_request
    def _inject_warning(response):
        if _override_warning and response.content_type == "application/json":
            try:
                data = response.get_json()
                if isinstance(data, dict):
                    data["warning"] = "; ".join(_override_warning)
                    response.set_data(json.dumps(data))
            except Exception:
                pass
        return response

    try:
        if tool in ("code", "general"):
            task = "code" if tool == "code" else "general"
            try:
                model, backend = _pick(task)
            except ValueError:
                model, backend = _pick("code" if task == "general" else "general")
            result = _chat(model, backend,
                           [{"role": "user", "content": body["prompt"]}],
                           tool=tool)
            return jsonify({"result": result, "model": model})

        elif tool == "review":
            model, backend = _pick("reasoning")
            result = _chat(model, backend, [
                {"role": "system", "content": "Review this code. Be concise."},
                {"role": "user", "content": body["code"]},
            ], tool="review")
            return jsonify({"result": result, "model": model})

        elif tool == "vision":
            import base64
            model, backend = _pick("vision")
            if not _is_safe_test_path(body["image_path"]):
                return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
            image_data = Path(body["image_path"]).read_bytes()
            image_b64 = base64.b64encode(image_data).decode()
            prompt = body.get("prompt", "Describe this image.")
            with _track_playground("vision", model, backend):
                if backend == "mlx":
                    resp = requests.post(f"{MLX_URL}/v1/chat/completions", json={
                        "model": model, "stream": False,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"}},
                        ]}],
                    }, timeout=120)
                    resp.raise_for_status()
                    result = resp.json()["choices"][0]["message"]["content"]
                else:
                    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
                        "model": model, "stream": False,
                        "messages": [{"role": "user",
                                      "content": prompt,
                                      "images": [image_b64]}],
                    }, timeout=120)
                    resp.raise_for_status()
                    result = resp.json()["message"]["content"]
            return jsonify({"result": result, "model": model})

        elif tool == "computer_use":
            import base64
            model, backend = _pick("computer_use")
            if body.get("image_path"):
                if not _is_safe_test_path(body["image_path"]):
                    return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
                image_data = Path(body["image_path"]).read_bytes()
            else:
                return jsonify({"error": "Screenshot required"}), 400
            image_b64 = base64.b64encode(image_data).decode()
            intent = body.get("intent", "Describe what actions to take")
            system_prompt = (
                "You are a GUI automation assistant. Given a screenshot and an intent, "
                "return a JSON array of actions to accomplish the intent.\n\n"
                "Each action is one of:\n"
                '- {"action": "click", "x": <int>, "y": <int>, "description": "<what>"}\n'
                '- {"action": "type", "text": "<text>", "description": "<where>"}\n'
                '- {"action": "scroll", "direction": "up"|"down", "amount": <int>, "description": "<why>"}\n'
                '- {"action": "key", "key": "<key combo>", "description": "<why>"}\n'
                '- {"action": "wait", "seconds": <float>, "description": "<why>"}\n\n'
                "Return ONLY the JSON array."
            )
            with _track_playground("computer_use", model, backend):
                if backend == "mlx":
                    resp = requests.post(f"{MLX_URL}/v1/chat/completions", json={
                        "model": model, "stream": False,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "text", "text": intent},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"}},
                            ]},
                        ],
                    }, timeout=300)
                    resp.raise_for_status()
                    result = resp.json()["choices"][0]["message"]["content"]
                else:
                    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
                        "model": model, "stream": False,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": intent,
                             "images": [image_b64]},
                        ],
                    }, timeout=300)
                    resp.raise_for_status()
                    result = resp.json()["message"]["content"]
            return jsonify({"result": result, "model": model})

        elif tool == "image_edit":
            model, backend = _pick("image_edit")
            image_path = body.get("image_path", "")
            if not _is_safe_test_path(image_path):
                return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
            prompt = body.get("prompt", "")
            import time as _time
            out_path = f"/tmp/playground_edit_{int(_time.time())}.png"
            with _track_playground("image_edit", model, backend):
                try:
                    result = subprocess.run(
                        ["mflux-generate-kontext",
                         "--image-path", image_path,
                         "--prompt", prompt,
                         "--output", out_path,
                         "--steps", "8",
                         "--image-strength", "0.75"],
                        capture_output=True, text=True, timeout=600,
                        env={**os.environ,
                             "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
                    )
                    if result.returncode != 0:
                        return jsonify({"error": f"image_edit: mflux-generate-kontext failed:\n{result.stderr[-300:]}"})
                    return jsonify({
                        "result": f"Saved to {out_path}",
                        "image_path": out_path,
                        "model": model,
                    })
                except Exception as e:
                    return jsonify({"error": _friendly_error(e, "image_edit")})

        elif tool == "image_gen":
            model, backend = _pick("image_gen")
            import time as _time
            out = f"/tmp/test_image_{int(_time.time())}.png"

            if backend == "mflux":
                with _track_playground("image_gen", model, backend):
                    steps = "4" if any(k in model.lower() for k in ("schnell", "turbo", "klein")) else "20"
                    try:
                        result = subprocess.run(
                            ["mflux-generate", "--model", model,
                             "--prompt", body["prompt"],
                             "--output", out, "--steps", steps],
                            capture_output=True, text=True, timeout=600,
                            env={**os.environ,
                                 "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
                        )
                    except FileNotFoundError:
                        return jsonify({"error": _MISSING_TOOL_HELP["mflux-generate"]})
                    if result.returncode != 0:
                        return jsonify({"error": f"image_gen: mflux-generate failed:\n{result.stderr[-300:]}"})
                if not Path(out).exists():
                    return jsonify({"error": f"image_gen: output image was not created at {out}"})
                return jsonify({"result": f"Saved to {out}", "image_path": out, "model": model})
            else:
                with _track_playground("image_gen", model, backend):
                    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
                        "model": model, "prompt": body["prompt"], "stream": False,
                    }, timeout=300)
                    resp.raise_for_status()
                    import base64
                    image_b64 = resp.json().get("image", "")
                if not image_b64:
                    return jsonify({"error": f"image_gen: {model} did not return an image — "
                                             f"this model may not support image generation."})
                Path(out).write_bytes(base64.b64decode(image_b64))
                return jsonify({"result": f"Saved to {out}", "image_path": out, "model": model})

        elif tool == "transcribe":
            model, backend = _pick("transcription")
            if not model:
                model = "whisper-v3"
            if not _is_safe_test_path(body["audio_path"]):
                return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
            audio_path = Path(body["audio_path"])
            suffix = audio_path.suffix.lstrip(".")

            # Whisper needs wav/mp3/m4a — convert webm via ffmpeg
            if suffix == "webm":
                wav_path = audio_path.with_suffix(".wav")
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(audio_path), str(wav_path)],
                        capture_output=True, timeout=30)
                except FileNotFoundError:
                    return jsonify({"error": _MISSING_TOOL_HELP["ffmpeg"]})
                audio_path = wav_path
                suffix = "wav"

            audio_data = audio_path.read_bytes()
            ct_map = {"mp3": "audio/mpeg", "wav": "audio/wav",
                      "m4a": "audio/mp4", "ogg": "audio/ogg"}
            ct = ct_map.get(suffix, "application/octet-stream")
            url = MLX_URL if backend == "mlx" else OLLAMA_URL
            with _track_playground("transcribe", model, backend):
                resp = requests.post(f"{url}/v1/audio/transcriptions",
                                     files={"file": (audio_path.name, audio_data, ct)},
                                     data={"model": model}, timeout=300)
                resp.raise_for_status()
            return jsonify({"result": resp.json().get("text", resp.text), "model": model})

        elif tool == "speak":
            model, backend = _pick("tts")
            model = body.get("model") or model
            voice = body.get("voice", "casual_male")
            lang = body.get("language", "en")
            text = body.get("text", "")
            ref_audio = body.get("ref_audio")
            if ref_audio and not _is_safe_test_path(ref_audio):
                return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
            if ref_audio:
                model = body.get("model") or "mlx-community/chatterbox-fp16"
            import time as _time
            out_path = f"/tmp/playground_tts_{int(_time.time())}.wav"
            out_dir = os.path.dirname(out_path)
            prefix = Path(out_path).stem
            with _track_playground("speak", model, backend):
                try:
                    from mlx_audio.tts.generate import generate_audio
                    kwargs = dict(
                        text=text, model=model, voice=voice,
                        lang_code=lang, output_path=out_dir,
                        file_prefix=prefix, audio_format="wav",
                        verbose=False, play=False,
                    )
                    if ref_audio:
                        kwargs["ref_audio"] = ref_audio
                    generate_audio(**kwargs)
                    actual = os.path.join(out_dir, f"{prefix}_000.wav")
                    if os.path.exists(actual):
                        os.rename(actual, out_path)
                    return jsonify({
                        "result": f"Audio saved to {out_path}",
                        "audio_path": out_path, "model": model.split("/")[-1],
                    })
                except Exception as e:
                    return jsonify({"error": f"speak: {e}"})

        elif tool == "translate":
            model, backend = _pick("translation")
            result = _chat(model, backend, [
                {"role": "system",
                 "content": f"Translate to {body['target']}. Output only the translation."},
                {"role": "user", "content": body["text"]},
            ], tool="translate")
            return jsonify({"result": result, "model": model})

        elif tool == "summarize":
            model, backend = _pick("long_context")
            if not _is_safe_test_path(body["file_path"]):
                return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
            text = Path(body["file_path"]).read_text(errors="replace")[:50000]
            result = _chat(model, backend, [
                {"role": "system", "content": "Summarize this content concisely."},
                {"role": "user", "content": text},
            ], tool="summarize")
            return jsonify({"result": result, "model": model})

        elif tool == "embed":
            model, backend = _pick("embedding")
            if not model:
                model, backend = "mxbai-embed-large", "ollama"
            with _track_playground("embed", model, backend):
                if backend == "mlx":
                    resp = requests.post(f"{MLX_URL}/v1/embeddings", json={
                        "model": model, "input": [body["text"]],
                    }, timeout=60)
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    embeddings = [d["embedding"] for d in data]
                else:
                    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
                        "model": model, "input": [body["text"]],
                    }, timeout=60)
                    resp.raise_for_status()
                    embeddings = resp.json().get("embeddings", [])
            return jsonify({
                "embeddings": embeddings,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "count": len(embeddings),
                "model": model,
            })

        else:
            return jsonify({"error": f"Unknown tool: {tool}"}), 400

    except requests.RequestException as e:
        return jsonify({"error": f"{tool}: {_requests_error_detail(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"{tool}: {e}"}), 500


@app.route("/api/test/screenshot", methods=["POST"])
def api_test_screenshot():
    """Interactive screenshot via system UI (Cmd-Shift-5 style).

    Uses osascript to invoke the system screenshot, which inherits screen
    recording permission from the frontmost app rather than requiring the
    profile server's Python binary to be individually authorized.
    """
    import time as _time
    dest = f"/tmp/screenshot_{int(_time.time())}.png"
    result = subprocess.run(
        ["screencapture", "-i", dest],
        capture_output=True, text=True, timeout=60)
    if not Path(dest).exists():
        stderr = (result.stderr or "").strip()
        if "not allowed" in stderr or "could not create image" in stderr:
            return jsonify({
                "error": "Screen recording permission needed. "
                         "System Settings → Privacy & Security → Screen Recording "
                         "→ enable the terminal or app you launched Super Puppy from."
            }), 403
        return jsonify({"error": "Screenshot cancelled."})
    return jsonify({"path": dest})


@app.route("/api/test/upload", methods=["POST"])
def api_test_upload():
    """Save an uploaded file to /tmp and return its path."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    import time as _time
    ext = Path(f.filename).suffix or ".bin"
    dest = f"/tmp/test_upload_{int(_time.time())}{ext}"
    f.save(dest)
    return jsonify({"path": dest})


_PLAYGROUND_PATH_ERROR = (
    "Playground file access is restricted to uploaded files in /tmp/. "
    "Use the file picker to upload your file, or use the equivalent MCP tool "
    "via Claude Code to access files in your home directory."
)


def _is_safe_test_path(path: str) -> bool:
    """Only allow serving files from /tmp/ (test outputs, screenshots, uploads)."""
    try:
        resolved = str(Path(path).resolve())
        return resolved.startswith("/tmp/") or resolved.startswith("/private/tmp/")
    except (ValueError, OSError):
        return False


@app.route("/api/test/image")
def api_test_image():
    path = request.args.get("path", "")
    if not path or not _is_safe_test_path(path) or not Path(path).exists():
        return "Not found", 404
    return send_file(path)


@app.route("/api/test/audio")
def api_test_audio():
    path = request.args.get("path", "")
    if not path or not _is_safe_test_path(path) or not Path(path).exists():
        return "Not found", 404
    return send_file(path, mimetype="audio/wav")


MCP_PORT = int(os.environ.get("MCP_PORT", "8100"))


@app.route("/api/gpu")
def api_gpu():
    """Report playground activity and GPU contention."""
    proxied = _proxy_to_desktop("/api/gpu", method="GET")
    if proxied is not None:
        return proxied
    # What the playground is doing right now (pick most recent if multiple)
    with _playground_lock:
        own = None
        if _playground_active:
            own = max(_playground_active.values(), key=lambda x: x["started"])

    # Check if other things are using the GPU (MCP server tasks)
    other_active = False
    try:
        resp = requests.get(f"http://127.0.0.1:{MCP_PORT}/gpu", timeout=2)
        mcp = resp.json()
        other_active = (mcp.get("ollama", {}).get("active", 0) > 0
                        or mcp.get("mlx", {}).get("active", 0) > 0)
    except Exception:
        pass

    return jsonify({
        "playground": own,
        "other_active": other_active,
    })


@app.route("/api/activity")
def api_activity():
    """Proxy the MCP server's /activity endpoint for the dashboard."""
    proxied = _proxy_to_desktop("/api/activity", method="GET")
    if proxied is not None:
        return proxied
    try:
        resp = requests.get(f"http://127.0.0.1:{MCP_PORT}/activity", timeout=3)
        data = resp.json()
    except Exception:
        data = {"active": [], "history": [], "stats": {}, "server_uptime_s": 0}

    # Merge playground activity into active list
    now = time.time()
    with _playground_lock:
        for task in _playground_active.values():
            data["active"].append({
                "description": f"playground:{task['tool']}:{task['model']}",
                "backend": task["backend"],
                "started": task["started"],
                "elapsed_ms": int((now - task["started"]) * 1000),
                "source": "playground",
            })
    return jsonify(data)


@app.route("/activity")
def activity_page():
    return send_file(os.path.join(SCRIPT_DIR, "activity.html"))


# ── PWA assets ───────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/manifest.json")
def pwa_manifest():
    return send_file(os.path.join(SCRIPT_DIR, "manifest.json"),
                     mimetype="application/manifest+json")


@app.route("/sw.js")
def pwa_service_worker():
    resp = send_file(os.path.join(SCRIPT_DIR, "sw.js"),
                     mimetype="application/javascript")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Service-Worker-Allowed"] = "/"
    return resp


@app.route("/pwa/<path:filename>")
def pwa_assets(filename):
    pwa_dir = os.path.join(SCRIPT_DIR, "pwa")
    return send_from_directory(pwa_dir, filename)


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    validate_network_conf(logger=logging.getLogger())
    threading.Thread(target=_idle_watcher, daemon=True).start()

    import socket
    if PORT == 0:
        s = socket.socket()
        s.bind((HOST, 0))
        PORT = s.getsockname()[1]
        s.close()

    # HTTPS only when binding to all interfaces (remote access).
    # Tailscale certs are issued for the FQDN, not 127.0.0.1, so TLS
    # on localhost would fail with a hostname mismatch.
    # Plain HTTP always. Tailscale encrypts the WireGuard tunnel for remote
    # access. HTTPS would break the local webview (cert is for the Tailscale
    # FQDN, not 127.0.0.1).
    print(f"http://{HOST}:{PORT}", flush=True)  # menu bar reads this
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
