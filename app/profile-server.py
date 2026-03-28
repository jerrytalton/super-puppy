# /// script
# requires-python = ">=3.12"
# dependencies = ["flask>=3.0", "pyyaml", "requests"]
# ///
"""
Model Profile Server for Super Puppy.

Web-based preference pane for managing which models are loaded
and which back each MCP tool task. Launched from the menu bar app.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
import yaml
from flask import Flask, jsonify, request, send_file

# ── Config ───────────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
MLX_CONFIG = Path("~/.config/mlx-server/config.yaml").expanduser()
PROFILES_FILE = Path("~/.config/local-models/profiles.json").expanduser()
MCP_PREFS_FILE = Path("~/.config/local-models/mcp_preferences.json").expanduser()
HTML_FILE = Path(__file__).parent / "profiles.html"

IDLE_TIMEOUT = 600  # 10 minutes
PORT = int(os.environ.get("PROFILE_SERVER_PORT", "0"))  # 0 = random

# ── Shared constants (mirrored from MCP server + menubar) ───────────

KNOWN_ACTIVE = {
    "nemotron_h_moe": {124: 12},
    "deepseek2": {671: 37},
}

TASK_LABELS = {
    "code": "Code",
    "general": "General",
    "reasoning": "Reasoning",
    "long_context": "Long Context",
    "translation": "Translation",
}

SPECIAL_TASKS = {
    "vision": {"label": "Vision", "prefixes": ["qwen3-vl", "llava", "moondream"]},
    "image_gen": {"label": "Image Gen", "prefixes": ["x/flux2", "x/z-image", "flux", "stable-diffusion"]},
    "transcription": {"label": "Transcription", "prefixes": ["whisper"]},
    "embedding": {"label": "Embedding", "prefixes": ["mxbai-embed", "nomic-embed", "snowflake-arctic", "all-minilm"]},
}

TASK_FILTERS = {
    "code": {
        "priority_names": ["coder"],
        "include_names": ["qwen3.5", "deepseek", "cogito", "nemotron", "gpt-oss", "llama3.3", "glm"],
        "exclude_names": ["vl", "flux", "z-image", "whisper", "ocr", "tinyllama", "goonsai", "nsfw"],
        "min_active_b": 3,
    },
    "general": {
        "exclude_names": ["coder", "vl", "flux", "z-image", "whisper", "ocr", "tinyllama", "goonsai", "nsfw"],
        "min_active_b": 3,
    },
    "reasoning": {
        "exclude_names": ["coder", "vl", "flux", "z-image", "whisper", "ocr", "tinyllama", "goonsai", "nsfw"],
        "min_active_b": 10,
    },
    "long_context": {
        "exclude_names": ["vl", "flux", "z-image", "whisper", "ocr", "tinyllama", "goonsai", "nsfw"],
        "min_ctx": 64000,
    },
    "translation": {
        "exclude_names": ["coder", "vl", "flux", "z-image", "whisper", "ocr", "tinyllama", "goonsai", "nsfw"],
        "min_active_b": 3,
    },
}

def load_default_prefs() -> dict[str, list[str]]:
    """Load ranked model preferences from config file."""
    if MCP_PREFS_FILE.exists():
        try:
            prefs = json.loads(MCP_PREFS_FILE.read_text())
            return {k: (v if isinstance(v, list) else [v]) for k, v in prefs.items()}
        except Exception:
            pass
    return {}

# ── Idle shutdown ────────────────────────────────────────────────────

_last_request = time.time()


def _idle_watcher():
    while True:
        time.sleep(30)
        if time.time() - _last_request > IDLE_TIMEOUT:
            print("Idle timeout — shutting down.", file=sys.stderr)
            os._exit(0)


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


def get_system_info():
    try:
        raw = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
        return {"total_ram_bytes": raw, "total_ram_gb": raw >> 30}  # GiB
    except Exception:
        return {"total_ram_bytes": 0, "total_ram_gb": 0}


def compute_active_params(model_name, total_b, mi, family):
    """Multi-strategy active param calculation for MoE models."""
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

    if not expert_count or not expert_used or expert_count <= 1:
        return total_b

    total_rounded = round(total_b)

    # Strategy 1: parse AXB from name
    match = re.search(r'[_-]A(\d+(?:\.\d+)?)B', model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Strategy 2: known hybrid lookup
    if family in KNOWN_ACTIVE and total_rounded in KNOWN_ACTIVE[family]:
        return KNOWN_ACTIVE[family][total_rounded]

    # Strategy 3: FFN subtraction
    expert_ffn = _get(".expert_feed_forward_length", 0)
    embed_len = _get(".embedding_length", 0)
    block_count = _get(".block_count", 0)
    if expert_ffn and embed_len and block_count:
        total_raw = int(total_b * 1e9)
        total_moe = block_count * expert_count * expert_ffn * embed_len * 3
        active_moe = block_count * expert_used * expert_ffn * embed_len * 3
        computed = total_raw - total_moe + active_moe
        if 0 < computed < total_raw:
            return round(computed / 1e9)

    # Strategy 4: simple ratio
    return round(total_b * expert_used / expert_count)


def get_all_models():
    """Aggregate all Ollama + MLX models with metadata."""
    models = {}

    # Ollama installed models
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
        active_b = compute_active_params(name, total_b, mi, family)

        # For models where Ollama doesn't report params (image gen), estimate from disk
        if not total_b and disk_bytes:
            total_b = disk_bytes / 1e9 / 0.5  # rough: 0.5 bytes per param at 4-bit
        active_b = compute_active_params(name, total_b, mi, family) if total_b else 0

        models[name] = {
            "name": name,
            "backend": "ollama",
            "disk_bytes": disk_bytes,
            "vram_bytes": disk_bytes,  # estimate; overridden if loaded
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
    if MLX_CONFIG.exists():
        try:
            cfg = yaml.safe_load(MLX_CONFIG.read_text())
            for entry in cfg.get("models", []):
                served_name = entry.get("served_model_name", "")
                mlx_config[served_name] = entry
        except Exception:
            pass

    for m in mlx_models:
        mid = m["id"]
        cfg = mlx_config.get(mid, {})
        on_demand = cfg.get("on_demand", False)
        model_path = cfg.get("model_path", "")

        # Parse params from model path (e.g. "Qwen3.5-397B-A17B-4bit")
        total_b, active_b = 0, 0
        total_match = re.search(r'(\d+)B', model_path)
        if total_match:
            total_b = int(total_match.group(1))
        active_match = re.search(r'A(\d+)B', model_path)
        if active_match:
            active_b = int(active_match.group(1))
        if not active_b:
            active_b = total_b

        # Estimate VRAM: ~0.5 bytes per param at 4-bit quant
        est_bytes = int(total_b * 1e9 * 0.5) if total_b else 0

        models[mid] = {
            "name": mid,
            "backend": "mlx",
            "disk_bytes": est_bytes,
            "vram_bytes": est_bytes,
            "total_params_b": total_b,
            "active_params_b": active_b,
            "context": cfg.get("context_length", 0),
            "has_vision": "vision" in model_path.lower() or "vl" in model_path.lower(),
            "family": "mlx",
            "quant": "4bit" if "4bit" in model_path else "",
            "is_loaded": not on_demand,  # always-on models are loaded
            "expires_at": None,
            "on_demand": on_demand,
        }

    return models


def model_matches_filter(name, model_info, task_filter):
    """Check if model is suitable for a task."""
    name_lower = name.lower()

    excludes = task_filter.get("exclude_names", [])
    if any(p.lower() in name_lower for p in excludes):
        return False

    priority = task_filter.get("priority_names", [])
    if any(p.lower() in name_lower for p in priority):
        return True

    includes = task_filter.get("include_names")
    if includes and not any(p.lower() in name_lower for p in includes):
        return False

    active = model_info.get("active_params_b", 0)
    min_active = task_filter.get("min_active_b", 0)
    if min_active and active > 0 and active < min_active:
        return False

    ctx = model_info.get("context", 0)
    min_ctx = task_filter.get("min_ctx", 0)
    if min_ctx and ctx > 0 and ctx < min_ctx:
        return False

    return True


def get_eligible_tasks(name, model_info):
    """Return list of task keys this model qualifies for."""
    tasks = []
    for task, filt in TASK_FILTERS.items():
        if model_matches_filter(name, model_info, filt):
            tasks.append(task)
    for task, spec in SPECIAL_TASKS.items():
        if any(name.startswith(p) for p in spec["prefixes"]):
            tasks.append(task)
    return tasks


# ── Profiles ─────────────────────────────────────────────────────────

def load_profiles():
    if PROFILES_FILE.exists():
        try:
            return json.loads(PROFILES_FILE.read_text())
        except Exception:
            pass
    return {"active": None, "profiles": {}}


def save_profiles(data):
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(json.dumps(data, indent=2))


def save_mcp_prefs(prefs):
    MCP_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MCP_PREFS_FILE.write_text(json.dumps(prefs, indent=2))


# ── Flask app ────────────────────────────────────────────────────────

app = Flask(__name__)


@app.before_request
def _touch_idle():
    global _last_request
    _last_request = time.time()


@app.route("/")
def index():
    return send_file(str(HTML_FILE))


@app.route("/api/system")
def api_system():
    return jsonify(get_system_info())


@app.route("/api/models")
def api_models():
    models = get_all_models()
    for name, info in models.items():
        info["eligible_tasks"] = get_eligible_tasks(name, info)
    return jsonify(list(models.values()))


@app.route("/api/tasks")
def api_tasks():
    prefs = load_default_prefs()
    all_tasks = {}
    for key, label in TASK_LABELS.items():
        all_tasks[key] = {"label": label, "defaults": prefs.get(key, [])}
    for key, spec in SPECIAL_TASKS.items():
        all_tasks[key] = {"label": spec["label"], "defaults": prefs.get(key, []), "prefixes": spec["prefixes"]}
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

    if profile.get("tasks"):
        current = load_default_prefs()
        for task, pick in profile["tasks"].items():
            existing = current.get(task, [])
            current[task] = [pick] + [m for m in existing if m != pick]
        save_mcp_prefs(current)

    data["active"] = name
    save_profiles(data)

    return jsonify({"ok": True})


@app.route("/api/profiles/<name>/warm", methods=["POST"])
def api_profiles_warm(name):
    """Pre-load the preferred models into server memory."""
    data = load_profiles()
    profile = data["profiles"].get(name)
    if not profile:
        return jsonify({"error": f"Profile '{name}' not found"}), 404

    tasks = profile.get("tasks", {})
    if not tasks:
        return jsonify({"ok": True, "loaded": []})

    # Unique models to load
    to_load = list(dict.fromkeys(tasks.values()))

    ps = ollama_get("/api/ps") or {}
    already = {m["name"] for m in ps.get("models", [])}
    to_load = [m for m in to_load if m not in already]

    for model in to_load:
        try:
            requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": model, "prompt": "", "keep_alive": "10m"},
                          timeout=300)
        except Exception:
            pass

    return jsonify({"ok": True, "loaded": to_load})


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=_idle_watcher, daemon=True).start()

    import socket
    if PORT == 0:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        PORT = s.getsockname()[1]
        s.close()

    print(f"http://127.0.0.1:{PORT}", flush=True)  # menu bar reads this
    app.run(host="127.0.0.1", port=PORT, debug=False)
