"""Shared model discovery constants, types, and helpers.

Single source of truth for MoE active-parameter tables, task definitions,
task filters, and config file paths. Imported by menubar, MCP server, and
profile server.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# ── Config file paths ─────────────────────────────────────────────────

CONFIG_DIR = Path("~/.config/local-models").expanduser()
PROFILES_FILE = CONFIG_DIR / "profiles.json"
MCP_PREFS_FILE = CONFIG_DIR / "mcp_preferences.json"
MODEL_PREFS_FILE = CONFIG_DIR / "model_preferences.json"
NETWORK_CONF = CONFIG_DIR / "network.conf"
ACTIVITY_DB = CONFIG_DIR / "activity.db"
MLX_SERVER_CONFIG = Path("~/.config/mlx-server/config.yaml").expanduser()
CLAUDE_CONFIG_FILE = Path("~/.claude.json").expanduser()

# ── Default network config (must match config/local-models/network.conf) ──

_NETWORK_DEFAULTS = {
    "TAILSCALE_HOSTNAME": "super-puppy",
    "OLLAMA_PORT": "11434",
    "MLX_PORT": "8000",
    "SERVER_RAM_GB": "0",
    "PROBE_TIMEOUT": "2",
    "PROFILE_PORT": "8101",
    "OP_REF": "",
    "IS_SERVER": "false",
    "AUTO_UPDATE": "true",
}

_NUMERIC_KEYS = {"OLLAMA_PORT", "MLX_PORT", "SERVER_RAM_GB", "PROBE_TIMEOUT", "PROFILE_PORT"}


def validate_network_conf(logger=None) -> list[str]:
    """Validate ~/.config/local-models/network.conf. Returns list of warnings.

    Repairs what it can: missing file gets defaults, non-numeric values
    get stripped to digits. Logs all issues if a logger is provided.
    """
    import json
    import shutil

    warnings: list[str] = []

    def warn(msg: str):
        warnings.append(msg)
        if logger:
            logger.warning("config: %s", msg)

    # 1. network.conf: must exist and not be empty
    if not NETWORK_CONF.exists() or NETWORK_CONF.stat().st_size == 0:
        warn(f"{NETWORK_CONF} is missing or empty — writing defaults")
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        template = Path(__file__).parent.parent / "config" / "local-models" / "network.conf"
        if template.exists():
            shutil.copy2(template, NETWORK_CONF)
        else:
            lines = [f"{k}={v}" for k, v in _NETWORK_DEFAULTS.items()]
            NETWORK_CONF.write_text("\n".join(lines) + "\n")

    # 2. Parse and validate values
    conf: dict[str, str] = {}
    dirty = False
    raw_lines = NETWORK_CONF.read_text().splitlines()
    repaired_lines = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            repaired_lines.append(line)
            continue
        key, _, val = stripped.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")

        if key in _NUMERIC_KEYS:
            digits = "".join(c for c in val if c.isdigit())
            if digits != val:
                warn(f"{key}={val!r} has non-numeric characters — using {digits or '0'}")
                val = digits or "0"
                line = f"{key}={val}"
                dirty = True

        conf[key] = val
        repaired_lines.append(line)

    if dirty:
        NETWORK_CONF.write_text("\n".join(repaired_lines) + "\n")

    # 3. mcp_preferences.json: must be valid JSON if it exists
    if MCP_PREFS_FILE.exists() and MCP_PREFS_FILE.stat().st_size > 0:
        try:
            json.loads(MCP_PREFS_FILE.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            warn(f"{MCP_PREFS_FILE} is not valid JSON: {e}")

    return warnings


# ── MoE active parameter table ───────────────────────────────────────
# Keyed by Ollama family name → {total_b_rounded: active_b}.
# For hybrid MoE architectures where auto-detection fails.

KNOWN_ACTIVE_PARAMS: dict[str, dict[int, int]] = {
    "nemotron_h_moe": {124: 12},
    "deepseek2": {671: 37},
}

# ── Task definitions ──────────────────────────────────────────────────

STANDARD_TASKS: dict[str, str] = {
    "code": "Code",
    "general": "General",
    "reasoning": "Reasoning",
    "long_context": "Long Context",
    "translation": "Translation",
}

# Tasks routed through a chat LLM, where chain-of-thought (and the
# `chat_template_kwargs.enable_thinking` knob on Qwen3) actually applies.
# Everything outside this set (image/video/audio/embedding) hits a
# specialized model that doesn't reason, so the UI's "think" toggle is
# meaningless for them.
THINK_CAPABLE_TASKS: frozenset[str] = frozenset({
    "code", "general", "reasoning", "long_context", "translation",
    "vision", "computer_use", "unfiltered",
})

# Tasks whose backend downloads weights on first use (mlx-audio, mflux,
# mlx-video). A profile-assigned HF path for one of these is NOT "stale"
# just because the HF cache doesn't have it yet — pick() honors it and
# the backend pulls on demand.
DOWNLOAD_ON_DEMAND_TASKS: frozenset[str] = frozenset({
    "tts", "image_gen", "image_edit", "video",
})


# Task → backend mapping for HuggingFace-cached models that are NOT
# served by Ollama or mlx-openai-server. These models live in the HF
# cache and are dispatched via dedicated subprocesses (mflux,
# mlx-audio, mlx-video). Used by the discovery and dispatch paths in
# both the MCP server and the profile server — kept here so a new
# entry flows through to both consumers.
HF_TASK_BACKENDS: dict[str, str] = {
    "tts": "mlx-audio",
    "transcription": "mlx",
    "image_edit": "mflux",
    "image_gen": "mflux",
    "video": "mlx-video",
}


def resolve_pref_candidate(
    candidate: str,
    models: dict[str, dict],
    *,
    task: str | None = None,
    allow_hf_on_demand: bool = False,
) -> tuple[str, str] | None:
    """Resolve a single preference candidate to (name, backend).

    Tried in order:
    1. Exact match in `models`.
    2. Prefix match (e.g. candidate "qwen3.5" matches "qwen3.5:8b").
    3. HF download-on-demand: if the candidate looks like an HF repo id
       ("org/repo") and the task uses an HF backend, return the repo
       id and the appropriate backend so the caller can fetch on
       first use.

    `models` is `name -> {"backend": ...}` (other keys ignored).
    Returns None if nothing matches.
    """
    if candidate in models:
        return candidate, models[candidate]["backend"]
    suffix_target = candidate + ":"
    latest = candidate + ":latest"
    for name in models:
        if name == latest or name.startswith(suffix_target):
            return name, models[name]["backend"]
    if (allow_hf_on_demand and task in HF_TASK_BACKENDS
            and "/" in candidate):
        return candidate, HF_TASK_BACKENDS[task]
    return None


def pick_model_from_prefs(
    task: str,
    models: dict[str, dict],
    prefs: dict,
    *,
    override: str | None = None,
    allow_hf_on_demand: bool = False,
    fallback_to_general: bool = False,
) -> tuple[str, str] | None:
    """Pick the first model matching a task from preferences.

    Used by both the MCP server's `pick_model` and the profile server's
    `_pick_model_for_task`. The caller passes the model registry and
    prefs it has already loaded — that keeps this function pure (no
    I/O) and lets each caller cache reads how it likes.

    `override` short-circuits prefs entirely. If set, the override is
    resolved with the same candidate-resolution rules and returned —
    or, if it fails to resolve, prefs are NOT tried (the caller asked
    for a specific model and shouldn't silently get a different one).

    Set `fallback_to_general=True` for the MCP path: when a task has
    no usable prefs, fall through to `prefs["general"]` rather than
    failing immediately. The profile server keeps this False because
    its callers have higher-level fallback logic.

    Set `allow_hf_on_demand=True` to accept candidates that look like
    HF repo ids (containing "/") for tasks served by HF backends —
    even when the model isn't downloaded yet. The profile server
    enables this so a freshly-set profile can dispatch a download on
    first use; the MCP server keeps it disabled because its registry
    only contains models that are actually loaded.

    Returns (name, backend) or None.
    """
    if override:
        return resolve_pref_candidate(
            override, models, task=task,
            allow_hf_on_demand=allow_hf_on_demand)

    keys: list[str] = [task]
    if fallback_to_general and task != "general":
        keys.append("general")

    for key in keys:
        candidates = prefs.get(key, [])
        if isinstance(candidates, str):
            candidates = [candidates]
        for c in candidates:
            result = resolve_pref_candidate(
                c, models, task=task,
                allow_hf_on_demand=allow_hf_on_demand)
            if result:
                return result
        # If the task itself had prefs but none resolved, don't fall
        # through to general — the user expressed a preference and we
        # owe them the error rather than a surprise pick.
        if key == task and candidates:
            return None
    return None


def model_has_vision(
    name: str,
    *,
    ollama_model_info: dict | None = None,
    hf_config: dict | None = None,
) -> bool:
    """Single source of truth: does this model accept image input?

    Checks four signals in order, any one is sufficient:

    1. Ollama model_info contains a "vision" capability key (the
       conventional way Ollama reports VL models).
    2. HF config.json declares a `vision_config` block, either at the
       top level or nested under `text_config` (Qwen3.5 family).
    3. HF config.json's `architectures` list contains a known
       vision-language architecture name.
    4. The model's name contains a vision substring (`vl`, `vision`,
       `vlm`) — last-resort heuristic that matches the Qwen3.5 /
       Qwen-VL / Llama-VL families when their HF cache hasn't been
       scanned yet.

    Pass whatever you have. The caller need not collect both ollama
    and HF signals — one is enough when it matches.
    """
    name_lower = (name or "").lower()

    if ollama_model_info:
        for k, v in ollama_model_info.items():
            # Ollama exposes architecture metadata as dotted keys, e.g.
            # `qwen2vl.vision.image_size`, `qwen3.vision_model.config`,
            # `mllama.vision_*`. Substring match catches all patterns.
            if "vision" in k:
                return True
            if k == "capabilities" and isinstance(v, list) and "vision" in v:
                return True

    if hf_config:
        if "vision_config" in hf_config:
            return True
        text_cfg = hf_config.get("text_config")
        if isinstance(text_cfg, dict) and "vision_config" in text_cfg:
            return True
        for arch in hf_config.get("architectures", []) or []:
            arch_lower = str(arch).lower()
            if any(s in arch_lower for s in ("vl", "vision", "multimodal")):
                return True

    # Name heuristic — useful when HF cache hasn't been scanned but the
    # name itself is a strong signal (qwen3.5-fast, qwen3-vl-7b, etc).
    # Match on word-boundary `vl`/`vlm` so we don't false-positive on
    # `nemotron`, `phi`, etc.
    if re.search(r"(?:^|[-_.:/])(?:vl|vlm|vision)(?:[-_.:/]|$)", name_lower):
        return True
    if "vision" in name_lower:
        return True

    return False


SPECIAL_TASKS: dict[str, dict[str, Any]] = {
    "vision": {
        "label": "Vision",
        "prefixes": ["qwen3-vl", "llava", "moondream"],
    },
    "image_gen": {
        "label": "Image Gen",
        "prefixes": ["x/flux2", "x/z-image", "FLUX.1-dev", "FLUX.2", "stable-diffusion"],
    },
    "transcription": {
        "label": "Transcription",
        "prefixes": ["whisper"],
    },
    "tts": {
        "label": "Text-to-Speech",
        "prefixes": ["voxtral", "chatterbox"],
    },
    "image_edit": {
        "label": "Image Edit",
        "prefixes": ["FLUX.1-Kontext", "FLUX.1-Fill"],
    },
    "embedding": {
        "label": "Embedding",
        "prefixes": ["mxbai-embed", "nomic-embed", "snowflake-arctic", "all-minilm"],
    },
    "unfiltered": {
        "label": "Unfiltered",
        "prefixes": ["wizard-vicuna-uncensored", "dolphin", "nous-hermes"],
    },
    "computer_use": {
        "label": "Computer Use",
        "prefixes": ["ui-tars", "fara"],
    },
    "video": {
        "label": "Video",
        "prefixes": ["wan2", "ltx"],
    },
}

# ── Task filters ──────────────────────────────────────────────────────
# Model names excluded from all general LLM tasks (non-language models).

ALWAYS_EXCLUDE: list[str] = [
    "vl", "flux", "z-image", "whisper", "ocr", "embed", "minilm",
    "tinyllama", "goonsai", "nsfw", "dolphin",
    "wan2", "ltx",
]

TASK_FILTERS: dict[str, dict[str, Any]] = {
    "code": {
        "priority_names": ["coder"],
        "include_names": [
            "qwen3.5", "deepseek", "cogito", "nemotron",
            "gpt-oss", "llama3.3", "glm",
        ],
        "exclude_names": ALWAYS_EXCLUDE,
        "min_active_b": 3,
    },
    "general": {
        "exclude_names": ["coder"] + ALWAYS_EXCLUDE,
        "min_active_b": 3,
    },
    "reasoning": {
        "exclude_names": ["coder"] + ALWAYS_EXCLUDE,
        "min_active_b": 10,
    },
    "long_context": {
        "exclude_names": ALWAYS_EXCLUDE,
        "min_ctx": 64000,
    },
    "translation": {
        "exclude_names": ["coder"] + ALWAYS_EXCLUDE,
        "min_active_b": 3,
    },
}

# ── Active param computation ──────────────────────────────────────────

_AXB_PATTERN = re.compile(r"[_-]A(\d+(?:\.\d+)?)B", re.IGNORECASE)


def active_params_b(
    model_name: str,
    total_b: float,
    family: str,
    expert_count: int | None,
    expert_used: int | None,
    expert_ffn: int = 0,
    embed_len: int = 0,
    block_count: int = 0,
) -> float:
    """Compute active parameter count (in billions) for a model.

    Uses a four-strategy cascade for MoE architectures:
      1. Parse "AXB" suffix from model name (e.g. qwen3-coder_A3B).
      2. Look up in KNOWN_ACTIVE_PARAMS table.
      3. FFN subtraction from architecture metadata.
      4. Simple expert ratio as last resort.

    Returns total_b unchanged for non-MoE models.
    """
    if not expert_count or not expert_used or expert_count <= 1:
        return total_b

    total_rounded = round(total_b)

    # Strategy 1: parse AXB from name
    match = _AXB_PATTERN.search(model_name)
    if match:
        return float(match.group(1))

    # Strategy 2: known hybrid lookup
    family_table = KNOWN_ACTIVE_PARAMS.get(family)
    if family_table and total_rounded in family_table:
        return float(family_table[total_rounded])

    # Strategy 3: FFN subtraction
    if expert_ffn and embed_len and block_count:
        total_raw = int(total_b * 1e9)
        total_moe = block_count * expert_count * expert_ffn * embed_len * 3
        active_moe = block_count * expert_used * expert_ffn * embed_len * 3
        computed = total_raw - total_moe + active_moe
        if 0 < computed < total_raw:
            return round(computed / 1e9)

    # Strategy 4: simple ratio
    return round(total_b * expert_used / expert_count)


def model_matches_filter(
    name: str,
    active_params_b: float,
    context: int,
    task_filter: dict[str, Any],
) -> bool:
    """Check if a model passes a task filter.

    Accepts explicit active_params_b and context values so callers
    don't need to agree on dict key names.
    """
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

    min_active = task_filter.get("min_active_b", 0)
    if min_active and active_params_b > 0 and active_params_b < min_active:
        return False

    min_ctx = task_filter.get("min_ctx", 0)
    if min_ctx and context > 0 and context < min_ctx:
        return False

    return True


# ── mflux dispatch ────────────────────────────────────────────────────
#
# mflux 0.17+ ships family-specific binaries and a `--base-model` enum.
# Passing an HF path like "black-forest-labs/FLUX.2-klein-9B" to the
# generic `mflux-generate` makes it load the weights as FLUX.1 (two text
# encoders) and die looking for `text_encoder_2/`. Dispatch by family.

_MFLUX_DISPATCH: tuple[tuple[str, str, str], ...] = (
    # (substring match on lowercased id, binary, --base-model value)
    ("flux2-klein-base-9b",   "mflux-generate-flux2",         "flux2-klein-base-9b"),
    ("flux2-klein-base-4b",   "mflux-generate-flux2",         "flux2-klein-base-4b"),
    ("flux.2-klein-base-9b",  "mflux-generate-flux2",         "flux2-klein-base-9b"),
    ("flux.2-klein-base-4b",  "mflux-generate-flux2",         "flux2-klein-base-4b"),
    ("flux2-klein-9b",        "mflux-generate-flux2",         "flux2-klein-9b"),
    ("flux2-klein-4b",        "mflux-generate-flux2",         "flux2-klein-4b"),
    ("flux.2-klein-9b",       "mflux-generate-flux2",         "flux2-klein-9b"),
    ("flux.2-klein-4b",       "mflux-generate-flux2",         "flux2-klein-4b"),
    ("flux.2-klein",          "mflux-generate-flux2",         "flux2-klein-9b"),
    ("flux2-klein",           "mflux-generate-flux2",         "flux2-klein-9b"),
    ("z-image-turbo",         "mflux-generate-z-image-turbo", "z-image-turbo"),
    ("z-image",               "mflux-generate-z-image",       "z-image"),
    ("qwen-image",            "mflux-generate-qwen",          "qwen"),
    ("fibo-edit",             "mflux-generate-fibo",          "fibo-edit"),
    ("fibo-lite",             "mflux-generate-fibo",          "fibo-lite"),
    ("fibo",                  "mflux-generate-fibo",          "fibo"),
    ("krea-dev",              "mflux-generate",               "krea-dev"),
    ("flux.1-schnell",        "mflux-generate",               "schnell"),
    ("flux1-schnell",         "mflux-generate",               "schnell"),
    ("flux.1-dev",            "mflux-generate",               "dev"),
    ("flux1-dev",             "mflux-generate",               "dev"),
)


def mflux_command(model_id: str) -> tuple[str, list[str]]:
    """Return (binary, extra_args) for an image-gen model identifier.

    For recognized families, dispatches to the specialized binary and sets
    `--base-model` so mflux loads the right weight layout. When the id looks
    like an HF repo path we also pass `--model`, so a custom fork is honored
    instead of being silently replaced by mflux's default. Unrecognized ids
    fall through to `mflux-generate --model <id>`.
    """
    normalized = model_id.lower().replace("_", "-")
    for needle, binary, base in _MFLUX_DISPATCH:
        if needle in normalized:
            args = ["--base-model", base]
            if "/" in model_id:
                args += ["--model", model_id]
            return binary, args
    return "mflux-generate", ["--model", model_id]


def mflux_is_turbo(model_id: str) -> bool:
    """Few-step turbo/schnell variants. Used to pick a sane `--steps` default."""
    m = model_id.lower()
    return any(k in m for k in ("schnell", "turbo"))
