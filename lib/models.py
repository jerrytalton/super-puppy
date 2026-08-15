"""Shared model discovery constants, types, and helpers.

Single source of truth for MoE active-parameter tables, task definitions,
task filters, and config file paths. Imported by menubar, MCP server, and
profile server.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

# ── Config file paths ─────────────────────────────────────────────────

CONFIG_DIR = Path("~/.config/local-models").expanduser()
PROFILES_FILE = CONFIG_DIR / "profiles.json"
MCP_PREFS_FILE = CONFIG_DIR / "mcp_preferences.json"
MODEL_PREFS_FILE = CONFIG_DIR / "model_preferences.json"
NETWORK_CONF = CONFIG_DIR / "network.conf"
ACTIVITY_DB = Path(os.environ["SP_ACTIVITY_DB"]).expanduser() if os.environ.get("SP_ACTIVITY_DB") else CONFIG_DIR / "activity.db"
MLX_SERVER_CONFIG = Path("~/.config/mlx-server/config.yaml").expanduser()
CLAUDE_CONFIG_FILE = Path("~/.claude.json").expanduser()

# ── Default network config (must match config/local-models/network.conf) ──

_NETWORK_DEFAULTS = {
    "TAILSCALE_HOSTNAME": "super-puppy",
    "OLLAMA_PORT": "11434",
    "MLX_PORT": "8000",
    "DS4_PORT": "8002",
    "DS4_DIR": "",
    "SERVER_RAM_GB": "0",
    "PROBE_TIMEOUT": "2",
    "PROFILE_PORT": "8101",
    "OP_REF": "",
    "IS_SERVER": "false",
    "AUTO_UPDATE": "true",
    "AUTO_PULL": "true",
}

_NUMERIC_KEYS = {"OLLAMA_PORT", "MLX_PORT", "DS4_PORT", "SERVER_RAM_GB",
                 "PROBE_TIMEOUT", "PROFILE_PORT"}


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


def set_network_conf_value(key: str, value: str) -> None:
    """Update or append one key in network.conf.

    network.conf is hand-edited — every other line (comments included)
    is preserved verbatim.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = (NETWORK_CONF.read_text().splitlines()
             if NETWORK_CONF.exists() else [])
    prefix = f"{key}="
    out, replaced = [], False
    for line in lines:
        if line.strip().startswith(prefix):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    NETWORK_CONF.write_text("\n".join(out) + "\n")


# ── Chat backends & ds4 ──────────────────────────────────────────────
# Three chat-LLM backends. "ds4" is antirez/ds4 serving glm-5.2 on the
# 512GB tier (OpenAI-compatible, localhost:8002, internal-only — never
# tailscale-served; client-mode traffic is brokered by the desktop's MCP
# and profile server).

LLM_BACKENDS: frozenset[str] = frozenset({"ollama", "mlx", "ds4"})

# ds4's /v1/models returns NO params/vision metadata, and its GGUF lives
# outside every existing sizing path (not an HF snapshot, not an Ollama
# blob) — those fields must stay hardcoded here or TASK_FILTERS
# min_active_b/min_ctx silently drop glm-5.2 from every task list.
# It DOES report context_length (verified live 2026-07-23) — see
# ds4_live_context() below, which discovery prefers over this constant so
# a --ctx launch-flag drift can't silently diverge from what's advertised.
# DS4_MODEL_BYTES is the exact on-disk size of GLM-5.2-UD-Q2_K_RoutedQ2K.gguf.
DS4_MODEL_NAME = "glm-5.2"
DS4_MODEL_BYTES = 262_036_650_048
# Computed from mlx-community/GLM-5.2-4bit's config.json (glm_moe_dsa: 78
# layers, first_k_dense_replace=3, 256 routed + 1 shared experts/layer,
# moe_intermediate_size=2048, hidden_size=6144): summing embeddings+head,
# MLA attention, the DSA indexer, dense-layer FFN, and every expert's FFN
# (routed + shared — TOTAL counts every weight on disk, not just the
# active ones) gives ~743.6B. Rounded down. Corrects a stale 380B guess
# that undercounted by roughly 2x.
DS4_TOTAL_PARAMS_B = 740
DS4_ACTIVE_PARAMS_B = 32
DS4_CONTEXT = 131072


def ds4_live_context(models_response: dict) -> int:
    """Extract the live context_length from ds4's /v1/models response.

    Falls back to DS4_CONTEXT when the response is missing, malformed, or
    doesn't carry a usable context_length for DS4_MODEL_NAME — this is
    what stops a --ctx launch-flag change from silently drifting away
    from what discovery (and the min_ctx task-eligibility gate) believes
    glm-5.2 can serve.
    """
    try:
        for entry in models_response.get("data", []):
            if entry.get("id") == DS4_MODEL_NAME:
                ctx = entry.get("context_length")
                if isinstance(ctx, int) and ctx > 0:
                    return ctx
    except (AttributeError, TypeError):
        pass
    return DS4_CONTEXT


DS4_DIR_DEFAULT = "~/.local/share/super-puppy/ds4"


def ds4_dir() -> Path:
    """The ds4 checkout/build/gguf directory.

    network.conf's DS4_DIR overrides; default is DS4_DIR_DEFAULT. install.sh
    provisions this directory on 512GB machines (and symlinks an existing
    ~/experiments/ds4 checkout when present).
    """
    if NETWORK_CONF.exists():
        for line in NETWORK_CONF.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("DS4_DIR="):
                val = stripped.partition("=")[2].strip().strip('"').strip("'")
                if val:
                    return Path(val).expanduser()
    return Path(DS4_DIR_DEFAULT).expanduser()


def ds4_installed() -> bool:
    """True only where install.sh actually provisioned ds4 (512GB tier).

    Gates every ds4 surface that would otherwise show a permanently-red
    service on machines that never run it.
    """
    return (ds4_dir() / "ds4-server").exists()


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


def merge_profile_picks(prefs: dict, profile: dict) -> dict:
    """Promote a profile's task picks to the front of each preference list.

    `mcp_preferences.json` — not profiles.json — is what the pickers actually
    read, and a PROFILES_VERSION bump only rewrites the presets. Without this,
    a preset fix (say, moving image_gen off a backend that stopped working)
    lands on disk and changes nothing: the stale pref still wins, and the new
    model is never even a candidate.

    Non-destructive: previously-configured models stay in the list behind the
    profile's pick, so a machine keeps its fallbacks. Same operation the
    Activate button performs, shared so the two can't drift.
    """
    merged = dict(prefs)
    for task, pick in (profile.get("tasks") or {}).items():
        existing = merged.get(task, [])
        if isinstance(existing, str):
            existing = [existing]
        merged[task] = [pick] + [m for m in existing if m != pick]
    if profile.get("thinking"):
        thinking = dict(merged.get("thinking") or {})
        thinking.update(profile["thinking"])
        merged["thinking"] = thinking
    return merged


def image_backend_eligible(_name: str, backend: str) -> bool:
    """Gate image_gen/image_edit candidates to the mflux backend.

    Ollama 0.32 rejects /api/generate for image models with a 400 while
    /api/show still advertises `capabilities: ["image"]` — the same lie its
    `-mlx` tags tell about vision. Both image tasks dispatch through mflux,
    so an Ollama-backed candidate resolves by name and then dies at
    dispatch with a backend error the user can do nothing about. Gating
    here makes the picker fall through to a pref that can actually serve
    the task, exactly as the vision gate does for tower-less tags.
    """
    return backend == HF_TASK_BACKENDS["image_gen"]


def resolve_pref_candidate(
    candidate: str,
    models: dict[str, dict],
    *,
    task: str | None = None,
    allow_hf_on_demand: bool = False,
    is_eligible: Callable[[str, str], bool] | None = None,
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

    `is_eligible(name, backend)` gates capability-matched tasks: a
    candidate that resolves by name but fails the predicate is treated
    as "no match" (returns None) so the caller can try the next pref.
    This is what keeps a vision pref that points at a tower-less tag
    (e.g. an Ollama `-mlx` tag advertising vision it can't do) from
    being handed back for the vision tool to reject.

    Returns None if nothing matches.
    """
    def _ok(name: str, backend: str) -> bool:
        return not is_eligible or is_eligible(name, backend)

    if candidate in models and _ok(candidate, models[candidate]["backend"]):
        return candidate, models[candidate]["backend"]
    suffix_target = candidate + ":"
    latest = candidate + ":latest"
    for name in models:
        if (name == latest or name.startswith(suffix_target)) \
                and _ok(name, models[name]["backend"]):
            return name, models[name]["backend"]
    # "/" alone does not mean "HF repo id" — Ollama namespaced tags look the
    # same ("x/z-image-turbo:bf16"). Two extra conditions keep those out:
    #   * `":" not in candidate` — HF repo ids are alphanumeric plus '-', '_'
    #     and '.' (huggingface_hub's validate_repo_id rejects a colon), while
    #     an Ollama tag always carries one.
    #   * `candidate not in models` — download-on-demand is only for models
    #     the registry doesn't have. Without it, a known Ollama model that
    #     is_eligible just rejected gets re-admitted here and relabelled with
    #     the HF backend.
    # Miss either and the caller is handed an Ollama tag to feed to mflux,
    # which dies with HFValidationError instead of falling through to a pref
    # that works.
    if (allow_hf_on_demand and task in HF_TASK_BACKENDS
            and candidate not in models
            and "/" in candidate and ":" not in candidate
            and _ok(candidate, HF_TASK_BACKENDS[task])):
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
    is_eligible: Callable[[str, str], bool] | None = None,
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
                allow_hf_on_demand=allow_hf_on_demand,
                is_eligible=is_eligible)
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
    ollama_projector_info: dict | None = None,
    hf_config: dict | None = None,
) -> bool:
    """Single source of truth: does this model have a working vision tower?

    Checks five signals in order, any one is sufficient:

    1. Ollama model_info contains a "vision" architecture key (e.g.
       `qwen35.vision.embedding_length`, `qwen2vl.vision.image_size`).
       These keys are present only when the vision encoder weights are
       actually in the model — they are the reliable signal.
    1b. Ollama /api/show `projector_info` declares a vision encoder.
       Ollama ≥0.32 ships some models' vision encoders as a separate
       projector blob (qwen3.8's is an 888MB mmproj layer), so their
       model_info carries zero `*.vision.*` keys and the signal moves
       here (`clip.has_vision_encoder`, `clip.vision.*`). Tower-less
       tags return projector_info: null, so this stays honest.
    2. HF config.json declares a `vision_config` block, either at the
       top level or nested under `text_config` (Qwen3.5 family).
    3. HF config.json's `architectures` list contains a known
       vision-language architecture name.
    4. The model's name contains a vision substring (`vl`, `vision`,
       `vlm`) — last-resort heuristic that matches the Qwen3.5 /
       Qwen-VL / Llama-VL families when their HF cache hasn't been
       scanned yet.

    NOTE: Ollama's top-level `capabilities` array is deliberately NOT
    used. Its MLX-converted tags (`*-mlx`, `*-mlx-bf16`) advertise
    `capabilities: ["vision"]` while shipping no vision tower — their
    model_info has zero `*.vision.*` keys and they silently ignore
    image input (verified on Ollama 0.30.10, qwen3.6:27b-mlx-bf16).
    Trusting `capabilities` turns a loud "not vision-capable" error
    into silent hallucination. The model_info vision keys, present only
    when the encoder weights exist, are the honest signal.

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

    if ollama_projector_info:
        for k, v in ollama_projector_info.items():
            # An explicit encoder boolean decides for itself: mmproj
            # metadata ships both encoder booleans, so an audio-only
            # projector carries `clip.has_vision_encoder: False` — the
            # key name alone must not count as a vision signal.
            if k == "clip.has_vision_encoder":
                if v:
                    return True
                continue
            if "vision" in k:
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
        "prefixes": ["ui-tars", "fara", "holo"],
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
        # "coding" catches fine-tune tags like qwen3.6:27b-coding-mxfp8,
        # which "coder" misses.
        "priority_names": ["coder", "coding"],
        # "qwen3" covers the whole line (3.5/3.6/3.8); ALWAYS_EXCLUDE
        # still drops qwen3-vl / qwen3-embedding.
        "include_names": [
            "qwen3", "deepseek", "cogito", "nemotron",
            "gpt-oss", "llama3.3", "glm", "muse-glimmer",
        ],
        "exclude_names": ALWAYS_EXCLUDE,
        "min_active_b": 3,
    },
    "general": {
        "exclude_names": ["coder", "coding"] + ALWAYS_EXCLUDE,
        "min_active_b": 3,
    },
    "reasoning": {
        "exclude_names": ["coder", "coding"] + ALWAYS_EXCLUDE,
        "min_active_b": 10,
    },
    "long_context": {
        "exclude_names": ALWAYS_EXCLUDE,
        "min_ctx": 64000,
    },
    "translation": {
        "exclude_names": ["coder", "coding"] + ALWAYS_EXCLUDE,
        "min_active_b": 3,
    },
}


# ── Default model profiles ──────────────────────────────────────────
# Single source of truth for preset profiles. Consumed by profile-server
# (serves/migrates them), the menu bar app (seeds them on startup), and
# install.sh (seeds them before pulling models, via lib.models).
#
# Four RAM-tier presets (32gb / 64gb / 128gb / 512gb) replace the old
# named profiles (laptop / desktop / everyday / maximum). Each tier's
# max_ram_gb cap gates model-pull validation in install.sh and the profile
# server. The active default is 64gb (fits M5 / mid GPU class).

PROFILES_VERSION = 36  # bump to force-refresh preset profiles on all machines

DEFAULT_PROFILES = {
    "version": PROFILES_VERSION,
    "active": "64gb",
    "profiles": {
        "32gb": {
            "label": "32 GB",
            "description": "Base M5 / M1 Max class — small, fast models",
            "max_ram_gb": 32,
            "warm": ["general", "embedding"],
            "tasks": {
                "code": "qwen3.5-small",
                "general": "qwen3.5-small",
                "reasoning": "qwen3.5-small",
                "long_context": "qwen3.5-small",
                "translation": "qwen3.5-small",
                # qwen3.5-small can't serve vision: mlx-openai-server's
                # multimodal (VLM) path is broken by the mlx 0.31.2 stream
                # bug (generation hangs, mlx-lm #1256). Vision routes to the
                # GGUF tag that works — qwen3.8:27b ships its encoder as a
                # separate projector blob (detected via /api/show
                # projector_info). ~17GB, loaded on demand for vision.
                "vision": "qwen3.8:27b",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
                "embedding": "embeddinggemma:300m",
                # image_gen runs on mflux, not Ollama. Ollama 0.32 rejects
                # /api/generate for image models (400 "image generation
                # models are not currently supported") while /api/show still
                # advertises capabilities: ["image"] — so an Ollama tag here
                # resolves fine and then dies at dispatch. HF repo ids only:
                # resolve_pref_candidate's on-demand path requires a "/".
                # 4B (23.7GB, ungated) on the small tiers; the 9B is 52.9GB.
                "image_gen": "black-forest-labs/FLUX.2-klein-4B",
            },
        },
        "64gb": {
            "label": "64 GB",
            "description": "M5 / mid GPU — dense 27B workhorse",
            "max_ram_gb": 64,
            "warm": ["general", "embedding"],
            "tasks": {
                "code": "qwen3.6:27b-coding-mxfp8",
                "general": "qwen3.8:27b-mlx",
                "reasoning": "qwen3.8:27b-mlx",
                "long_context": "qwen3.8:27b-mlx",
                "translation": "qwen3.8:27b-mlx",
                "vision": "qwen3.8:27b",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
                "embedding": "qwen3-embedding:8b",
                "unfiltered": "dolphin3:8b",
                "computer_use": "ui-venus",
                "image_gen": "black-forest-labs/FLUX.2-klein-4B",
                # FLUX.2-edit on klein-4B replaced Kontext (2026-08-15
                # bake-off + corrected rerun): quality indistinguishable
                # from true 9B across recolor/remove/add-sign, at 56s and
                # 13.6GB peak (9B: 106s/24GB; Kontext: 145s/31GB). Shares
                # weights with this tier's klein-4B gen pick.
                "image_edit": "black-forest-labs/FLUX.2-klein-4B",
            },
        },
        "128gb": {
            "label": "128 GB",
            "description": "M5 Max class — dense 27B + strong vision",
            "max_ram_gb": 128,
            "warm": ["general", "embedding"],
            "tasks": {
                "code": "qwen3-coder-next:latest",
                # Text tasks share the ~18GB -mlx quant, not -mlx-bf16: the
                # bf16 tag's ~54GB footprint got evicted/reloaded under
                # concurrent image+video+MLX load, corrupting context (it
                # echoed a prior request's system prompt, observed on
                # qwen3.6:27b-mlx-bf16). Same tag the 64gb tier uses.
                "general": "qwen3.8:27b-mlx",
                "reasoning": "qwen3.8:27b-mlx",
                "long_context": "qwen3.8:27b-mlx",
                "translation": "qwen3.8:27b-mlx",
                "vision": "qwen3.8:27b",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
                "embedding": "qwen3-embedding:8b",
                "unfiltered": "dolphin3:8b",
                "computer_use": "ui-venus",
                "image_gen": "black-forest-labs/FLUX.2-klein-9B",
                "image_edit": "black-forest-labs/FLUX.2-klein-4B",
                "video": "AITRADER/Wan2.2-T2V-A14B-mlx-bf16",
            },
        },
        "512gb": {
            "label": "512 GB",
            "description": "M3 Ultra class — frontier",
            "max_ram_gb": 512,
            "warm": ["general", "embedding"],
            "tasks": {
                "code": "qwen3-coder-next:latest",
                "general": "glm-5.2",
                "reasoning": "glm-5.2",
                "long_context": "glm-5.2",
                "translation": "glm-5.2",
                # A dense 27B GGUF (formerly qwen3.6:27b, now qwen3.8:27b)
                # beats the 35B-A3B MoE on vision benchmarks and actually
                # serves images end-to-end; the prior qwen3.5:122b pick
                # wasn't even a served model.
                "vision": "qwen3.8:27b",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit",
                "embedding": "qwen3-embedding:8b",
                "unfiltered": "dolphin3:8b",
                "computer_use": "ui-venus",
                "image_gen": "black-forest-labs/FLUX.2-klein-9B",
                "image_edit": "black-forest-labs/FLUX.2-klein-4B",
                "video": "AITRADER/Wan2.2-T2V-A14B-mlx-bf16",
            },
        },
    },
}

RETIRED_PROFILE_NAMES = frozenset({"laptop", "desktop", "everyday", "maximum"})


def migrate_profiles(data: dict) -> dict:
    """Bring an on-disk profiles dict up to the current presets.

    Refreshes the preset profiles, drops retired preset names, preserves
    genuinely-custom profiles (anything not a current or retired preset), and
    repairs `active` if it no longer names a profile that exists.
    """
    refreshed = {**DEFAULT_PROFILES, "profiles": dict(DEFAULT_PROFILES["profiles"])}
    for name, profile in (data.get("profiles") or {}).items():
        if name in DEFAULT_PROFILES["profiles"] or name in RETIRED_PROFILE_NAMES:
            continue
        refreshed["profiles"][name] = profile
    active = data.get("active")
    refreshed["active"] = active if active in refreshed["profiles"] else DEFAULT_PROFILES["active"]
    return refreshed


WARM_BUDGET_FRACTION = 0.65  # warm set should fit under this fraction of tier RAM


def warm_task_keys(name: str, profile: dict) -> list[str]:
    """The warm task keys for a profile, self-healing for stale presets.

    A profiles.json written before `warm` existed (or otherwise missing it) can
    sit at the current version and never re-migrate. So when a *preset* profile
    has no `warm` key, fall back to the canonical `DEFAULT_PROFILES` warm list
    rather than treating it as "nothing warm" (which renders the bar as 0B warm
    and keeps no models resident). Custom profiles legitimately default to [].
    """
    keys = profile.get("warm")
    if keys is None and name in DEFAULT_PROFILES["profiles"]:
        keys = DEFAULT_PROFILES["profiles"][name].get("warm", [])
    return keys or []


def warm_model_names(data: dict) -> set[str]:
    """Model names kept warm for the active profile (its `warm` task keys).

    Returns an empty set when there is no active profile. A task key that isn't
    present in the profile's tasks is skipped. Stale presets missing `warm` are
    healed via warm_task_keys.
    """
    name = data.get("active")
    prof = (data.get("profiles") or {}).get(name)
    if not prof:
        return set()
    tasks = prof.get("tasks", {})
    return {tasks[k] for k in warm_task_keys(name, prof) if k in tasks}


def profile_ollama_models(profile: dict) -> set[str]:
    """The Ollama tags a profile's task picks need pulled locally.

    Same classification warm_ping_targets and install.sh use: an Ollama
    tag always carries ':' — including namespaced ones like
    "x/z-image-turbo:bf16" — while HF repo ids never do (huggingface_hub
    rejects a colon) and bare names are MLX- or ds4-served.
    """
    return {
        model for model in (profile.get("tasks") or {}).values()
        if ":" in model
    }


def profile_hf_models(profile: dict) -> set[str]:
    """The HF repo picks a profile needs downloaded locally (image
    gen/edit, video, TTS) — the same set install.sh fetches via
    `hf download`. A '/' without ':' means an HF repo id; anything
    carrying ':' is an Ollama tag (namespaced tags have both).
    """
    return {
        model for model in (profile.get("tasks") or {}).values()
        if "/" in model and ":" not in model
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


# ── mflux edit dispatch ──────────────────────────────────────────────
#
# Editing is a different binary set from generation, with three different
# interfaces: kontext takes a single --image-path and honours
# --image-strength; the flux2/qwen/fibo edit binaries take a variadic
# --image-paths and have no strength knob. So this can't reuse
# _MFLUX_DISPATCH — feeding an edit model to mflux_command() picks a
# *generate* binary that has no image input at all.

class MfluxEditCommand(NamedTuple):
    binary: str
    extra_args: list[str]
    image_flag: str          # "--image-path" (single) or "--image-paths" (variadic)
    supports_strength: bool


_MFLUX_EDIT_DISPATCH: tuple[tuple[str, str, str, str, bool], ...] = (
    # (substring match on lowercased id, binary, --base-model, image flag, strength)
    ("kontext",          "mflux-generate-kontext",    "",               "--image-path",  True),
    ("flux.2-klein-4b",  "mflux-generate-flux2-edit", "flux2-klein-4b", "--image-paths", False),
    ("flux2-klein-4b",   "mflux-generate-flux2-edit", "flux2-klein-4b", "--image-paths", False),
    ("flux.2-klein-9b",  "mflux-generate-flux2-edit", "flux2-klein-9b", "--image-paths", False),
    ("flux2-klein-9b",   "mflux-generate-flux2-edit", "flux2-klein-9b", "--image-paths", False),
    ("flux.2-klein",     "mflux-generate-flux2-edit", "flux2-klein-9b", "--image-paths", False),
    ("flux2-klein",      "mflux-generate-flux2-edit", "flux2-klein-9b", "--image-paths", False),
    ("qwen-image-edit",  "mflux-generate-qwen-edit",  "qwen",           "--image-paths", False),
    ("fibo-edit",        "mflux-generate-fibo-edit",  "fibo-edit",      "--image-paths", False),
)


def mflux_edit_command(model_id: str) -> MfluxEditCommand:
    """Return how to invoke mflux for an image-edit model.

    Both servers used to hardcode `mflux-generate-kontext` and drop the
    resolved model on the floor, so a profile naming Qwen-Image-Edit
    silently got FLUX Kontext instead — wrong weights, no error.

    Unrecognized ids fall back to the kontext binary but still carry the
    caller's `--model`, so an incompatible choice fails loudly in mflux
    rather than being silently swapped for something else.
    """
    normalized = model_id.lower().replace("_", "-")
    for needle, binary, base, image_flag, strength in _MFLUX_EDIT_DISPATCH:
        if needle in normalized:
            args = ["--base-model", base] if base else []
            if "/" in model_id:
                args += ["--model", model_id]
            return MfluxEditCommand(binary, args, image_flag, strength)
    return MfluxEditCommand(
        "mflux-generate-kontext", ["--model", model_id], "--image-path", True)


def mflux_is_turbo(model_id: str) -> bool:
    """Few-step turbo/schnell variants. Used to pick a sane `--steps` default."""
    m = model_id.lower()
    return any(k in m for k in ("schnell", "turbo"))
