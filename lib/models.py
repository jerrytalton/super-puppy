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
MLX_SERVER_CONFIG = Path("~/.config/mlx-server/config.yaml").expanduser()
CLAUDE_CONFIG_FILE = Path("~/.claude.json").expanduser()

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
}

# ── Task filters ──────────────────────────────────────────────────────
# Model names excluded from all general LLM tasks (non-language models).

ALWAYS_EXCLUDE: list[str] = [
    "vl", "flux", "z-image", "whisper", "ocr", "embed", "minilm",
    "tinyllama", "goonsai", "nsfw", "dolphin",
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
