"""Scan HuggingFace cache for downloaded models and extract metadata.

Reads config.json for model_type/architecture classification, and
safetensors/npz headers for accurate param counts and dtype. No network
calls — purely local filesystem reads.
"""

import json
import struct
from pathlib import Path

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# model_type → task mapping (from HuggingFace config.json)
_MODEL_TYPE_TASKS = {
    "voxtral_tts": "tts",
    "chatterbox": "tts",
    "whisper": "transcription",
}

# diffusers _class_name → task mapping (from transformer/config.json)
_DIFFUSERS_CLASS_TASKS = {
    "FluxTransformer2DModel": "image_gen",
    "Flux2Transformer2DModel": "image_gen",
    "WanTransformer3DModel": "video",
    "LTXVideoTransformer3DModel": "video",
}

# HF model ID substrings that refine task classification
_NAME_TASK_OVERRIDES = {
    "Kontext": "image_edit",
    "Fill": "image_edit",
    "ltx-video": "video",
    "ltx2": "video",
    "wan2": "video",
}

# Bytes per element for each dtype
_DTYPE_BYTES = {
    "F32": 4, "F16": 2, "BF16": 2,
    "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U8": 1, "BOOL": 1,
    # numpy dtype names
    "float16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1,
}


def hf_snapshot_dir(model_path: str) -> Path:
    """Path to the snapshots/ directory for an HF repo id, regardless of
    whether it exists yet. Encapsulates the `models--{x}--{y}` mangling
    that used to be inlined in 6+ places."""
    return HF_CACHE / f"models--{model_path.replace('/', '--')}" / "snapshots"


def hf_newest_snapshot(model_path: str) -> Path | None:
    """Newest snapshot directory by mtime, or None if nothing exists.

    Use this whenever the caller needs the snapshot the user most
    recently pulled. Lex sort of HF snapshot hashes is essentially
    random and was producing inconsistent results across the codebase
    (vision detection in MCP vs. profile-server reading different
    snapshots of the same model)."""
    snapshots = hf_snapshot_dir(model_path)
    if not snapshots.exists():
        return None
    dirs = sorted(snapshots.iterdir(),
                  key=lambda p: p.stat().st_mtime,
                  reverse=True)
    return dirs[0] if dirs else None


# Files that must be resolved (i.e. their snapshot symlinks point to a
# fully-downloaded blob) before mlx_video.generate_wan can load a Wan
# checkpoint. HF only materializes the symlink after the blob finishes
# downloading, so a partial pull leaves a snapshot directory that *looks*
# present but fails with a cryptic load_safetensors error.
_WAN_REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "t5_encoder.safetensors",
    "vae.safetensors",
)
_WAN_MODEL_LAYOUTS: tuple[tuple[str, ...], ...] = (
    # Either a single-model layout or a dual-expert MoE layout.
    ("model.safetensors",),
    ("high_noise_model.safetensors", "low_noise_model.safetensors"),
)


def check_wan_snapshot_ready(snapshot: Path) -> str | None:
    """Verify a Wan MLX snapshot has every weight file resolved.

    Returns None when the snapshot is ready, otherwise a human-readable
    description of what's missing — caller can drop that straight into
    a user-facing error message.
    """
    missing = [f for f in _WAN_REQUIRED_FILES if not (snapshot / f).exists()]
    has_complete_layout = any(
        all((snapshot / f).exists() for f in layout)
        for layout in _WAN_MODEL_LAYOUTS
    )
    if not has_complete_layout:
        missing.append(
            "model weights (model.safetensors or "
            "high_noise_model.safetensors + low_noise_model.safetensors)"
        )
    if missing:
        return "missing " + ", ".join(missing)
    return None


def read_newest_hf_config(model_path: str) -> dict | None:
    """Parse config.json from the newest HF snapshot. Returns None when
    nothing is downloaded, the snapshot lacks a config, or the JSON is
    malformed. Centralizes a pattern repeated across discovery, vision
    detection, and capability probing."""
    snap = hf_newest_snapshot(model_path)
    if snap is None:
        return None
    cfg = snap / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text())
    except Exception:
        return None


def _latest_snapshot(model_dir: Path) -> Path | None:
    """Internal helper used by scan_hf_model. Takes the model's top-level
    cache dir (containing the `snapshots/` subdir) so it can be reused
    while iterating HF_CACHE."""
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    dirs = sorted(snapshots.iterdir(),
                  key=lambda p: p.stat().st_mtime,
                  reverse=True)
    return dirs[0] if dirs else None


def _read_config(snap_dir: Path) -> dict:
    """Read config.json, checking top-level and transformer/ subdirectory."""
    for candidate in (snap_dir / "config.json", snap_dir / "transformer" / "config.json"):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                pass
    return {}


def _count_safetensor_params(snap_dir: Path) -> tuple[int, set[str], int]:
    """Count params from safetensors headers. Returns (total_params, dtypes, file_bytes)."""
    total_params = 0
    dtypes: set[str] = set()
    total_bytes = 0

    safetensor_files = list(snap_dir.glob("*.safetensors"))
    for subdir in ("transformer", "text_encoder", "text_encoder_2"):
        safetensor_files.extend((snap_dir / subdir).glob("*.safetensors"))

    for sf in safetensor_files:
        total_bytes += sf.stat().st_size
        try:
            with open(sf, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_size))
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                params = 1
                for s in info["shape"]:
                    params *= s
                total_params += params
                dtypes.add(info["dtype"])
        except Exception:
            continue

    return total_params, dtypes, total_bytes


def _count_npz_params(snap_dir: Path) -> tuple[int, set[str], int]:
    """Count params from numpy .npz files (used by whisper-mlx etc)."""
    total_params = 0
    dtypes: set[str] = set()
    total_bytes = 0

    npz_files = list(snap_dir.glob("*.npz"))
    if not npz_files:
        return total_params, dtypes, total_bytes

    try:
        import numpy as np
    except ImportError:
        # numpy not available — return file sizes only
        for npz_file in npz_files:
            total_bytes += npz_file.stat().st_size
        return total_params, dtypes, total_bytes

    for npz_file in npz_files:
        total_bytes += npz_file.stat().st_size
        try:
            data = np.load(npz_file)
            for arr in data.values():
                total_params += arr.size
                dtypes.add(str(arr.dtype))
        except Exception:
            continue

    return total_params, dtypes, total_bytes


def _estimate_vram(total_params: int, dtypes: set[str], quant_config: dict | None) -> int:
    """Estimate VRAM usage in bytes."""
    if quant_config:
        bits = quant_config.get("bits", 4)
        return int(total_params * bits / 8)
    if dtypes:
        bytes_per = max(_DTYPE_BYTES.get(d, 2) for d in dtypes)
        return total_params * bytes_per
    return total_params * 2


def _classify_model(config: dict, hf_id: str) -> str | None:
    """Determine task type from config.json contents and model name."""
    model_type = config.get("model_type", "")
    task = _MODEL_TYPE_TASKS.get(model_type)

    if not task:
        class_name = config.get("_class_name", "")
        task = _DIFFUSERS_CLASS_TASKS.get(class_name)

    if not task:
        for arch in config.get("architectures", []):
            arch_lower = arch.lower()
            if "whisper" in arch_lower:
                task = "transcription"
                break
            if "tts" in arch_lower or "speech" in arch_lower:
                task = "tts"
                break

    # Kokoro TTS: no model_type, but has istftnet + plbert vocoder config
    if not task and "istftnet" in config and "plbert" in config:
        task = "tts"

    # Refine or assign by model name (e.g. Flux Kontext → image_edit, not image_gen)
    for substring, override_task in _NAME_TASK_OVERRIDES.items():
        if substring.lower() in hf_id.lower():
            task = override_task
            break

    return task


def _hf_id_from_dir(model_dir: Path) -> str:
    dir_name = model_dir.name
    if dir_name.startswith("models--"):
        return dir_name[len("models--"):].replace("--", "/", 1)
    return dir_name


def scan_hf_model(model_dir: Path) -> dict | None:
    """Scan a single HuggingFace cache directory and return model metadata.

    Returns dict with: name, task, model_type, total_params, total_params_b,
    dtypes, quant_bits, disk_bytes, vram_bytes. Returns None if unclassifiable
    or still downloading (any .incomplete blob means some snapshot symlinks
    are unresolved — treating the repo as installed would mark a partially-
    downloaded model as 'active' in the UI and let the task-dispatch layer
    pick a model that can't actually run)."""
    blobs_dir = model_dir / "blobs"
    if blobs_dir.exists():
        for b in blobs_dir.iterdir():
            if b.name.endswith(".incomplete"):
                return None

    snap = _latest_snapshot(model_dir)
    if not snap:
        return None

    hf_id = _hf_id_from_dir(model_dir)
    config = _read_config(snap)
    task = _classify_model(config, hf_id)
    if not task:
        return None

    # Try safetensors first, fall back to npz
    total_params, dtypes, disk_bytes = _count_safetensor_params(snap)
    if not total_params:
        total_params, dtypes, disk_bytes = _count_npz_params(snap)

    # Last resort: directory size
    if not disk_bytes:
        disk_bytes = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file())

    quant_config = config.get("quantization_config") or config.get("quantization")
    vram_bytes = _estimate_vram(total_params, dtypes, quant_config)

    model_type = config.get("model_type", config.get("_class_name", ""))
    quant_bits = quant_config.get("bits") if quant_config else None
    params_b = round(total_params / 1e9, 2)

    return {
        "name": hf_id,
        "task": task,
        "model_type": model_type,
        "total_params": total_params,
        "total_params_b": params_b,
        "dtypes": sorted(dtypes),
        "quant_bits": quant_bits,
        "disk_bytes": disk_bytes,
        "vram_bytes": vram_bytes,
    }


def scan_hf_cache(tasks: set[str] | None = None) -> list[dict]:
    """Scan the HuggingFace cache for all downloaded models.

    Args:
        tasks: If provided, only return models matching these task types.
               e.g. {"tts", "transcription", "image_gen", "image_edit"}

    Returns list of model metadata dicts, sorted by name.
    """
    if not HF_CACHE.exists():
        return []

    results = []
    for model_dir in sorted(HF_CACHE.iterdir()):
        if not model_dir.name.startswith("models--"):
            continue
        info = scan_hf_model(model_dir)
        if info and (tasks is None or info["task"] in tasks):
            results.append(info)

    return results
