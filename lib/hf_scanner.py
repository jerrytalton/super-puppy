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
}

# HF model ID substrings that refine task classification
_NAME_TASK_OVERRIDES = {
    "Kontext": "image_edit",
    "Fill": "image_edit",
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


def _latest_snapshot(model_dir: Path) -> Path | None:
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    dirs = sorted(snapshots.iterdir(), reverse=True)
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
    import numpy as np

    total_params = 0
    dtypes: set[str] = set()
    total_bytes = 0

    for npz_file in snap_dir.glob("*.npz"):
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

    if not task:
        return None

    # Refine by model name (e.g. Flux Kontext → image_edit, not image_gen)
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
    dtypes, quant_bits, disk_bytes, vram_bytes. Returns None if unclassifiable.
    """
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
