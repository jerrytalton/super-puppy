"""One-shot ``mlx_vlm`` subprocess dispatch for MLX multimodal models.

mlx-openai-server's persistent :8000 server loads a model on one thread
and generates on another; mlx 0.31.2 made compute streams thread-local
(mlx-lm #1256), so multimodal generation hangs/RPC-times-out there. A
one-shot ``mlx_vlm generate`` subprocess loads AND generates in a single
fresh process/thread, sidestepping the bug — the same pattern mflux
(image) and mlx-audio (TTS) already use.

Shared so the MCP server (async, via run_in_executor) and the profile
server (sync, Flask) run identical dispatch. `generate` is synchronous;
async callers wrap it.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
from pathlib import Path

# Coordinate space UI-Venus / Qwen-VL grounding models emit box coords in.
GROUNDING_COORD_SPACE = 1000


def parse_output(raw: str) -> str:
    """Extract the generated text from ``mlx_vlm generate`` CLI output.

    The CLI wraps output in ``====`` fences: a header (Files:/Prompt: with
    the echoed chat-templated prompt), the generation, a closing fence,
    then timing stats. The generation follows the prompt echo's
    assistant-turn marker.
    """
    parts = raw.split("==========")
    block = parts[1] if len(parts) >= 3 else (parts[-1] if len(parts) > 1 else raw)
    for marker in ("<|im_start|>assistant", "\nassistant\n", "assistant\n"):
        i = block.rfind(marker)
        if i != -1:
            return block[i + len(marker):].strip()
    lines = [ln for ln in block.splitlines()
             if not ln.startswith(("Files:", "Prompt:"))]
    return "\n".join(lines).strip()


def repo_for(served_name: str, config_path: str | Path) -> str:
    """Resolve an MLX served-model name to its HuggingFace repo path via
    the mlx-server config, falling back to the name itself."""
    try:
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text())
        for m in cfg.get("models", []):
            if m.get("served_model_name") == served_name:
                return m.get("model_path", served_name)
    except Exception:
        pass
    return served_name


def command() -> list[str]:
    """Interpreter + module prefix for ``mlx_vlm generate``.

    Honors $MLX_VLM_PYTHON; else prefers the dedicated ``mlx-vlm`` uv tool
    env (installed by install.sh with torch), then the mlx-openai-server
    tool env (also has mlx_vlm + torch), then an isolated uvx env.
    """
    override = os.environ.get("MLX_VLM_PYTHON")
    if override:
        return [override, "-m", "mlx_vlm", "generate"]
    for tool in ("mlx-vlm", "mlx-openai-server"):
        py = os.path.expanduser(f"~/.local/share/uv/tools/{tool}/bin/python")
        if os.path.exists(py):
            return [py, "-m", "mlx_vlm", "generate"]
    return ["uvx", "--from", "mlx-vlm==0.4.4", "--with", "torch",
            "python", "-m", "mlx_vlm", "generate"]


def generate(repo: str, image_path: str, system: str, prompt: str,
             max_tokens: int = 1024, timeout: int = 600) -> str:
    """Run a one-shot mlx_vlm subprocess; return the generated text.

    Raises RuntimeError on a non-zero exit and propagates
    subprocess.TimeoutExpired.
    """
    cmd = [*command(), "--model", repo, "--image", image_path,
           "--system", system, "--prompt", prompt,
           "--max-tokens", str(max_tokens), "--temperature", "0.0"]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        env={**os.environ,
             "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"})
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_vlm failed: {proc.stderr[-500:]}")
    return parse_output(proc.stdout)


def image_dimensions(path: str | Path) -> tuple[int, int] | None:
    """Read (width, height) from a PNG or JPEG header using only stdlib."""
    try:
        data = Path(path).read_bytes()
    except Exception:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)
    if data[:2] == b"\xff\xd8":  # JPEG
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            # SOF markers carry the dimensions (skip DHT/DAC/DRI/RST).
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return int(w), int(h)
            if i + 4 > len(data):
                break
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + seg_len
    return None


_BOX_RE = re.compile(
    r"[Cc]lick\s*\(\s*box\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
    r"|\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def normalize_grounding(raw: str, image_width: int | None,
                        image_height: int | None,
                        coord_space: int = GROUNDING_COORD_SPACE) -> str:
    """Convert a model-native grounding action to a standard JSON click
    action with pixel coordinates.

    UI-Venus / Qwen-VL grounding models emit e.g.
    ``<answer>Click(box=(529,719))</answer>`` with coordinates in a
    0-`coord_space` normalized space. This denormalizes to actual pixels
    and wraps it as ``[{"action":"click","x":..,"y":..,"description":..}]``.
    Returns `raw` unchanged when no coordinate pair is found or image
    dimensions are unknown (so unrecognized formats pass through).
    """
    if not image_width or not image_height:
        return raw
    m = _BOX_RE.search(raw)
    if not m:
        return raw
    nx = int(m.group(1) if m.group(1) is not None else m.group(3))
    ny = int(m.group(2) if m.group(2) is not None else m.group(4))
    px = round(nx / coord_space * image_width)
    py = round(ny / coord_space * image_height)
    action = {"action": "click", "x": px, "y": py,
              "description": raw.strip()[:120]}
    return json.dumps([action], indent=2)
