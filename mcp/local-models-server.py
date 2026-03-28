# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]>=1.0", "httpx>=0.28"]
# ///
"""
Local Models MCP Server for Super Puppy.

Exposes Ollama and MLX models as tools for Claude Code.
Claude reasons; local models do heavy lifting.
"""

import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

logging.getLogger("httpx").setLevel(logging.WARNING)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")

mcp = FastMCP("local-models")

# ── Model Discovery ─────────────────────────────────────────────────

_models: dict = {}  # populated at startup


async def discover_models():
    """Query Ollama and MLX for available models and capabilities."""
    models = {}
    async with httpx.AsyncClient(timeout=10) as client:
        # Ollama
        try:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            for m in resp.json().get("models", []):
                name = m["name"]
                details = m.get("details", {})
                total_b = 0.0
                try:
                    total_b = float(details.get("parameter_size", "0").rstrip("B"))
                except (ValueError, AttributeError):
                    pass

                ctx, has_vision = 0, False
                expert_count, expert_used = None, None
                try:
                    show = await client.post(
                        f"{OLLAMA_URL}/api/show",
                        json={"name": name}, timeout=5,
                    )
                    mi = show.json().get("model_info", {})
                    has_vision = any("vision" in k for k in mi)
                    for k, v in mi.items():
                        if "context_length" in k:
                            ctx = int(v)
                        elif k.endswith(".expert_count"):
                            expert_count = int(v)
                        elif k.endswith(".expert_used_count"):
                            expert_used = int(v)
                except Exception:
                    pass

                active_b = total_b
                if expert_count and expert_used and expert_count > 1:
                    active_b = round(total_b * expert_used / expert_count, 1)

                models[name] = {
                    "backend": "ollama",
                    "total_params_b": total_b,
                    "active_params_b": active_b,
                    "context": ctx,
                    "vision": has_vision,
                }
                base = name.split(":")[0]
                if base not in models or total_b > models[base]["total_params_b"]:
                    models[base] = models[name]
        except Exception as e:
            print(f"Ollama discovery failed: {e}", file=sys.stderr, flush=True)

        # MLX
        try:
            resp = await client.get(f"{MLX_URL}/v1/models")
            for m in resp.json().get("data", []):
                mid = m["id"]
                models[mid] = {
                    "backend": "mlx",
                    "total_params_b": 0,
                    "active_params_b": 0,
                    "context": 0,
                    "vision": False,
                }
        except Exception as e:
            print(f"MLX discovery failed: {e}", file=sys.stderr, flush=True)

    return models


def pick_model(task: str, override: str | None = None) -> tuple[str, str]:
    """Pick best model for a task. Returns (model_name, backend)."""
    if override and override in _models:
        return override, _models[override]["backend"]

    preferences = {
        "code": ["qwen3-coder", "qwen2.5-coder:32b", "qwen2.5-coder", "qwen-coder", "qwen3.5"],
        "general": ["qwen3.5", "qwen3.5-fast", "qwen3.5-large"],
        "reasoning": ["qwen3.5-large", "nemotron-super", "DeepSeek-R1", "qwen3.5"],
        "vision": ["qwen3-vl"],
        "long_context": ["qwen3.5", "qwen3.5-large"],
    }
    for name in preferences.get(task, preferences["general"]):
        if name in _models:
            return name, _models[name]["backend"]

    # Last resort: pick anything
    for name, info in _models.items():
        if info["backend"] in ("ollama", "mlx"):
            return name, info["backend"]

    raise ValueError("No local models available")


async def chat_ollama(model: str, messages: list[dict], max_tokens: int = 4096) -> str:
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"num_predict": max_tokens}},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def chat_mlx(model: str, messages: list[dict], max_tokens: int = 4096) -> str:
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{MLX_URL}/v1/chat/completions",
            json={"model": model, "messages": messages, "max_tokens": max_tokens,
                  "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def chat(model: str, backend: str, messages: list[dict],
               max_tokens: int = 4096) -> str:
    if backend == "ollama":
        return await chat_ollama(model, messages, max_tokens)
    return await chat_mlx(model, messages, max_tokens)


# ── Tools ────────────────────────────────────────────────────────────

@mcp.tool()
async def local_models_status() -> str:
    """Show available local models and their capabilities.

    Returns the current state of Ollama and MLX backends: which models
    are available, their parameter counts, context lengths, vision
    capability, and backend type.
    """
    global _models
    _models = await discover_models()

    ollama = {k: v for k, v in _models.items() if v["backend"] == "ollama"}
    mlx = {k: v for k, v in _models.items() if v["backend"] == "mlx"}

    lines = [f"Ollama ({OLLAMA_URL}): {len(ollama)} models",
             f"MLX ({MLX_URL}): {len(mlx)} models", ""]

    for name, info in sorted(_models.items(), key=lambda x: -x[1]["total_params_b"]):
        parts = [f"  {name}"]
        if info["total_params_b"]:
            if info["active_params_b"] != info["total_params_b"]:
                parts.append(f"{info['total_params_b']}B ({info['active_params_b']}B active)")
            else:
                parts.append(f"{info['total_params_b']}B")
        if info["context"]:
            parts.append(f"{info['context']//1024}K ctx")
        if info["vision"]:
            parts.append("vision")
        parts.append(f"[{info['backend']}]")
        lines.append(" | ".join(parts))

    return "\n".join(lines)


@mcp.tool()
async def local_generate(
    prompt: str,
    context_files: list[str] | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Generate code or text using a local model.

    Use for bulk generation, boilerplate, file transformations, and migrations.
    Automatically selects the best model (code-specialist for code, general for text).

    Args:
        prompt: What to generate.
        context_files: Optional list of absolute file paths to include as context.
        model: Optional model name override. Use local_models_status to see available models.
        system_prompt: Optional system prompt.
        max_tokens: Maximum tokens to generate (default 4096).
    """
    is_code = any(w in prompt.lower() for w in [
        "code", "function", "class", "implement", "refactor", "convert",
        "generate", "write", "create", "test", "fix",
    ])
    task = "code" if is_code else "general"
    model_name, backend = pick_model(task, model)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = prompt
    if context_files:
        file_contents = []
        for fp in context_files:
            try:
                text = Path(fp).read_text(errors="replace")
                file_contents.append(f"--- {fp} ---\n{text}")
            except Exception as e:
                file_contents.append(f"--- {fp} --- (error: {e})")
        user_content = "\n\n".join(file_contents) + "\n\n" + prompt

    messages.append({"role": "user", "content": user_content})

    result = await chat(model_name, backend, messages, max_tokens)
    return f"[{model_name} via {backend}]\n\n{result}"


@mcp.tool()
async def local_review(
    code: str | None = None,
    file_paths: list[str] | None = None,
    focus: str | None = None,
    model: str | None = None,
) -> str:
    """Get a second opinion on code from a local model.

    Use when you want a different model's perspective on code quality,
    security, performance, or correctness. Different training data
    surfaces different issues.

    Args:
        code: Inline code to review.
        file_paths: Absolute paths to files to review (alternative to code).
        focus: Optional focus area: "security", "performance", "correctness", "style".
        model: Optional model name override.
    """
    model_name, backend = pick_model("reasoning", model)

    review_content = code or ""
    if file_paths:
        parts = []
        for fp in file_paths:
            try:
                text = Path(fp).read_text(errors="replace")
                parts.append(f"--- {fp} ---\n{text}")
            except Exception as e:
                parts.append(f"--- {fp} --- (error: {e})")
        review_content = "\n\n".join(parts)

    if not review_content.strip():
        return "Error: provide either code or file_paths to review."

    focus_instruction = ""
    if focus:
        focus_instruction = f" Focus specifically on {focus} issues."

    messages = [
        {"role": "system",
         "content": f"You are a thorough code reviewer.{focus_instruction} "
                    "Be specific, cite line numbers, and prioritize actionable findings."},
        {"role": "user",
         "content": f"Review this code:\n\n{review_content}"},
    ]

    result = await chat(model_name, backend, messages, 4096)
    return f"[{model_name} via {backend}]\n\n{result}"


@mcp.tool()
async def local_vision(
    image_paths: list[str],
    prompt: str = "Describe what you see in this image.",
) -> str:
    """Analyze images from disk using a local vision model.

    Claude cannot see local files without attachment. This tool reads
    images directly from the filesystem and analyzes them with a local
    vision model (qwen3-vl via Ollama).

    Args:
        image_paths: List of absolute paths to image files (PNG, JPG, etc).
        prompt: What to analyze about the image(s).
    """
    vision_models = [n for n, i in _models.items() if i.get("vision")]
    if not vision_models:
        return "Error: no vision-capable model available. Need qwen3-vl in Ollama."

    model_name = vision_models[0]

    images = []
    for ip in image_paths:
        try:
            data = Path(ip).read_bytes()
            images.append(base64.b64encode(data).decode())
        except Exception as e:
            return f"Error reading {ip}: {e}"

    # Ollama native multimodal API
    messages = [{"role": "user", "content": prompt, "images": images}]

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model_name, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        result = resp.json()["message"]["content"]

    return f"[{model_name} via ollama]\n\n{result}"


@mcp.tool()
async def local_transcribe(
    audio_path: str,
    language: str | None = None,
) -> str:
    """Transcribe audio to text using local Whisper v3.

    Args:
        audio_path: Absolute path to an audio file (mp3, wav, m4a, etc).
        language: Optional language hint (e.g. "en", "es", "ja").
    """
    if "whisper-v3" not in _models:
        return "Error: whisper-v3 not available on MLX server."

    try:
        audio_data = Path(audio_path).read_bytes()
    except Exception as e:
        return f"Error reading {audio_path}: {e}"

    suffix = Path(audio_path).suffix.lstrip(".")
    content_types = {"mp3": "audio/mpeg", "wav": "audio/wav",
                     "m4a": "audio/mp4", "ogg": "audio/ogg",
                     "flac": "audio/flac"}
    ct = content_types.get(suffix, "application/octet-stream")

    async with httpx.AsyncClient(timeout=300) as client:
        files = {"file": (Path(audio_path).name, audio_data, ct)}
        data = {"model": "whisper-v3"}
        if language:
            data["language"] = language

        resp = await client.post(
            f"{MLX_URL}/v1/audio/transcriptions",
            files=files, data=data,
        )
        resp.raise_for_status()
        result = resp.json().get("text", resp.text)

    return f"[whisper-v3 via mlx]\n\n{result}"


@mcp.tool()
async def local_candidates(
    prompt: str,
    models: list[str] | None = None,
    system_prompt: str | None = None,
    max_models: int = 3,
) -> str:
    """Run the same prompt against multiple local models in parallel.

    Use when you're uncertain between approaches and want diverse
    perspectives. Different model architectures catch different things.

    Args:
        prompt: The question or task.
        models: Optional list of specific model names. Defaults to a diverse set.
        system_prompt: Optional system prompt applied to all models.
        max_models: Maximum number of models to query (default 3).
    """
    if models:
        selected = [(m, _models[m]["backend"]) for m in models if m in _models]
    else:
        # Pick a diverse set across backends
        diverse = ["qwen3.5", "nemotron-super", "qwen-coder", "DeepSeek-R1",
                    "qwen3.5-large", "qwen3.5-fast"]
        selected = [(m, _models[m]["backend"]) for m in diverse if m in _models]

    selected = selected[:max_models]
    if not selected:
        return "Error: no models available for candidates."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    async def query_one(model_name: str, backend: str) -> str:
        try:
            result = await chat(model_name, backend, messages, 4096)
            return f"### {model_name} [{backend}]\n\n{result}"
        except Exception as e:
            return f"### {model_name} [{backend}]\n\nError: {e}"

    results = await asyncio.gather(
        *[query_one(m, b) for m, b in selected]
    )

    header = f"Queried {len(selected)} models: {', '.join(m for m, _ in selected)}"
    return header + "\n\n" + "\n\n---\n\n".join(results)


@mcp.tool()
async def local_summarize(
    file_paths: list[str],
    prompt: str = "Summarize this content concisely, focusing on the key points.",
    model: str | None = None,
    max_tokens: int = 4096,
) -> str:
    """Summarize large files using a local model with long context.

    Use instead of feeding huge files directly to Claude. The local model
    condenses them first so Claude gets a concise overview.

    Args:
        file_paths: Absolute paths to files to summarize.
        prompt: Focus for the summary (e.g. "summarize the architecture",
                "find all TODO items", "list all API endpoints").
        model: Optional model override. Defaults to the longest-context model.
        max_tokens: Maximum tokens for the summary (default 4096).
    """
    model_name, backend = pick_model("long_context", model)

    contents = []
    for fp in file_paths:
        try:
            text = Path(fp).read_text(errors="replace")
            contents.append(f"--- {fp} ---\n{text}")
        except Exception as e:
            contents.append(f"--- {fp} --- (error: {e})")

    full_text = "\n\n".join(contents)

    messages = [
        {"role": "system",
         "content": "You summarize content precisely and concisely."},
        {"role": "user",
         "content": f"{prompt}\n\n{full_text}"},
    ]

    result = await chat(model_name, backend, messages, max_tokens)
    return f"[{model_name} via {backend}]\n\n{result}"


# ── Main ─────────────────────────────────────────────────────────────

async def _startup():
    global _models
    _models = await discover_models()
    ollama_count = sum(1 for v in _models.values() if v["backend"] == "ollama")
    mlx_count = sum(1 for v in _models.values() if v["backend"] == "mlx")
    print(f"local-models MCP: {ollama_count} Ollama + {mlx_count} MLX models",
          file=sys.stderr, flush=True)


def main():
    asyncio.run(_startup())
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
