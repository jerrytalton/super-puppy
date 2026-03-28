# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]>=1.0", "httpx>=0.28", "sentence-transformers>=3.0", "torch"]
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
import uuid
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

logging.getLogger("httpx").setLevel(logging.WARNING)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
MCP_PREFS_FILE = Path("~/.config/local-models/mcp_preferences.json").expanduser()

mcp = FastMCP("local-models")

_KNOWN_ACTIVE = {
    "nemotron_h_moe": {124: 12},
    "deepseek2": {671: 37},
}

# ── Model Discovery ─────────────────────────────────────────────────

_models: dict = {}  # populated at startup

# Background job store for async dispatch/collect pattern
_jobs: dict[str, dict] = {}  # job_id -> {task, status, result, model, created}


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
                    family = show.json().get("details", {}).get("family", "")
                    has_vision = any("vision" in k for k in mi)
                    expert_ffn, embed_len, block_count = 0, 0, 0
                    for k, v in mi.items():
                        if "context_length" in k:
                            ctx = int(v)
                        elif k.endswith(".expert_count"):
                            expert_count = int(v)
                        elif k.endswith(".expert_used_count"):
                            expert_used = int(v)
                        elif k.endswith(".expert_feed_forward_length"):
                            expert_ffn = int(v)
                        elif k.endswith(".embedding_length") and ".vision." not in k:
                            embed_len = int(v)
                        elif k.endswith(".block_count") and ".vision." not in k:
                            block_count = int(v)
                except Exception:
                    pass

                active_b = total_b
                if expert_count and expert_used and expert_count > 1:
                    import re as _re
                    total_raw = int(total_b * 1e9)
                    # Strategy 1: parse AXB from name
                    m = _re.search(r'[_-]A(\d+(?:\.\d+)?)B', name, _re.IGNORECASE)
                    if m:
                        active_b = float(m.group(1))
                    # Strategy 2: known hybrid lookup
                    elif family in _KNOWN_ACTIVE:
                        known = _KNOWN_ACTIVE[family]
                        if round(total_b) in known:
                            active_b = known[round(total_b)]
                    # Strategy 3: FFN subtraction
                    elif expert_ffn and embed_len and block_count:
                        total_moe = block_count * expert_count * expert_ffn * embed_len * 3
                        active_moe = block_count * expert_used * expert_ffn * embed_len * 3
                        computed = total_raw - total_moe + active_moe
                        if 0 < computed < total_raw:
                            active_b = round(computed / 1e9)
                        else:
                            active_b = round(total_b * expert_used / expert_count)
                    else:
                        active_b = round(total_b * expert_used / expert_count)
                active_b = round(active_b)
                total_b = round(total_b)

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


def load_mcp_prefs() -> dict[str, str | list[str]]:
    """Load task→model preferences from config file."""
    if MCP_PREFS_FILE.exists():
        try:
            return json.loads(MCP_PREFS_FILE.read_text())
        except Exception:
            pass
    return {}


def _resolve_model(name: str) -> tuple[str, str] | None:
    """Try exact match, then prefix match (e.g. 'qwen3-vl' → 'qwen3-vl:235b')."""
    if name in _models:
        return name, _models[name]["backend"]
    for full_name in _models:
        if full_name.startswith(name + ":"):
            return full_name, _models[full_name]["backend"]
    return None


def pick_model(task: str, override: str | None = None) -> tuple[str, str]:
    """Pick best model for a task. Returns (model_name, backend).

    Resolution order:
      1. Explicit override from the tool call
      2. Preferences from mcp_preferences.json (list or single string)
      3. 'general' task preferences as last-ditch fallback
      4. Any available model
    """
    if override:
        result = _resolve_model(override)
        if result:
            return result

    prefs = load_mcp_prefs()
    for key in (task, "general"):
        candidates = prefs.get(key, [])
        if isinstance(candidates, str):
            candidates = [candidates]
        for pref in candidates:
            result = _resolve_model(pref)
            if result:
                return result
        if key == task and candidates:
            break  # task had preferences but none matched; try general

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
    model: str | None = None,
) -> str:
    """Analyze images from disk using a local vision model.

    Claude cannot see local files without attachment. This tool reads
    images directly from the filesystem and analyzes them with a local
    vision model (qwen3-vl via Ollama).

    Args:
        image_paths: List of absolute paths to image files (PNG, JPG, etc).
        prompt: What to analyze about the image(s).
        model: Optional model name override. Must be a vision-capable model.
    """
    model_name, backend = pick_model("vision", model)
    if not _models.get(model_name, {}).get("vision"):
        return f"Error: {model_name} is not vision-capable. Need a model like qwen3-vl."

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
async def local_image(
    prompt: str,
    output_path: str | None = None,
    model: str | None = None,
) -> str:
    """Generate an image using a local model (Flux, Z-Image, etc).

    Creates images locally on the M3 Ultra. The generated image is saved
    to disk and the path is returned.

    Args:
        prompt: Description of the image to generate.
        output_path: Where to save the image. Defaults to /tmp/local_image_<timestamp>.png.
        model: Optional model override. Defaults to best available image model.
    """
    try:
        selected, _ = pick_model("image_gen", model)
    except ValueError:
        return "Error: no image generation model available. Need flux2 or similar in Ollama."

    if not output_path:
        import time as _time
        output_path = f"/tmp/local_image_{int(_time.time())}.png"

    print(f"  → generate image {selected}: {prompt[:50]}", file=sys.stderr, flush=True)

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": selected, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        image_b64 = resp.json().get("image", "")

    if not image_b64:
        return f"Error: {selected} did not return an image."

    image_data = base64.b64decode(image_b64)
    Path(output_path).write_bytes(image_data)

    return f"[{selected} via ollama]\n\nImage saved to {output_path} ({len(image_data)} bytes)"


@mcp.tool()
async def local_transcribe(
    audio_path: str,
    language: str | None = None,
    model: str | None = None,
) -> str:
    """Transcribe audio to text using local Whisper v3.

    Args:
        audio_path: Absolute path to an audio file (mp3, wav, m4a, etc).
        language: Optional language hint (e.g. "en", "es", "ja").
        model: Optional model name override. Must be a whisper model.
    """
    try:
        whisper_model, _ = pick_model("transcription", model)
    except ValueError:
        return "Error: no whisper model available on MLX server."

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
        data = {"model": whisper_model}
        if language:
            data["language"] = language

        resp = await client.post(
            f"{MLX_URL}/v1/audio/transcriptions",
            files=files, data=data,
        )
        resp.raise_for_status()
        result = resp.json().get("text", resp.text)

    return f"[{whisper_model} via mlx]\n\n{result}"


@mcp.tool()
async def local_translate(
    text: str,
    target_language: str,
    source_language: str | None = None,
    file_paths: list[str] | None = None,
    model: str | None = None,
) -> str:
    """Translate text between languages using a local multilingual model.

    Uses models like Cogito (30+ languages) or Qwen3.5 for translation.
    Handles inline text or entire files.

    Args:
        text: Text to translate (or instructions if using file_paths).
        target_language: Target language (e.g. "Spanish", "Japanese", "zh-CN").
        source_language: Optional source language. Auto-detected if omitted.
        file_paths: Optional list of file paths to translate instead of inline text.
        model: Optional model override.
    """
    model_name, backend = pick_model("translation", model)

    content = text
    if file_paths:
        parts = []
        for fp in file_paths:
            try:
                parts.append(f"--- {fp} ---\n{Path(fp).read_text(errors='replace')}")
            except Exception as e:
                parts.append(f"--- {fp} --- (error: {e})")
        content = "\n\n".join(parts)

    source_hint = f" from {source_language}" if source_language else ""
    messages = [
        {"role": "system",
         "content": f"You are a professional translator. Translate{source_hint} "
                    f"to {target_language}. Preserve formatting, code blocks, and "
                    "structure. Output only the translation."},
        {"role": "user", "content": content},
    ]

    result = await chat(model_name, backend, messages, 8192)
    return f"[{model_name} via {backend}]\n\n{result}"


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


# ── Embeddings ───────────────────────────────────────────────────────

# HuggingFace embedding models (loaded lazily)
_hf_embed_models: dict = {}

HF_EMBED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-small-v2": "intfloat/e5-small-v2",
}

# Ollama embedding models (discovered at startup)
OLLAMA_EMBED_NAMES = ["mxbai-embed-large", "nomic-embed-text",
                      "snowflake-arctic-embed", "all-minilm"]


def _get_hf_model(name: str):
    """Lazy-load a HuggingFace embedding model."""
    if name not in _hf_embed_models:
        from sentence_transformers import SentenceTransformer
        _hf_embed_models[name] = SentenceTransformer(HF_EMBED_MODELS[name])
    return _hf_embed_models[name]


async def embed_ollama(model: str, texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


def embed_hf(model_name: str, texts: list[str]) -> list[list[float]]:
    model = _get_hf_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


@mcp.tool()
async def local_embed(
    texts: list[str],
    model: str | None = None,
    file_paths: list[str] | None = None,
) -> str:
    """Generate embeddings for text or files using a local embedding model.

    Use for semantic code search, finding similar files, clustering, or RAG.
    Returns cosine-similarity-ready normalized vectors.

    Available models: bge-m3 (best quality, multilingual), mxbai-embed-large,
    nomic-embed-text, snowflake-arctic-embed, all-minilm (fastest), e5-small-v2.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name. Default: mxbai-embed-large (Ollama) or bge-m3 (HF).
        file_paths: Optional file paths — contents are read and embedded.
    """
    if file_paths:
        for fp in file_paths:
            try:
                texts.append(Path(fp).read_text(errors="replace"))
            except Exception as e:
                texts.append(f"Error reading {fp}: {e}")

    if not texts:
        return "Error: provide texts or file_paths to embed."

    # Pick model
    chosen = model
    if not chosen:
        # Prefer Ollama models (faster startup), fall back to HF
        for name in OLLAMA_EMBED_NAMES:
            if name in _models:
                chosen = name
                break
        if not chosen:
            chosen = "bge-m3"

    # Route to appropriate backend
    if chosen in HF_EMBED_MODELS:
        import asyncio
        embeddings = await asyncio.to_thread(embed_hf, chosen, texts)
        dim = len(embeddings[0]) if embeddings else 0
        return (
            f"[{chosen} via huggingface · {len(embeddings)} embeddings · {dim}d]\n\n"
            + json.dumps(embeddings)
        )
    else:
        embeddings = await embed_ollama(chosen, texts)
        dim = len(embeddings[0]) if embeddings else 0
        return (
            f"[{chosen} via ollama · {len(embeddings)} embeddings · {dim}d]\n\n"
            + json.dumps(embeddings)
        )


@mcp.tool()
async def local_similarity_search(
    query: str,
    file_paths: list[str],
    model: str | None = None,
    top_k: int = 5,
) -> str:
    """Find files most semantically similar to a query.

    Embeds the query and all files, then ranks by cosine similarity.
    Use for "which files relate to X?" questions.

    Args:
        query: The search query (e.g. "authentication logic", "database connection handling").
        file_paths: List of absolute file paths to search through.
        model: Embedding model name. Default: best available.
        top_k: Number of top results to return (default 5).
    """
    if not file_paths:
        return "Error: provide file_paths to search."

    # Read files
    file_texts = []
    valid_paths = []
    for fp in file_paths:
        try:
            text = Path(fp).read_text(errors="replace")
            file_texts.append(text[:8000])  # truncate for embedding context
            valid_paths.append(fp)
        except Exception:
            pass

    if not file_texts:
        return "Error: no readable files."

    all_texts = [query] + file_texts

    # Pick model
    chosen = model
    if not chosen:
        for name in OLLAMA_EMBED_NAMES:
            if name in _models:
                chosen = name
                break
        if not chosen:
            chosen = "bge-m3"

    # Embed
    if chosen in HF_EMBED_MODELS:
        import asyncio
        embeddings = await asyncio.to_thread(embed_hf, chosen, all_texts)
    else:
        embeddings = await embed_ollama(chosen, all_texts)

    # Cosine similarity (embeddings are normalized)
    query_emb = embeddings[0]
    results = []
    for i, (fp, emb) in enumerate(zip(valid_paths, embeddings[1:])):
        sim = sum(a * b for a, b in zip(query_emb, emb))
        results.append((sim, fp))

    results.sort(reverse=True)
    top = results[:top_k]

    lines = [f"[{chosen} · {len(valid_paths)} files searched]\n"]
    for sim, fp in top:
        lines.append(f"  {sim:.3f}  {fp}")

    return "\n".join(lines)


# ── Background dispatch/collect ──────────────────────────────────────

@mcp.tool()
async def local_dispatch(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    context_files: list[str] | None = None,
    task_type: str = "reasoning",
) -> str:
    """Start a local model working on a task in the background. Returns immediately.

    Use this to get a parallel second opinion while you continue thinking.
    Call local_collect with the returned job_id when you're ready for the result.

    This is true parallelism: the local model runs on GPU while you run on
    Anthropic's servers. Use it for architecture review, code review,
    alternative approaches — anything where a second perspective helps.

    Args:
        prompt: The task or question for the local model.
        system_prompt: Optional system prompt.
        model: Optional model override. Defaults to best reasoning model.
        context_files: Optional file paths to include as context.
        task_type: Task type for model selection (reasoning, code, general). Default: reasoning.
    """
    model_name, backend = pick_model(task_type, model)
    job_id = uuid.uuid4().hex[:8]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = prompt
    if context_files:
        file_parts = []
        for fp in context_files:
            try:
                file_parts.append(f"--- {fp} ---\n{Path(fp).read_text(errors='replace')}")
            except Exception as e:
                file_parts.append(f"--- {fp} --- (error: {e})")
        user_content = "\n\n".join(file_parts) + "\n\n" + prompt

    messages.append({"role": "user", "content": user_content})

    _jobs[job_id] = {
        "status": "running",
        "model": model_name,
        "backend": backend,
        "result": None,
        "created": asyncio.get_event_loop().time(),
    }

    async def _run():
        try:
            result = await chat(model_name, backend, messages, 8192)
            _jobs[job_id]["result"] = f"[{model_name} via {backend}]\n\n{result}"
            _jobs[job_id]["status"] = "done"
        except Exception as e:
            _jobs[job_id]["result"] = f"Error: {e}"
            _jobs[job_id]["status"] = "failed"

    asyncio.create_task(_run())

    return (
        f"Job dispatched: {job_id}\n"
        f"Model: {model_name} ({backend})\n"
        f"Continue your work, then call local_collect('{job_id}') for the result."
    )


@mcp.tool()
async def local_collect(job_id: str) -> str:
    """Collect the result of a background job started with local_dispatch.

    Args:
        job_id: The job ID returned by local_dispatch.
    """
    if job_id not in _jobs:
        available = [f"{jid} ({j['status']})" for jid, j in _jobs.items()]
        return f"Unknown job ID: {job_id}\nActive jobs: {', '.join(available) or 'none'}"

    job = _jobs[job_id]

    if job["status"] == "running":
        elapsed = asyncio.get_event_loop().time() - job["created"]
        return (
            f"Job {job_id} still running ({elapsed:.0f}s elapsed).\n"
            f"Model: {job['model']} ({job['backend']})\n"
            f"Call local_collect('{job_id}') again shortly."
        )

    result = job["result"]
    del _jobs[job_id]
    return result


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
