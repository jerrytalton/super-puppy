# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]==1.26.0", "httpx==0.28.1", "sentence-transformers==5.3.0", "torch==2.11.0", "pyyaml==6.0.3", "mlx-audio[tts] @ git+https://github.com/Blaizzy/mlx-audio.git@e42e1431fcf89af313375296c46d03a0153c4aa7", "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git@9ab4826d20e39286af13a26615c33b403d48be72", "mlx-video-with-audio==0.1.33"]
# ///
"""
Local Models MCP Server for Super Puppy.

Exposes Ollama and MLX models as tools for Claude Code.
Claude reasons; local models do heavy lifting.
"""

import anyio
import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib import activity
from lib.models import MCP_PREFS_FILE, MLX_SERVER_CONFIG, NETWORK_CONF, active_params_b
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
MCP_HOST = os.environ.get("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("MCP_PORT", "8100"))

# --- Bearer token auth (token comes from 1Password via wrapper) ---
MCP_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "")
if not MCP_AUTH_TOKEN:
    print("local-models MCP: ERROR: MCP_AUTH_TOKEN not set. Refusing to start without auth.",
          file=sys.stderr, flush=True)
    sys.exit(1)

# Allow Tailscale FQDN in Host header (tailscale serve proxies with original Host)
_EXTRA_HOSTS = os.environ.get("MCP_ALLOWED_HOSTS", "")
_allowed_hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
if _EXTRA_HOSTS:
    _allowed_hosts.extend(h.strip() for h in _EXTRA_HOSTS.split(",") if h.strip())

from mcp.server.transport_security import TransportSecuritySettings
_transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=_allowed_hosts,
)
mcp = FastMCP("local-models", host=MCP_HOST, port=MCP_PORT,
              transport_security=_transport_security)


# ── Path validation ─────────────────────────────────────────────────
# All tools that accept file paths must validate them to prevent
# path traversal attacks (reading /etc/passwd, ~/.ssh/id_rsa, etc.).

def _load_allowed_roots() -> tuple[Path, ...]:
    """Load allowed path roots from MCP_ALLOWED_PATHS in network.conf, or use defaults."""
    defaults = (Path.home(), Path("/tmp"), Path("/private/tmp"))
    if not NETWORK_CONF.exists():
        return defaults
    for line in NETWORK_CONF.read_text().splitlines():
        line = line.strip()
        if line.startswith("MCP_ALLOWED_PATHS="):
            val = line.partition("=")[2].strip().strip('"').strip("'")
            if val:
                roots = [Path(p).resolve() for p in val.split(":") if p]
                roots.extend([Path("/tmp"), Path("/private/tmp")])
                return tuple(roots)
    return defaults


_ALLOWED_ROOTS: tuple[Path, ...] = _load_allowed_roots()


def _validate_path(path: str, must_exist: bool = True) -> str | None:
    """Validate a file path is under $HOME or /tmp.

    Returns None if safe, or an error message string if not.
    """
    try:
        resolved = Path(path).resolve()
    except (ValueError, OSError) as e:
        return f"Invalid path '{path}': {e}"
    if not any(resolved == root or resolved.is_relative_to(root) for root in _ALLOWED_ROOTS):
        return f"Path not allowed (must be under $HOME or /tmp): {path}"
    if must_exist and not resolved.exists():
        return f"File not found: {path}"
    return None


def _validate_paths(paths: list[str], must_exist: bool = True) -> str | None:
    """Validate a list of file paths. Returns first error or None."""
    for p in paths:
        err = _validate_path(p, must_exist=must_exist)
        if err:
            return f"Error: {err}"
    return None


_AUTH_EXEMPT_PATHS = {"/gpu", "/activity", "/api/mcp-models"}
_MAX_SESSIONS = 1000
_authenticated_sessions: OrderedDict[str, None] = OrderedDict()
_session_lock = threading.Lock()


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not MCP_AUTH_TOKEN:
            return await call_next(request)
        path = request.url.path
        if path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        if ".well-known" in path or path == "/register":
            return await call_next(request)
        # Session-bound requests: validate against authenticated session set
        session_id = request.query_params.get("session_id")
        if path.startswith("/messages") and session_id:
            with _session_lock:
                if session_id in _authenticated_sessions:
                    return await call_next(request)
            return JSONResponse({"error": "unauthorized"}, status_code=403)
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {MCP_AUTH_TOKEN}":
            # Track session ID from authenticated /mcp init requests
            if path == "/mcp" and session_id:
                with _session_lock:
                    if len(_authenticated_sessions) >= _MAX_SESSIONS and _authenticated_sessions:
                        _authenticated_sessions.popitem(last=False)
                    _authenticated_sessions[session_id] = None
            response = await call_next(request)
            # Also capture session IDs from response headers (MCP protocol)
            new_sid = response.headers.get("mcp-session-id")
            if new_sid:
                with _session_lock:
                    _authenticated_sessions[new_sid] = None
            return response
        return JSONResponse({"error": "unauthorized"}, status_code=403)

# ── GPU activity tracking ────────────────────────────────────────────
# Tracks concurrent requests per backend so tools can warn about contention.
# Also maintains a ring buffer of completed requests for the activity dashboard.

_gpu_active: dict[str, int] = {"ollama": 0, "mlx": 0}
_gpu_active_details: dict[str, list[dict]] = {"ollama": [], "mlx": []}
_gpu_lock = threading.Lock()
_REQUEST_HISTORY_MAX = 200
_request_history: list[dict] = []
_request_stats: dict[str, int] = {}  # tool:count
_server_start_time = time.time()


class _gpu_request:
    """Context manager to track active GPU requests per backend."""

    def __init__(self, backend: str, description: str):
        self.backend = backend
        self.description = description
        self.started = 0.0

    def __enter__(self):
        self.started = time.time()
        entry = {"description": self.description, "started": self.started}
        with _gpu_lock:
            _gpu_active[self.backend] += 1
            _gpu_active_details[self.backend].append(entry)
        return self

    def __exit__(self, exc_type, exc_val, _tb):
        completed_at = time.time()
        elapsed_ms = int((completed_at - self.started) * 1000)
        status = "error" if exc_type else "ok"
        tool = self.description.split(":")[0]
        model = ":".join(self.description.split(":")[1:])
        with _gpu_lock:
            _gpu_active[self.backend] -= 1
            _gpu_active_details[self.backend] = [
                e for e in _gpu_active_details[self.backend]
                if e["description"] != self.description
            ]
            _request_history.append({
                "description": self.description,
                "backend": self.backend,
                "duration_ms": elapsed_ms,
                "completed_at": completed_at,
                "status": status,
            })
            if len(_request_history) > _REQUEST_HISTORY_MAX:
                _request_history.pop(0)
            _request_stats[tool] = _request_stats.get(tool, 0) + 1
        activity.log_request(
            tool=tool, model=model, backend=self.backend, source="mcp",
            status=status, duration_ms=elapsed_ms,
            started_at=self.started, completed_at=completed_at,
            error_msg=str(exc_val) if exc_val else None,
        )


def _gpu_contention_warning(backend: str) -> str:
    """Return a warning string if there are other active requests on the backend."""
    with _gpu_lock:
        others = _gpu_active[backend] - 1
        if others <= 0:
            return ""
        descs = [e["description"] for e in _gpu_active_details[backend]]
        queue_info = f" ({', '.join(descs[:3])})" if descs else ""
        return (f"⚠ GPU contention: {others} other request{'s' if others > 1 else ''} "
                f"active on {backend}{queue_info}. Response may be slow.\n\n")


# ── Model Discovery ─────────────────────────────────────────────────

_models: dict = {}  # populated at startup

# Background job store for async dispatch/collect pattern
_jobs: dict[str, dict] = {}  # job_id -> {task, status, result, model, created}
_JOB_TTL = 3600  # expire uncollected jobs after 1 hour


async def discover_models():
    """Query Ollama and MLX for available models and capabilities."""
    models = {}
    async with httpx.AsyncClient(timeout=10) as client:
        # Ollama
        try:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            tag_models = resp.json().get("models", [])

            async def _show_model(m):
                name = m["name"]
                details = m.get("details", {})
                total_b = 0.0
                try:
                    total_b = float(details.get("parameter_size", "0").rstrip("B"))
                except (ValueError, AttributeError):
                    pass

                ctx, has_vision = 0, False
                expert_count, expert_used = None, None
                family = ""
                expert_ffn, embed_len, block_count = 0, 0, 0
                try:
                    show = await client.post(
                        f"{OLLAMA_URL}/api/show",
                        json={"name": name}, timeout=5,
                    )
                    mi = show.json().get("model_info", {})
                    family = show.json().get("details", {}).get("family", "")
                    has_vision = any("vision" in k for k in mi)
                    for k, v in mi.items():
                        if k.endswith(".context_length"):
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

                ab = active_params_b(
                    name, total_b, family,
                    expert_count, expert_used,
                    expert_ffn, embed_len, block_count,
                )
                return name, {
                    "backend": "ollama",
                    "total_params_b": round(total_b),
                    "active_params_b": round(ab),
                    "context": ctx,
                    "vision": has_vision,
                }

            results = await asyncio.gather(
                *[_show_model(m) for m in tag_models],
                return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    continue
                name, info = r
                models[name] = info
                base = name.split(":")[0]
                if base not in models or info["total_params_b"] > models[base]["total_params_b"]:
                    models[base] = info
        except Exception as e:
            logging.warning("Ollama discovery failed: %s", e)

        # MLX
        # Load MLX server config to map served names → HuggingFace paths
        _mlx_cfg_map = {}
        if MLX_SERVER_CONFIG.exists():
            try:
                import yaml
                _mcfg = yaml.safe_load(MLX_SERVER_CONFIG.read_text())
                for entry in _mcfg.get("models", []):
                    sn = entry.get("served_model_name", "")
                    mp = entry.get("model_path", "")
                    if sn and mp:
                        _mlx_cfg_map[sn] = mp
            except Exception:
                pass
        _hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        try:
            resp = await client.get(f"{MLX_URL}/v1/models")
            for m in resp.json().get("data", []):
                mid = m["id"]
                has_vision = False
                # Check HuggingFace config.json for vision_config
                model_path = _mlx_cfg_map.get(mid, mid)
                cache_dir = _hf_cache / f"models--{model_path.replace('/', '--')}" / "snapshots"
                if cache_dir.exists():
                    for snap in sorted(cache_dir.iterdir(), reverse=True):
                        hf_cfg = snap / "config.json"
                        if hf_cfg.exists():
                            try:
                                hf = json.loads(hf_cfg.read_text())
                                has_vision = (
                                    "vision_config" in hf
                                    or "vision_config" in hf.get("text_config", {})
                                )
                            except Exception:
                                pass
                            break
                models[mid] = {
                    "backend": "mlx",
                    "total_params_b": 0,
                    "active_params_b": 0,
                    "context": 0,
                    "vision": has_vision,
                }
        except Exception as e:
            logging.warning("MLX discovery failed: %s", e)

        # Register on-demand MLX models from config that aren't loaded yet
        for sn, mp in _mlx_cfg_map.items():
            if sn not in models:
                has_vision = False
                cache_dir = _hf_cache / f"models--{mp.replace('/', '--')}" / "snapshots"
                if cache_dir.exists():
                    for snap in sorted(cache_dir.iterdir(), reverse=True):
                        hf_cfg = snap / "config.json"
                        if hf_cfg.exists():
                            try:
                                hf = json.loads(hf_cfg.read_text())
                                has_vision = (
                                    "vision_config" in hf
                                    or "vision_config" in hf.get("text_config", {})
                                )
                            except Exception:
                                pass
                            break
                models[sn] = {
                    "backend": "mlx",
                    "total_params_b": 0,
                    "active_params_b": 0,
                    "context": 0,
                    "vision": has_vision,
                }

        # HuggingFace cache: TTS, transcription, image_edit, image_gen
        _TASK_BACKENDS = {
            "tts": "mlx-audio",
            "transcription": "mlx",
            "image_edit": "mflux",
            "image_gen": "mflux",
            "video": "mlx-video",
        }
        from lib.hf_scanner import scan_hf_cache
        for hf_model in scan_hf_cache(_TASK_BACKENDS.keys()):
            name = hf_model["name"]
            if name not in models:
                models[name] = {
                    "backend": _TASK_BACKENDS[hf_model["task"]],
                    "total_params_b": hf_model["total_params_b"],
                    "active_params_b": hf_model["total_params_b"],
                    "context": 0,
                    "vision": False,
                    "task": hf_model["task"],
                }

    return models


def load_mcp_prefs() -> dict[str, str | list[str]]:
    """Load task→model preferences from config file."""
    if MCP_PREFS_FILE.exists():
        try:
            return json.loads(MCP_PREFS_FILE.read_text())
        except Exception:
            pass
    return {}


def thinking_enabled(task: str) -> bool:
    """Check if thinking mode is enabled for a task."""
    prefs = load_mcp_prefs()
    thinking = prefs.get("thinking", {})
    return bool(thinking.get(task, True))


def _resolve_model(name: str) -> tuple[str, str] | None:
    """Try exact match, then prefix match (e.g. 'qwen3-vl' → 'qwen3-vl:235b').

    Prefers tagged names over base-name aliases so backends get a name
    they recognize (e.g. 'qwen3.5:9b' instead of 'qwen3.5').
    """
    # Exact match on a tagged name
    if name in _models and ":" in name:
        return name, _models[name]["backend"]
    # Prefix match: bare name → tagged variant
    for full_name in _models:
        if full_name == name + ":latest" or full_name.startswith(name + ":"):
            return full_name, _models[full_name]["backend"]
    # Exact match on base-name alias (still valid for HF/MLX models without tags)
    if name in _models:
        return name, _models[name]["backend"]
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

    # Fall back: models tagged with a specific task, then any LLM
    for name, info in _models.items():
        if info.get("task") == task:
            return name, info["backend"]
    for name, info in _models.items():
        if info["backend"] in ("ollama", "mlx"):
            return name, info["backend"]

    # Build actionable error message
    available = [n for n, m in _models.items() if m["backend"] in ("ollama", "mlx")]
    parts = [f"No model available for task '{task}'."]
    if override:
        parts.append(f"Requested model '{override}' not found.")
    prefs = load_mcp_prefs()
    task_prefs = prefs.get(task, [])
    if task_prefs:
        if isinstance(task_prefs, str):
            task_prefs = [task_prefs]
        parts.append(f"Tried preferences: {', '.join(task_prefs)} — none matched.")
    if available:
        shown = available[:5]
        suffix = f" (+{len(available) - 5} more)" if len(available) > 5 else ""
        parts.append(f"Available models: {', '.join(shown)}{suffix}.")
    else:
        parts.append("No models loaded — is Ollama/MLX running?")
    parts.append("Check ~/.config/local-models/mcp_preferences.json or pull a model with 'ollama pull'.")
    raise ValueError(" ".join(parts))


def _http_error_detail(e: httpx.HTTPStatusError, action: str) -> str:
    body = e.response.text[:500] if e.response.text else "(empty body)"
    return f"{action}: HTTP {e.response.status_code} from {e.request.url} — {body}"


async def chat_ollama(model: str, messages: list[dict],
                      max_tokens: int = 4096, think: bool = True) -> str:
    body = {"model": model, "messages": messages, "stream": False,
            "options": {"num_predict": max_tokens}}
    if not think:
        body["think"] = False
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(_http_error_detail(e, f"Ollama chat ({model})")) from e
    except httpx.ConnectError:
        raise RuntimeError(f"Ollama chat ({model}): cannot connect to {OLLAMA_URL} — is Ollama running?")
    except httpx.TimeoutException:
        raise RuntimeError(f"Ollama chat ({model}): request timed out after 300s")


async def chat_mlx(model: str, messages: list[dict],
                   max_tokens: int = 4096, think: bool = True) -> str:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": False}
    if not think:
        # MLX OpenAI-compatible: some models respect this
        body["temperature"] = 0.3
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{MLX_URL}/v1/chat/completions", json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(_http_error_detail(e, f"MLX chat ({model})")) from e
    except httpx.ConnectError:
        raise RuntimeError(f"MLX chat ({model}): cannot connect to {MLX_URL} — is mlx-openai-server running?")
    except httpx.TimeoutException:
        raise RuntimeError(f"MLX chat ({model}): request timed out after 300s")


async def chat(model: str, backend: str, messages: list[dict],
               max_tokens: int = 4096, think: bool = True) -> str:
    with _gpu_request(backend, f"chat:{model}"):
        warning = _gpu_contention_warning(backend)
        if backend == "ollama":
            result = await chat_ollama(model, messages, max_tokens, think)
        else:
            result = await chat_mlx(model, messages, max_tokens, think)
        return warning + result


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
        err = _validate_paths(context_files)
        if err:
            return err
        file_contents = []
        for fp in context_files:
            try:
                text = Path(fp).read_text(errors="replace")
                file_contents.append(f"--- {fp} ---\n{text}")
            except Exception as e:
                file_contents.append(f"--- {fp} --- (error: {e})")
        user_content = "\n\n".join(file_contents) + "\n\n" + prompt

    messages.append({"role": "user", "content": user_content})

    result = await chat(model_name, backend, messages, max_tokens,
                        think=thinking_enabled(task))
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
        err = _validate_paths(file_paths)
        if err:
            return err
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

    result = await chat(model_name, backend, messages, 4096,
                        think=thinking_enabled("reasoning"))
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

    err = _validate_paths(image_paths)
    if err:
        return err
    images_b64 = []
    for ip in image_paths:
        try:
            data = Path(ip).read_bytes()
            images_b64.append(base64.b64encode(data).decode())
        except Exception as e:
            return f"Error reading {ip}: {e}"

    logging.info("vision %s (%s): %s", model_name, backend, prompt[:50])

    try:
        with _gpu_request(backend, f"vision:{model_name}"):
            warning = _gpu_contention_warning(backend)
            async with httpx.AsyncClient(timeout=300) as client:
                if backend == "ollama":
                    messages = [{"role": "user", "content": prompt,
                                 "images": images_b64}]
                    body = {"model": model_name, "messages": messages,
                            "stream": False}
                    if not thinking_enabled("vision"):
                        body["think"] = False
                    resp = await client.post(f"{OLLAMA_URL}/api/chat", json=body)
                    resp.raise_for_status()
                    result = resp.json()["message"]["content"]
                else:
                    content = [{"type": "text", "text": prompt}]
                    for b64 in images_b64:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        })
                    body = {"model": model_name,
                            "messages": [{"role": "user", "content": content}],
                            "max_tokens": 4096, "stream": False}
                    resp = await client.post(
                        f"{MLX_URL}/v1/chat/completions", json=body)
                    resp.raise_for_status()
                    result = resp.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        return f"Error: {_http_error_detail(e, f'Vision ({model_name} via {backend})')}"
    except httpx.ConnectError:
        url = OLLAMA_URL if backend == "ollama" else MLX_URL
        return f"Error: Vision ({model_name}): cannot connect to {url} — is the backend running?"
    except httpx.TimeoutException:
        return f"Error: Vision ({model_name}): request timed out after 300s"

    return f"{warning}[{model_name} via {backend}]\n\n{result}"


_COMPUTER_USE_SYSTEM = """You are a GUI automation assistant. Given a screenshot and an intent, return a JSON array of actions to accomplish the intent.

Each action is one of:
- {"action": "click", "x": <int>, "y": <int>, "description": "<what you're clicking>"}
- {"action": "type", "text": "<text to type>", "description": "<where you're typing>"}
- {"action": "scroll", "direction": "up"|"down"|"left"|"right", "amount": <int>, "description": "<why>"}
- {"action": "key", "key": "<key combo e.g. cmd+s>", "description": "<why>"}
- {"action": "wait", "seconds": <float>, "description": "<why>"}

Coordinates are absolute pixel positions from the top-left of the screen.
Return ONLY the JSON array. No explanation, no markdown."""


async def _take_screenshot() -> str:
    """Take a full-screen screenshot silently. Returns the file path."""
    path = f"/tmp/sp_screenshot_{int(time.time())}.png"
    proc = await asyncio.create_subprocess_exec(
        "screencapture", "-x", path,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE)
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError("Screenshot timed out after 10s")
    if not Path(path).exists():
        raise RuntimeError("Screenshot failed — check Screen Recording permissions")
    return path


@mcp.tool()
async def local_computer_use(
    intent: str,
    screenshot_path: str | None = None,
    model: str | None = None,
) -> str:
    """Analyze a screenshot and return structured GUI actions for an intent.

    Takes a screenshot (or uses a provided one), sends it to a GUI-aware
    vision model (UI-TARS, Fara), and returns a JSON array of actions
    (click, type, scroll, key, wait) to accomplish the intent.

    The tool observes and plans — it does NOT execute the actions.
    """
    model_name, backend = pick_model("computer_use", model)

    # Take or read screenshot
    if screenshot_path:
        err = _validate_path(screenshot_path)
        if err:
            return f"Error: {err}"
        img_bytes = Path(screenshot_path).read_bytes()
    else:
        try:
            path = await _take_screenshot()
            img_bytes = Path(path).read_bytes()
            screenshot_path = path
        except RuntimeError as e:
            return f"Error: {e}"

    img_b64 = base64.b64encode(img_bytes).decode()

    with _gpu_request(backend, f"computer_use:{model_name}"):
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                if backend == "ollama":
                    messages = [
                        {"role": "system", "content": _COMPUTER_USE_SYSTEM},
                        {"role": "user", "content": intent, "images": [img_b64]},
                    ]
                    body = {"model": model_name, "messages": messages,
                            "stream": False, "think": False}
                    resp = await client.post(
                        f"{OLLAMA_URL}/api/chat", json=body)
                    resp.raise_for_status()
                    result = resp.json()["message"]["content"]
                else:
                    content = [
                        {"type": "text", "text": intent},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ]
                    body = {"model": model_name,
                            "messages": [
                                {"role": "system", "content": _COMPUTER_USE_SYSTEM},
                                {"role": "user", "content": content},
                            ],
                            "max_tokens": 4096, "stream": False}
                    resp = await client.post(
                        f"{MLX_URL}/v1/chat/completions", json=body)
                    resp.raise_for_status()
                    result = resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            return f"Error: Computer use ({model_name}): HTTP {e.response.status_code} — {e.response.text[:200]}"
        except httpx.ConnectError:
            url = OLLAMA_URL if backend == "ollama" else MLX_URL
            return f"Error: Computer use ({model_name}): cannot connect to {url}"
        except httpx.TimeoutException:
            return f"Error: Computer use ({model_name}): timed out after 300s"

    # Try to validate the response is JSON
    try:
        actions = json.loads(result)
        result = json.dumps(actions, indent=2)
    except json.JSONDecodeError:
        pass  # model returned text instead of JSON — return as-is

    warning = _gpu_contention_warning(backend)
    meta = f"[{model_name} via {backend}]"
    if screenshot_path:
        meta += f" screenshot: {screenshot_path}"
    return f"{warning}{meta}\n\n{result}"


@mcp.tool()
async def local_image(
    prompt: str,
    output_path: str | None = None,
    model: str | None = None,
) -> str:
    """Generate an image using a local model (Flux, Z-Image, etc).

    Creates images locally using a diffusion model. The generated image is saved
    to disk and the path is returned.

    Args:
        prompt: Description of the image to generate.
        output_path: Where to save the image. Defaults to /tmp/local_image_<timestamp>.png.
        model: Optional model override. Defaults to best available image model.
    """
    try:
        selected, backend = pick_model("image_gen", model)
    except ValueError:
        return "Error: no image generation model available. Need flux2 or similar."

    if not output_path:
        import time as _time
        output_path = f"/tmp/local_image_{int(_time.time())}.png"
    err = _validate_path(output_path, must_exist=False)
    if err:
        return f"Error: {err}"

    logging.info("generate image %s (%s): %s", selected, backend, prompt[:50])

    if backend == "mflux":
        with _gpu_request("mlx", f"image:{selected}"):
            warning = _gpu_contention_warning("mlx")
            loop = asyncio.get_event_loop()
            try:
                # Dev models need more steps; schnell/turbo/klein are fast
                steps = "4" if any(k in selected.lower() for k in ("schnell", "turbo", "klein")) else "20"
                proc = await loop.run_in_executor(None, lambda: subprocess.run(
                    ["mflux-generate", "--model", selected, "--prompt", prompt,
                     "--output", output_path, "--steps", steps],
                    capture_output=True, text=True, timeout=600,
                    env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
                ))
                if proc.returncode != 0:
                    return f"Error: mflux-generate failed:\n{proc.stderr[-500:]}"
            except subprocess.TimeoutExpired:
                return "Error: image generation timed out after 10 minutes."

        if not Path(output_path).exists():
            return f"Error: output image was not created at {output_path}"
        size = Path(output_path).stat().st_size
        return f"{warning}[{selected} via mflux]\n\nImage saved to {output_path} ({size} bytes)"

    # Ollama backend
    try:
        with _gpu_request("ollama", f"image:{selected}"):
            warning = _gpu_contention_warning("ollama")
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": selected, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                image_b64 = resp.json().get("image", "")
    except httpx.HTTPStatusError as e:
        return f"Error: {_http_error_detail(e, f'Image generation ({selected})')}"
    except httpx.ConnectError:
        return f"Error: Image generation ({selected}): cannot connect to {OLLAMA_URL} — is Ollama running?"
    except httpx.TimeoutException:
        return f"Error: Image generation ({selected}): request timed out after 300s"

    if not image_b64:
        return f"Error: {selected} did not return an image."

    try:
        image_data = base64.b64decode(image_b64)
    except Exception as e:
        return f"Error: Invalid image data from {selected}: {e}"
    Path(output_path).write_bytes(image_data)

    return f"{warning}[{selected} via ollama]\n\nImage saved to {output_path} ({len(image_data)} bytes)"


@mcp.tool()
async def local_image_edit(
    image_path: str,
    prompt: str,
    output_path: str | None = None,
    strength: float = 0.8,
    steps: int = 4,
    seed: int | None = None,
    model: str | None = None,
) -> str:
    """Edit an image using a text prompt with Flux Kontext (local, on Apple Silicon).

    Takes an existing image and modifies it according to the prompt.
    Good for recoloring, adding/removing elements, style transfer, etc.

    Args:
        image_path: Absolute path to the input image to edit.
        prompt: Description of the desired changes (e.g. "make the dog white with a red cape").
        output_path: Where to save the result. Defaults to /tmp/local_edit_<timestamp>.png.
        strength: How much to change the image (0.0 = no change, 1.0 = ignore input). Default 0.8.
        steps: Number of diffusion steps (more = higher quality, slower). Default 4.
        seed: Optional random seed for reproducibility.
        model: Optional model override. Defaults to best available image edit model.
    """
    try:
        selected, backend = pick_model("image_edit", model)
    except ValueError:
        return "Error: no image editing model available. Need mflux-generate-kontext installed."

    err = _validate_path(image_path)
    if err:
        return f"Error: {err}"

    if not output_path:
        import time as _time
        output_path = f"/tmp/local_edit_{int(_time.time())}.png"
    err = _validate_path(output_path, must_exist=False)
    if err:
        return f"Error: {err}"

    logging.info("edit image %s: %s", selected, prompt[:60])

    with _gpu_request(backend, f"image_edit:{selected}"):
        warning = _gpu_contention_warning(backend)

        cmd = [
            "mflux-generate-kontext",
            "--image-path", image_path,
            "--prompt", prompt,
            "--output", output_path,
            "--steps", str(steps),
            "--image-strength", str(strength),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        loop = asyncio.get_event_loop()
        try:
            proc = await loop.run_in_executor(None, lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
            ))
            if proc.returncode != 0:
                return f"Error: mflux-generate-kontext failed:\n{proc.stderr[-500:]}"
        except subprocess.TimeoutExpired:
            return "Error: image editing timed out after 10 minutes."

    if not Path(output_path).exists():
        return f"Error: output image was not created at {output_path}"

    size = Path(output_path).stat().st_size
    return f"{warning}[{selected} via {backend}]\n\nEdited image saved to {output_path} ({size} bytes)"


@mcp.tool()
async def local_video(
    prompt: str,
    image_path: str | None = None,
    output_path: str | None = None,
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
    audio_genre: str | None = None,
) -> str:
    """Generate video locally using MLX (Wan2.2, LTX-2).

    Auto-detects mode from parameters:
    - Text prompt only → text-to-video
    - image_path provided → image-to-video (animates the image)
    - audio_genre provided → video with synchronized audio

    Output is an MP4 file saved to disk.

    Args:
        prompt: Description of the video to generate.
        image_path: Optional input image for image-to-video mode.
        output_path: Where to save the video. Defaults to /tmp/local_video_<timestamp>.mp4.
        model: Optional model override. Defaults to best available video model.
        width: Output width in pixels (must be divisible by 64). Default: model's native resolution.
        height: Output height in pixels (must be divisible by 64). Default: model's native resolution.
        num_frames: Number of frames to generate. Default: 65 (~2.7s at 24fps).
        audio_genre: Music genre for audio-synced video (e.g. 'electronic', 'jazz', 'ambient').
            When provided, uses mlx-video-with-audio for synchronized audio generation.
    """
    try:
        selected, backend = pick_model("video", model)
    except ValueError:
        return ("Error: no video model available. Install mlx-video or "
                "mlx-video-with-audio and download a model (e.g. Wan2.2-T2V-14B).")

    if image_path:
        err = _validate_path(image_path)
        if err:
            return f"Error: {err}"

    if not output_path:
        import time as _time
        output_path = f"/tmp/local_video_{int(_time.time())}.mp4"
    err = _validate_path(output_path, must_exist=False)
    if err:
        return f"Error: {err}"

    mode = "audio" if audio_genre else ("i2v" if image_path else "t2v")
    logging.info("generate video %s (%s, %s): %s", selected, backend, mode, prompt[:50])

    with _gpu_request("mlx", f"video:{selected}"):
        warning = _gpu_contention_warning("mlx")
        loop = asyncio.get_event_loop()

        if mode == "audio":
            # mlx-video-with-audio: LTX-2 with synchronized audio
            cmd = [
                sys.executable, "-m", "mlx_video.generate_av",
                "--prompt", prompt,
                "--output-path", output_path,
            ]
            if width:
                cmd.extend(["--width", str(width)])
            if height:
                cmd.extend(["--height", str(height)])
            if num_frames:
                cmd.extend(["--num-frames", str(num_frames)])
            timeout = 1200
        elif "ltx" in selected.lower():
            # LTX-2 via mlx-video
            cmd = [
                sys.executable, "-m", "mlx_video.generate",
                "--prompt", prompt,
                "--output-path", output_path,
            ]
            if image_path:
                cmd.extend(["--image", image_path])
            if width:
                cmd.extend(["--width", str(width)])
            if height:
                cmd.extend(["--height", str(height)])
            if num_frames:
                cmd.extend(["--num-frames", str(num_frames)])
            timeout = 900
        else:
            # Wan2.x via mlx-video — generate_wan wants a filesystem path,
            # so resolve the HF repo id to its local snapshot directory.
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            snap_root = hf_cache / f"models--{selected.replace('/', '--')}" / "snapshots"
            snapshot = None
            if snap_root.exists():
                snaps = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                if snaps:
                    snapshot = snaps[0]
            if snapshot is None:
                return (f"Error: video model {selected} is not downloaded locally. "
                        f"Pull it with `hf download {selected}` or from the profiles page.")
            cmd = [
                sys.executable, "-m", "mlx_video.generate_wan",
                "--model-dir", str(snapshot),
                "--prompt", prompt,
                "--output-path", output_path,
            ]
            if image_path:
                cmd.extend(["--image", image_path])
            if width:
                cmd.extend(["--width", str(width)])
            if height:
                cmd.extend(["--height", str(height)])
            if num_frames:
                cmd.extend(["--num-frames", str(num_frames)])
            timeout = 900

        try:
            proc = await loop.run_in_executor(None, lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
            ))
            if proc.returncode != 0:
                return f"Error: video generation failed:\n{proc.stderr[-500:]}"
        except subprocess.TimeoutExpired:
            return f"Error: video generation timed out after {timeout // 60} minutes."

    if not Path(output_path).exists():
        return f"Error: output video was not created at {output_path}"

    size = Path(output_path).stat().st_size
    mb = size / (1024 * 1024)
    return f"{warning}[{selected} via {backend}]\n\nVideo saved to {output_path} ({mb:.1f} MB)"


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
        whisper_model, backend = pick_model("transcription", model)
    except ValueError:
        return "Error: no transcription model available."

    err = _validate_path(audio_path)
    if err:
        return f"Error: {err}"
    try:
        audio_data = Path(audio_path).read_bytes()
    except Exception as e:
        return f"Error reading {audio_path}: {e}"

    suffix = Path(audio_path).suffix.lstrip(".")
    content_types = {"mp3": "audio/mpeg", "wav": "audio/wav",
                     "m4a": "audio/mp4", "ogg": "audio/ogg",
                     "flac": "audio/flac"}
    ct = content_types.get(suffix, "application/octet-stream")

    url = MLX_URL if backend == "mlx" else OLLAMA_URL
    try:
        with _gpu_request(backend, f"transcribe:{whisper_model}"):
            warning = _gpu_contention_warning(backend)
            async with httpx.AsyncClient(timeout=300) as client:
                files = {"file": (Path(audio_path).name, audio_data, ct)}
                data = {"model": whisper_model}
                if language:
                    data["language"] = language

                resp = await client.post(
                    f"{url}/v1/audio/transcriptions",
                    files=files, data=data,
                )
                resp.raise_for_status()
                result = resp.json().get("text", resp.text)
    except httpx.HTTPStatusError as e:
        return f"Error: {_http_error_detail(e, f'Transcription ({whisper_model})')}"
    except httpx.ConnectError:
        return f"Error: Transcription ({whisper_model}): cannot connect to {url} — is the backend running?"
    except httpx.TimeoutException:
        return f"Error: Transcription ({whisper_model}): request timed out after 300s"

    return f"{warning}[{whisper_model} via {backend}]\n\n{result}"


@mcp.tool()
async def local_speak(
    text: str,
    output_path: str | None = None,
    voice: str = "casual_male",
    model: str | None = None,
    language: str = "en",
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> str:
    """Generate speech from text using a local TTS model.

    Supports Voxtral (20 preset voices, 9 languages) and Chatterbox
    (voice cloning, 23 languages). Audio is saved to disk.

    Args:
        text: Text to speak.
        output_path: Where to save audio. Defaults to /tmp/local_tts_<timestamp>.wav.
        voice: Voice preset. Voxtral voices: casual_male, casual_female,
               cheerful_female, neutral_male, neutral_female, fr_male, fr_female,
               es_male, es_female, de_male, de_female, it_male, it_female,
               pt_male, pt_female, nl_male, nl_female, ar_male, hi_male, hi_female.
               Ignored when ref_audio is provided (Chatterbox voice cloning).
        model: Optional model override. Defaults to profile-selected TTS model.
               Use "mlx-community/chatterbox-fp16" for voice cloning.
        language: Language code (e.g. "en", "fr", "es", "de"). Default "en".
        ref_audio: Path to a reference audio file for voice cloning (Chatterbox).
        ref_text: Optional transcript of the reference audio (improves cloning).
    """
    try:
        model, backend = pick_model("tts", model)
    except ValueError:
        return "Error: no TTS model available. Need Voxtral or Chatterbox downloaded."
    if not output_path:
        import time as _time
        output_path = f"/tmp/local_tts_{int(_time.time())}.wav"
    err = _validate_path(output_path, must_exist=False)
    if err:
        return f"Error: {err}"
    if ref_audio:
        err = _validate_path(ref_audio)
        if err:
            return f"Error: {err}"

    out_dir = os.path.dirname(output_path) or "/tmp"
    prefix = Path(output_path).stem
    fmt = Path(output_path).suffix.lstrip(".") or "wav"

    logging.info("TTS %s: %s", model.split("/")[-1], text[:60])

    def _generate():
        from mlx_audio.tts.generate import generate_audio
        kwargs = dict(
            text=text,
            model=model,
            voice=voice,
            lang_code=language,
            output_path=out_dir,
            file_prefix=prefix,
            audio_format=fmt,
            verbose=False,
            play=False,
        )
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
        generate_audio(**kwargs)

    with _gpu_request(backend, f"tts:{model.split('/')[-1]}"):
        warning = _gpu_contention_warning(backend)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _generate)
        except Exception as e:
            return f"Error generating speech: {e}"

    # generate_audio appends _000 to the prefix
    actual_path = os.path.join(out_dir, f"{prefix}_000.{fmt}")
    if os.path.exists(actual_path):
        final_path = output_path
        os.rename(actual_path, final_path)
    elif os.path.exists(output_path):
        final_path = output_path
    else:
        import glob
        candidates = glob.glob(os.path.join(out_dir, f"{prefix}*"))
        if candidates:
            final_path = max(candidates, key=os.path.getmtime)
        else:
            return f"Error: audio file was not created at {output_path}"

    size = Path(final_path).stat().st_size
    return f"{warning}[{model.split('/')[-1]} via {backend}]\n\nAudio saved to {final_path} ({size} bytes)"


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
        err = _validate_paths(file_paths)
        if err:
            return err
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

    result = await chat(model_name, backend, messages, 8192,
                        think=thinking_enabled("translation"))
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
            result = await chat(model_name, backend, messages, 4096,
                                think=thinking_enabled("reasoning"))
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

    err = _validate_paths(file_paths)
    if err:
        return err
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

    result = await chat(model_name, backend, messages, max_tokens,
                        think=thinking_enabled("long_context"))
    return f"[{model_name} via {backend}]\n\n{result}"


# ── Embeddings ───────────────────────────────────────────────────────

# HuggingFace embedding models (loaded lazily)
_hf_embed_models: dict = {}
_hf_embed_lock = threading.Lock()

HF_EMBED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-small-v2": "intfloat/e5-small-v2",
}

# Ollama embedding models (discovered at startup)
OLLAMA_EMBED_NAMES = ["qwen3-embedding", "mxbai-embed-large",
                      "nomic-embed-text", "snowflake-arctic-embed", "all-minilm"]


def _get_hf_model(name: str):
    """Lazy-load a HuggingFace embedding model."""
    with _hf_embed_lock:
        if name not in _hf_embed_models:
            from sentence_transformers import SentenceTransformer
            _hf_embed_models[name] = SentenceTransformer(HF_EMBED_MODELS[name])
        return _hf_embed_models[name]


async def embed_ollama(model: str, texts: list[str]) -> list[list[float]]:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]
    except httpx.HTTPStatusError as e:
        raise RuntimeError(_http_error_detail(e, f"Embedding ({model})")) from e
    except httpx.ConnectError:
        raise RuntimeError(f"Embedding ({model}): cannot connect to {OLLAMA_URL} — is Ollama running?")
    except httpx.TimeoutException:
        raise RuntimeError(f"Embedding ({model}): request timed out after 60s")


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

    Available models: qwen3-embedding (best quality, 32K context, multilingual),
    bge-m3, nomic-embed-text, snowflake-arctic-embed, all-minilm (fastest), e5-small-v2.

    Args:
        texts: List of text strings to embed.
        model: Embedding model name. Default: qwen3-embedding:8b (Ollama) or bge-m3 (HF).
        file_paths: Optional file paths — contents are read and embedded.
    """
    if file_paths:
        err = _validate_paths(file_paths)
        if err:
            return err
        for fp in file_paths:
            try:
                texts.append(Path(fp).read_text(errors="replace"))
            except Exception as e:
                texts.append(f"Error reading {fp}: {e}")

    if not texts:
        return "Error: provide texts or file_paths to embed."

    # Pick model — HF models are a separate pool, check both
    if model and model in HF_EMBED_MODELS:
        chosen, backend = model, "huggingface"
    elif model and model in _models:
        chosen, backend = model, _models[model]["backend"]
    else:
        try:
            chosen, backend = pick_model("embedding", model)
        except ValueError:
            chosen, backend = "bge-m3", "huggingface"

    if backend == "huggingface" or chosen in HF_EMBED_MODELS:
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
            f"[{chosen} via {backend} · {len(embeddings)} embeddings · {dim}d]\n\n"
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
    err = _validate_paths(file_paths)
    if err:
        return err
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

    if model and model in HF_EMBED_MODELS:
        chosen, backend = model, "huggingface"
    elif model and model in _models:
        chosen, backend = model, _models[model]["backend"]
    else:
        try:
            chosen, backend = pick_model("embedding", model)
        except ValueError:
            chosen, backend = "bge-m3", "huggingface"

    if backend == "huggingface" or chosen in HF_EMBED_MODELS:
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
        err = _validate_paths(context_files)
        if err:
            return err
        file_parts = []
        for fp in context_files:
            try:
                file_parts.append(f"--- {fp} ---\n{Path(fp).read_text(errors='replace')}")
            except Exception as e:
                file_parts.append(f"--- {fp} --- (error: {e})")
        user_content = "\n\n".join(file_parts) + "\n\n" + prompt

    messages.append({"role": "user", "content": user_content})

    # Evict stale jobs before adding new ones
    now = asyncio.get_event_loop().time()
    stale = [jid for jid, j in _jobs.items()
             if j["status"] != "running" and now - j["created"] > _JOB_TTL]
    for jid in stale:
        del _jobs[jid]

    _jobs[job_id] = {
        "status": "running",
        "model": model_name,
        "backend": backend,
        "result": None,
        "created": now,
    }

    async def _run():
        try:
            result = await chat(model_name, backend, messages, 8192,
                                think=thinking_enabled(task_type))
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
    activity.init_db()
    _models = await discover_models()
    ollama_count = sum(1 for v in _models.values() if v["backend"] == "ollama")
    mlx_count = sum(1 for v in _models.values() if v["backend"] == "mlx")
    logging.info("local-models MCP: %d Ollama + %d MLX models", ollama_count, mlx_count)


async def _gpu_status(request):
    """Lightweight endpoint for Playground to poll GPU activity."""
    now = time.time()
    with _gpu_lock:
        data = {
            "ollama": {
                "active": _gpu_active["ollama"],
                "tasks": [
                    {**e, "elapsed_ms": int((now - e["started"]) * 1000)}
                    for e in _gpu_active_details["ollama"]
                ],
            },
            "mlx": {
                "active": _gpu_active["mlx"],
                "tasks": [
                    {**e, "elapsed_ms": int((now - e["started"]) * 1000)}
                    for e in _gpu_active_details["mlx"]
                ],
            },
        }
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            await client.get(f"{MLX_URL}/v1/models")
            data["mlx"]["responsive"] = True
    except Exception:
        data["mlx"]["responsive"] = False
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            await client.get(f"{OLLAMA_URL}/api/tags")
            data["ollama"]["responsive"] = True
    except Exception:
        data["ollama"]["responsive"] = False
    return JSONResponse(data)


async def _activity_status(request):
    """Activity dashboard endpoint: current + recent requests + stats."""
    now = time.time()
    period = int(request.query_params.get("period", 86400))
    with _gpu_lock:
        active = []
        for backend in ("ollama", "mlx"):
            for e in _gpu_active_details[backend]:
                active.append({
                    "description": e["description"],
                    "backend": backend,
                    "started": e["started"],
                    "elapsed_ms": int((now - e["started"]) * 1000),
                })
    db_data = activity.query_activity(period)
    return JSONResponse({
        "active": active,
        "server_uptime_s": int(now - _server_start_time),
        **db_data,
    })


async def _mcp_models(request):
    """Return HF-cache models (TTS, image gen/edit, transcription)."""
    hf_backends = {"mlx-audio", "mflux"}
    names = [name for name, info in _models.items()
             if info.get("backend") in hf_backends]
    return JSONResponse({"models": names})




def main():
    asyncio.run(_startup())
    if MCP_AUTH_TOKEN:
        import uvicorn
        from starlette.routing import Route
        app = mcp.streamable_http_app()
        app.add_middleware(BearerAuthMiddleware)
        # Add unauthenticated status endpoint
        app.routes.append(Route("/gpu", _gpu_status))
        app.routes.append(Route("/activity", _activity_status))
        app.routes.append(Route("/api/mcp-models", _mcp_models))
        config = uvicorn.Config(
            app, host=mcp.settings.host, port=mcp.settings.port,
            log_level=mcp.settings.log_level.lower(),
            proxy_headers=True, forwarded_allow_ips="*")
        anyio.run(uvicorn.Server(config).serve)
    else:
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
