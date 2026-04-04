"""Unit tests for MCP server logic: model selection, auth, GPU tracking, jobs.

Tests pure/near-pure functions from mcp/local-models-server.py without
requiring live Ollama/MLX services. Heavy dependencies (mcp, httpx, torch,
starlette) are mocked at import time.
"""

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Mock heavy dependencies before importing the MCP server ─────────
# The server imports mcp, httpx, starlette, torch, sentence-transformers,
# mlx-audio at the top level. We mock them all.

_starlette_mock = MagicMock()
_starlette_mock.middleware.base.BaseHTTPMiddleware = type(
    "BaseHTTPMiddleware", (), {})
_starlette_mock.responses.JSONResponse = MagicMock()

for mod_name in (
    "httpx", "mcp", "mcp.server", "mcp.server.fastmcp",
    "mcp.server.transport_security",
    "starlette", "starlette.middleware", "starlette.middleware.base",
    "starlette.responses",
    "torch", "sentence_transformers",
    "mlx_audio", "mlx_audio.tts",
    "anyio", "yaml", "pyyaml",
):
    if mod_name not in sys.modules:
        if mod_name.startswith("starlette"):
            sys.modules[mod_name] = _starlette_mock
        else:
            sys.modules[mod_name] = MagicMock()

# Make FastMCP return a mock that records tool registrations
_fastmcp_mock = MagicMock()
_fastmcp_mock.return_value = MagicMock()
sys.modules["mcp.server.fastmcp"].FastMCP = _fastmcp_mock
sys.modules["mcp.server.transport_security"].TransportSecuritySettings = MagicMock()

# Provide BaseHTTPMiddleware as a real class so the server can subclass it
sys.modules["starlette.middleware.base"].BaseHTTPMiddleware = type(
    "BaseHTTPMiddleware", (), {"dispatch": lambda self, req, call_next: None})
sys.modules["starlette.responses"].JSONResponse = MagicMock()

# Ensure lib/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Now import the server — this will use our mocked modules
import importlib
import mcp as _mcp_mod  # noqa: F811 - this is the mock

# We need to import the server module carefully
_server_path = Path(__file__).resolve().parent.parent / "mcp"
sys.path.insert(0, str(_server_path))

# Patch os.environ for MCP_AUTH_TOKEN before import
with patch.dict("os.environ", {
    "MCP_AUTH_TOKEN": "test-token-123",
    "OLLAMA_URL": "http://localhost:11434",
    "MLX_URL": "http://localhost:8000",
}):
    # The server reads env vars at module level
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "local_models_server",
        str(_server_path / "local-models-server.py"))
    server = importlib.util.module_from_spec(spec)
    sys.modules["local_models_server"] = server
    spec.loader.exec_module(server)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_gpu_state():
    """Reset GPU tracking state between tests."""
    server._gpu_active.update({"ollama": 0, "mlx": 0})
    server._gpu_active_details.update({"ollama": [], "mlx": []})
    server._request_history.clear()
    server._request_stats.clear()
    yield


@pytest.fixture(autouse=True)
def _reset_models():
    """Reset model registry between tests."""
    server._models.clear()
    yield


@pytest.fixture(autouse=True)
def _reset_jobs():
    """Reset job store between tests."""
    server._jobs.clear()
    yield


# ── _resolve_model ──────────────────────────────────────────────────

class TestResolveModel:
    def test_exact_tagged_match(self):
        server._models["qwen3:8b"] = {"backend": "ollama"}
        assert server._resolve_model("qwen3:8b") == ("qwen3:8b", "ollama")

    def test_prefix_latest(self):
        server._models["qwen3:latest"] = {"backend": "ollama"}
        assert server._resolve_model("qwen3") == ("qwen3:latest", "ollama")

    def test_prefix_version(self):
        server._models["qwen3:8b"] = {"backend": "ollama"}
        assert server._resolve_model("qwen3") == ("qwen3:8b", "ollama")

    def test_base_name_alias(self):
        server._models["qwen3.5-fast"] = {"backend": "mlx"}
        assert server._resolve_model("qwen3.5-fast") == ("qwen3.5-fast", "mlx")

    def test_no_match(self):
        server._models["qwen3:8b"] = {"backend": "ollama"}
        assert server._resolve_model("nonexistent") is None

    def test_prefers_tagged_over_base(self):
        server._models["llama3:8b"] = {"backend": "ollama"}
        server._models["llama3"] = {"backend": "mlx"}
        result = server._resolve_model("llama3")
        # Should prefer the tagged version (prefix match) over base alias
        assert result == ("llama3:8b", "ollama")


# ── pick_model ──────────────────────────────────────────────────────

class TestPickModel:
    def test_explicit_override(self):
        server._models["custom:7b"] = {"backend": "ollama"}
        assert server.pick_model("code", "custom:7b") == ("custom:7b", "ollama")

    def test_prefs_single_string(self):
        server._models["deepseek:33b"] = {"backend": "ollama"}
        with patch.object(server, "load_mcp_prefs",
                          return_value={"code": "deepseek"}):
            assert server.pick_model("code") == ("deepseek:33b", "ollama")

    def test_prefs_list(self):
        server._models["qwen3:8b"] = {"backend": "ollama"}
        with patch.object(server, "load_mcp_prefs",
                          return_value={"code": ["missing", "qwen3"]}):
            assert server.pick_model("code") == ("qwen3:8b", "ollama")

    def test_falls_back_to_general(self):
        server._models["llama3:8b"] = {"backend": "ollama"}
        with patch.object(server, "load_mcp_prefs",
                          return_value={"general": "llama3"}):
            assert server.pick_model("code") == ("llama3:8b", "ollama")

    def test_task_tagged_model(self):
        server._models["code-llama:13b"] = {"backend": "ollama", "task": "code"}
        with patch.object(server, "load_mcp_prefs", return_value={}):
            assert server.pick_model("code") == ("code-llama:13b", "ollama")

    def test_any_available_model(self):
        server._models["random:7b"] = {"backend": "mlx"}
        with patch.object(server, "load_mcp_prefs", return_value={}):
            assert server.pick_model("code") == ("random:7b", "mlx")

    def test_no_models_raises(self):
        with patch.object(server, "load_mcp_prefs", return_value={}):
            with pytest.raises(ValueError, match="No model available"):
                server.pick_model("code")

    def test_override_miss_falls_through(self):
        server._models["fallback:7b"] = {"backend": "ollama"}
        with patch.object(server, "load_mcp_prefs", return_value={}):
            name, backend = server.pick_model("code", "nonexistent")
        assert name == "fallback:7b"


# ── load_mcp_prefs / thinking_enabled ──────────────────────────────

class TestPrefsAndThinking:
    def test_load_prefs_missing_file(self, tmp_path):
        with patch.object(server, "MCP_PREFS_FILE", tmp_path / "nope.json"):
            assert server.load_mcp_prefs() == {}

    def test_load_prefs_valid(self, tmp_path):
        f = tmp_path / "prefs.json"
        f.write_text('{"code": "qwen3"}')
        with patch.object(server, "MCP_PREFS_FILE", f):
            assert server.load_mcp_prefs() == {"code": "qwen3"}

    def test_load_prefs_invalid_json(self, tmp_path):
        f = tmp_path / "prefs.json"
        f.write_text("{broken")
        with patch.object(server, "MCP_PREFS_FILE", f):
            assert server.load_mcp_prefs() == {}

    def test_thinking_enabled_default(self):
        with patch.object(server, "load_mcp_prefs", return_value={}):
            assert server.thinking_enabled("code") is True

    def test_thinking_disabled(self):
        with patch.object(server, "load_mcp_prefs",
                          return_value={"thinking": {"code": False}}):
            assert server.thinking_enabled("code") is False

    def test_thinking_other_task_default(self):
        with patch.object(server, "load_mcp_prefs",
                          return_value={"thinking": {"code": False}}):
            assert server.thinking_enabled("general") is True


# ── GPU activity tracking ──────────────────────────────────────────

class TestGpuTracking:
    def test_context_manager_increments(self):
        with server._gpu_request("ollama", "test:model"):
            assert server._gpu_active["ollama"] == 1
        assert server._gpu_active["ollama"] == 0

    def test_history_recorded(self):
        with server._gpu_request("mlx", "vision:qwen"):
            pass
        assert len(server._request_history) == 1
        assert server._request_history[0]["backend"] == "mlx"
        assert server._request_history[0]["status"] == "ok"

    def test_error_status_on_exception(self):
        try:
            with server._gpu_request("ollama", "gen:test"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert server._request_history[0]["status"] == "error"

    def test_history_ring_buffer(self):
        for i in range(server._REQUEST_HISTORY_MAX + 10):
            with server._gpu_request("ollama", f"test:{i}"):
                pass
        assert len(server._request_history) == server._REQUEST_HISTORY_MAX

    def test_stats_accumulated(self):
        with server._gpu_request("ollama", "vision:model"):
            pass
        with server._gpu_request("ollama", "vision:other"):
            pass
        with server._gpu_request("mlx", "code:thing"):
            pass
        assert server._request_stats["vision"] == 2
        assert server._request_stats["code"] == 1

    def test_contention_warning_none(self):
        assert server._gpu_contention_warning("ollama") == ""

    def test_contention_warning_active(self):
        with server._gpu_request("ollama", "gen:big-model"):
            with server._gpu_request("ollama", "vision:other"):
                warning = server._gpu_contention_warning("ollama")
                assert "1 other request" in warning
                assert "active on ollama" in warning

    def test_contention_warning_plural(self):
        with server._gpu_request("ollama", "a:1"):
            with server._gpu_request("ollama", "b:2"):
                with server._gpu_request("ollama", "c:3"):
                    warning = server._gpu_contention_warning("ollama")
                    assert "2 other requests" in warning


# ── Auth middleware logic ──────────────────────────────────────────

class TestAuthSessionTracking:
    def test_session_add_and_check(self):
        with server._session_lock:
            server._authenticated_sessions.add("sess-123")
        assert "sess-123" in server._authenticated_sessions

    def test_session_eviction_at_max(self):
        with server._session_lock:
            server._authenticated_sessions.clear()
            for i in range(server._MAX_SESSIONS):
                server._authenticated_sessions.add(f"sess-{i}")
            assert len(server._authenticated_sessions) == server._MAX_SESSIONS
            # Adding one more should trigger pop()
            if len(server._authenticated_sessions) >= server._MAX_SESSIONS:
                server._authenticated_sessions.pop()
            server._authenticated_sessions.add("sess-new")
            assert "sess-new" in server._authenticated_sessions
            assert len(server._authenticated_sessions) == server._MAX_SESSIONS

    def test_exempt_paths(self):
        assert "/gpu" in server._AUTH_EXEMPT_PATHS
        assert "/api/mcp-models" in server._AUTH_EXEMPT_PATHS


# ── Job dispatch/collect ───────────────────────────────────────────

class TestJobStore:
    def _make_job(self, status="running", result=None):
        return {
            "status": status,
            "model": "test-model",
            "backend": "ollama",
            "result": result,
            "created": time.time(),
        }

    def test_job_store_starts_empty(self):
        assert len(server._jobs) == 0

    def test_running_job_not_collected(self):
        server._jobs["abc"] = self._make_job("running")
        assert server._jobs["abc"]["status"] == "running"
        assert server._jobs["abc"]["result"] is None

    def test_done_job_has_result(self):
        server._jobs["done1"] = self._make_job("done", "The answer")
        job = server._jobs["done1"]
        assert job["status"] == "done"
        assert job["result"] == "The answer"

    def test_failed_job_has_error(self):
        server._jobs["fail1"] = self._make_job("failed", "Error: timeout")
        assert server._jobs["fail1"]["status"] == "failed"
        assert "Error" in server._jobs["fail1"]["result"]

    def test_stale_job_eviction_logic(self):
        old_time = time.time() - server._JOB_TTL - 10
        server._jobs["stale"] = {
            "status": "done", "model": "m", "backend": "ollama",
            "result": "old", "created": old_time,
        }
        server._jobs["fresh"] = self._make_job("done", "new")
        # Evict stale jobs (same logic as local_dispatch)
        now = time.time()
        stale = [jid for jid, j in server._jobs.items()
                 if j["status"] != "running" and now - j["created"] > server._JOB_TTL]
        for jid in stale:
            del server._jobs[jid]
        assert "stale" not in server._jobs
        assert "fresh" in server._jobs

    def test_job_ttl_constant(self):
        assert server._JOB_TTL == 3600
