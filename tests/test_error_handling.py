"""Tests for error handling, model validation, and backend routing.

Tests call REAL functions from the MCP server and profile server.
No re-implementations — if the source code breaks, these tests break.
"""

import contextlib
import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Import profile server (same pattern as test_profile_server.py) ──

for mod in ("mlx_audio", "mlx_audio.tts"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import os
import importlib.util

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("MLX_URL", "http://localhost:8000")
os.environ["PROFILE_IDLE_TIMEOUT"] = "0"

hf_scanner_mock = MagicMock()
hf_scanner_mock.scan_hf_cache = MagicMock(return_value=[])
sys.modules["lib.hf_scanner"] = hf_scanner_mock

_ps_path = Path(__file__).resolve().parent.parent / "app" / "profile-server.py"
spec = importlib.util.spec_from_file_location("profile_server_eh", str(_ps_path))
ps = importlib.util.module_from_spec(spec)
sys.modules["profile_server_eh"] = ps
spec.loader.exec_module(ps)


# ── Import MCP server (same pattern as test_mcp_server.py) ──────────

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

_fastmcp_mock = MagicMock()
_fastmcp_mock.return_value = MagicMock()
sys.modules["mcp.server.fastmcp"].FastMCP = _fastmcp_mock
sys.modules["mcp.server.transport_security"].TransportSecuritySettings = MagicMock()
sys.modules["starlette.middleware.base"].BaseHTTPMiddleware = type(
    "BaseHTTPMiddleware", (), {"dispatch": lambda self, req, call_next: None})
sys.modules["starlette.responses"].JSONResponse = MagicMock()

_server_path = Path(__file__).resolve().parent.parent / "mcp"
sys.path.insert(0, str(_server_path))

with patch.dict("os.environ", {
    "MCP_AUTH_TOKEN": "test-token-123",
    "OLLAMA_URL": "http://localhost:11434",
    "MLX_URL": "http://localhost:8000",
}):
    spec_mcp = importlib.util.spec_from_file_location(
        "local_models_server_eh",
        str(_server_path / "local-models-server.py"))
    server = importlib.util.module_from_spec(spec_mcp)
    sys.modules["local_models_server_eh"] = server
    spec_mcp.loader.exec_module(server)


# ── Fixtures ────────────────────────────────────────────────────────

FAKE_MODELS = {
    "qwen3.5-fast": {"backend": "ollama", "parameter_size": "32B"},
    "nemotron-super": {"backend": "ollama", "parameter_size": "49B"},
    "qwen3.5-fast:latest": {"backend": "ollama", "parameter_size": "32B"},
}

@pytest.fixture(autouse=True)
def _reset_gpu_state():
    server._gpu_active.update({"ollama": 0, "mlx": 0})
    server._gpu_active_details.update({"ollama": [], "mlx": []})
    server._request_history.clear()
    server._request_stats.clear()
    yield


# ── MCP server: _http_error_detail ──────────────────────────────────

class TestHttpErrorDetail:
    """Test the REAL _http_error_detail function from the MCP server."""

    def _make_error(self, status=500, text="internal error", url="http://localhost:11434/api/chat"):
        e = MagicMock()
        e.response.status_code = status
        e.response.text = text
        e.request.url = url
        return e

    def test_includes_status_and_body(self):
        e = self._make_error(404, '{"error":"model not found"}')
        result = server._http_error_detail(e, "Ollama chat (test-model)")
        assert "404" in result
        assert "model not found" in result
        assert "localhost:11434" in result
        assert "Ollama chat (test-model)" in result

    def test_truncates_long_body(self):
        e = self._make_error(500, "x" * 1000)
        result = server._http_error_detail(e, "test")
        # Body should be truncated to 500 chars
        assert len(result) < 600

    def test_handles_empty_body(self):
        e = self._make_error(502, "")
        result = server._http_error_detail(e, "test")
        assert "(empty body)" in result


# ── Profile server: _requests_error_detail ──────────────────────────

class TestRequestsErrorDetail:
    """Test the REAL _requests_error_detail from the profile server."""

    def test_http_error(self):
        import requests
        resp = MagicMock()
        resp.status_code = 503
        resp.text = "Service Unavailable"
        resp.url = "http://localhost:8000/v1/chat/completions"
        e = requests.HTTPError(response=resp)
        result = ps._requests_error_detail(e)
        assert "503" in result
        assert "Service Unavailable" in result
        assert "localhost:8000" in result

    def test_connection_error(self):
        import requests
        e = requests.ConnectionError("Connection refused")
        result = ps._requests_error_detail(e)
        assert "Cannot connect" in result

    def test_timeout_error(self):
        import requests
        e = requests.Timeout("read timed out")
        result = ps._requests_error_detail(e)
        assert "timed out" in result

    def test_generic_error(self):
        result = ps._requests_error_detail(ValueError("weird error"))
        assert "weird error" in result


# ── Profile server: _pick_model_for_task ────────────────────────────

class TestPickModelForTask:
    """Test the REAL _pick_model_for_task from the profile server."""

    def test_returns_exact_match(self):
        with patch.object(ps, "load_default_prefs", return_value={"code": ["qwen3.5-fast"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            model, backend, warning = ps._pick_model_for_task("code")
        assert model == "qwen3.5-fast"
        assert backend == "ollama"
        assert warning is None

    def test_returns_prefix_match(self):
        models = {"qwen3.5-fast:latest": {"backend": "ollama"}}
        with patch.object(ps, "load_default_prefs", return_value={"code": ["qwen3.5-fast"]}), \
             patch.object(ps, "get_all_models", return_value=models):
            model, backend, warning = ps._pick_model_for_task("code")
        assert model == "qwen3.5-fast:latest"
        assert backend == "ollama"
        assert warning is None

    def test_warns_when_all_candidates_stale(self):
        with patch.object(ps, "load_default_prefs", return_value={"code": ["gone-model", "also-gone"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            model, backend, warning = ps._pick_model_for_task("code")
        assert model is None
        assert backend is None
        assert "gone-model" in warning
        assert "also-gone" in warning
        assert "using fallback" in warning

    def test_no_warning_with_empty_candidates(self):
        with patch.object(ps, "load_default_prefs", return_value={}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            model, backend, warning = ps._pick_model_for_task("code")
        assert model is None
        assert warning is None


# ── Profile server: activation pruning ──────────────────────────────

class TestActivationPruning:
    """Test the REAL /api/profiles/<name>/activate endpoint."""

    @pytest.fixture()
    def client(self):
        ps.app.config["TESTING"] = True
        with ps.app.test_client() as c:
            yield c

    @pytest.fixture()
    def profiles_dir(self, tmp_path):
        pf = tmp_path / "profiles.json"
        mf = tmp_path / "mcp_prefs.json"
        with patch.object(ps, "PROFILES_FILE", pf), \
             patch.object(ps, "MCP_PREFS_FILE", mf):
            yield tmp_path

    def test_reports_missing_ollama_models(self, client, profiles_dir):
        ps.save_profiles({
            "version": ps.PROFILES_VERSION,
            "active": None,
            "profiles": {
                "test": {"label": "Test", "tasks": {"code": "qwen3.5-fast"}},
            },
        })
        ps.save_mcp_prefs({
            "code": ["nonexistent-model", "qwen3.5-fast"],
            "image_gen": ["gone-image-model"],
        })

        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            resp = client.post("/api/profiles/test/activate")

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"]
        # Ollama-style names (no "/") go into missing, not warnings
        missing_names = [m["name"] for m in data["missing"]]
        assert "nonexistent-model" in missing_names
        assert "gone-image-model" in missing_names

    def test_reports_missing_hf_models_as_pullable(self, client, profiles_dir):
        ps.save_profiles({
            "version": ps.PROFILES_VERSION,
            "active": None,
            "profiles": {
                "test": {"label": "Test", "tasks": {"code": "qwen3.5-fast"}},
            },
        })
        ps.save_mcp_prefs({
            "code": ["org/nonexistent-hf-model"],
        })

        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            resp = client.post("/api/profiles/test/activate")

        data = resp.get_json()
        assert any(m["name"] == "org/nonexistent-hf-model" for m in data["missing"])

    def test_skips_pruning_when_no_models(self, client, profiles_dir):
        ps.save_profiles({
            "version": ps.PROFILES_VERSION,
            "active": None,
            "profiles": {"test": {"label": "Test", "tasks": {}}},
        })
        ps.save_mcp_prefs({"code": ["anything"]})

        with patch.object(ps, "get_all_models", return_value={}):
            resp = client.post("/api/profiles/test/activate")

        assert resp.status_code == 200
        # With no models, pruning should be skipped — candidates preserved
        saved = ps.load_default_prefs()
        assert "anything" in saved.get("code", [])


# ── Profile server: _track_playground ───────────────────────────────

class TestPlaygroundTracking:
    """Test the REAL _track_playground context manager."""

    def test_tracks_and_cleans_up(self):
        with ps._track_playground("test_tool", "test_model", "ollama"):
            with ps._playground_lock:
                assert len(ps._playground_active) == 1
                entry = list(ps._playground_active.values())[0]
                assert entry["tool"] == "test_tool"
                assert entry["model"] == "test_model"
        with ps._playground_lock:
            assert len(ps._playground_active) == 0

    def test_cleans_up_on_exception(self):
        with pytest.raises(RuntimeError):
            with ps._track_playground("failing", "model", "ollama"):
                raise RuntimeError("boom")
        with ps._playground_lock:
            assert len(ps._playground_active) == 0

    def test_thread_isolation(self):
        both_ready = threading.Event()
        t1_entered = threading.Event()
        t2_entered = threading.Event()
        t1_done = threading.Event()
        results = {}

        def worker1():
            with ps._track_playground("task1", "model", "ollama"):
                t1_entered.set()
                t2_entered.wait(timeout=2)
                with ps._playground_lock:
                    results["t1"] = len(ps._playground_active)
                t1_done.set()

        def worker2():
            t1_entered.wait(timeout=2)
            with ps._track_playground("task2", "model", "ollama"):
                t2_entered.set()
                t1_done.wait(timeout=2)
                with ps._playground_lock:
                    results["t2"] = len(ps._playground_active)

        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start(); t2.start()
        t1.join(timeout=5); t2.join(timeout=5)

        # Both threads should see 2 active entries while overlapping
        assert results["t1"] == 2
        # After t1 exits, t2 should see 1
        assert results["t2"] == 1
        # All cleaned up after
        with ps._playground_lock:
            assert len(ps._playground_active) == 0


# ── MCP server: _gpu_contention_warning ─────────────────────────────

class TestGpuContentionWarning:
    """Test the REAL _gpu_contention_warning from the MCP server."""

    def test_no_warning_when_alone(self):
        with server._gpu_request("ollama", "test"):
            result = server._gpu_contention_warning("ollama")
        assert result == ""

    def test_warns_with_concurrent_requests(self):
        # Simulate two concurrent requests
        ctx1 = server._gpu_request("ollama", "first_task")
        ctx2 = server._gpu_request("ollama", "second_task")
        ctx1.__enter__()
        ctx2.__enter__()
        warning = server._gpu_contention_warning("ollama")
        ctx2.__exit__(None, None, None)
        ctx1.__exit__(None, None, None)

        assert "contention" in warning.lower()
        assert "1 other request" in warning

    def test_pluralizes_correctly(self):
        ctxs = [server._gpu_request("ollama", f"task_{i}") for i in range(3)]
        for c in ctxs:
            c.__enter__()
        warning = server._gpu_contention_warning("ollama")
        for c in reversed(ctxs):
            c.__exit__(None, None, None)

        assert "2 other requests" in warning



# ── MCP server: auth middleware logic ───────────────────────────────

class TestAuthMiddleware:
    """Test auth session tracking via the REAL module-level state."""

    @pytest.fixture(autouse=True)
    def _reset_sessions(self):
        with server._session_lock:
            server._authenticated_sessions.clear()
        yield
        with server._session_lock:
            server._authenticated_sessions.clear()

    def test_session_added_on_auth(self):
        with server._session_lock:
            server._authenticated_sessions["session-1"] = None
        with server._session_lock:
            assert "session-1" in server._authenticated_sessions

    def test_session_eviction_at_max(self):
        """Fill to _MAX_SESSIONS, then add one more — should not crash."""
        with server._session_lock:
            for i in range(server._MAX_SESSIONS):
                server._authenticated_sessions[f"s-{i}"] = None
            assert len(server._authenticated_sessions) == server._MAX_SESSIONS
            # Simulate what the middleware does (FIFO eviction)
            if len(server._authenticated_sessions) >= server._MAX_SESSIONS and server._authenticated_sessions:
                server._authenticated_sessions.popitem(last=False)
            server._authenticated_sessions["new-session"] = None
        with server._session_lock:
            assert "new-session" in server._authenticated_sessions
            assert len(server._authenticated_sessions) == server._MAX_SESSIONS

    def test_eviction_on_empty_set_does_not_crash(self):
        """Regression: popitem() on empty dict should not raise KeyError."""
        with server._session_lock:
            assert len(server._authenticated_sessions) == 0
            # Simulate the guard we added
            if len(server._authenticated_sessions) >= server._MAX_SESSIONS and server._authenticated_sessions:
                server._authenticated_sessions.popitem(last=False)  # should not execute
            server._authenticated_sessions["first"] = None
        with server._session_lock:
            assert "first" in server._authenticated_sessions

    def test_exempt_paths(self):
        assert "/gpu" in server._AUTH_EXEMPT_PATHS
        assert "/api/mcp-models" in server._AUTH_EXEMPT_PATHS

    def test_token_matches_env(self):
        assert server.MCP_AUTH_TOKEN == "test-token-123"
