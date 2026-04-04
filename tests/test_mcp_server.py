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

class TestAuthMiddlewareDispatch:
    """Test the REAL BearerAuthMiddleware.dispatch method."""

    @pytest.fixture(autouse=True)
    def _reset_sessions(self):
        with server._session_lock:
            server._authenticated_sessions.clear()
        yield
        with server._session_lock:
            server._authenticated_sessions.clear()

    def _make_request(self, path="/mcp", headers=None, query_params=None):
        req = MagicMock()
        req.url.path = path
        req.headers = headers or {}
        req.query_params = query_params or {}
        return req

    def _make_response(self, headers=None):
        resp = MagicMock()
        resp.headers = headers or {}
        return resp

    def _call(self, path, headers=None, query_params=None, resp_headers=None):
        import asyncio

        middleware = server.BearerAuthMiddleware.__new__(server.BearerAuthMiddleware)
        req = self._make_request(path, headers, query_params)
        resp = self._make_response(resp_headers)

        async def call_next(r):
            return resp
        call_next_mock = MagicMock(side_effect=call_next)

        async def run():
            return await middleware.dispatch(req, call_next_mock)

        result = asyncio.run(run())
        return result, call_next_mock, req

    def test_rejects_missing_token(self):
        result, call_next, _ = self._call("/mcp", headers={})
        call_next.assert_not_called()

    def test_rejects_wrong_token(self):
        result, call_next, _ = self._call("/mcp", headers={"authorization": "Bearer wrong-token"})
        call_next.assert_not_called()

    def test_allows_correct_token(self):
        result, call_next, req = self._call(
            "/mcp", headers={"authorization": f"Bearer {server.MCP_AUTH_TOKEN}"})
        call_next.assert_called_once_with(req)

    def test_exempt_paths_skip_auth(self):
        for path in ("/gpu", "/api/mcp-models"):
            result, call_next, req = self._call(path, headers={})
            call_next.assert_called_once_with(req)

    def test_well_known_skips_auth(self):
        result, call_next, req = self._call(
            "/.well-known/oauth-authorization-server", headers={})
        call_next.assert_called_once_with(req)

    def test_mcp_init_tracks_session(self):
        self._call(
            "/mcp",
            headers={"authorization": f"Bearer {server.MCP_AUTH_TOKEN}"},
            query_params={"session_id": "sess-abc"})
        with server._session_lock:
            assert "sess-abc" in server._authenticated_sessions

    def test_messages_with_valid_session_passes(self):
        with server._session_lock:
            server._authenticated_sessions["sess-ok"] = None
        result, call_next, req = self._call(
            "/messages", headers={}, query_params={"session_id": "sess-ok"})
        call_next.assert_called_once_with(req)

    def test_messages_with_unknown_session_rejects(self):
        result, call_next, _ = self._call(
            "/messages", headers={}, query_params={"session_id": "sess-unknown"})
        call_next.assert_not_called()

    def test_response_header_session_tracked(self):
        self._call(
            "/mcp",
            headers={"authorization": f"Bearer {server.MCP_AUTH_TOKEN}"},
            resp_headers={"mcp-session-id": "from-resp"})
        with server._session_lock:
            assert "from-resp" in server._authenticated_sessions


    def test_session_eviction_is_fifo(self):
        old_max = server._MAX_SESSIONS
        server._MAX_SESSIONS = 3
        try:
            for sid in ["first", "second", "third"]:
                self._call(
                    "/mcp",
                    headers={"authorization": f"Bearer {server.MCP_AUTH_TOKEN}"},
                    query_params={"session_id": sid})
            with server._session_lock:
                assert "first" in server._authenticated_sessions
                assert "second" in server._authenticated_sessions
                assert "third" in server._authenticated_sessions

            self._call(
                "/mcp",
                headers={"authorization": f"Bearer {server.MCP_AUTH_TOKEN}"},
                query_params={"session_id": "fourth"})
            with server._session_lock:
                assert "first" not in server._authenticated_sessions
                assert "second" in server._authenticated_sessions
                assert "fourth" in server._authenticated_sessions
        finally:
            server._MAX_SESSIONS = old_max


class TestPathValidation:
    """Path traversal prevention in MCP tools."""

    def test_home_directory_allowed(self):
        # Create a temp file under $HOME to test
        import tempfile
        home = Path.home()
        with tempfile.NamedTemporaryFile(dir=home, suffix=".txt", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            assert server._validate_path(path) is None
        finally:
            Path(path).unlink()

    def test_tmp_directory_allowed(self):
        import tempfile
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".txt", delete=False) as f:
            f.write(b"test")
            path = f.name
        try:
            assert server._validate_path(path) is None
        finally:
            Path(path).unlink()

    def test_etc_passwd_rejected(self):
        result = server._validate_path("/etc/passwd")
        assert result is not None
        assert "not allowed" in result

    def test_ssh_keys_rejected(self):
        result = server._validate_path("/root/.ssh/id_rsa")
        assert result is not None
        assert "not allowed" in result

    def test_proc_environ_rejected(self):
        result = server._validate_path("/proc/self/environ")
        assert result is not None
        assert "not allowed" in result

    def test_traversal_via_dotdot_rejected(self):
        # Try to escape from $HOME via ../
        result = server._validate_path(str(Path.home() / ".." / "etc" / "passwd"))
        assert result is not None
        assert "not allowed" in result

    def test_nonexistent_file_rejected_by_default(self):
        result = server._validate_path("/tmp/nonexistent_file_abc123.txt")
        assert result is not None
        assert "not found" in result.lower()

    def test_nonexistent_file_allowed_for_writes(self):
        result = server._validate_path("/tmp/new_output_file.png", must_exist=False)
        assert result is None

    def test_validate_paths_rejects_bad_in_list(self):
        import tempfile
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".txt", delete=False) as f:
            f.write(b"test")
            good_path = f.name
        try:
            result = server._validate_paths([good_path, "/etc/passwd"])
            assert result is not None
            assert "not allowed" in result
        finally:
            Path(good_path).unlink()


class TestConfigurablePathRestrictions:
    """MCP_ALLOWED_PATHS in network.conf restricts file access."""

    def test_custom_allowed_roots(self, tmp_path):
        project_dir = tmp_path / "projects"
        project_dir.mkdir()
        test_file = project_dir / "test.txt"
        test_file.write_text("hello")

        old_roots = server._ALLOWED_ROOTS
        server._ALLOWED_ROOTS = (project_dir, Path("/tmp"), Path("/private/tmp"))
        try:
            assert server._validate_path(str(test_file)) is None
            result = server._validate_path(str(Path.home() / ".ssh" / "id_rsa"))
            assert result is not None
            assert "not allowed" in result
        finally:
            server._ALLOWED_ROOTS = old_roots

    def test_tmp_always_included(self, tmp_path):
        project_dir = tmp_path / "projects"
        project_dir.mkdir()

        old_roots = server._ALLOWED_ROOTS
        server._ALLOWED_ROOTS = (project_dir, Path("/tmp"), Path("/private/tmp"))
        try:
            result = server._validate_path("/tmp/some_file.txt", must_exist=False)
            assert result is None
        finally:
            server._ALLOWED_ROOTS = old_roots

    def test_load_allowed_roots_from_config(self, tmp_path):
        conf_file = tmp_path / "network.conf"
        conf_file.write_text('MCP_ALLOWED_PATHS="/Users/test/projects:/Users/test/data"\n')
        with patch.object(server, "NETWORK_CONF", conf_file):
            roots = server._load_allowed_roots()
        root_strs = [str(r) for r in roots]
        assert "/Users/test/projects" in root_strs
        assert "/Users/test/data" in root_strs
        assert str(Path("/tmp")) in root_strs

    def test_load_allowed_roots_defaults_without_config(self, tmp_path):
        conf_file = tmp_path / "nonexistent.conf"
        with patch.object(server, "NETWORK_CONF", conf_file):
            roots = server._load_allowed_roots()
        assert Path.home() in roots
        assert Path("/tmp") in roots
