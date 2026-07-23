"""Unit tests for MCP server logic: model selection, auth, GPU tracking, jobs.

Tests pure/near-pure functions from mcp/local-models-server.py without
requiring live Ollama/MLX services. Heavy dependencies (mcp, httpx, torch,
starlette) are mocked at import time.
"""

import asyncio
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
    "anyio",
    # NB: do NOT mock yaml here — it's only imported lazily inside
    # functions no MCP test exercises, and mocking it globally poisons
    # lib.mlx_vlm.repo_for's yaml.safe_load in other test modules.
):
    if mod_name not in sys.modules:
        if mod_name.startswith("starlette"):
            sys.modules[mod_name] = _starlette_mock
        else:
            sys.modules[mod_name] = MagicMock()

# Make FastMCP return a mock whose .tool() decorator returns the function
# unchanged — that way @mcp.tool()-decorated functions remain callable from
# tests instead of being replaced with a MagicMock. Other attribute access
# falls through to MagicMock as before.
_fastmcp_instance = MagicMock()
_fastmcp_instance.tool = lambda *a, **kw: (lambda fn: fn)
# Default: no active MCP request → _current_client() returns "" (the real
# get_context() yields a context whose request is None outside a tool call).
# Tests that need attribution patch server._current_client or server.mcp.get_context.
_fastmcp_instance.get_context.return_value.request_context.request = None
_fastmcp_mock = MagicMock(return_value=_fastmcp_instance)
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

    def test_no_models_error_is_actionable(self):
        """Error message includes what was tried and what's available."""
        # Only non-LLM backend models — pick_model won't fall back to these
        server._models.clear()
        server._models["whisper:v3"] = {"backend": "whisper"}
        with patch.object(server, "load_mcp_prefs",
                          return_value={"code": ["nonexistent-model"]}):
            with pytest.raises(ValueError) as exc_info:
                server.pick_model("code")
            msg = str(exc_info.value)
            assert "code" in msg
            assert "nonexistent-model" in msg  # shows what was tried
            assert "No models loaded" in msg  # no ollama/mlx available
            assert "mcp_preferences.json" in msg  # suggests fix

    def test_vision_skips_non_vision_prefs(self):
        """A vision pref that resolves to a model without a vision tower
        must be skipped, not handed back for the vision tool to reject.
        This is the qwen3.6:27b-mlx-bf16 case: it resolves by name but
        can't see, so the picker must fall through to a working one."""
        server._models["qwen3.6:27b-mlx-bf16"] = {"backend": "ollama", "vision": False}
        server._models["qwen3.6:27b"] = {"backend": "ollama", "vision": True}
        with patch.object(server, "load_mcp_prefs",
                          return_value={"vision": ["qwen3.6:27b-mlx-bf16", "qwen3.6:27b"]}):
            assert server.pick_model("vision") == ("qwen3.6:27b", "ollama")

    def test_vision_no_eligible_model_raises(self):
        """When nothing has a vision tower, raise — never fall back to a
        text model via the any-LLM fallback (that produces hallucinations)."""
        server._models["qwen3.6:27b-mlx-bf16"] = {"backend": "ollama", "vision": False}
        server._models["dolphin3:8b"] = {"backend": "ollama", "vision": False}
        with patch.object(server, "load_mcp_prefs",
                          return_value={"vision": ["qwen3.6:27b-mlx-bf16"]}):
            with pytest.raises(ValueError, match="vision"):
                server.pick_model("vision")

    def test_override_miss_raises_not_silent_fallback(self):
        """An override that doesn't resolve must error, not silently
        substitute an arbitrary model. A vision request for an unpulled
        model used to land on an image-gen model instead."""
        server._models["fallback:7b"] = {"backend": "ollama"}
        with patch.object(server, "load_mcp_prefs", return_value={}):
            with pytest.raises(ValueError) as exc_info:
                server.pick_model("vision", "nonexistent")
        msg = str(exc_info.value)
        assert "nonexistent" in msg
        assert "not found" in msg


# ── ds4 chat dispatch ───────────────────────────────────────────────

class _FakeDs4Response:
    """Mimics httpx.Response for ds4: .text carries raw control chars, so
    strict .json() raises exactly like the real failure mode."""

    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.text)  # strict — raises on control chars


def _fake_async_client(response_text, captured):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None):
            captured["url"] = url
            captured["body"] = json
            return _FakeDs4Response(response_text)

    return _FakeAsyncClient


class TestChatDs4:
    def test_chat_routes_ds4_and_parses_unescaped_control_chars(self):
        """The real 2026-07-22 failure: ds4's encoder emits raw control
        chars inside long reasoning_content; strict JSON parsing (resp.json /
        plain json.loads) raises and crashes dispatch. chat() must route
        backend='ds4' to chat_ds4 (the old else-branch misrouted it to MLX
        :8000) and parse with strict=False."""
        raw = ('{"choices":[{"message":{"content":"OK then",'
               '"reasoning_content":"thinking\x01hard"}}]}')
        captured = {}
        with patch.object(server.httpx, "AsyncClient",
                          _fake_async_client(raw, captured)):
            result = asyncio.run(server.chat(
                "glm-5.2", "ds4",
                [{"role": "user", "content": "hi"}]))
        assert "OK then" in result
        assert captured["url"].endswith(":8002/v1/chat/completions")

    def test_chat_ds4_timeout_is_600_not_300(self):
        """At ~11.5 tok/s a 4096-token generation takes ~356s, and ds4
        serializes requests — 300s cuts off real in-flight requests. Pin the
        timeout the way test_chat_routes_ds4_and_parses_unescaped_control_chars
        pins the URL, so a regression back to 300 fails loudly."""
        raw = '{"choices":[{"message":{"content":"hi"}}]}'
        captured = {}
        with patch.object(server.httpx, "AsyncClient",
                          _fake_async_client(raw, captured)):
            asyncio.run(server.chat_ds4(
                "glm-5.2", [{"role": "user", "content": "hi"}]))
        assert captured["client_kwargs"]["timeout"] == 600

    def test_chat_ds4_never_forwards_think_toggle(self):
        """enable_thinking is VERIFIED BROKEN on ds4 (200 OK but reasoning
        migrates into content, no think-block markers to strip). think=False
        must NOT put chat_template_kwargs on the wire."""
        raw = '{"choices":[{"message":{"content":"hi"}}]}'
        captured = {}
        with patch.object(server.httpx, "AsyncClient",
                          _fake_async_client(raw, captured)):
            asyncio.run(server.chat_ds4(
                "glm-5.2", [{"role": "user", "content": "hi"}], think=False))
        assert "chat_template_kwargs" not in captured["body"]
        assert "think" not in captured["body"]

    def test_chat_ds4_falls_back_to_reasoning_content(self):
        """glm-5.2 on ds4 always thinks and can burn its whole token budget
        thinking — content comes back empty. Surface reasoning_content
        instead of returning an empty string."""
        raw = ('{"choices":[{"message":{"content":"",'
               '"reasoning_content":"the answer is 42"}}]}')
        with patch.object(server.httpx, "AsyncClient",
                          _fake_async_client(raw, {})):
            result = asyncio.run(server.chat_ds4(
                "glm-5.2", [{"role": "user", "content": "hi"}]))
        assert result == "the answer is 42"

    def test_gpu_tracking_accepts_ds4_key(self):
        """_gpu_active is a defaultdict for exactly this reason (a plain
        dict KeyError'd on mlx-audio and broke local_speak) — pin the
        guarantee for the new backend."""
        with server._gpu_request("ds4", "chat:glm-5.2"):
            assert server._gpu_active["ds4"] == 1
        assert server._gpu_active["ds4"] == 0

    def test_pick_model_any_llm_fallback_includes_ds4(self):
        """With no prefs and no task match, pick_model falls back to 'any
        LLM'. The old ('ollama', 'mlx') tuple silently excluded a
        ds4-backed registry — glm-5.2 would be invisible to fallback."""
        server._models["glm-5.2"] = {
            "backend": "ds4", "total_params_b": 380,
            "active_params_b": 32, "context": 131072, "vision": False,
        }
        with patch.object(server, "load_mcp_prefs", return_value={}):
            assert server.pick_model("general") == ("glm-5.2", "ds4")


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

    def test_hf_subprocess_backends_dont_keyerror(self):
        """TTS/image/video use backends beyond ollama+mlx (mlx-audio,
        mflux, mlx-video). Tracking must not KeyError on them — that's
        what broke local_speak with a bare 'mlx-audio' error."""
        for backend in ("mlx-audio", "mflux", "mlx-video"):
            with server._gpu_request(backend, f"tts:{backend}"):
                assert server._gpu_active[backend] == 1
            assert server._gpu_active[backend] == 0


# mlx_vlm dispatch parsing/normalization lives in lib.mlx_vlm and is
# unit-tested in test_core.py::TestMlxVlmDispatch.

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

    def test_gpu_tracker_stamps_client_machine(self):
        from lib import activity
        activity.init_db()
        with patch.object(server, "_current_client", return_value="jerry-laptop"):
            with server._gpu_request("ollama", "code:model-x"):
                pass
        row = activity.query_activity(60)["history"][0]
        assert row["machine"] == "jerry-laptop"

    def test_gpu_tracker_stamps_unknown_client_when_unset(self):
        from lib import activity
        activity.init_db()
        row_before = len(activity.query_activity(60)["history"])
        with patch.object(server, "_current_client", return_value=""):
            with server._gpu_request("ollama", "code:model-y"):
                pass
        history = activity.query_activity(60)["history"]
        assert len(history) == row_before + 1
        assert history[0]["machine"] == "unknown-client"


# ── X-SP-Client validation ──────────────────────────────────────────

class TestValidatedClient:
    def test_accepts_good_hostnames(self):
        assert server._validated_client("jerry-laptop") == "jerry-laptop"
        assert server._validated_client("MacBook-Pro.local") == "MacBook-Pro.local"

    def test_rejects_injection_attempt(self):
        assert server._validated_client("<img onerror=x>") == ""

    def test_rejects_too_long(self):
        assert server._validated_client("a" * 65) == ""

    def test_rejects_empty(self):
        assert server._validated_client("") == ""


class TestCurrentClient:
    """_current_client reads X-SP-Client off the current request context.

    These stub the request context to exercise the read/validation logic in
    isolation; the cross-task behavior over the real transport is proven by
    tests/test_mcp_attribution_e2e.py.
    """

    def _ctx_with_headers(self, headers):
        request = MagicMock()
        request.headers = headers
        ctx = MagicMock()
        ctx.request_context.request = request
        return ctx

    def test_reads_valid_header(self):
        ctx = self._ctx_with_headers({"x-sp-client": "jerry-laptop"})
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server._current_client() == "jerry-laptop"

    def test_invalid_header_is_empty(self):
        ctx = self._ctx_with_headers({"x-sp-client": "<img onerror=x>"})
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server._current_client() == ""

    def test_absent_header_is_empty(self):
        ctx = self._ctx_with_headers({})
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server._current_client() == ""

    def test_no_request_is_empty(self):
        ctx = MagicMock()
        ctx.request_context.request = None
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server._current_client() == ""

    def test_outside_request_context_is_empty(self):
        ctx = MagicMock()
        type(ctx).request_context = property(
            lambda self: (_ for _ in ()).throw(ValueError("no request")))
        with patch.object(server.mcp, "get_context", return_value=ctx):
            assert server._current_client() == ""


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

    def test_extension_allowlist_rejects_id_rsa(self):
        """The extension gate stops prompt-injected calls like
        local_image_edit(image_path='~/.ssh/id_rsa') from passing
        validation just because the file is under $HOME."""
        import tempfile
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix="", delete=False) as f:
            f.write(b"junk")
            sketchy = f.name
        try:
            err = server._validate_path(sketchy, allowed_exts=server._IMAGE_EXTS)
            assert err is not None
            assert "extension" in err.lower()
        finally:
            Path(sketchy).unlink()

    def test_extension_allowlist_accepts_image(self):
        import tempfile
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            good = f.name
        try:
            assert server._validate_path(
                good, allowed_exts=server._IMAGE_EXTS) is None
        finally:
            Path(good).unlink()

    def test_extension_allowlist_case_insensitive(self):
        import tempfile
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".PNG", delete=False) as f:
            f.write(b"x")
            good = f.name
        try:
            assert server._validate_path(
                good, allowed_exts=server._IMAGE_EXTS) is None
        finally:
            Path(good).unlink()

    def test_text_exts_rejects_secret_files(self):
        """Tools that ingest arbitrary files (translate, summarize, embed,
        etc.) gate on _TEXT_EXTS so a prompt-injected call can't read e.g.
        ~/.ssh/id_rsa or ~/.aws/credentials and feed contents to the model."""
        import tempfile
        # Files an attacker would target — extensionless or non-text ext.
        targets = [
            ("id_rsa", ""),         # ssh private key (no ext)
            ("credentials", ""),    # .aws/credentials (no ext)
            ("Cookies.binarycookies", ".binarycookies"),
            ("passwd", ""),         # /etc/passwd-style (we have to use $HOME)
        ]
        for name, suffix in targets:
            with tempfile.NamedTemporaryFile(
                    dir="/tmp", prefix=name + "_", suffix=suffix,
                    delete=False) as f:
                f.write(b"secret\n")
                path = f.name
            try:
                err = server._validate_path(
                    path, allowed_exts=server._TEXT_EXTS)
                assert err is not None, f"{name} should have been rejected"
                assert "extension" in err.lower()
            finally:
                Path(path).unlink()

    def test_text_exts_accepts_common_code_and_docs(self):
        """The allowlist must cover everything a developer actually wants
        to summarize/embed, otherwise users will set SP_ALLOW_NO_AUTH-style
        escapes and we lose the gate."""
        import tempfile
        for suffix in (".py", ".md", ".json", ".yaml", ".sql", ".ts",
                       ".rs", ".go", ".log", ".txt"):
            with tempfile.NamedTemporaryFile(
                    dir="/tmp", suffix=suffix, delete=False) as f:
                f.write(b"hello\n")
                path = f.name
            try:
                err = server._validate_path(
                    path, allowed_exts=server._TEXT_EXTS)
                assert err is None, (
                    f"{suffix} should be permitted but got: {err}")
            finally:
                Path(path).unlink()


class TestComputerUseScreenshotRequired:
    """local_computer_use must NOT auto-capture the screen. Auto-capture
    on the tailnet means one stolen token = silent screen harvesting."""

    def test_silent_screencapture_helper_removed(self):
        """_take_screenshot was the silent-capture helper.  Make sure it's
        gone — its presence (even unused) is a footgun."""
        assert not hasattr(server, "_take_screenshot"), (
            "Silent screen capture must not be reachable from MCP code.")

    def test_empty_screenshot_path_rejected(self):
        """The runtime guard catches empty strings even though the type
        annotation already forbids missing args at the schema level."""
        import asyncio
        result = asyncio.run(server.local_computer_use(
            intent="click submit", screenshot_path=""))
        assert "screenshot_path is required" in result.lower()

    def test_screenshot_path_extension_gated(self):
        """Even when the file exists and is under $HOME, a non-image
        extension is rejected — same threat as the text-tool case."""
        import asyncio
        import tempfile
        with tempfile.NamedTemporaryFile(
                dir="/tmp", suffix=".txt", delete=False) as f:
            f.write(b"not really an image")
            sketchy = f.name
        try:
            result = asyncio.run(server.local_computer_use(
                intent="click submit", screenshot_path=sketchy))
            assert "extension" in result.lower(), (
                f"expected extension rejection, got: {result!r}")
        finally:
            Path(sketchy).unlink()


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


class TestUvicornGracefulShutdown:
    """The MCP server runs under uvicorn. On SIGTERM, uvicorn's default is to
    wait forever for in-flight requests — which never completes for the SSE
    long-poll Claude Code holds open. The menubar then SIGKILLs after 5s,
    yanking the SSE stream mid-byte. Claude's transport hits the unhandled
    error path and the CLI exits cleanly without an error.

    Setting timeout_graceful_shutdown bounds the drain window so uvicorn
    closes the connection itself (still abrupt from the wire's perspective,
    but with a defined boundary) and runs the lifespan shutdown — which
    triggers FastMCP's session_manager.__aexit__ for any application-level
    session cleanup.
    """

    def test_build_uvicorn_config_sets_graceful_shutdown_timeout(self):
        captured = {}

        def fake_config(*args, **kwargs):
            captured.update(kwargs)
            return MagicMock()

        fake_uvicorn = MagicMock()
        fake_uvicorn.Config = fake_config
        with patch.dict(sys.modules, {"uvicorn": fake_uvicorn}):
            server._build_uvicorn_config(MagicMock(), "127.0.0.1", 8100, "info")

        assert "timeout_graceful_shutdown" in captured, \
            "uvicorn.Config must be built with a finite graceful-shutdown timeout"
        timeout = captured["timeout_graceful_shutdown"]
        assert isinstance(timeout, (int, float)) and 1 <= timeout <= 10, \
            f"timeout_graceful_shutdown should be 1..10s, got {timeout}"

    def test_main_wires_through_build_uvicorn_config(self):
        """main() must use the helper — otherwise the timeout drifts away
        the next time someone edits the inline uvicorn.Config call."""
        import inspect
        src = inspect.getsource(server.main)
        assert "_build_uvicorn_config" in src, \
            "main() must call _build_uvicorn_config so the graceful-shutdown " \
            "timeout stays wired in"


class TestGpuStatusShape:
    """app/menubar.py::gpu_active_counts consumes {backend: {"active": int}} —
    guard the real endpoint's shape so a server-side change can't silently
    break the warm-ping busy gate."""

    def test_gpu_status_shape_matches_menubar_probe(self):
        import asyncio
        with patch.object(server, "JSONResponse", side_effect=lambda d: d):
            data = asyncio.run(server._gpu_status(None))
        for backend in ("ollama", "mlx", "ds4"):
            assert isinstance(data[backend]["active"], int)


# ── ds4 discovery ───────────────────────────────────────────────────

def _fake_discovery_client(ds4_up, ds4_context_length=None):
    """AsyncClient stub: Ollama and MLX unreachable; ds4 configurable.
    Exercises the real discover_models control flow, not a mock of it."""
    class _Resp:
        status_code = 200
        def json(self):
            if ds4_context_length is None:
                return {}
            return {"data": [{"id": "glm-5.2",
                               "context_length": ds4_context_length}]}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            if url.startswith(server.DS4_URL):
                if ds4_up:
                    return _Resp()
                raise ConnectionError("ds4 down")
            raise ConnectionError("backend down")

        async def post(self, url, **kwargs):
            raise ConnectionError("backend down")

    return _Client


class TestDs4Discovery:
    def _discover(self, ds4_up, tmp_path, ds4_context_length=None):
        # MLX_SERVER_CONFIG must point away from the real user yaml: an
        # unmigrated ~/.config/mlx-server/config.yaml still lists glm-5.2
        # as an MLX served-name and would leak into the registry.
        from unittest.mock import MagicMock
        from pathlib import Path
        import lib.hf_scanner as hf_scanner
        with patch.object(server.httpx, "AsyncClient",
                          _fake_discovery_client(ds4_up, ds4_context_length)), \
             patch.object(server, "MLX_SERVER_CONFIG",
                          Path(tmp_path) / "absent.yaml"), \
             patch.object(hf_scanner, "scan_hf_cache",
                          MagicMock(return_value=[])):
            return asyncio.run(server.discover_models())

    def test_ds4_up_inserts_glm52_with_hardcoded_metadata(self, tmp_path):
        """ds4's /v1/models returns no params/vision metadata (this stub
        returns no body at all, exercising the DS4_CONTEXT fallback).
        Without the hardcoded params/vision, TASK_FILTERS min_active_b
        (reasoning: 10) and min_ctx (long_context: 64000) silently drop
        glm-5.2 from every task list."""
        models = self._discover(True, tmp_path)
        assert models["glm-5.2"] == {
            "backend": "ds4",
            "total_params_b": 380,
            "active_params_b": 32,
            "context": 131072,
            "vision": False,
        }

    def test_ds4_up_prefers_live_context_length(self, tmp_path):
        """A --ctx launch-flag drift must surface through discovery, not
        hide behind the DS4_CONTEXT constant."""
        models = self._discover(True, tmp_path, ds4_context_length=65536)
        assert models["glm-5.2"]["context"] == 65536

    def test_ds4_down_means_glm52_absent(self, tmp_path):
        """Same semantics as MLX-down today: unreachable backend, no model."""
        models = self._discover(False, tmp_path)
        assert "glm-5.2" not in models


class TestDs4YamlGate:
    """I2: mirrors app/profile-server.py's
    test_ds4_down_but_installed_keeps_glm52_absent. If a user's MLX yaml
    still lists glm-5.2 (unmigrated, or manually re-added per the rollback
    doc) and ds4 is provisioned but down, the MCP layer must not register
    it as an MLX-backed model — that would invite a stale 418GB cold load
    on a machine where ds4 is supposed to own the name."""

    def _discover(self, tmp_path, ds4_installed_value):
        from unittest.mock import MagicMock
        from pathlib import Path
        import lib.hf_scanner as hf_scanner

        yaml_path = Path(tmp_path) / "config.yaml"
        yaml_path.write_text(
            "models:\n"
            "  - model_path: mlx-community/GLM-5.2-4bit\n"
            "    model_type: lm\n"
            "    served_model_name: glm-5.2\n"
            "    context_length: 131072\n"
            "    on_demand: true\n")
        with patch.object(server.httpx, "AsyncClient",
                          _fake_discovery_client(False)), \
             patch.object(server, "MLX_SERVER_CONFIG", yaml_path), \
             patch.object(server, "ds4_installed",
                          return_value=ds4_installed_value), \
             patch.object(hf_scanner, "scan_hf_cache",
                          MagicMock(return_value=[])):
            return asyncio.run(server.discover_models())

    def test_ds4_down_but_installed_keeps_glm52_absent(self, tmp_path):
        models = self._discover(tmp_path, ds4_installed_value=True)
        assert "glm-5.2" not in models

    def test_ds4_not_installed_allows_mlx_glm52_fallback(self, tmp_path):
        models = self._discover(tmp_path, ds4_installed_value=False)
        assert "glm-5.2" in models
        assert models["glm-5.2"]["backend"] == "mlx"
