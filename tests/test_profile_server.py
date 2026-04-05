"""Unit tests for the profile server: routes, model selection, profiles CRUD.

Tests use Flask's test client — no live Ollama/MLX needed.
"""

import contextlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub mlx_audio before importing profile-server (it imports at top level
# on macOS but isn't needed for testing)
for mod in ("mlx_audio", "mlx_audio.tts"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

# Patch env vars and heavy I/O before import
import importlib.util
import os

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("MLX_URL", "http://localhost:8000")
os.environ["PROFILE_IDLE_TIMEOUT"] = "0"  # disable idle shutdown

# Stub hf_scanner to avoid scanning the real HF cache
hf_scanner_mock = MagicMock()
hf_scanner_mock.scan_hf_cache = MagicMock(return_value=[])
sys.modules["lib.hf_scanner"] = hf_scanner_mock

_ps_path = Path(__file__).resolve().parent.parent / "app" / "profile-server.py"
spec = importlib.util.spec_from_file_location("profile_server", str(_ps_path))
ps = importlib.util.module_from_spec(spec)
sys.modules["profile_server"] = ps
spec.loader.exec_module(ps)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Flask test client."""
    ps.app.config["TESTING"] = True
    with ps.app.test_client() as c:
        yield c


@pytest.fixture()
def profiles_dir(tmp_path):
    """Temp directory for profile and pref files."""
    pf = tmp_path / "profiles.json"
    mf = tmp_path / "mcp_prefs.json"
    with patch.object(ps, "PROFILES_FILE", pf), \
         patch.object(ps, "MCP_PREFS_FILE", mf):
        yield tmp_path


FAKE_MODELS = {
    "qwen3:8b": {
        "name": "qwen3:8b", "backend": "ollama",
        "active_params_b": 8, "context": 32768,
        "has_vision": False, "family": "qwen3",
        "disk_bytes": 5_000_000_000, "vram_bytes": 6_000_000_000,
        "total_params_b": 8, "quant": "Q4_K_M",
        "is_loaded": True, "expires_at": None,
    },
    "llama3:70b": {
        "name": "llama3:70b", "backend": "ollama",
        "active_params_b": 70, "context": 8192,
        "has_vision": False, "family": "llama",
        "disk_bytes": 40_000_000_000, "vram_bytes": 45_000_000_000,
        "total_params_b": 70, "quant": "Q4_K_M",
        "is_loaded": False, "expires_at": None,
    },
    "qwen3-vl:32b": {
        "name": "qwen3-vl:32b", "backend": "ollama",
        "active_params_b": 32, "context": 32768,
        "has_vision": True, "family": "qwen3",
        "disk_bytes": 18_000_000_000, "vram_bytes": 20_000_000_000,
        "total_params_b": 32, "quant": "Q4_K_M",
        "is_loaded": False, "expires_at": None,
    },
    "whisper-v3": {
        "name": "whisper-v3", "backend": "mlx",
        "active_params_b": 1.5, "context": 0,
        "has_vision": False, "family": "transcription",
        "disk_bytes": 1_500_000_000, "vram_bytes": 1_500_000_000,
        "total_params_b": 1.5, "quant": "",
        "is_loaded": True, "expires_at": None,
    },
    "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16": {
        "name": "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16", "backend": "mlx-audio",
        "active_params_b": 4, "context": 0,
        "has_vision": False, "family": "tts",
        "disk_bytes": 4_000_000_000, "vram_bytes": 4_000_000_000,
        "total_params_b": 4, "quant": "bf16",
        "is_loaded": False, "expires_at": None,
    },
}


# ── Pure functions ──────────────────────────────────────────────────

class TestChatUrl:
    def test_mlx_backend(self):
        assert ps._chat_url("mlx") == "http://localhost:8000/v1/chat/completions"

    def test_ollama_backend(self):
        assert ps._chat_url("ollama") == "http://localhost:11434/api/chat"


class TestIsRemoteOllama:
    def test_localhost(self):
        with patch.object(ps, "OLLAMA_URL", "http://localhost:11434"):
            assert ps._is_remote_ollama() is False

    def test_loopback(self):
        with patch.object(ps, "OLLAMA_URL", "http://127.0.0.1:11434"):
            assert ps._is_remote_ollama() is False

    def test_remote(self):
        with patch.object(ps, "OLLAMA_URL", "http://100.64.0.2:11434"):
            assert ps._is_remote_ollama() is True


class TestRequestsErrorDetail:
    def test_http_error_model_not_found(self):
        resp = MagicMock()
        resp.status_code = 404
        resp.text = '{"error":"model \'qwen3:8b\' not found, try pulling it"}'
        resp.url = "http://localhost:11434/api/chat"
        exc = ps.requests.HTTPError(response=resp)
        result = ps._requests_error_detail(exc)
        assert "not downloaded" in result
        assert "ollama pull qwen3:8b" in result

    def test_http_error_generic(self):
        resp = MagicMock()
        resp.status_code = 503
        resp.text = "Service Unavailable"
        resp.url = "http://localhost:8000"
        exc = ps.requests.HTTPError(response=resp)
        result = ps._requests_error_detail(exc)
        assert "503" in result
        assert "localhost:8000" in result

    def test_connection_error(self):
        exc = ps.requests.ConnectionError("refused")
        result = ps._requests_error_detail(exc)
        assert "Cannot connect" in result

    def test_timeout(self):
        exc = ps.requests.Timeout("timed out")
        result = ps._requests_error_detail(exc)
        assert "timed out" in result


class TestGetEligibleTasks:
    def test_small_ollama_model(self):
        tasks = ps.get_eligible_tasks("qwen3:8b", FAKE_MODELS["qwen3:8b"])
        assert "code" in tasks or "general" in tasks

    def test_vision_model_includes_vision(self):
        tasks = ps.get_eligible_tasks("qwen3-vl:32b", FAKE_MODELS["qwen3-vl:32b"])
        assert "vision" in tasks

    def test_whisper_includes_transcription(self):
        tasks = ps.get_eligible_tasks("whisper-v3", FAKE_MODELS["whisper-v3"])
        assert "transcription" in tasks

    def test_non_llm_backend_skips_task_filters(self):
        model = {**FAKE_MODELS["whisper-v3"], "backend": "mlx-audio"}
        tasks = ps.get_eligible_tasks("whisper-v3", model)
        assert "code" not in tasks
        assert "general" not in tasks


class TestPickModelForTask:
    def test_picks_preferred_model(self):
        with patch.object(ps, "load_default_prefs",
                          return_value={"code": ["qwen3:8b"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("code")
        assert name == "qwen3:8b"
        assert backend == "ollama"
        assert warning is None

    def test_prefix_match(self):
        with patch.object(ps, "load_default_prefs",
                          return_value={"code": ["qwen3"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("code")
        assert name == "qwen3:8b"

    def test_fallback_when_missing(self):
        with patch.object(ps, "load_default_prefs",
                          return_value={"code": ["nonexistent"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("code")
        assert name is None
        assert "not available" in warning

    def test_no_prefs_returns_none(self):
        with patch.object(ps, "load_default_prefs", return_value={}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("code")
        assert name is None
        assert warning is None


# ── Profiles CRUD ───────────────────────────────────────────────────

class TestProfilesCRUD:
    def test_load_creates_default(self, profiles_dir):
        data = ps.load_profiles()
        assert "profiles" in data
        assert "everyday" in data["profiles"]
        assert (profiles_dir / "profiles.json").exists()

    def test_save_and_load_roundtrip(self, profiles_dir):
        custom = {"version": ps.PROFILES_VERSION, "active": "test",
                  "profiles": {"test": {"label": "Test", "tasks": {}}}}
        ps.save_profiles(custom)
        loaded = ps.load_profiles()
        assert loaded["active"] == "test"
        assert "test" in loaded["profiles"]

    def test_version_bump_refreshes_presets(self, profiles_dir):
        old = {"version": 1, "active": "everyday",
               "profiles": {"everyday": {"tasks": {}}}}
        (profiles_dir / "profiles.json").write_text(json.dumps(old))
        loaded = ps.load_profiles()
        assert loaded["version"] == ps.PROFILES_VERSION
        assert "everyday" in loaded["profiles"]


# ── Flask routes ────────────────────────────────────────────────────

class TestRoutes:
    def test_api_system(self, client):
        with patch.object(ps, "get_system_info",
                          return_value={"total_ram_gb": 512, "mode": "server"}):
            resp = client.get("/api/system")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_ram_gb"] == 512

    def test_api_tasks(self, client):
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "code" in data
        assert "vision" in data
        assert "label" in data["code"]

    def test_api_models(self, client):
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.get_json()
        names = {m["name"] for m in data}
        assert "qwen3:8b" in names
        assert all("eligible_tasks" in m for m in data)

    def test_api_profiles_get(self, client, profiles_dir):
        resp = client.get("/api/profiles")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "profiles" in data

    def test_api_profiles_save(self, client, profiles_dir):
        resp = client.post("/api/profiles", json={
            "name": "custom",
            "label": "Custom",
            "description": "Test profile",
            "tasks": {"code": "qwen3:8b"},
        })
        assert resp.status_code == 200
        data = ps.load_profiles()
        assert "custom" in data["profiles"]
        assert data["profiles"]["custom"]["tasks"]["code"] == "qwen3:8b"

    def test_api_profiles_save_requires_name(self, client, profiles_dir):
        resp = client.post("/api/profiles", json={"label": "No name"})
        assert resp.status_code == 400

    def test_api_profiles_delete(self, client, profiles_dir):
        ps.save_profiles({
            "version": ps.PROFILES_VERSION,
            "active": "doomed",
            "profiles": {"doomed": {"label": "Doomed", "tasks": {}}},
        })
        resp = client.delete("/api/profiles/doomed")
        assert resp.status_code == 200
        data = ps.load_profiles()
        assert "doomed" not in data["profiles"]
        assert data["active"] is None  # cleared since active was deleted

    def test_api_profiles_activate(self, client, profiles_dir):
        ps.save_profiles({
            "version": ps.PROFILES_VERSION,
            "active": None,
            "profiles": {"test": {"label": "Test", "tasks": {"code": "qwen3:8b"}}},
        })
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            resp = client.post("/api/profiles/test/activate")
        assert resp.status_code == 200
        data = ps.load_profiles()
        assert data["active"] == "test"

    def test_api_profiles_activate_not_found(self, client, profiles_dir):
        resp = client.post("/api/profiles/nonexistent/activate")
        assert resp.status_code == 404

    def test_api_profiles_activate_prunes_stale(self, client, profiles_dir):
        ps.save_profiles({
            "version": ps.PROFILES_VERSION,
            "active": None,
            "profiles": {"test": {"label": "Test",
                                  "tasks": {"code": "gone-model:7b"}}},
        })
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            resp = client.post("/api/profiles/test/activate")
        data = resp.get_json()
        assert len(data["warnings"]) > 0
        assert "gone-model" in data["warnings"][0]

    def test_api_test_unknown_tool(self, client):
        resp = client.post("/api/test", json={"tool": "nonexistent"})
        assert resp.status_code == 400
        assert "Unknown tool" in resp.get_json()["error"]

    def test_api_test_code_dispatches_to_chat(self, client):
        prefs = {"code": ["qwen3:8b"], "general": ["qwen3:8b"]}
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs), \
             patch.object(ps, "_chat", return_value="Hello!") as mock_chat:
            resp = client.post("/api/test", json={"tool": "code", "prompt": "say hi"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["result"] == "Hello!"
        mock_chat.assert_called_once()
        args = mock_chat.call_args
        assert args[0][0] == "qwen3:8b"
        assert args[0][1] == "ollama"

    def test_api_test_review_dispatches_to_chat(self, client):
        prefs = {"reasoning": ["llama3:70b"]}
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs), \
             patch.object(ps, "_chat", return_value="Looks good") as mock_chat:
            resp = client.post("/api/test", json={"tool": "review", "code": "x = 1"})
        assert resp.status_code == 200
        assert resp.get_json()["result"] == "Looks good"
        mock_chat.assert_called_once()

    def test_api_test_override_warns_on_missing_model(self, client):
        prefs = {"code": ["qwen3:8b"]}
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs), \
             patch.object(ps, "_chat", return_value="result"):
            resp = client.post("/api/test", json={
                "tool": "code", "prompt": "hi", "model": "nonexistent"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "warning" in data
        assert "not found" in data["warning"]

    def test_api_test_override_uses_specified_model(self, client):
        prefs = {"code": ["qwen3:8b"]}
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs), \
             patch.object(ps, "_chat", return_value="result") as mock_chat:
            resp = client.post("/api/test", json={
                "tool": "code", "prompt": "hi", "model": "llama3:70b"})
        assert resp.status_code == 200
        args = mock_chat.call_args
        assert args[0][0] == "llama3:70b"

    def test_api_test_no_model_available_returns_error(self, client):
        with patch.object(ps, "get_all_models", return_value={}), \
             patch.object(ps, "load_default_prefs", return_value={"code": []}):
            resp = client.post("/api/test", json={"tool": "code", "prompt": "hi"})
        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_api_test_speak_ref_audio_rejects_bad_path(self, client):
        prefs = {"tts": ["mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"]}
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs):
            resp = client.post("/api/test", json={
                "tool": "speak", "text": "hello",
                "ref_audio": "/Users/jerry/.ssh/id_rsa",
            })
        assert resp.status_code == 403
        assert "restricted" in resp.get_json()["error"].lower()

    def test_api_test_speak_ref_audio_selects_chatterbox(self, client):
        prefs = {"tts": ["mlx-community/Voxtral-4B-TTS-2603-mlx-bf16"]}
        mock_gen = MagicMock()
        mock_module = MagicMock()
        mock_module.generate_audio = mock_gen
        import sys
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs), \
             patch.object(ps, "_is_safe_test_path", return_value=True), \
             patch.dict(sys.modules, {"mlx_audio": MagicMock(),
                                      "mlx_audio.tts": MagicMock(),
                                      "mlx_audio.tts.generate": mock_module}):
            resp = client.post("/api/test", json={
                "tool": "speak", "text": "hello",
                "ref_audio": "/tmp/ref.wav",
            })
        mock_gen.assert_called_once()
        kwargs = mock_gen.call_args[1]
        assert "chatterbox" in kwargs.get("model", "")
        assert kwargs.get("ref_audio") == "/tmp/ref.wav"

    def test_tools_page(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200


class TestReadServerRamGb:
    def test_reads_value(self, tmp_path):
        conf = tmp_path / "network.conf"
        conf.write_text("SERVER_RAM_GB=512\n")
        with patch.object(ps, "NETWORK_CONF", conf):
            assert ps._read_server_ram_gb() == 512

    def test_zero_returns_none(self, tmp_path):
        conf = tmp_path / "network.conf"
        conf.write_text("SERVER_RAM_GB=0\n")
        with patch.object(ps, "NETWORK_CONF", conf):
            assert ps._read_server_ram_gb() is None

    def test_missing_file(self, tmp_path):
        with patch.object(ps, "NETWORK_CONF", tmp_path / "nope"):
            assert ps._read_server_ram_gb() is None

    def test_strips_quotes(self, tmp_path):
        conf = tmp_path / "network.conf"
        conf.write_text('SERVER_RAM_GB="256"\n')
        with patch.object(ps, "NETWORK_CONF", conf):
            assert ps._read_server_ram_gb() == 256

    def test_strips_unit_suffix(self, tmp_path):
        conf = tmp_path / "network.conf"
        conf.write_text("SERVER_RAM_GB=512GB\n")
        with patch.object(ps, "NETWORK_CONF", conf):
            assert ps._read_server_ram_gb() == 512


class TestLoadDefaultPrefs:
    def test_missing_file(self, tmp_path):
        with patch.object(ps, "MCP_PREFS_FILE", tmp_path / "nope.json"):
            assert ps.load_default_prefs() == {}

    def test_string_promoted_to_list(self, tmp_path):
        f = tmp_path / "prefs.json"
        f.write_text('{"code": "qwen3"}')
        with patch.object(ps, "MCP_PREFS_FILE", f):
            prefs = ps.load_default_prefs()
        assert prefs["code"] == ["qwen3"]

    def test_list_preserved(self, tmp_path):
        f = tmp_path / "prefs.json"
        f.write_text('{"code": ["a", "b"]}')
        with patch.object(ps, "MCP_PREFS_FILE", f):
            prefs = ps.load_default_prefs()
        assert prefs["code"] == ["a", "b"]


class TestProfileServerAuth:
    """Bearer token auth for remote access."""

    def test_localhost_skips_auth(self, client):
        """Requests from 127.0.0.1 should work without a token."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system")
        assert resp.status_code == 200

    def test_remote_without_token_rejected(self, client):
        """Remote requests without a token should get 403."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            # Flask test client defaults to 127.0.0.1, so we need to
            # simulate a remote request by patching request.remote_addr
            with client.application.test_request_context(
                    "/api/system", environ_base={"REMOTE_ADDR": "100.64.0.5"}):
                from flask import request as flask_req
                resp = client.application.full_dispatch_request()
                # The before_request hook checks remote_addr
        # Direct test: simulate via the app's test machinery
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system", headers={},
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 403

    def test_remote_with_correct_token_allowed(self, client):
        """Remote requests with correct bearer token should work."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system",
                              headers={"Authorization": "Bearer secret-token"},
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 200

    def test_remote_with_wrong_token_rejected(self, client):
        """Remote requests with wrong token should get 403."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system",
                              headers={"Authorization": "Bearer wrong"},
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 403

    def test_static_pages_skip_auth(self, client):
        """HTML pages should load without auth even from remote."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            for path in ("/", "/profiles", "/tools"):
                resp = client.get(path, environ_base={"REMOTE_ADDR": "100.64.0.5"})
                assert resp.status_code == 200, f"{path} should skip auth"

    def test_no_token_configured_allows_all(self, client):
        """When no token is set, all requests are allowed (dev mode)."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", ""):
            resp = client.get("/api/system",
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 200

    def test_auth_token_endpoint_localhost_only(self, client):
        """Token bootstrap endpoint only works from localhost."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            # Localhost: should return token
            resp = client.get("/api/auth-token")
            assert resp.status_code == 200
            assert resp.get_json()["token"] == "secret-token"

            # Remote: should be rejected
            resp = client.get("/api/auth-token",
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
            assert resp.status_code == 403
