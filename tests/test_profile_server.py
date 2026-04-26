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
os.environ["SP_ALLOW_NO_AUTH"] = "1"  # tests run without a real token

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


class TestRemoteFetchAuth:
    """Client-mode fetches against the desktop must forward the bearer
    token — otherwise the desktop's auth-required profile server 403s."""

    def test_fetch_remote_models_forwards_auth(self):
        captured = {}
        def fake_get(url, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = []
            return resp
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "shared-secret"), \
             patch.object(ps, "OLLAMA_URL", "https://desk.tail.ts.net:11434"), \
             patch.object(ps.requests, "get", side_effect=fake_get):
            ps._fetch_remote_models()
        assert captured["url"].endswith("/api/models")
        assert captured["url"].startswith("https://"), (
            "tailscale serve only listens on https — http would always fail")
        assert captured["headers"] == {"Authorization": "Bearer shared-secret"}

    def test_fetch_remote_models_returns_none_on_403(self):
        """403 means we have no token / wrong token; don't pretend success."""
        def fake_get(url, headers=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 403
            return resp
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "wrong"), \
             patch.object(ps, "OLLAMA_URL", "https://desk.tail.ts.net:11434"), \
             patch.object(ps.requests, "get", side_effect=fake_get):
            assert ps._fetch_remote_models() is None

    def test_fetch_all_models_skips_local_hf_cache_in_remote_mode(self):
        """The laptop's HF cache is for the laptop, not the desktop. When
        we're routing to the desktop, surfacing local-only HF models as if
        they were available on the desktop misleads the user."""
        ollama_called = []
        mlx_called = []
        hf_called = []
        with patch.object(ps, "OLLAMA_URL", "https://desk.tail.ts.net:11434"), \
             patch.object(ps, "_fetch_remote_models", return_value=None), \
             patch.object(ps, "_fetch_ollama_models",
                          side_effect=lambda: (ollama_called.append(1) or {})), \
             patch.object(ps, "_fetch_mlx_models",
                          side_effect=lambda existing: (mlx_called.append(1) or {})), \
             patch.object(ps, "_fetch_hf_cache_models",
                          side_effect=lambda existing: (hf_called.append(1) or {})):
            ps._fetch_all_models()
        assert ollama_called and mlx_called
        assert not hf_called, (
            "Remote-mode fallback must not scan the local HF cache.")

    def test_fetch_all_models_includes_local_hf_in_offline_mode(self):
        called = []
        with patch.object(ps, "OLLAMA_URL", "http://localhost:11434"), \
             patch.object(ps, "_fetch_ollama_models", return_value={}), \
             patch.object(ps, "_fetch_mlx_models", return_value={}), \
             patch.object(ps, "_fetch_hf_cache_models",
                          side_effect=lambda existing: (called.append(1) or {})):
            ps._fetch_all_models()
        assert called, "Local mode should still scan local HF cache."


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

    def test_version_bump_preserves_custom_profiles(self, profiles_dir):
        old = {"version": 1, "active": "myconfig",
               "profiles": {
                   "everyday": {"tasks": {}},
                   "myconfig": {"label": "My Config", "tasks": {"code": "custom-model"},
                                "max_ram_gb": 128, "thinking": {"code": True}},
               }}
        (profiles_dir / "profiles.json").write_text(json.dumps(old))
        loaded = ps.load_profiles()
        assert loaded["version"] == ps.PROFILES_VERSION
        assert "myconfig" in loaded["profiles"]
        assert loaded["profiles"]["myconfig"]["tasks"]["code"] == "custom-model"
        assert loaded["profiles"]["myconfig"]["max_ram_gb"] == 128
        assert loaded["active"] == "myconfig"


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

    def test_api_profiles_save_persists_max_ram_and_thinking(self, client, profiles_dir):
        resp = client.post("/api/profiles", json={
            "name": "custom",
            "label": "Custom",
            "tasks": {"code": "qwen3:8b"},
            "thinking": {"code": True},
            "max_ram_gb": 128,
        })
        assert resp.status_code == 200
        data = ps.load_profiles()
        assert data["profiles"]["custom"]["max_ram_gb"] == 128
        assert data["profiles"]["custom"]["thinking"]["code"] is True

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

    def test_api_profiles_activate_reports_missing_ollama(self, client, profiles_dir):
        ps.save_profiles({
            "version": ps.PROFILES_VERSION,
            "active": None,
            "profiles": {"test": {"label": "Test",
                                  "tasks": {"code": "gone-model:7b"}}},
        })
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            resp = client.post("/api/profiles/test/activate")
        data = resp.get_json()
        missing_names = [m["name"] for m in data["missing"]]
        assert "gone-model:7b" in missing_names

    def test_api_test_unknown_tool(self, client):
        resp = client.post("/api/test", json={"tool": "nonexistent"})
        assert resp.status_code == 400
        assert "Unknown tool" in resp.get_json()["error"]

    def test_api_test_code_round_trip(self, client):
        """End-to-end: /api/test?tool=code picks the profile's code model,
        posts to Ollama's chat endpoint with the user prompt, and returns the
        parsed content. Mocks at the HTTP boundary, not at _chat."""
        prefs = {"code": ["qwen3:8b"], "general": ["qwen3:8b"]}
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"message": {"content": "Hello!"}}
        fake_resp.raise_for_status = MagicMock()
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs), \
             patch.object(ps.requests, "post", return_value=fake_resp) as mock_post:
            resp = client.post("/api/test", json={"tool": "code", "prompt": "say hi"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["result"] == "Hello!"
        assert data["model"] == "qwen3:8b"
        assert mock_post.call_count == 1
        url = mock_post.call_args[0][0]
        payload = mock_post.call_args[1]["json"]
        assert url.endswith("/api/chat")
        assert payload["model"] == "qwen3:8b"
        assert payload["messages"] == [{"role": "user", "content": "say hi"}]
        assert payload["stream"] is False
        assert payload["keep_alive"] == "30m"

    def test_api_test_override_round_trip(self, client):
        """Override model flows all the way through to the HTTP request."""
        prefs = {"code": ["qwen3:8b"]}
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {"message": {"content": "override result"}}
        fake_resp.raise_for_status = MagicMock()
        with patch.object(ps, "get_all_models", return_value=FAKE_MODELS), \
             patch.object(ps, "load_default_prefs", return_value=prefs), \
             patch.object(ps.requests, "post", return_value=fake_resp) as mock_post:
            resp = client.post("/api/test", json={
                "tool": "code", "prompt": "hi", "model": "llama3:70b"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["result"] == "override result"
        assert data["model"] == "llama3:70b"
        assert mock_post.call_args[1]["json"]["model"] == "llama3:70b"

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
    """Bearer token auth — required on every request, no localhost shortcut.

    Tailscale serve forwards remote requests as if they came from 127.0.0.1,
    so trusting the loopback address would silently bypass auth for any
    tailnet peer.
    """

    def test_localhost_without_token_rejected(self, client):
        """No localhost shortcut — a request without the token gets 403
        even from 127.0.0.1, because Tailscale serve makes remote requests
        look local."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system")
        assert resp.status_code == 403

    def test_localhost_with_token_allowed(self, client):
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get(
                "/api/system",
                headers={"Authorization": "Bearer secret-token"})
        assert resp.status_code == 200

    def test_remote_without_token_rejected(self, client):
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system", headers={},
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 403

    def test_remote_with_correct_token_allowed(self, client):
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system",
                              headers={"Authorization": "Bearer secret-token"},
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 200

    def test_remote_with_wrong_token_rejected(self, client):
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/system",
                              headers={"Authorization": "Bearer wrong"},
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 403

    def test_static_pages_require_auth(self, client):
        """HTML pages also require the bearer.  The menubar's WKWebView sets
        Authorization on the initial NSURLRequest; tailnet peers without the
        token see 403."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            for path in ("/", "/profiles", "/tools"):
                resp = client.get(path,
                                  environ_base={"REMOTE_ADDR": "100.64.0.5"})
                assert resp.status_code == 403, f"{path} must require auth"

    def test_identity_route_exempt(self, client):
        """/api/identity is the orphan-detection handshake — its per-launch
        token is the auth, so the route is exempt from bearer checks."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get("/api/identity",
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 200

    def test_query_param_token_accepted_for_get(self, client):
        """Native <img>/<audio>/<video> can't set headers — accept ?token=
        for GETs only."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.get(
                "/api/system?token=secret-token",
                environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 200

    def test_query_param_token_rejected_for_post(self, client):
        """POSTs (mutation) ignore ?token= — only header counts."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "secret-token"):
            resp = client.post(
                "/api/profiles?token=secret-token",
                json={},
                environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 403

    def test_no_token_configured_fails_closed(self, client):
        """When no token is set and SP_ALLOW_NO_AUTH is not enabled, every
        request gets 503 (refused).  Production startup also exits before
        reaching this state — this is the runtime defense."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", ""), \
             patch.object(ps, "_ALLOW_NO_AUTH", False):
            resp = client.get("/api/system",
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 503

    def test_no_token_configured_with_explicit_dev_flag_allows_all(self, client):
        """SP_ALLOW_NO_AUTH=1 is the explicit escape hatch for unit tests
        and local dev."""
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", ""), \
             patch.object(ps, "_ALLOW_NO_AUTH", True):
            resp = client.get("/api/system",
                              environ_base={"REMOTE_ADDR": "100.64.0.5"})
        assert resp.status_code == 200


class TestClientModeMediaProxy:
    """In client mode, media-serving endpoints (/api/test/image|audio|video)
    must forward to the desktop's profile server. The path returned by the
    desktop's tool handlers refers to a file on the desktop's filesystem,
    so the laptop has to fetch it through the desktop, not from local /tmp.
    """

    @staticmethod
    def _fake_media_response(content_type, payload=b"\x89PNG\r\n\x1a\nFAKE"):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": content_type}
        resp.iter_content = lambda chunk_size=4096: iter([payload])
        resp.content = payload
        return resp

    def test_image_route_proxies_in_client_mode(self, client):
        fake = self._fake_media_response("image/png")
        with patch.object(ps, "OLLAMA_URL", "http://100.64.0.2:11434"), \
             patch.object(ps.requests, "get", return_value=fake) as mock_get:
            resp = client.get("/api/test/image?path=/tmp/test_image_1.png")
        assert resp.status_code == 200
        assert resp.data == b"\x89PNG\r\n\x1a\nFAKE"
        assert resp.headers["Content-Type"] == "image/png"
        url = mock_get.call_args[0][0]
        assert url == "https://100.64.0.2:8101/api/test/image"
        params = mock_get.call_args[1]["params"]
        assert params["path"] == "/tmp/test_image_1.png"
        assert mock_get.call_args[1]["headers"]["X-SP-Proxy-Hops"] == "1"

    def test_audio_route_proxies_in_client_mode(self, client):
        fake = self._fake_media_response("audio/wav", b"RIFF\x00\x00\x00\x00WAVE")
        with patch.object(ps, "OLLAMA_URL", "http://100.64.0.2:11434"), \
             patch.object(ps.requests, "get", return_value=fake) as mock_get:
            resp = client.get("/api/test/audio?path=/tmp/test_speech.wav")
        assert resp.status_code == 200
        assert resp.data == b"RIFF\x00\x00\x00\x00WAVE"
        assert mock_get.call_args[0][0] == "https://100.64.0.2:8101/api/test/audio"

    def test_video_route_proxies_in_client_mode(self, client):
        fake = self._fake_media_response("video/mp4", b"\x00\x00\x00\x18ftypmp42FAKE")
        with patch.object(ps, "OLLAMA_URL", "http://100.64.0.2:11434"), \
             patch.object(ps.requests, "get", return_value=fake) as mock_get:
            resp = client.get("/api/test/video?path=/tmp/test_video.mp4")
        assert resp.status_code == 200
        assert resp.data == b"\x00\x00\x00\x18ftypmp42FAKE"
        assert mock_get.call_args[0][0] == "https://100.64.0.2:8101/api/test/video"

    def test_local_mode_serves_from_disk(self, client, tmp_path):
        """When OLLAMA_URL is local, the route should NOT proxy — it should
        read the file from /tmp directly. (We use /tmp to satisfy
        _is_safe_test_path.)"""
        target = Path("/tmp/test_image_local.png")
        target.write_bytes(b"localdata")
        try:
            with patch.object(ps, "OLLAMA_URL", "http://localhost:11434"), \
                 patch.object(ps.requests, "get") as mock_get:
                resp = client.get(f"/api/test/image?path={target}")
            assert resp.status_code == 200
            assert resp.data == b"localdata"
            assert mock_get.call_count == 0
        finally:
            target.unlink(missing_ok=True)

    def test_proxy_loop_guard(self, client):
        """If a proxied request comes back to us with too many hops, we
        refuse to proxy again."""
        with patch.object(ps, "OLLAMA_URL", "http://100.64.0.2:11434"):
            resp = client.get("/api/test/image?path=/tmp/x.png",
                              headers={"X-SP-Proxy-Hops": "3"})
        assert resp.status_code == 502
        assert "loop" in resp.get_json()["error"].lower()


class TestClientModeUploadProxy:
    """In client mode, /api/test/upload must forward the multipart body to
    the desktop so the saved path is on the desktop's filesystem (where the
    backends will read it from)."""

    def test_upload_proxies_multipart_in_client_mode(self, client):
        fake = MagicMock()
        fake.status_code = 200
        fake.headers = {"content-type": "application/json"}
        fake.content = b'{"path": "/tmp/test_upload_999.png"}'
        with patch.object(ps, "OLLAMA_URL", "http://100.64.0.2:11434"), \
             patch.object(ps.requests, "post", return_value=fake) as mock_post:
            resp = client.post(
                "/api/test/upload",
                data={"file": (Path("/dev/null").open("rb"), "screenshot.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert resp.get_json()["path"] == "/tmp/test_upload_999.png"
        url = mock_post.call_args[0][0]
        assert url == "https://100.64.0.2:8101/api/test/upload"
        # The raw multipart body must be forwarded, not re-encoded as JSON
        kwargs = mock_post.call_args[1]
        assert "data" in kwargs and kwargs["data"]
        assert "json" not in kwargs
        assert kwargs["headers"]["Content-Type"].startswith("multipart/form-data")
        assert kwargs["headers"]["X-SP-Proxy-Hops"] == "1"

    def test_upload_local_mode_saves_to_tmp(self, client):
        """In local mode, the upload is saved to /tmp and its path returned —
        no proxy involved."""
        with patch.object(ps, "OLLAMA_URL", "http://localhost:11434"), \
             patch.object(ps.requests, "post") as mock_post:
            resp = client.post(
                "/api/test/upload",
                data={"file": (Path("/dev/null").open("rb"), "screenshot.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        path = resp.get_json()["path"]
        assert path.startswith("/tmp/test_upload_")
        assert path.endswith(".png")
        assert mock_post.call_count == 0
        Path(path).unlink(missing_ok=True)

    def test_upload_loop_guard(self, client):
        with patch.object(ps, "OLLAMA_URL", "http://100.64.0.2:11434"):
            resp = client.post(
                "/api/test/upload",
                data={"file": (Path("/dev/null").open("rb"), "x.png")},
                content_type="multipart/form-data",
                headers={"X-SP-Proxy-Hops": "3"},
            )
        assert resp.status_code == 502
        assert "loop" in resp.get_json()["error"].lower()


class TestPlaygroundModelAllowlist:
    """The Playground model override used to accept any HF repo path —
    POST {"tool": "speak", "model": "evil/repo"} would force an HF download
    of an arbitrary model into the user's cache.  These tests pin the
    allowlist gate (downloaded ∪ in-profile ∪ in-prefs)."""

    def test_known_downloaded_model_accepted(self, tmp_path):
        """A model that's already in the HF cache is accepted."""
        with patch.object(ps, "_hf_model_downloaded", return_value=True):
            assert ps._hf_model_is_known("foo/bar") is True

    def test_unknown_repo_rejected(self):
        """An HF repo that's neither cached nor configured is rejected."""
        with patch.object(ps, "_hf_model_downloaded", return_value=False), \
             patch.object(ps, "load_profiles",
                          return_value={"profiles": {}}), \
             patch.object(ps, "load_default_prefs", return_value={}):
            assert ps._hf_model_is_known("evil/random-repo") is False

    def test_model_in_profile_accepted(self):
        """A model listed in some profile's task is accepted even if not
        cached — the operator opted into it."""
        with patch.object(ps, "_hf_model_downloaded", return_value=False), \
             patch.object(ps, "load_profiles", return_value={
                 "profiles": {
                     "test": {"tasks": {"tts": "org/special-tts"}},
                 },
             }), \
             patch.object(ps, "load_default_prefs", return_value={}):
            assert ps._hf_model_is_known("org/special-tts") is True

    def test_model_in_prefs_accepted(self):
        """A model listed in mcp_preferences is accepted (user added it
        to the candidate list, even if not currently downloaded)."""
        with patch.object(ps, "_hf_model_downloaded", return_value=False), \
             patch.object(ps, "load_profiles",
                          return_value={"profiles": {}}), \
             patch.object(ps, "load_default_prefs",
                          return_value={"tts": ["org/voice"]}):
            assert ps._hf_model_is_known("org/voice") is True

    def test_non_hf_path_rejected(self):
        """Names without `/` aren't HF repos — gate doesn't apply."""
        assert ps._hf_model_is_known("qwen3.5:9b") is False
        assert ps._hf_model_is_known("") is False
        assert ps._hf_model_is_known(None) is False


class TestUploadHardening:
    """The /api/test/upload route was a sharp edge — it took a multipart
    filename and used its suffix verbatim as the on-disk extension, with
    no size cap.  These tests pin the new defences."""

    def _post(self, client, name, data=b"x"):
        from io import BytesIO
        with patch.object(ps, "OLLAMA_URL", "http://localhost:11434"):
            return client.post(
                "/api/test/upload",
                data={"file": (BytesIO(data), name)},
                content_type="multipart/form-data",
            )

    def test_random_basename(self, client):
        """Saved filename must NOT be predictable from the upload — random
        token, not a timestamp."""
        resp = self._post(client, "screenshot.png", b"PNGDATA")
        assert resp.status_code == 200
        path = resp.get_json()["path"]
        # 16-char hex token (secrets.token_hex(8) → 16 chars)
        assert path.startswith("/tmp/test_upload_")
        assert path.endswith(".png")
        # Reject the old timestamp-based pattern: those were all digits
        basename = path.removeprefix("/tmp/test_upload_").removesuffix(".png")
        assert not basename.isdigit(), \
            f"basename {basename!r} looks like the old timestamp pattern"
        Path(path).unlink(missing_ok=True)

    def test_path_traversal_in_filename_stripped(self, client):
        """A multipart filename like '../../etc/foo.png' must NOT escape /tmp."""
        resp = self._post(client, "../../etc/passwd.png", b"x")
        assert resp.status_code == 200
        path = resp.get_json()["path"]
        assert path.startswith("/tmp/test_upload_")
        assert "/etc/" not in path
        assert ".." not in path
        Path(path).unlink(missing_ok=True)

    def test_disallowed_extension_rejected(self, client):
        for bad_ext in (".dylib", ".plist", ".so", ".sh", ".py", ""):
            resp = self._post(client, f"evil{bad_ext}", b"x")
            assert resp.status_code == 400, \
                f"{bad_ext!r} should be rejected, got {resp.status_code}"
            assert "not allowed" in resp.get_json()["error"]

    def test_extension_check_is_case_insensitive(self, client):
        resp = self._post(client, "PHOTO.PNG", b"x")
        assert resp.status_code == 200
        Path(resp.get_json()["path"]).unlink(missing_ok=True)

    def test_oversize_payload_rejected(self, client):
        """Payload over the size cap is rejected and the partial file is
        deleted, not left dangling."""
        big = b"\x00" * (ps._UPLOAD_MAX_BYTES + 1024)
        resp = self._post(client, "big.png", big)
        assert resp.status_code == 413
        assert "limit" in resp.get_json()["error"].lower()
        # No leftover from the truncated write
        leftover = list(Path("/tmp").glob("test_upload_*.png"))
        for p in leftover:
            # Anything left over should be small (from other tests), not the big payload
            assert p.stat().st_size <= ps._UPLOAD_MAX_BYTES, \
                f"oversize file {p} was not cleaned up"
