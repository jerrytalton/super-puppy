"""Unit tests for the profile server: routes, model selection, profiles CRUD.

Tests use Flask's test client — no live Ollama/MLX needed.
"""

import contextlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# NB: do NOT stub mlx_audio here. profile-server imports it lazily inside
# the TTS handler, so the import below doesn't need it — and a module-level
# stub never comes back out of sys.modules. This file sorts before
# test_tools_*, so the leaked MagicMock made the smoke suite's real-mlx_audio
# guard pass against a stub, running "live" TTS against a mock in the release
# gate (and skipping when run standalone). The one test that exercises TTS
# patches sys.modules itself.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

# Patch env vars and heavy I/O before import
import importlib.util
import os

os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
os.environ.setdefault("MLX_URL", "http://localhost:8000")
os.environ["PROFILE_IDLE_TIMEOUT"] = "0"  # disable idle shutdown
os.environ["SP_ALLOW_NO_AUTH"] = "1"  # tests run without a real token

# Stub hf_scanner to avoid scanning the real HF cache — scoped to the import
# below, since the smoke suite deliberately wants the REAL scanner (it stubs
# only the slow full-cache scan) and a leaked MagicMock would hand its video
# handler a mock snapshot helper.
from tests._stub_helpers import stubbed_modules  # noqa: E402

hf_scanner_mock = MagicMock()
hf_scanner_mock.scan_hf_cache = MagicMock(return_value=[])

_ps_path = Path(__file__).resolve().parent.parent / "app" / "profile-server.py"
with stubbed_modules({"lib.hf_scanner": hf_scanner_mock}):
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
    # An Ollama tag that advertises capabilities: ["image"] and cannot serve
    # it — Ollama 0.32 returns 400 from /api/generate for these.
    "x/flux2-klein:latest": {
        "name": "x/flux2-klein:latest", "backend": "ollama",
        "active_params_b": 9, "context": 0,
        "has_vision": False, "family": "flux",
        "disk_bytes": 6_000_000_000, "vram_bytes": 6_000_000_000,
        "total_params_b": 9, "quant": "bf16",
        "is_loaded": False, "expires_at": None,
    },
    "black-forest-labs/FLUX.2-klein-4B": {
        "name": "black-forest-labs/FLUX.2-klein-4B", "backend": "mflux",
        "active_params_b": 4, "context": 0,
        "has_vision": False, "family": "image_gen",
        "disk_bytes": 23_700_000_000, "vram_bytes": 23_700_000_000,
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

    def test_ds4_backend(self):
        assert ps._chat_url("ds4") == "http://localhost:8002/v1/chat/completions"


class TestChatDs4Dispatch:
    def _resp(self, text):
        r = MagicMock()
        r.text = text
        r.raise_for_status = MagicMock()
        return r

    def test_chat_ds4_posts_to_ds4_without_mlx_shim(self):
        """think=False must NOT forward chat_template_kwargs to ds4
        (verified broken: reasoning migrates into content), and the reply
        must survive raw control chars in reasoning_content (ds4 encoder
        bug — strict resp.json() raises)."""
        raw = ('{"choices":[{"message":{"content":"pong",'
               '"reasoning_content":"pondering\x01deeply"}}]}')
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json
            return self._resp(raw)

        with patch.object(ps.requests, "post", side_effect=fake_post):
            out = ps._chat("glm-5.2", "ds4",
                           [{"role": "user", "content": "ping"}], think=False)
        assert out == "pong"
        assert captured["url"] == "http://localhost:8002/v1/chat/completions"
        assert "chat_template_kwargs" not in captured["body"]

    def test_chat_ds4_think_false_sends_native_thinking_disabled(self):
        """ds4's native thinking control (verified live 2026-07-23):
        thinking is default-on, and think=False disables it via
        `"thinking": {"type": "disabled"}`."""
        raw = '{"choices":[{"message":{"content":"pong"}}]}'
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["body"] = json
            return self._resp(raw)

        with patch.object(ps.requests, "post", side_effect=fake_post):
            ps._chat("glm-5.2", "ds4",
                     [{"role": "user", "content": "ping"}], think=False)
        assert captured["body"]["thinking"] == {"type": "disabled"}

    def test_chat_ds4_think_true_omits_thinking_key(self):
        """think=True (or the default) must omit the `thinking` key so
        ds4's default (thinking on) applies."""
        raw = '{"choices":[{"message":{"content":"pong"}}]}'
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["body"] = json
            return self._resp(raw)

        with patch.object(ps.requests, "post", side_effect=fake_post):
            ps._chat("glm-5.2", "ds4",
                     [{"role": "user", "content": "ping"}], think=True)
        assert "thinking" not in captured["body"]

    def test_chat_ds4_falls_back_to_reasoning_content(self):
        raw = ('{"choices":[{"message":{"content":"",'
               '"reasoning_content":"all reasoning, no answer"}}]}')
        with patch.object(ps.requests, "post",
                          return_value=self._resp(raw)):
            out = ps._chat("glm-5.2", "ds4",
                           [{"role": "user", "content": "hi"}])
        assert out == "all reasoning, no answer"

    def test_chat_stream_ds4_think_false_sends_native_thinking_disabled(self):
        """Streaming ds4 dispatch must wire the same native thinking
        control as non-streaming: think=False → `"thinking": {"type":
        "disabled"}`, never the broken MLX chat_template_kwargs shim."""
        lines = [b"data: [DONE]"]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = iter(lines)
        captured = {}

        def fake_post(url, json=None, stream=None, timeout=None):
            captured["body"] = json
            return resp

        with patch.object(ps.requests, "post", side_effect=fake_post):
            list(ps._chat_stream(
                "glm-5.2", "ds4", [{"role": "user", "content": "hi"}],
                think=False))
        assert captured["body"]["thinking"] == {"type": "disabled"}
        assert "chat_template_kwargs" not in captured["body"]

    def test_chat_stream_ds4_think_true_omits_thinking_key(self):
        lines = [b"data: [DONE]"]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = iter(lines)
        captured = {}

        def fake_post(url, json=None, stream=None, timeout=None):
            captured["body"] = json
            return resp

        with patch.object(ps.requests, "post", side_effect=fake_post):
            list(ps._chat_stream(
                "glm-5.2", "ds4", [{"role": "user", "content": "hi"}],
                think=True))
        assert "thinking" not in captured["body"]

    def test_chat_stream_ds4_yields_tokens_with_tolerant_parse(self):
        """The streaming branch parses each SSE chunk; a chunk with a raw
        control char must yield its token, not be dropped by a strict
        parser (long thinking answers stream reasoning first)."""
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hel\x01lo"}}]}',
            b"data: [DONE]",
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = iter(lines)
        with patch.object(ps.requests, "post", return_value=resp):
            events = list(ps._chat_stream(
                "glm-5.2", "ds4", [{"role": "user", "content": "hi"}]))
        joined = "".join(events)
        assert "Hel\\u0001lo" in joined or "Hello" in joined.replace("\\u0001", "")
        assert '"done": true' in joined

    def test_chat_stream_ds4_falls_back_to_reasoning_content(self):
        """The streaming branch must read reasoning_content when content is empty.
        glm-5.2 on ds4 always thinks, so deltas often have text only in reasoning_content."""
        lines = [
            b'data: {"choices":[{"delta":{"reasoning_content":"Think"}}]}',
            b'data: {"choices":[{"delta":{"reasoning_content":"ing"}}]}',
            b"data: [DONE]",
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = iter(lines)
        with patch.object(ps.requests, "post", return_value=resp):
            events = list(ps._chat_stream(
                "glm-5.2", "ds4", [{"role": "user", "content": "hi"}]))
        joined = "".join(events)
        assert '"token": "Think"' in joined
        assert '"token": "ing"' in joined
        assert '"done": true' in joined

    def test_chat_stream_ds4_timeout_is_600_not_300(self):
        """No SSE chunk flows during prefill, and a 131072-ctx prompt measured
        323s of prefill — a 300s first-chunk read timeout kills exactly the
        long requests the raised context exists for (ds4 also serializes, so
        queueing adds to first-byte time). Pin 600 so a regression fails."""
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = iter([b"data: [DONE]"])
        with patch.object(ps.requests, "post", return_value=resp) as mock_post:
            list(ps._chat_stream(
                "glm-5.2", "ds4", [{"role": "user", "content": "hi"}]))
        assert mock_post.call_args.kwargs["timeout"] == 600


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

    def test_fetch_mlx_models_sees_local_dir_entries(self, tmp_path):
        """MLX entries whose model_path is a local dir (subfolder repos)
        must be discovered once the checkpoint is complete — the HF-cache
        name mangling can never match an absolute path, and a skipped
        entry means the Playground's unfiltered tool 400s even with
        weights on disk."""
        d = tmp_path / "Qwen3.8-27B-Uncensored-MLX" / "8-bit"
        d.mkdir(parents=True)
        for f in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (d / f).write_text("{}")
        (d / "model.safetensors.index.json").write_text(
            '{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}')
        entry = {"served_model_name": "qwen3.8-uncensored-8bit",
                 "model_path": str(d), "model_type": "lm",
                 "context_length": 131072, "on_demand": True}
        with patch.object(ps, "_load_mlx_config",
                          return_value={"qwen3.8-uncensored-8bit": entry}), \
             patch.object(ps, "_mlx_loaded_ids", return_value=set()):
            # Incomplete checkpoint (shard missing): entry must be absent.
            assert ps._fetch_mlx_models({}) == {}
            (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 64)
            models = ps._fetch_mlx_models({})
        info = models["qwen3.8-uncensored-8bit"]
        assert info["backend"] == "mlx"
        assert info["quant"] == "8-bit"
        assert info["disk_bytes"] > 0
        assert info["has_vision"] is False

    def test_uncensored_model_is_unfiltered_only(self):
        """An abliterated model must qualify for the unfiltered task and
        for NOTHING else — a picker falling back to it for `general` (or
        vision, which its checkpoint nominally carries but the lm serving
        path can't reach) would silently serve unaligned output."""
        model = {"backend": "mlx", "active_params_b": 27, "context": 131072,
                 "has_vision": True}
        tasks = ps.get_eligible_tasks("qwen3.8-uncensored-8bit", model)
        assert tasks == ["unfiltered"]


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

    def test_vision_skips_non_vision_pref(self):
        """A vision pref pointing at a tower-less model is skipped in
        favour of one that can actually see (the -mlx-bf16 trap)."""
        with patch.object(ps, "load_default_prefs",
                          return_value={"vision": ["qwen3:8b", "qwen3-vl:32b"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("vision")
        assert name == "qwen3-vl:32b"

    def test_vision_no_capable_model_returns_none(self):
        with patch.object(ps, "load_default_prefs",
                          return_value={"vision": ["qwen3:8b"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("vision")
        assert name is None

    def test_image_gen_skips_ollama_backed_pref(self):
        """An Ollama image tag is skipped for the mflux model behind it.

        Ollama 0.32 answers /api/generate with 400 "image generation models
        are not currently supported" while still advertising
        capabilities: ["image"], so resolving by name alone hands back a
        model that cannot serve the request.
        """
        with patch.object(ps, "load_default_prefs",
                          return_value={"image_gen": [
                              "x/flux2-klein:latest",
                              "black-forest-labs/FLUX.2-klein-4B"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("image_gen")
        assert name == "black-forest-labs/FLUX.2-klein-4B"
        assert backend == "mflux"

    def test_unpulled_ollama_tag_is_not_mistaken_for_an_hf_repo(self):
        """An Ollama tag that isn't pulled locally must not slip through the
        HF download-on-demand path just because it contains a slash.

        `x/z-image-turbo:bf16` is absent from the registry, so the
        "not in models" guard passes; only the colon test keeps it out.
        Letting it through hands mflux a repo id huggingface_hub rejects
        (HFValidationError) instead of falling through to a usable pref.
        """
        with patch.object(ps, "load_default_prefs",
                          return_value={"image_gen": [
                              "x/z-image-turbo:bf16",
                              "black-forest-labs/FLUX.2-klein-4B"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("image_gen")
        assert name == "black-forest-labs/FLUX.2-klein-4B"
        assert backend == "mflux"

    def test_image_gen_ollama_only_pref_returns_none(self):
        """With nothing but an Ollama image tag, report no model rather
        than dispatching into a backend that 400s."""
        with patch.object(ps, "load_default_prefs",
                          return_value={"image_gen": ["x/flux2-klein:latest"]}), \
             patch.object(ps, "get_all_models", return_value=FAKE_MODELS):
            name, backend, warning = ps._pick_model_for_task("image_gen")
        assert name is None
        assert warning is not None


# ── Profiles CRUD ───────────────────────────────────────────────────

class TestProfilesCRUD:
    def test_load_creates_default(self, profiles_dir):
        data = ps.load_profiles()
        assert "profiles" in data
        assert "64gb" in data["profiles"]
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
        assert "everyday" not in loaded["profiles"]
        assert "64gb" in loaded["profiles"]
        assert loaded["active"] == ps.DEFAULT_PROFILES["active"]

    def test_load_profiles_migration_drops_retired(self, tmp_path):
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps({"version": 25, "active": "everyday",
                                    "profiles": {"everyday": {"tasks": {}},
                                                 "custom": {"max_ram_gb": 8, "tasks": {}}}}))
        with patch.object(ps, "PROFILES_FILE", path):
            out = ps.load_profiles()
        assert out["version"] == ps.PROFILES_VERSION
        assert "everyday" not in out["profiles"]
        assert "custom" in out["profiles"]
        assert "64gb" in out["profiles"]
        assert out["active"] in out["profiles"]

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

    def test_saving_profile_does_not_mutate_shared_default_presets(self, client, profiles_dir):
        # Regression: load_profiles() shallow-copied DEFAULT_PROFILES, so the
        # returned dict's ["profiles"] aliased the shared constant. Saving a
        # custom profile then mutated the in-process presets — corrupting the
        # canonical map that migrate_profiles uses to decide what's a preset
        # (so the custom profile would later be dropped) and that warm-set
        # logic falls back to (a bogus preset with warm=None renders 0B warm).
        probe = "pollution_probe_9f3a"
        assert probe not in ps.DEFAULT_PROFILES["profiles"]  # sanity: name is unique
        resp = client.post("/api/profiles", json={"name": probe, "tasks": {}})
        assert resp.status_code == 200
        assert probe not in ps.DEFAULT_PROFILES["profiles"], \
            "save leaked a custom profile into the shared DEFAULT_PROFILES constant"

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

    def test_api_profiles_get_missing_is_active_picks_only(self, client, profiles_dir):
        """The page-load download prompt lists only the active profile's chosen
        models — not every fallback candidate from the default prefs."""
        ps.save_profiles({
            "version": ps.PROFILES_VERSION, "active": "t",
            "profiles": {"t": {"label": "T", "tasks": {
                "code": "pick-coder:7b", "general": "pick-gen:7b"}}},
        })
        fallbacks = {"code": ["pick-coder:7b", "fallback-coder:70b"],
                     "general": ["pick-gen:7b", "fallback-gen:70b"],
                     "reasoning": ["unrelated:70b"]}
        with patch.object(ps, "load_default_prefs", return_value=fallbacks), \
             patch.object(ps, "get_all_models",
                          return_value={"installed:1b": {"backend": "ollama"}}), \
             patch.object(ps, "_resolve_model_sizes",
                          side_effect=lambda names: [{"name": n, "size_gb": None} for n in names]):
            resp = client.get("/api/profiles")
        names = {m["name"] for m in resp.get_json().get("missing", [])}
        assert names == {"pick-coder:7b", "pick-gen:7b"}

    def test_memory_warm_falls_back_to_canonical_for_stale_preset(self, client):
        """A preset profile stored WITHOUT a `warm` key (e.g. a v27 file written
        before warm existed) must still resolve its canonical warm set, not an
        empty one — otherwise the bar shows 0B warm and everything hatched."""
        GB = 1 << 30
        # '128gb' preset, but the stored dict has NO 'warm' key.
        prof = {"max_ram_gb": 128,
                "tasks": {"general": "qwen3.6:27b-mlx-bf16", "embedding": "qwen3-embedding:8b",
                          "code": "qwen3-coder-next:latest"}}
        models = {"qwen3.6:27b-mlx-bf16": {"backend": "ollama", "vram_bytes": 55 * GB},
                  "qwen3-embedding:8b": {"backend": "ollama", "vram_bytes": 8 * GB},
                  "qwen3-coder-next:latest": {"backend": "ollama", "vram_bytes": 52 * GB}}
        with patch.object(ps, "load_profiles",
                          return_value={"active": "128gb", "profiles": {"128gb": prof}}), \
             patch.object(ps, "get_all_models", return_value=models):
            d = client.get("/api/profiles/128gb/memory").get_json()
        assert {m["name"] for m in d["warm"]} == {"qwen3.6:27b-mlx-bf16", "qwen3-embedding:8b"}
        assert d["warm_bytes"] == 63 * GB

    def test_memory_estimates_undownloaded_warm_model(self, client, profiles_dir):
        # Viewing a higher tier whose models aren't downloaded here: the warm
        # model must be estimated from HF and flagged, not counted as 0.
        ps.save_profiles({"version": ps.PROFILES_VERSION, "active": "big",
            "profiles": {"big": {"max_ram_gb": 512, "warm": ["general"],
                "tasks": {"general": "qwen3.5-fast"}}}})
        with patch.object(ps, "get_all_models", return_value={}), \
             patch.object(ps, "_load_mlx_config",
                          return_value={"qwen3.5-fast": {"model_path": "mlx-community/Qwen3.5-35B-A3B-4bit"}}), \
             patch.object(ps, "_get_hf_model_size", return_value=180.0):
            ps._model_size_cache.clear()
            d = client.get("/api/profiles/big/memory").get_json()
        warm = {w["name"]: w for w in d["warm"]}
        assert warm["qwen3.5-fast"]["bytes"] == int(180 * 1e9)
        assert warm["qwen3.5-fast"]["estimated"] is True
        assert warm["qwen3.5-fast"]["downloaded"] is False
        assert d["warm_bytes"] == int(180 * 1e9)

    def test_mlx_downloaded_model_sized_from_disk_not_name(self):
        # A model path without a parseable param count that isn't in the
        # known-params table, so the name-based estimate is 0. A downloaded model
        # must be sized from its real on-disk bytes instead (else the memory bar
        # shows "0 bit" for it).
        with patch.object(ps, "_load_mlx_config",
                          return_value={"big-model": {"model_path": "mlx-community/BigModel-4bit"}}), \
             patch.object(ps, "_mlx_loaded_ids", return_value=set()), \
             patch.object(ps, "_hf_model_downloaded", return_value=True), \
             patch.object(ps, "_hf_cache_bytes", return_value=180 * 10**9), \
             patch.object(ps, "_mlx_model_has_vision", return_value=False):
            out = ps._fetch_mlx_models(existing=set())
        assert out["big-model"]["disk_bytes"] == 180 * 10**9
        assert out["big-model"]["vram_bytes"] == 180 * 10**9

    def test_pull_resolves_bare_mlx_served_name_to_hf_repo(self, client):
        # A bare MLX served-name (no "/", no ":") must be pulled as its HF
        # model_path via hf, not `ollama pull <served-name>` (which 404s on the
        # manifest — "file does not exist").
        from contextlib import contextmanager
        captured = {}

        def fake_worker(name, kind, total):
            captured["name"] = name
            captured["kind"] = kind
            return 4321

        @contextmanager
        def fake_lock():
            yield

        with patch.object(ps, "_refuse_if_client", return_value=None), \
             patch.object(ps, "_load_mlx_config",
                          return_value={"qwen3.5-fast": {"model_path": "mlx-community/Qwen3.5-35B-A3B-4bit"}}), \
             patch.object(ps, "_pulls_lock", fake_lock), \
             patch.object(ps, "_pulls_read", return_value={"pulls": {}, "dismissed": []}), \
             patch.object(ps, "_pulls_write"), \
             patch.object(ps, "_get_hf_model_size", return_value=None), \
             patch.object(ps, "_start_pull_worker", side_effect=fake_worker):
            resp = client.post("/api/models/pull", json={"models": ["qwen3.5-fast"]})
        assert resp.status_code == 202
        assert captured["name"] == "mlx-community/Qwen3.5-35B-A3B-4bit"
        assert captured["kind"] == "hf"

    def test_load_profiles_does_not_clobber_unparseable_file(self, client, profiles_dir):
        # A non-empty but unparseable file (e.g. a transient torn read from a
        # concurrent writer) must NOT be overwritten with defaults.
        from pathlib import Path
        pf = Path(ps.PROFILES_FILE)
        pf.write_text('{ "version": 27, "profiles": {  TORN')
        before = pf.read_text()
        out = ps.load_profiles()
        assert pf.read_text() == before           # file left intact
        assert "profiles" in out                  # defaults served in-memory only

    def test_save_profiles_is_atomic_no_tmp_left(self, client, profiles_dir):
        from pathlib import Path
        ps.save_profiles({"version": ps.PROFILES_VERSION, "active": None, "profiles": {}})
        d = Path(ps.PROFILES_FILE).parent
        assert not any(p.name.endswith(".tmp") for p in d.iterdir())
        assert ps.load_profiles()["version"] == ps.PROFILES_VERSION

    def test_keep_alive_for_does_not_write_on_version_mismatch(self, client, profiles_dir):
        # The inference hot path must not migrate-and-write the file.
        import json as _json
        from pathlib import Path
        pf = Path(ps.PROFILES_FILE)
        stale = {"version": 1, "active": "t",
                 "profiles": {"t": {"max_ram_gb": 8, "warm": ["general"],
                                    "tasks": {"general": "warm-model:1b", "code": "cold:1b"}}}}
        pf.write_text(_json.dumps(stale))
        before = pf.read_text()
        assert ps.keep_alive_for("warm-model:1b") == ps.OLLAMA_KEEP_ALIVE
        assert ps.keep_alive_for("cold:1b") == ps.OLLAMA_KEEP_ALIVE_ONDEMAND
        assert pf.read_text() == before           # no migrate/save side-effect

    def test_memory_route_proxies_as_get_in_client_mode(self, client):
        sentinel = ps.Response("{}", content_type="application/json")
        with patch.object(ps, "_proxy_to_desktop", return_value=sentinel) as prox:
            client.get("/api/profiles/128gb/memory")
        assert prox.called
        assert prox.call_args.kwargs.get("method") == "GET"

    def test_api_profiles_activate_missing_only_picks_but_saves_fallbacks(self, client, profiles_dir):
        """Activate prompts to pull only the profile's picks, but still saves
        the full fallback lists into prefs for the MCP server's runtime use."""
        ps.save_profiles({
            "version": ps.PROFILES_VERSION, "active": None,
            "profiles": {"t": {"label": "T", "tasks": {"code": "pick-coder:7b"}}},
        })
        fallbacks = {"code": ["pick-coder:7b", "fallback-coder:70b"]}
        saved = {}
        with patch.object(ps, "load_default_prefs", return_value=dict(fallbacks)), \
             patch.object(ps, "get_all_models",
                          return_value={"installed:1b": {"backend": "ollama"}}), \
             patch.object(ps, "_resolve_model_sizes",
                          side_effect=lambda names: [{"name": n, "size_gb": None} for n in names]), \
             patch.object(ps, "save_mcp_prefs", side_effect=lambda p: saved.update(p)):
            resp = client.post("/api/profiles/t/activate")
        names = {m["name"] for m in resp.get_json()["missing"]}
        assert names == {"pick-coder:7b"}
        assert "fallback-coder:70b" in saved.get("code", [])

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
        # qwen3:8b is not in the warm set for this test, so keep_alive is short
        assert payload["keep_alive"] == "30s"

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

    def test_warm_loads_only_warm_set(self, client):
        prof = {"max_ram_gb": 128, "warm": ["general", "embedding"],
                "tasks": {"general": "work:bf16", "code": "coder:bf16",
                          "embedding": "embed:8b", "image_gen": "x/img:latest"}}
        with patch.object(ps, "load_profiles", return_value={"active": "t", "profiles": {"t": prof}}), \
             patch.object(ps, "get_all_models", return_value={
                 "work:bf16": {"backend": "ollama"}, "coder:bf16": {"backend": "ollama"},
                 "embed:8b": {"backend": "ollama"}, "x/img:latest": {"backend": "ollama"}}), \
             patch.object(ps, "ollama_get", return_value={"models": []}), \
             patch("requests.post") as post:
            post.return_value.status_code = 200
            r = client.post("/api/profiles/t/warm")
        assert r.status_code == 200
        warmed = {c.kwargs["json"]["model"] for c in post.call_args_list}
        assert warmed == {"work:bf16", "embed:8b"}      # general + embedding only
        assert "coder:bf16" not in warmed and "x/img:latest" not in warmed

    def _mem(self, client, max_ram_gb, tasks, warm, sizes):
        prof = {"max_ram_gb": max_ram_gb, "warm": warm, "tasks": tasks}
        models = {n: {"backend": "ollama", "vram_bytes": b, "disk_bytes": b}
                  for n, b in sizes.items()}
        with patch.object(ps, "load_profiles", return_value={"active": "t", "profiles": {"t": prof}}), \
             patch.object(ps, "get_all_models", return_value=models):
            return client.get("/api/profiles/t/memory").get_json()

    def test_memory_ok(self, client):
        GB = 1 << 30
        d = self._mem(client, 128,
                      {"general": "w", "embedding": "e", "code": "c"},
                      ["general", "embedding"],
                      {"w": 55 * GB, "e": 8 * GB, "c": 52 * GB})
        assert d["warm_bytes"] == 63 * GB
        assert d["largest_on_demand_bytes"] == 52 * GB
        assert d["peak_bytes"] == 115 * GB           # 63 + 52 ≤ 128
        assert d["state"] == "ok"

    def test_memory_tight_dominant_over_budget(self, client):
        GB = 1 << 30
        d = self._mem(client, 512,
                      {"general": "glm", "embedding": "e", "vision": "v"},
                      ["general", "embedding"],
                      {"glm": 418 * GB, "e": 8 * GB, "v": 95 * GB})
        assert d["warm_bytes"] == 426 * GB           # > 333 budget, ≤ 512 cap
        assert d["peak_bytes"] == 521 * GB           # 426 + 95 > 512 cap
        assert d["state"] == "tight"

    def test_memory_tight_peak_exceeds_cap_only(self, client):
        """tight via peak > cap alone — warm set fits comfortably within budget.

        This path requires warm_bytes ≤ budget_bytes so the tight state can only
        be reached via the peak_bytes > cap_bytes branch of the OR, not the
        warm > budget branch.  The dominant-over-budget test above cannot catch
        a regression here because it fires both OR conditions at once."""
        GB = 1 << 30
        # cap=100GB, budget=65GB (65%), warm=64GB ≤ budget, peak=114GB > cap
        d = self._mem(client, 100,
                      {"general": "w", "embedding": "e", "code": "c"},
                      ["general", "embedding"],
                      {"w": 60 * GB, "e": 4 * GB, "c": 50 * GB})
        assert d["warm_bytes"] == 64 * GB
        assert d["warm_bytes"] <= d["budget_bytes"], (
            "warm must stay within budget — otherwise this test proves nothing "
            "about the peak-only branch")
        assert d["peak_bytes"] > d["cap_bytes"]
        assert d["state"] == "tight"

    def test_memory_thrash_warm_exceeds_cap(self, client):
        GB = 1 << 30
        d = self._mem(client, 64,
                      {"general": "a", "embedding": "b"},
                      ["general", "embedding"],
                      {"a": 50 * GB, "b": 30 * GB})   # warm 80 > 64 cap
        assert d["state"] == "thrash"

    def test_memory_dedup_shared_model_not_double_counted(self, client):
        GB = 1 << 30
        d = self._mem(client, 32,
                      {"general": "s", "code": "s", "embedding": "e", "image_gen": "img"},
                      ["general", "embedding"],
                      {"s": 5 * GB, "e": 1 * GB, "img": 6 * GB})
        # 's' backs warm general AND non-warm code → counted warm only, not on-demand
        assert {m["name"] for m in d["on_demand"]} == {"img"}
        assert d["warm_bytes"] == 6 * GB


# ── ds4 discovery ───────────────────────────────────────────────────

class TestFetchDs4Models:
    def test_ds4_up_inserts_hardcoded_entry(self):
        """ds4 serves one pinned model with no params/vision metadata; those
        two fields must be fully hardcoded (sizes included — the GGUF is
        invisible to every existing sizing path) and marked always-resident,
        or the memory bar undercounts 244GiB and warm logic tries to
        keep-alive it. This mock's .json() has no usable context_length,
        exercising the DS4_CONTEXT fallback."""
        resp = MagicMock()
        resp.ok = True
        with patch.object(ps.requests, "get", return_value=resp):
            out = ps._fetch_ds4_models(existing={})
        entry = out["glm-5.2"]
        assert entry["backend"] == "ds4"
        assert entry["disk_bytes"] == 262_036_650_048
        assert entry["vram_bytes"] == 262_036_650_048
        assert entry["total_params_b"] == 740
        assert entry["active_params_b"] == 32
        assert entry["context"] == 131072
        assert entry["has_vision"] is False
        assert entry["is_loaded"] is True
        assert entry["on_demand"] is False

    def test_ds4_up_prefers_live_context_length(self):
        """A --ctx launch-flag drift must surface through discovery, not
        hide behind the DS4_CONTEXT constant."""
        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {
            "data": [{"id": "glm-5.2", "context_length": 65536}]}
        with patch.object(ps.requests, "get", return_value=resp):
            out = ps._fetch_ds4_models(existing={})
        assert out["glm-5.2"]["context"] == 65536

    def test_ds4_down_returns_empty(self):
        with patch.object(ps.requests, "get",
                          side_effect=ps.requests.ConnectionError("down")):
            assert ps._fetch_ds4_models(existing={}) == {}

    def test_existing_name_not_overwritten(self):
        resp = MagicMock()
        resp.ok = True
        with patch.object(ps.requests, "get", return_value=resp):
            out = ps._fetch_ds4_models(existing={"glm-5.2": {}})
        assert out == {}

    def test_ds4_model_is_eligible_for_llm_tasks(self):
        """The one-line bug this guards: _LLM_BACKENDS without 'ds4' gives
        glm-5.2 zero eligible tasks — invisible in every dropdown."""
        resp = MagicMock()
        resp.ok = True
        with patch.object(ps.requests, "get", return_value=resp):
            entry = ps._fetch_ds4_models(existing={})["glm-5.2"]
        tasks = ps.get_eligible_tasks("glm-5.2", entry)
        for task in ("code", "general", "reasoning", "long_context",
                     "translation"):
            assert task in tasks, f"glm-5.2 missing eligible task {task!r}"

    def test_missing_models_check_skips_ds4_served_name(self):
        """glm-5.2 is pre-provisioned by install.sh, not pullable. With the
        MLX yaml entry gone, an unpatched _check_missing_models would prompt
        the user to pull it, and the pull would 404 (`ollama pull glm-5.2`)."""
        with patch.object(ps, "get_all_models",
                          return_value={"qwen3.6:27b": {"backend": "ollama"}}):
            missing, _ = ps._check_missing_models({"general": ["glm-5.2"]})
        assert missing == []


class TestDs4InstalledGatesMlxGlm52Fallback:
    """When ds4 is provisioned on this machine, ds4 owns glm-5.2 outright:
    if ds4 is unreachable the model must be absent, never resurface as a
    stale MLX entry (wrong sizing, invites a 418GB cold load) from an
    unmigrated ~/.config/mlx-server/config.yaml."""

    def test_ds4_down_but_installed_keeps_glm52_absent(self):
        with patch.object(ps, "OLLAMA_URL", "http://localhost:11434"), \
             patch.object(ps, "_fetch_ollama_models", return_value={}), \
             patch.object(ps, "_fetch_ds4_models", return_value={}), \
             patch.object(ps, "ds4_installed", return_value=True), \
             patch.object(ps, "_load_mlx_config",
                          return_value={"glm-5.2": {"model_path": "mlx-community/GLM-5.2-4bit"}}), \
             patch.object(ps, "_mlx_loaded_ids", return_value=set()), \
             patch.object(ps, "_hf_model_downloaded", return_value=True), \
             patch.object(ps, "_hf_cache_bytes", return_value=180 * 10**9), \
             patch.object(ps, "_mlx_model_has_vision", return_value=False):
            models = ps._fetch_all_models()
        assert "glm-5.2" not in models

    def test_ds4_not_installed_allows_mlx_glm52_fallback(self):
        with patch.object(ps, "OLLAMA_URL", "http://localhost:11434"), \
             patch.object(ps, "_fetch_ollama_models", return_value={}), \
             patch.object(ps, "_fetch_ds4_models", return_value={}), \
             patch.object(ps, "ds4_installed", return_value=False), \
             patch.object(ps, "_load_mlx_config",
                          return_value={"glm-5.2": {"model_path": "mlx-community/GLM-5.2-4bit"}}), \
             patch.object(ps, "_mlx_loaded_ids", return_value=set()), \
             patch.object(ps, "_hf_model_downloaded", return_value=True), \
             patch.object(ps, "_hf_cache_bytes", return_value=180 * 10**9), \
             patch.object(ps, "_mlx_model_has_vision", return_value=False):
            models = ps._fetch_all_models()
        assert "glm-5.2" in models
        assert models["glm-5.2"]["backend"] == "mlx"


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


class TestKeepAliveFor:
    """keep_alive_for returns long keep_alive for warm models, short otherwise."""

    def test_keep_alive_for_warm_vs_on_demand(self, profiles_dir):
        # keep_alive_for reads profiles.json directly (read-only hot path), so
        # the warm set is whatever is on disk — not a patched load_profiles.
        ps.save_profiles({"version": ps.PROFILES_VERSION, "active": "t",
                          "profiles": {"t": {"warm": ["general"],
                                             "tasks": {"general": "w:bf16", "code": "c:bf16"}}}})
        assert ps.keep_alive_for("w:bf16") == "30m"
        assert ps.keep_alive_for("c:bf16") == "30s"
        assert ps.keep_alive_for("unknown:1b") == "30s"


class TestFleetReport:
    """POST /api/fleet/report ingest + GET /api/fleet query."""

    @pytest.fixture(autouse=True)
    def _fleet_setup(self):
        """Fresh fleet tables per test (per-test DB via conftest's
        _isolate_activity_db) and a clean rate-limit dict, since
        _fleet_rate is module-global state shared across tests."""
        ps.activity.init_db()
        ps._fleet_rate.clear()
        yield
        ps._fleet_rate.clear()

    def _payload(self, machine="laptop"):
        return {"machine": machine, "version": "v1.2.0", "mode": "client",
                "sent_at": 1, "audit": [{"id": "claude-mcp", "status": "pass"}],
                "usage": [{"day": "2026-07-10", "tool": "vision", "source": "mcp",
                           "count": 3, "errors": 0, "avg_ms": 100}]}

    def test_report_accepts_valid(self, client):
        r = client.post("/api/fleet/report", json=self._payload(machine="report-valid"))
        assert r.status_code == 200
        got = client.get("/api/fleet").get_json()
        assert got["machines"][0]["machine"] == "report-valid"

    def test_report_rejects_bad_machine(self, client):
        p = self._payload(machine="<script>")
        r = client.post("/api/fleet/report", json=p)
        assert r.status_code == 400

    def test_report_rate_limited(self, client):
        payload = self._payload(machine="report-rate-limited")
        assert client.post("/api/fleet/report", json=payload).status_code == 200
        assert client.post("/api/fleet/report", json=payload).status_code == 429

    def test_report_rejects_malformed_usage_item(self, client):
        p = self._payload(machine="report-bad-usage")
        p["usage"] = [{"day": "2026-07-10", "tool": "vision", "source": "mcp",
                       "count": "not-an-int", "errors": 0, "avg_ms": 100}]
        r = client.post("/api/fleet/report", json=p)
        assert r.status_code == 400

    def test_report_rejects_usage_item_missing_day(self, client):
        p = self._payload(machine="report-missing-day")
        p["usage"] = [{"tool": "vision", "source": "mcp",
                       "count": 3, "errors": 0, "avg_ms": 100}]
        r = client.post("/api/fleet/report", json=p)
        assert r.status_code == 400

    def test_report_rejects_usage_item_missing_source(self, client):
        p = self._payload(machine="report-missing-source")
        p["usage"] = [{"day": "2026-07-10", "tool": "vision",
                       "count": 3, "errors": 0, "avg_ms": 100}]
        r = client.post("/api/fleet/report", json=p)
        assert r.status_code == 400

    def test_report_rejects_usage_item_xss_source(self, client):
        p = self._payload(machine="report-xss-source")
        p["usage"] = [{"day": "2026-07-10", "tool": "vision", "source": "<script>",
                       "count": 3, "errors": 0, "avg_ms": 100}]
        r = client.post("/api/fleet/report", json=p)
        assert r.status_code == 400

    def test_report_rejects_non_dict_body_list(self, client):
        r = client.post("/api/fleet/report", json=[1, 2, 3])
        assert r.status_code == 400

    def test_report_rejects_non_dict_body_string(self, client):
        r = client.post("/api/fleet/report", json="hi")
        assert r.status_code == 400

    def test_report_accepts_missing_usage_key(self, client):
        """body.get("usage", []) validates an absent key as [] and passes,
        so the ingest call must not KeyError on body["usage"] — a client
        with nothing to report yet (e.g. no usage since last heartbeat)
        must not 500."""
        p = self._payload(machine="report-no-usage")
        del p["usage"]
        r = client.post("/api/fleet/report", json=p)
        assert r.status_code == 200
        got = client.get("/api/fleet").get_json()
        assert any(m["machine"] == "report-no-usage" for m in got["machines"])


def test_api_activity_includes_last_activity(client):
    """Activity API response includes last_activity_at timestamp."""
    import time
    from lib import activity
    activity.init_db()
    now = time.time()
    activity.log_request(tool="code", model="x", backend="ollama", source="mcp",
                         status="ok", duration_ms=5, started_at=now-1, completed_at=now)
    data = client.get("/api/activity?period=1").get_json()  # 1-second window → empty history
    assert data["last_activity_at"] is not None


def test_api_activity_carries_csp_header(client):
    """Every response (defense-in-depth for the Fleet view's XSS-safe render)
    must carry a restrictive Content-Security-Policy header."""
    from lib import activity
    activity.init_db()
    resp = client.get("/api/activity")
    csp = resp.headers.get("Content-Security-Policy")
    assert csp is not None
    assert "default-src 'none'" in csp
    assert "script-src 'self' 'unsafe-inline'" in csp


class TestAuditRoutes:
    def test_api_audit_returns_check_list(self, client):
        r = client.get("/api/audit")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list) and data
        assert {"id", "tool", "status", "detail", "fixable"} <= set(data[0])

    def test_api_audit_fix_rejects_bad_group(self, client):
        r = client.post("/api/audit/fix", json={"group": "../etc"})
        assert r.status_code == 400


class TestShareUrl:
    """/api/share-url hands out the canonical tokened Playground URL so
    phones can bookmark a link that actually authenticates (the page strips
    ?token= from the address bar, so address-bar bookmarks are tokenless)."""

    def test_share_url_uses_tailscale_fqdn_and_token(self, client):
        with patch.object(ps, "_tailscale_fqdn", return_value="box.tail.ts.net"), \
             patch.object(ps, "_PROFILE_AUTH_TOKEN", "sekret"), \
             patch.object(ps, "PORT", 8101):
            resp = client.get("/api/share-url",
                              headers={"Authorization": "Bearer sekret"})
        assert resp.status_code == 200
        assert resp.get_json() == {
            "url": "https://box.tail.ts.net:8101/tools?token=sekret"}

    def test_share_url_falls_back_to_localhost_without_fqdn(self, client):
        with patch.object(ps, "_tailscale_fqdn", return_value=""), \
             patch.object(ps, "_PROFILE_AUTH_TOKEN", "sekret"), \
             patch.object(ps, "PORT", 8101):
            resp = client.get("/api/share-url",
                              headers={"Authorization": "Bearer sekret"})
        assert resp.status_code == 200
        assert resp.get_json()["url"].endswith("/tools?token=sekret")
        assert resp.get_json()["url"].startswith("http://localhost")

    def test_share_url_requires_auth(self, client):
        with patch.object(ps, "_PROFILE_AUTH_TOKEN", "sekret"):
            resp = client.get("/api/share-url")
        assert resp.status_code == 403


class TestDiagnosticsDs4:
    def test_diagnostics_reports_ds4_when_installed(self, client):
        resp_ok = MagicMock()
        resp_ok.ok = True
        with patch.object(ps, "ds4_installed", return_value=True), \
             patch.object(ps.requests, "get", return_value=resp_ok), \
             patch.object(ps, "ollama_get", return_value=None), \
             patch.object(ps, "get_all_models", return_value={}):
            d = client.get("/api/diagnostics").get_json()
        assert d["services"]["ds4"] is True

    def test_diagnostics_ds4_null_when_not_installed(self, client):
        """Laptops never run ds4 — a permanently-red 'Down' row would be
        noise. null tells the UI to omit the row entirely."""
        with patch.object(ps, "ds4_installed", return_value=False), \
             patch.object(ps.requests, "get",
                          side_effect=ps.requests.ConnectionError("x")), \
             patch.object(ps, "ollama_get", return_value=None), \
             patch.object(ps, "get_all_models", return_value={}):
            d = client.get("/api/diagnostics").get_json()
        assert d["services"]["ds4"] is None
