"""Tests for pure/near-pure functions in app/menubar.py."""

import importlib
import importlib.util
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The menubar module imports macOS-only packages (rumps, objc, AppKit, WebKit)
# at the top level. We mock them so tests run in any environment.
import sys

_MACOS_STUBS = {}
for mod_name in ("rumps", "objc", "AppKit", "WebKit"):
    if mod_name not in sys.modules:
        _MACOS_STUBS[mod_name] = MagicMock()
        sys.modules[mod_name] = _MACOS_STUBS[mod_name]

# objc.typedSelector must return a passthrough decorator
sys.modules["objc"].typedSelector = lambda sig: lambda fn: fn

import app.menubar as menubar


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_ts_cache():
    """Reset the Tailscale resolution cache between tests."""
    menubar._ts_cache.update({"ip": "", "ts": 0})
    yield


@pytest.fixture()
def mode_conf(tmp_path):
    """Provide a temp file to use as MODE_CONF."""
    path = tmp_path / "mode.conf"
    return path


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------

class TestGetVersion:
    def test_exact_tag(self):
        """Returns the tag name when HEAD is exactly on a tag."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.return_value = "v1.0.0\n"
            mock_sub.DEVNULL = subprocess.DEVNULL
            assert menubar.get_version("HEAD") == "v1.0.0"

    def test_commits_past_tag(self):
        """Returns tag+distance when HEAD is past a tag."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.return_value = "v1.0.0-3-gabcdef0\n"
            mock_sub.DEVNULL = subprocess.DEVNULL
            assert menubar.get_version("HEAD") == "v1.0.0+3"

    def test_returns_dev_on_failure(self):
        """Any subprocess error yields 'dev'."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.side_effect = subprocess.CalledProcessError(1, "git")
            mock_sub.DEVNULL = subprocess.DEVNULL
            assert menubar.get_version("HEAD") == "dev"

    def test_no_tags(self):
        """Returns short hash when no tags exist."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.return_value = "abcdef0\n"
            mock_sub.DEVNULL = subprocess.DEVNULL
            assert menubar.get_version("HEAD") == "abcdef0"


# ---------------------------------------------------------------------------
# load_network_conf
# ---------------------------------------------------------------------------

class TestLoadNetworkConf:
    def test_defaults_when_missing(self, tmp_path):
        """Returns defaults when file doesn't exist."""
        with patch.object(menubar, "NETWORK_CONF", str(tmp_path / "nope.conf")):
            conf = menubar.load_network_conf()
        assert conf["OLLAMA_PORT"] == "11434"
        assert conf["MLX_PORT"] == "8000"
        assert conf["PROBE_TIMEOUT"] == "2"
        assert conf["MODEL_SERVER_HOST"] == ""

    def test_parses_plain_values(self, tmp_path):
        path = tmp_path / "network.conf"
        path.write_text("MODEL_SERVER_HOST=myhost\nOLLAMA_PORT=9999\n")
        with patch.object(menubar, "NETWORK_CONF", str(path)):
            conf = menubar.load_network_conf()
        assert conf["MODEL_SERVER_HOST"] == "myhost"
        assert conf["OLLAMA_PORT"] == "9999"

    def test_strips_quotes(self, tmp_path):
        path = tmp_path / "network.conf"
        path.write_text('MODEL_SERVER_HOST="studio.tail12345.ts.net"\n')
        with patch.object(menubar, "NETWORK_CONF", str(path)):
            conf = menubar.load_network_conf()
        assert conf["MODEL_SERVER_HOST"] == "studio.tail12345.ts.net"

    def test_strips_single_quotes(self, tmp_path):
        path = tmp_path / "network.conf"
        path.write_text("MODEL_SERVER_HOST='studio'\n")
        with patch.object(menubar, "NETWORK_CONF", str(path)):
            conf = menubar.load_network_conf()
        assert conf["MODEL_SERVER_HOST"] == "studio"

    def test_skips_comments_and_blank_lines(self, tmp_path):
        path = tmp_path / "network.conf"
        path.write_text("# comment\n\nOLLAMA_PORT=5555\n")
        with patch.object(menubar, "NETWORK_CONF", str(path)):
            conf = menubar.load_network_conf()
        assert conf["OLLAMA_PORT"] == "5555"
        assert conf["MODEL_SERVER_HOST"] == ""


# ---------------------------------------------------------------------------
# resolve_desktop_tailscale
# ---------------------------------------------------------------------------

def _make_ts_status(hostname, ipv4="100.64.0.2", dns_name="studio.tail.ts.net."):
    return json.dumps({
        "BackendState": "Running",
        "Peer": {
            "abc123": {
                "HostName": hostname,
                "DNSName": dns_name,
                "TailscaleIPs": [ipv4, "fd7a::1"],
            }
        },
    })


class TestResolveDesktopTailscale:
    def test_finds_peer(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = _make_ts_status("studio")
        with patch("app.menubar.subprocess.run", return_value=result):
            ip, fqdn = menubar.resolve_desktop_tailscale("studio")
        assert ip == "100.64.0.2"
        assert fqdn == "studio.tail.ts.net"

    def test_empty_hostname_returns_empty(self):
        ip, fqdn = menubar.resolve_desktop_tailscale("")
        assert ip == ""
        assert fqdn == ""

    def test_missing_peer_returns_empty(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = _make_ts_status("other-host")
        with patch("app.menubar.subprocess.run", return_value=result):
            ip, fqdn = menubar.resolve_desktop_tailscale("studio")
        assert ip == ""
        assert fqdn == ""

    def test_cache_hit_avoids_subprocess(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = _make_ts_status("studio")
        with patch("app.menubar.subprocess.run", return_value=result) as mock_run:
            menubar.resolve_desktop_tailscale("studio")
            menubar.resolve_desktop_tailscale("studio")
            assert mock_run.call_count == 1

    def test_cache_expires(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = _make_ts_status("studio")
        with patch("app.menubar.subprocess.run", return_value=result) as mock_run:
            menubar.resolve_desktop_tailscale("studio")
            menubar._ts_cache["ts"] = time.time() - 60  # expire
            menubar.resolve_desktop_tailscale("studio")
            assert mock_run.call_count == 2

    def test_backend_not_running_returns_empty(self):
        result = MagicMock()
        result.returncode = 0
        result.stdout = json.dumps({"BackendState": "Stopped", "Peer": {}})
        with patch("app.menubar.subprocess.run", return_value=result):
            ip, fqdn = menubar.resolve_desktop_tailscale("studio")
        assert ip == ""

    def test_subprocess_failure_returns_empty(self):
        with patch("app.menubar.subprocess.run", side_effect=OSError("no tailscale")):
            ip, fqdn = menubar.resolve_desktop_tailscale("studio")
        assert ip == ""
        assert fqdn == ""


# ---------------------------------------------------------------------------
# probe_port
# ---------------------------------------------------------------------------

class TestProbePort:
    def test_open_port(self):
        """probe_port returns True for a port that is listening."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            assert menubar.probe_port(port) is True
        finally:
            srv.close()

    def test_closed_port(self):
        """probe_port returns False for a port nothing is listening on."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.close()
        assert menubar.probe_port(port) is False

    def test_custom_host(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            assert menubar.probe_port(port, host="127.0.0.1") is True
        finally:
            srv.close()


# ---------------------------------------------------------------------------
# load_force_local / save_force_local
# ---------------------------------------------------------------------------

class TestForceLocal:
    def test_default_is_false(self, tmp_path):
        with patch.object(menubar, "MODE_CONF", str(tmp_path / "missing.conf")):
            assert menubar.load_force_local() is False

    def test_roundtrip_true(self, tmp_path):
        path = tmp_path / "mode.conf"
        with patch.object(menubar, "MODE_CONF", str(path)):
            menubar.save_force_local(True)
            assert menubar.load_force_local() is True

    def test_roundtrip_false(self, tmp_path):
        path = tmp_path / "mode.conf"
        with patch.object(menubar, "MODE_CONF", str(path)):
            menubar.save_force_local(False)
            assert menubar.load_force_local() is False

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "mode.conf"
        with patch.object(menubar, "MODE_CONF", str(path)):
            menubar.save_force_local(True)
            assert path.exists()
            assert menubar.load_force_local() is True

    def test_handles_quoted_value(self, tmp_path):
        path = tmp_path / "mode.conf"
        path.write_text('FORCE_LOCAL="true"\n')
        with patch.object(menubar, "MODE_CONF", str(path)):
            assert menubar.load_force_local() is True

    def test_handles_single_quoted_value(self, tmp_path):
        path = tmp_path / "mode.conf"
        path.write_text("FORCE_LOCAL='false'\n")
        with patch.object(menubar, "MODE_CONF", str(path)):
            assert menubar.load_force_local() is False


class TestComputerUseTask:
    def test_computer_use_in_special_tasks(self):
        from lib.models import SPECIAL_TASKS
        assert "computer_use" in SPECIAL_TASKS

    def test_computer_use_has_prefixes(self):
        from lib.models import SPECIAL_TASKS
        task = SPECIAL_TASKS["computer_use"]
        assert "label" in task
        assert "prefixes" in task
        assert len(task["prefixes"]) > 0

    def test_computer_use_matches_known_models(self):
        from lib.models import SPECIAL_TASKS
        prefixes = SPECIAL_TASKS["computer_use"]["prefixes"]
        assert any("ui-tars" in p for p in prefixes)
        assert any("fara" in p for p in prefixes)


class TestValidateNetworkConf:
    def test_empty_file_gets_defaults(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text("")
        template_dir = tmp_path / "config" / "local-models"
        template_dir.mkdir(parents=True)
        (template_dir / "network.conf").write_text("OLLAMA_PORT=11434\n")
        with patch.object(models, "NETWORK_CONF", conf), \
             patch.object(models, "CONFIG_DIR", tmp_path):
            warnings = models.validate_network_conf()
        assert any("missing or empty" in w for w in warnings)
        assert conf.stat().st_size > 0

    def test_non_numeric_port_repaired(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text("OLLAMA_PORT=11434abc\nMLX_PORT=8000\n")
        with patch.object(models, "NETWORK_CONF", conf), \
             patch.object(models, "CONFIG_DIR", tmp_path):
            warnings = models.validate_network_conf()
        assert any("non-numeric" in w for w in warnings)
        repaired = conf.read_text()
        assert "OLLAMA_PORT=11434" in repaired
        assert "abc" not in repaired

    def test_ram_with_suffix_repaired(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text("SERVER_RAM_GB=512GB\n")
        with patch.object(models, "NETWORK_CONF", conf), \
             patch.object(models, "CONFIG_DIR", tmp_path):
            warnings = models.validate_network_conf()
        assert any("non-numeric" in w for w in warnings)
        assert "SERVER_RAM_GB=512" in conf.read_text()

    def test_valid_config_no_warnings(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text("OLLAMA_PORT=11434\nMLX_PORT=8000\nSERVER_RAM_GB=512\n")
        with patch.object(models, "NETWORK_CONF", conf), \
             patch.object(models, "CONFIG_DIR", tmp_path):
            warnings = models.validate_network_conf()
        assert warnings == []

    def test_bad_json_prefs_warned(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text("OLLAMA_PORT=11434\n")
        prefs = tmp_path / "prefs.json"
        prefs.write_text("{broken json")
        with patch.object(models, "NETWORK_CONF", conf), \
             patch.object(models, "CONFIG_DIR", tmp_path), \
             patch.object(models, "MCP_PREFS_FILE", prefs):
            warnings = models.validate_network_conf()
        assert any("not valid JSON" in w for w in warnings)


class TestModelHasVision:
    def test_model_info_vision_keys_signal_vision(self):
        """GGUF vision models carry the vision tower; Ollama exposes it
        as `<arch>.vision.*` model_info keys. This is the honest signal."""
        from lib.models import model_has_vision
        assert model_has_vision(
            "qwen3.6:27b",
            ollama_model_info={"qwen35.vision.embedding_length": 1280},
        )
        assert model_has_vision(
            "qwen2-vl",
            ollama_model_info={"qwen2vl.vision.image_size": 448},
        )

    def test_mlx_tag_without_vision_tower_is_not_vision(self):
        """Ollama's MLX-converted tags advertise capabilities:[vision]
        but ship no vision tower — model_info has zero vision keys and
        the model silently ignores images. It must NOT be treated as
        vision-capable, or a loud error becomes silent hallucination."""
        from lib.models import model_has_vision
        assert not model_has_vision(
            "qwen3.6:27b-mlx-bf16",
            ollama_model_info={"qwen35.context_length": 262144},
        )

    def test_name_heuristic_still_works(self):
        from lib.models import model_has_vision
        assert model_has_vision("qwen3-vl:7b")
        assert not model_has_vision("nemotron:9b")


class TestMlxVlmDispatch:
    _SAMPLE = (
        "Fetching 13 files: 100%\n"
        "==========\n"
        "Files: ['/tmp/ui.png'] \n\n"
        "Prompt: <|im_start|>user\n"
        "...Click the Submit button.<|im_end|>\n"
        "<|im_start|>assistant\n\n\n"
        "<answer>\nClick(box=(529,719))\n</answer>\n\n"
        "==========\n"
        "Prompt: 697 tokens\nGeneration: 19 tokens\nPeak memory: 18.6 GB\n"
    )

    def test_parse_output_extracts_generation(self):
        from lib.mlx_vlm import parse_output
        out = parse_output(self._SAMPLE)
        assert "Click(box=(529,719))" in out
        assert "Submit button" not in out
        assert "tokens" not in out and "Peak memory" not in out

    def test_image_dimensions_png(self, tmp_path):
        import struct
        from lib.mlx_vlm import image_dimensions
        png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
               + struct.pack(">II", 800, 600))
        f = tmp_path / "x.png"
        f.write_bytes(png)
        assert image_dimensions(f) == (800, 600)

    def test_normalize_grounding_denormalizes_to_pixels(self):
        import json
        from lib.mlx_vlm import normalize_grounding
        # 529,719 in 0-1000 space on a 1000x700 image -> (529, 503)
        out = normalize_grounding("<answer>Click(box=(529,719))</answer>", 1000, 700)
        action = json.loads(out)[0]
        assert action["action"] == "click"
        assert action["x"] == 529
        assert action["y"] == 503

    def test_normalize_grounding_passes_through_unknown(self):
        from lib.mlx_vlm import normalize_grounding
        raw = "I cannot determine where to click."
        assert normalize_grounding(raw, 1000, 700) == raw

    def test_repo_for_resolves_served_name(self, tmp_path):
        from lib.mlx_vlm import repo_for
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "models:\n"
            "  - served_model_name: ui-venus\n"
            "    model_path: mlx-community/UI-Venus-1.5-8B-bf16\n")
        assert repo_for("ui-venus", cfg) == "mlx-community/UI-Venus-1.5-8B-bf16"
        assert repo_for("unknown", cfg) == "unknown"


class TestVideoTask:
    def test_video_in_special_tasks(self):
        from lib.models import SPECIAL_TASKS
        assert "video" in SPECIAL_TASKS

    def test_video_has_prefixes(self):
        from lib.models import SPECIAL_TASKS
        task = SPECIAL_TASKS["video"]
        assert "label" in task
        assert "prefixes" in task
        assert len(task["prefixes"]) > 0

    def test_video_prefixes_match_known_models(self):
        from lib.models import SPECIAL_TASKS
        prefixes = SPECIAL_TASKS["video"]["prefixes"]
        test_names = ["wan2.2-i2v", "ltx-video-2b", "Wan2.1-T2V"]
        for name in test_names:
            assert any(name.lower().startswith(p.lower()) for p in prefixes), (
                f"{name} should match a video prefix")

    def test_video_models_excluded_from_general_tasks(self):
        from lib.models import ALWAYS_EXCLUDE
        assert "wan2" in ALWAYS_EXCLUDE
        assert "ltx" in ALWAYS_EXCLUDE

    @staticmethod
    def _real_classify_model():
        """Get the real _classify_model, bypassing any test mock on the module."""
        # Other test files replace sys.modules["lib.hf_scanner"] with MagicMock.
        # Force a fresh import from the actual source file.
        saved = sys.modules.pop("lib.hf_scanner", None)
        try:
            spec = importlib.util.spec_from_file_location(
                "lib.hf_scanner",
                Path(__file__).resolve().parent.parent / "lib" / "hf_scanner.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod._classify_model
        finally:
            # Restore whatever was there so other tests aren't affected
            if saved is not None:
                sys.modules["lib.hf_scanner"] = saved

    def test_hf_scanner_classifies_wan_video(self):
        classify = self._real_classify_model()
        config = {"_class_name": "WanTransformer3DModel"}
        assert classify(config, "Wan2.2-T2V-14B") == "video"

    def test_hf_scanner_classifies_ltx_video(self):
        classify = self._real_classify_model()
        config = {"_class_name": "LTXVideoTransformer3DModel"}
        assert classify(config, "Lightricks/LTX-Video-2") == "video"

    def test_hf_scanner_classifies_ltx_by_name(self):
        classify = self._real_classify_model()
        config = {}
        assert classify(config, "ltx-video-2b-v0.9.5") == "video"

    def test_hf_scanner_classifies_wan_by_name(self):
        classify = self._real_classify_model()
        config = {}
        assert classify(config, "Wan2.1-T2V-1.3B") == "video"


class TestDefaultProfilesSeeding:
    """The installer and the menu bar app both seed profiles.json from
    DEFAULT_PROFILES (the profile server only runs when remote access is on).
    These guard the shared data contract and the menu bar seed path."""

    def test_every_preset_resolves_pullable_models(self):
        """Each preset must yield at least one Ollama (':') or HuggingFace
        ('/' without ':') model — otherwise the installer's model-pull step
        would silently resolve nothing for that profile."""
        for name, prof in menubar.DEFAULT_PROFILES["profiles"].items():
            tasks = prof.get("tasks", {})
            assert tasks, f"profile {name} has no tasks"
            pullable = [m for m in tasks.values()
                        if ":" in m or ("/" in m and ":" not in m)]
            assert pullable, f"profile {name} resolves no pullable models"

    def test_seed_writes_presets_when_missing(self, tmp_path):
        """A fresh machine with no profiles.json gets the presets seeded, so
        the installer can resolve models even with the profile server down.
        This is the regression guard for the install bug."""
        prof_path = tmp_path / "profiles.json"
        with patch.object(menubar, "PROFILES_FILE", str(prof_path)):
            assert not prof_path.exists()
            assert menubar.seed_profiles_if_missing() is True
            seeded = json.loads(prof_path.read_text())
        assert set(seeded["profiles"]) == set(menubar.DEFAULT_PROFILES["profiles"])
        assert seeded["active"] == menubar.DEFAULT_PROFILES["active"]

    def test_seed_leaves_existing_profiles_untouched(self, tmp_path):
        """If profiles.json is already at the current version, seeding must be
        a no-op and not clobber existing profiles."""
        prof_path = tmp_path / "profiles.json"
        custom = {"version": menubar.PROFILES_VERSION, "active": "mine",
                  "profiles": {"mine": {"label": "Mine", "max_ram_gb": 16,
                                        "tasks": {"code": "custom:7b"}}}}
        prof_path.write_text(json.dumps(custom))
        with patch.object(menubar, "PROFILES_FILE", str(prof_path)):
            assert menubar.seed_profiles_if_missing() is False
            result = json.loads(prof_path.read_text())
        assert result == custom

    def test_tiers_present_and_capped(self):
        profs = menubar.DEFAULT_PROFILES["profiles"]
        assert set(profs) == {"32gb", "64gb", "128gb", "512gb"}
        assert [profs[k]["max_ram_gb"] for k in ("32gb", "64gb", "128gb", "512gb")] == [32, 64, 128, 512]
        assert menubar.DEFAULT_PROFILES["active"] == "64gb"

    def test_migrate_drops_retired_presets_and_fixes_active(self):
        from lib.models import migrate_profiles, PROFILES_VERSION
        old = {"version": 25, "active": "everyday",
               "profiles": {"everyday": {"tasks": {"code": "x:1b"}},
                            "mine": {"label": "Mine", "max_ram_gb": 16, "tasks": {"code": "c:1b"}}}}
        out = migrate_profiles(old)
        assert out["version"] == PROFILES_VERSION
        assert "everyday" not in out["profiles"]          # retired preset dropped
        assert "mine" in out["profiles"]                  # real custom kept
        assert set(out["profiles"]) >= {"32gb", "64gb", "128gb", "512gb"}
        assert out["active"] in out["profiles"]           # active repaired (was "everyday")

    def test_migrate_preserves_valid_active(self):
        from lib.models import migrate_profiles
        out = migrate_profiles({"version": 25, "active": "mine",
                                "profiles": {"mine": {"max_ram_gb": 8, "tasks": {}}}})
        assert out["active"] == "mine"

    def test_seed_migrates_stale_version(self, tmp_path):
        prof = tmp_path / "profiles.json"
        prof.write_text(json.dumps({"version": 1, "active": "laptop",
                                    "profiles": {"laptop": {"tasks": {}}}}))
        with patch.object(menubar, "PROFILES_FILE", str(prof)):
            assert menubar.seed_profiles_if_missing() is True
            out = json.loads(prof.read_text())
        assert out["version"] == menubar.PROFILES_VERSION
        assert "laptop" not in out["profiles"]
        assert "64gb" in out["profiles"]

    def test_pick_profile_fallback_is_32gb(self):
        # no profile fits 8GB and none named in fallback set except presets
        assert menubar.pick_profile_for_ram(8, menubar.DEFAULT_PROFILES["profiles"]) == "32gb"

    def test_every_preset_has_valid_warm_keys(self):
        for name, prof in menubar.DEFAULT_PROFILES["profiles"].items():
            warm = prof.get("warm")
            assert warm == ["general", "embedding"], f"{name} warm={warm}"
            for key in warm:
                assert key in prof["tasks"], f"{name} warm key {key} not in tasks"

    def test_warm_model_names_resolves_active(self):
        from lib.models import warm_model_names, DEFAULT_PROFILES
        data = {"active": "128gb", "profiles": DEFAULT_PROFILES["profiles"]}
        names = warm_model_names(data)
        tasks = DEFAULT_PROFILES["profiles"]["128gb"]["tasks"]
        assert names == {tasks["general"], tasks["embedding"]}

    def test_warm_model_names_no_active(self):
        from lib.models import warm_model_names
        assert warm_model_names({"active": None, "profiles": {}}) == set()

    def test_migrate_adds_warm_to_presets_custom_absent(self):
        from lib.models import migrate_profiles
        out = migrate_profiles({"version": 26, "active": "64gb",
                                "profiles": {"mine": {"max_ram_gb": 8, "tasks": {"code": "c:1b"}}}})
        assert out["profiles"]["64gb"]["warm"] == ["general", "embedding"]
        assert "warm" not in out["profiles"]["mine"]   # custom untouched; absent ⇒ on-demand

    def test_warm_ping_targets_classifies_backend(self):
        data = {"active": "t", "profiles": {"t": {
            "warm": ["general", "embedding", "tts"],
            "tasks": {"general": "qwen3.6:27b-mlx", "embedding": "embed:8b",
                      "tts": "mlx-community/Some-TTS", "code": "coder:1b"}}}}
        targets = dict(menubar.warm_ping_targets(data))
        assert targets == {"qwen3.6:27b-mlx": "ollama", "embed:8b": "ollama"}
        # HF-repo TTS excluded (not a keep-warm server target); non-warm 'code' absent

    def test_warm_models_bare_names_are_mlx_served(self):
        """Every bare-name (no ':' and no '/') warm model in a shipped preset
        must appear as a served_model_name in config/mlx-server/config.yaml.
        This guards the string-shape heuristic in warm_ping_targets: a bare name
        is classified as 'mlx', so a bare name that isn't in the MLX config would
        silently ping the wrong backend (or no backend at all)."""
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "mlx-server" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        served = {m["served_model_name"] for m in cfg["models"]}
        for name, prof in menubar.DEFAULT_PROFILES["profiles"].items():
            tasks = prof["tasks"]
            for key in prof.get("warm", []):
                model = tasks[key]
                if ":" in model or "/" in model:
                    continue  # ollama tag or HF repo — fine, warm_ping_targets handles them
                assert model in served, (
                    f"profile {name!r} warm bare-name {model!r} "
                    f"is not a served_model_name in mlx-server/config.yaml"
                )


class TestWarmGate:
    """Contention-aware warm pings: refreshes are free, reloads are gated."""

    VM_STAT = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                97134.\n"
        "Pages active:                           15234131.\n"
        "Pages inactive:                         15856694.\n"
        "Pages speculative:                         12618.\n"
        "Pages purgeable:                           72004.\n"
    )

    def test_parse_vm_stat_available_sums_reclaimable(self):
        got = menubar.parse_vm_stat_available_gb(self.VM_STAT)
        assert got == pytest.approx(244.7, abs=0.1)

    def test_parse_vm_stat_respects_header_page_size(self):
        text = self.VM_STAT.replace("16384", "4096")
        got = menubar.parse_vm_stat_available_gb(text)
        assert got == pytest.approx(244.7 / 4, abs=0.1)

    def test_parse_vm_stat_garbage_is_none(self):
        assert menubar.parse_vm_stat_available_gb("no pages here") is None

    def test_cold_load_blocked_by_inflight(self):
        r = menubar.cold_load_skip_reason({"ollama": 2}, set(), 1, 400.0, 30.0)
        assert r is not None and "ollama=2" in r

    def test_cold_load_blocked_by_foreign_residents(self):
        r = menubar.cold_load_skip_reason({}, {"someone-elses:70b"}, 1, 400.0, 30.0)
        assert r is not None and "someone-elses:70b" in r

    def test_cold_load_blocked_by_pressure(self):
        r = menubar.cold_load_skip_reason({}, set(), 2, 400.0, 30.0)
        assert r is not None and "pressure" in r

    def test_cold_load_blocked_by_insufficient_available(self):
        r = menubar.cold_load_skip_reason({}, set(), 1, 100.0, 390.0)
        assert r is not None and "390" in r and "100" in r

    def test_cold_load_blocked_by_unknown_size(self):
        r = menubar.cold_load_skip_reason({}, set(), 1, 400.0, None)
        assert r is not None and "unknown" in r

    def test_cold_load_allowed_when_quiet_and_fits(self):
        assert menubar.cold_load_skip_reason({}, set(), 1, 400.0, 380.0) is None

    def test_cold_load_headroom_enforced(self):
        assert menubar.cold_load_skip_reason({}, set(), 1, 390.0, 380.0) is not None


class TestHeartbeatPayload:
    """build_heartbeat_payload is the pure shape used by the fleet heartbeat
    (Task 9) — the menu bar app POSTs this to /api/fleet/report every 15
    minutes so the fleet view can show per-machine usage + audit status."""

    def test_build_heartbeat_payload_shape(self):
        payload = menubar.build_heartbeat_payload(
            machine="laptop", version="v1.2.0", mode="client",
            summary=[{"day": "2026-07-10", "tool": "vision", "source": "mcp",
                      "count": 3, "errors": 0, "avg_ms": 100}],
            audit=[{"id": "claude-mcp", "status": "pass"}])
        assert payload["machine"] == "laptop"
        assert payload["version"] == "v1.2.0"
        assert payload["mode"] == "client"
        assert payload["usage"][0]["tool"] == "vision"
        assert payload["audit"][0]["status"] == "pass"
        assert "sent_at" in payload
