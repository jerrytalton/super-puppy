"""Tests for pure/near-pure functions in app/menubar.py."""

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

import importlib
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
    def test_simple_date(self):
        """Single commit on a date produces YYYY.M.D with no zero-padding."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.side_effect = [
                "2026-04-01 12:00:00 -0700\n",  # git log --format=%ai
                "1\n",                            # rev-list --count
            ]
            assert menubar.get_version("HEAD") == "2026.4.1"

    def test_build_number_for_multiple_commits(self):
        """Multiple commits on the same date appends a build number."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.side_effect = [
                "2026-01-15 09:00:00 +0000\n",
                "3\n",
            ]
            assert menubar.get_version("HEAD") == "2026.1.15.3"

    def test_returns_dev_on_failure(self):
        """Any subprocess error yields 'dev'."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.side_effect = subprocess.CalledProcessError(1, "git")
            assert menubar.get_version("HEAD") == "dev"

    def test_no_zero_padding(self):
        """Months and days must not be zero-padded."""
        with patch("app.menubar.subprocess") as mock_sub:
            mock_sub.check_output.side_effect = [
                "2026-09-05 00:00:00 +0000\n",
                "1\n",
            ]
            assert menubar.get_version("HEAD") == "2026.9.5"


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
