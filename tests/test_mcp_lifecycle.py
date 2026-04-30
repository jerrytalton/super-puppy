"""Tests for MCP server lifecycle management in app/menubar.py.

Background: every menubar startup ran `_start_mcp_server`, which unconditionally
SIGTERM'd anything holding port 8100 before spawning a new MCP. Combined with
auto-update (menubar exits, launchd respawns on the new tag), this meant every
release killed every Claude Code session connected to the local MCP — including
sessions on other machines connected through Tailscale serve to this one.

The fix: probe the orphan first. If it's a healthy MCP responding to our
`/api/mcp-models` endpoint, adopt it (just record the PID and update the
Claude config) instead of killing and respawning. Only kill orphans that
look broken.
"""

import sys
import subprocess
from unittest.mock import MagicMock, patch

import pytest


class _RumpsAppBase:
    def __init__(self, *a, **kw):
        pass


_rumps_stub = MagicMock()
_rumps_stub.App = _RumpsAppBase

for mod_name, stub in [("rumps", _rumps_stub), ("objc", MagicMock()),
                       ("AppKit", MagicMock()), ("WebKit", MagicMock())]:
    sys.modules.setdefault(mod_name, stub)
sys.modules["objc"].typedSelector = lambda sig: lambda fn: fn

import importlib
if "app.menubar" in sys.modules:
    importlib.reload(sys.modules["app.menubar"])
import app.menubar as menubar


def _mcp_app(**overrides):
    """Build an instance configured for `_start_mcp_server` paths."""
    inst = object.__new__(menubar.LocalModelsApp)
    inst.desktop = True            # skip the remote-mode early return
    inst.force_local = True
    inst.remote_reachable = False
    inst._mcp_proc = None
    inst._mcp_proc_pid = None
    inst._mcp_log = None
    inst.ts_hostname = ""
    inst.desktop_fqdn = ""
    inst.__dict__.update(overrides)
    return inst


class TestStartMcpServerAdoptsHealthyOrphan:
    """When something is already serving on :8100 with our auth and returning
    models, leave it alone — kill+respawn would take down all connected Claude
    sessions for no benefit. Adoption preserves them across menubar restarts.
    """

    def test_healthy_orphan_is_adopted_no_kill_no_spawn(self):
        """lsof finds pid 9999, /api/mcp-models returns models → no SIGTERM, no Popen."""
        app = _mcp_app()
        # lsof says pid 9999 holds 8100
        check_output = MagicMock(return_value="9999\n")
        # Health probe says it's our MCP
        get_models = MagicMock(return_value=["qwen3.5", "tinyllama"])
        # Spy on os.kill and Popen to assert they're never called
        os_kill = MagicMock()
        popen = MagicMock()
        configure_claude = MagicMock()

        with patch("app.menubar.subprocess.check_output", check_output), \
             patch("app.menubar.get_mcp_models", get_models), \
             patch("app.menubar.os.kill", os_kill), \
             patch("app.menubar.os.getpid", return_value=1), \
             patch("app.menubar.subprocess.Popen", popen), \
             patch.object(menubar.LocalModelsApp, "_configure_claude_mcp",
                          configure_claude):
            app._start_mcp_server()

        os_kill.assert_not_called()
        popen.assert_not_called()
        assert app._mcp_proc_pid == 9999, \
            "Adopted PID must be recorded so _stop_mcp_server can find it later"
        configure_claude.assert_called_once_with("http://127.0.0.1:8100/mcp")

    def test_unhealthy_orphan_is_killed_and_replaced(self):
        """lsof finds pid 9999, /api/mcp-models returns [] → SIGTERM, then spawn."""
        app = _mcp_app()
        check_output = MagicMock(return_value="9999\n")
        get_models = MagicMock(return_value=[])  # not our MCP, or broken
        os_kill = MagicMock()
        popen_proc = MagicMock(pid=12345)
        popen = MagicMock(return_value=popen_proc)
        configure_claude = MagicMock()

        with patch("app.menubar.subprocess.check_output", check_output), \
             patch("app.menubar.get_mcp_models", get_models), \
             patch("app.menubar.os.kill", os_kill), \
             patch("app.menubar.os.getpid", return_value=1), \
             patch("app.menubar.subprocess.Popen", popen), \
             patch("app.menubar.time.sleep", lambda *a, **k: None), \
             patch("builtins.open", MagicMock()), \
             patch.object(menubar.LocalModelsApp, "_configure_claude_mcp",
                          configure_claude):
            app._start_mcp_server()

        # Existing behavior preserved for unhealthy orphan: kill + spawn fresh
        assert any(call.args[0] == 9999 for call in os_kill.call_args_list), \
            "Unhealthy orphan must be SIGTERM'd"
        popen.assert_called_once()
        assert app._mcp_proc is popen_proc

    def test_stop_uses_pid_when_proc_handle_is_none(self):
        """After adoption we have a PID but no Popen — stop must still SIGTERM it."""
        app = _mcp_app(_mcp_proc=None, _mcp_proc_pid=9999)
        os_kill = MagicMock(side_effect=OSError())  # short-circuit alive-poll loop
        with patch("app.menubar.os.kill", os_kill), \
             patch("app.menubar.time.sleep", lambda *a, **k: None):
            app._stop_mcp_server()
        kill_calls = [c.args for c in os_kill.call_args_list]
        assert (9999, menubar.signal.SIGTERM) in kill_calls, \
            "_stop_mcp_server must SIGTERM the adopted PID"
        assert app._mcp_proc_pid is None, "PID must be cleared after stop"

    def test_no_orphan_just_spawns(self):
        """lsof returns nothing → no kill, just spawn fresh."""
        app = _mcp_app()
        # lsof exits nonzero when nothing matches
        check_output = MagicMock(
            side_effect=subprocess.CalledProcessError(1, ["lsof"]))
        os_kill = MagicMock()
        popen_proc = MagicMock(pid=12345)
        popen = MagicMock(return_value=popen_proc)
        configure_claude = MagicMock()

        with patch("app.menubar.subprocess.check_output", check_output), \
             patch("app.menubar.get_mcp_models", MagicMock(return_value=[])), \
             patch("app.menubar.os.kill", os_kill), \
             patch("app.menubar.os.getpid", return_value=1), \
             patch("app.menubar.subprocess.Popen", popen), \
             patch("app.menubar.time.sleep", lambda *a, **k: None), \
             patch("builtins.open", MagicMock()), \
             patch.object(menubar.LocalModelsApp, "_configure_claude_mcp",
                          configure_claude):
            app._start_mcp_server()

        os_kill.assert_not_called()
        popen.assert_called_once()
