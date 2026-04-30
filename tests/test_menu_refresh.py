"""Tests for the periodic menu-refresh path in app/menubar.py.

A bug introduced in bc63ba2 (Apr 9 2026) removed `self.menu_share_url` and
`self.menu_restart` from `__init__` but left dangling references in
`_update_menu`. The result was an `AttributeError` on every poll on the
desktop with Remote Access enabled. PyObjC converted it into an NSException;
after ~50 minutes of accumulated faults FrontBoard invalidated the workspace
client connection and the menubar exited cleanly (exit code 0). Because the
launchd KeepAlive policy is `SuccessfulExit: false`, the app stayed dead.

These tests pin `_update_menu` so that any future missing-attribute regression
fails loudly in CI instead of silently piling up in macOS unified logs.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class _RumpsAppBase:
    def __init__(self, *a, **kw):
        pass


_rumps_stub = MagicMock()
_rumps_stub.App = _RumpsAppBase

for mod_name, stub in [("rumps", _rumps_stub), ("objc", MagicMock()),
                       ("AppKit", MagicMock()), ("WebKit", MagicMock())]:
    sys.modules[mod_name] = stub
sys.modules["objc"].typedSelector = lambda sig: lambda fn: fn

import importlib
if "app.menubar" in sys.modules:
    importlib.reload(sys.modules["app.menubar"])
import app.menubar as menubar


def _bare_app(**overrides):
    """Build a minimal LocalModelsApp instance for menu-refresh tests."""
    inst = object.__new__(menubar.LocalModelsApp)
    inst.title = None
    inst.desktop = True
    inst.remote_access_enabled = True
    inst.force_local = False
    inst.mode = "server"
    inst.servers_started = True
    inst.app_version = "v1.0.0"
    inst.ollama_models = []
    inst.mlx_models = []
    inst.mcp_models = []
    inst.ollama_ok = False
    inst.mlx_ok = False
    inst.ollama_loading = False
    inst.mlx_loading = False
    inst.remote_reachable = False
    inst._mcp_proc = None
    inst._last_restart_attempt = 0
    inst.desktop_ip = ""
    inst.desktop_fqdn = ""
    for attr in ("menu_status", "menu_remote_access", "menu_profiles",
                 "menu_ollama", "menu_mlx", "menu_mcp",
                 "menu_ollama_restart", "menu_mlx_restart", "menu_mcp_restart",
                 "menu_version", "menu_mode_remote", "menu_mode_local",
                 "menu_tools_sub", "menu_playground", "menu_quit"):
        setattr(inst, attr, MagicMock())
    inst.__dict__.update(overrides)
    return inst


@pytest.fixture(autouse=True)
def _silence_rendering():
    """No-op out the AppKit-touching helpers so tests run on any machine."""
    with patch.object(menubar.LocalModelsApp, "_styled_menu",
                      lambda *a, **kw: None), \
         patch.object(menubar.LocalModelsApp, "_start_mcp_server",
                      lambda self: None), \
         patch("app.menubar.load_profiles",
               return_value={"active": None, "profiles": {}}), \
         patch("app.menubar.probe_port", return_value=False):
        yield


class TestUpdateMenuNoMissingAttrs:
    """`_update_menu` must not reference attributes that __init__ never sets.

    Each test exercises a code path that previously raised AttributeError
    after commit bc63ba2 dropped the corresponding menu items.
    """

    def test_desktop_remote_access_enabled(self):
        """Desktop + Remote Access on — used to die at self.menu_share_url."""
        app = _bare_app(remote_access_enabled=True)
        app._update_menu()  # must not raise

    def test_desktop_remote_access_disabled(self):
        """Desktop + Remote Access off — also references menu_share_url."""
        app = _bare_app(remote_access_enabled=False)
        app._update_menu()

    def test_desktop_full_path_runs_to_completion(self):
        """All the way through MCP probe + Restart wiring (line ~2200)."""
        app = _bare_app(remote_access_enabled=True, ollama_ok=True,
                        mlx_ok=True, ollama_models=["m1"], mlx_models=["m2"],
                        mcp_models=["x"])
        app._update_menu()

    def test_laptop_client_mode(self):
        """Laptop in client mode — different code path, must also not raise."""
        app = _bare_app(desktop=False, remote_access_enabled=False,
                        mode="client", remote_reachable=True,
                        mcp_models=["remote-model"])
        app._update_menu()


class TestRemoteReachabilityRequiresLiveBackend:
    """Tailscale serve listens on 8100 even when the MCP backend is dead.

    The laptop must not declare the desktop "connected" purely on the basis
    of a TCP probe — it has to confirm the backend actually serves. The user
    saw the wrong behavior (status: connected, models: 0) when the desktop
    menubar crashed but tailscale serve was still proxying.
    """

    def test_tcp_up_but_mcp_empty_marks_remote_unreachable(self):
        """probe_port=True + get_mcp_models=[] → remote_reachable=False."""
        app = object.__new__(menubar.LocalModelsApp)
        app.ts_hostname = "super-puppy"
        app.force_local = False
        app.remote_reachable = True  # was previously connected
        app.servers_started = True
        app.desktop = False
        app.desktop_ip = "100.64.0.1"
        app.desktop_fqdn = "super-puppy.tailnet.ts.net"
        app.mode = "client"
        app.ollama_ok = False
        app.mlx_ok = False
        app.ollama_models = []
        app.mlx_models = []
        app.mcp_models = ["stale"]

        with patch.object(menubar.LocalModelsApp, "_resolve_desktop",
                          return_value=True), \
             patch("app.menubar.http_get_json", return_value=None), \
             patch("app.menubar.get_mcp_models", return_value=[]), \
             patch("app.menubar.probe_service", return_value=False), \
             patch.object(menubar.LocalModelsApp, "_notify_connection",
                          lambda *a, **kw: None):
            app._refresh_client_mode()

        assert app.remote_reachable is False, \
            "Desktop with dead MCP backend must not count as reachable"

    def test_tcp_up_and_mcp_serving_marks_remote_reachable(self):
        """Healthy backend → remote_reachable stays True."""
        app = object.__new__(menubar.LocalModelsApp)
        app.ts_hostname = "super-puppy"
        app.force_local = False
        app.remote_reachable = False
        app.servers_started = True
        app.desktop = False
        app.desktop_ip = "100.64.0.1"
        app.desktop_fqdn = "super-puppy.tailnet.ts.net"
        app.mode = "offline"
        app.ollama_ok = False
        app.mlx_ok = False
        app.ollama_models = []
        app.mlx_models = []
        app.mcp_models = []

        live_models = [{"name": "qwen3.6:27b"}, {"name": "tinyllama"}]
        with patch.object(menubar.LocalModelsApp, "_resolve_desktop",
                          return_value=True), \
             patch("app.menubar.http_get_json", return_value=live_models), \
             patch.object(menubar.LocalModelsApp, "_notify_connection",
                          lambda *a, **kw: None):
            app._refresh_client_mode()

        assert app.remote_reachable is True
        assert app.mode == "client"
        assert app.mcp_models == ["qwen3.6:27b", "tinyllama"]


class TestForceLocalDoesNotTrapTheUser:
    """The Remote toggle is the only UI path that clears FORCE_LOCAL=true.
    `_select_remote` early-returns when `remote_reachable` is False, and the
    menu greys the toggle for the same reason. So if `_refresh_client_mode`
    skips the desktop probe whenever `force_local` is True, the laptop ends
    up in a state where the user *can never get out of local override*: the
    probe is gated on `not force_local`, which makes `remote_reachable=False`,
    which disables the toggle that would have cleared `force_local`. One-way
    trap, no UI escape. force_local should control whether we *use* the
    desktop, not whether we know it's there.
    """

    def test_force_local_with_live_desktop_still_marks_remote_reachable(self):
        """force_local=True + healthy desktop → remote_reachable=True, mode≠client."""
        app = object.__new__(menubar.LocalModelsApp)
        app.ts_hostname = "super-puppy"
        app.force_local = True  # the trap
        app.remote_reachable = False
        app.servers_started = True
        app.desktop = False
        app.desktop_ip = "100.64.0.1"
        app.desktop_fqdn = "super-puppy.tailnet.ts.net"
        app.mode = "offline"
        app.ollama_ok = False
        app.mlx_ok = False
        app.ollama_models = []
        app.mlx_models = []
        app.mcp_models = []

        live_models = [{"name": "qwen3.6:27b"}, {"name": "tinyllama"}]
        with patch.object(menubar.LocalModelsApp, "_resolve_desktop",
                          return_value=True), \
             patch("app.menubar.http_get_json", return_value=live_models), \
             patch("app.menubar.probe_service", return_value=False), \
             patch.object(menubar.LocalModelsApp, "_notify_connection",
                          lambda *a, **kw: None):
            app._refresh_client_mode()

        assert app.remote_reachable is True, \
            "Healthy desktop must register as reachable even under force_local"
        assert app.mode != "client", \
            "force_local must still keep us out of client mode"

    def test_force_local_with_dead_desktop_keeps_remote_unreachable(self):
        """force_local=True + dead desktop → remote_reachable=False (no false positive)."""
        app = object.__new__(menubar.LocalModelsApp)
        app.ts_hostname = "super-puppy"
        app.force_local = True
        app.remote_reachable = False
        app.servers_started = True
        app.desktop = False
        app.desktop_ip = ""
        app.desktop_fqdn = ""
        app.mode = "offline"
        app.ollama_ok = False
        app.mlx_ok = False
        app.ollama_models = []
        app.mlx_models = []
        app.mcp_models = []

        with patch.object(menubar.LocalModelsApp, "_resolve_desktop",
                          return_value=False), \
             patch("app.menubar.probe_service", return_value=False), \
             patch.object(menubar.LocalModelsApp, "_notify_connection",
                          lambda *a, **kw: None):
            app._refresh_client_mode()

        assert app.remote_reachable is False
