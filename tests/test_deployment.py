"""Tests for the auto-update, rollback, and post-update deployment pipeline.

Covers:
- get_latest_remote_tag: tag discovery and sorting
- check_repo_update_available: fetch, compare, ancestor detection
- apply_repo_update: checkout with stash/pop, failure handling
- _auto_update: post-update hook, crash marker, rollback on failure
- _startup_rollback_check: crash detection, rollback, skip marking
- _check_for_updates: skip list, idle gate, end-to-end dispatch
- _mcp_code_changed: diff detection
- _mark_startup_healthy: marker cleanup, skip unblock
- _post_update_health_check: regression detection
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Stub macOS-only modules with a real base class for rumps.App so that
# `class LocalModelsApp(rumps.App)` creates a real Python class with
# real methods (MagicMock as a base class swallows the class body).
class _RumpsAppBase:
    def __init__(self, *a, **kw): pass

_rumps_stub = MagicMock()
_rumps_stub.App = _RumpsAppBase

for mod_name, stub in [("rumps", _rumps_stub), ("objc", MagicMock()),
                        ("AppKit", MagicMock()), ("WebKit", MagicMock())]:
    sys.modules[mod_name] = stub
sys.modules["objc"].typedSelector = lambda sig: lambda fn: fn

# Force reimport so the module picks up our real rumps.App base class
# (another test file may have already imported with a MagicMock base).
import importlib
if "app.menubar" in sys.modules:
    importlib.reload(sys.modules["app.menubar"])
import app.menubar as menubar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run(returncode=0, stdout="", stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


@pytest.fixture()
def update_dir(tmp_path):
    """Temp directory for update marker files."""
    d = tmp_path / "config"
    d.mkdir()
    patches = {
        "UPDATE_STARTED_FILE": str(d / "update_started"),
        "UPDATE_SKIPPED_FILE": str(d / "update_skipped"),
        "UPDATE_PRE_HASH_FILE": str(d / "update_pre_hash"),
        "PRE_UPDATE_HEALTH_FILE": str(d / "pre_update_health.json"),
    }
    ctx = {}
    for attr, val in patches.items():
        ctx[attr] = patch.object(menubar, attr, val)
    for c in ctx.values():
        c.start()
    yield d
    for c in ctx.values():
        c.stop()


@pytest.fixture()
def app_instance(update_dir):
    """A minimal LocalModelsApp with enough state to test deployment methods."""
    inst = object.__new__(menubar.LocalModelsApp)
    inst.ollama_ok = True
    inst.mlx_ok = True
    inst.mcp_models = ["test-model"]
    inst.app_version = "v1.0.0"
    inst.update_available = 0
    inst.profile_server = None
    inst._health_checked = False
    return inst


# ---------------------------------------------------------------------------
# get_latest_remote_tag
# ---------------------------------------------------------------------------

class TestGetLatestRemoteTag:
    def test_returns_latest_tag(self):
        def mock_run(cmd, **kw):
            return _mock_run(stdout="v2.0.0\nv1.1.0\nv1.0.0\n")
        with patch("app.menubar.subprocess.run", side_effect=mock_run), \
             patch("app.menubar.subprocess.check_output", return_value="abc123\n"):
            tag, hash_ = menubar.get_latest_remote_tag()
        assert tag == "v2.0.0"
        assert hash_ == "abc123"

    def test_no_tags(self):
        with patch("app.menubar.subprocess.run", return_value=_mock_run(stdout="")):
            tag, hash_ = menubar.get_latest_remote_tag()
        assert tag == ""
        assert hash_ == ""

    def test_subprocess_failure(self):
        with patch("app.menubar.subprocess.run", side_effect=OSError("no git")):
            tag, hash_ = menubar.get_latest_remote_tag()
        assert tag == ""
        assert hash_ == ""


# ---------------------------------------------------------------------------
# check_repo_update_available
# ---------------------------------------------------------------------------

class TestCheckRepoUpdateAvailable:
    def test_no_tags_returns_zero(self):
        with patch("app.menubar.subprocess.run", return_value=_mock_run()), \
             patch("app.menubar.get_latest_remote_tag", return_value=("", "")):
            behind, tag, hash_ = menubar.check_repo_update_available()
        assert behind == 0

    def test_already_at_latest(self):
        with patch("app.menubar.subprocess.run", return_value=_mock_run()), \
             patch("app.menubar.get_latest_remote_tag", return_value=("v1.0.0", "abc123")), \
             patch("app.menubar.subprocess.check_output", return_value="abc123\n"):
            behind, _, _ = menubar.check_repo_update_available()
        assert behind == 0

    def test_head_ahead_of_tag(self):
        """HEAD is a descendant of the latest tag — no update needed."""
        def mock_run(cmd, **kw):
            if "merge-base" in cmd:
                return _mock_run(returncode=0)  # tag is ancestor of HEAD
            return _mock_run()
        with patch("app.menubar.subprocess.run", side_effect=mock_run), \
             patch("app.menubar.get_latest_remote_tag", return_value=("v1.0.0", "abc123")), \
             patch("app.menubar.subprocess.check_output", return_value="def456\n"):
            behind, _, _ = menubar.check_repo_update_available()
        assert behind == 0

    def test_behind_returns_one(self):
        """HEAD is behind the latest tag — update available."""
        def mock_run(cmd, **kw):
            if "merge-base" in cmd:
                return _mock_run(returncode=1)  # tag is NOT ancestor of HEAD
            return _mock_run()
        with patch("app.menubar.subprocess.run", side_effect=mock_run), \
             patch("app.menubar.get_latest_remote_tag", return_value=("v2.0.0", "new789")), \
             patch("app.menubar.subprocess.check_output", return_value="old123\n"):
            behind, tag, hash_ = menubar.check_repo_update_available()
        assert behind == 1
        assert tag == "v2.0.0"
        assert hash_ == "new789"

    def test_fetch_failure(self):
        with patch("app.menubar.subprocess.run",
                    return_value=_mock_run(returncode=1, stderr="network error")):
            behind, _, _ = menubar.check_repo_update_available()
        assert behind == 0


# ---------------------------------------------------------------------------
# apply_repo_update
# ---------------------------------------------------------------------------

class TestApplyRepoUpdate:
    """Tests for apply_repo_update. All patch verify_tag_signature to isolate checkout logic."""

    def _verified(self):
        return patch("app.menubar.verify_tag_signature", return_value=(True, "ok"))

    def test_clean_checkout(self):
        calls = []
        def mock_run(cmd, **kw):
            calls.append(cmd)
            if "status" in cmd:
                return _mock_run(stdout="")  # clean worktree
            if "checkout" in cmd:
                return _mock_run()
            return _mock_run()
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=mock_run):
            ok, msg = menubar.apply_repo_update("v2.0.0")
        assert ok is True
        assert "v2.0.0" in msg
        assert not any("stash" in str(c) for c in calls)

    def test_stash_and_pop(self):
        calls = []
        def mock_run(cmd, **kw):
            calls.append(cmd)
            if "status" in cmd:
                return _mock_run(stdout=" M file.py\n")  # dirty worktree
            return _mock_run()
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=mock_run):
            ok, _ = menubar.apply_repo_update("v2.0.0")
        assert ok is True
        stash_cmds = [c for c in calls if "stash" in c]
        assert len(stash_cmds) == 2  # stash + stash pop

    def test_checkout_failure(self):
        def mock_run(cmd, **kw):
            if "status" in cmd:
                return _mock_run(stdout="")
            if "checkout" in cmd:
                return _mock_run(returncode=1, stderr="error: pathspec")
            return _mock_run()
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=mock_run):
            ok, msg = menubar.apply_repo_update("v99.0.0")
        assert ok is False
        assert "pathspec" in msg

    def test_stash_pop_failure(self):
        def mock_run(cmd, **kw):
            if "status" in cmd:
                return _mock_run(stdout=" M file.py\n")
            if "stash" in cmd and "pop" in cmd:
                return _mock_run(returncode=1, stderr="CONFLICT")
            return _mock_run()
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=mock_run):
            ok, msg = menubar.apply_repo_update("v2.0.0")
        assert ok is False
        assert "CONFLICT" in msg

    def test_exception_returns_false(self):
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=OSError("disk full")):
            ok, msg = menubar.apply_repo_update("v2.0.0")
        assert ok is False
        assert "disk full" in msg

    def test_unsigned_tag_rejected(self):
        with patch("app.menubar.verify_tag_signature",
                   return_value=(False, "tag v99.0.0 is not signed")):
            ok, msg = menubar.apply_repo_update("v99.0.0")
        assert ok is False
        assert "signature" in msg.lower()

    def test_untrusted_key_rejected(self):
        with patch("app.menubar.verify_tag_signature",
                   return_value=(False, "no trusted public key for v2.0.0")):
            ok, msg = menubar.apply_repo_update("v2.0.0")
        assert ok is False
        assert "public key" in msg.lower()


# ---------------------------------------------------------------------------
# _startup_rollback_check
# ---------------------------------------------------------------------------

class TestStartupRollbackCheck:
    def test_no_marker_is_noop(self, app_instance, update_dir):
        """No update_started file → nothing happens."""
        app_instance._startup_rollback_check()
        # Should not raise or change anything

    def test_survived_crash_window(self, app_instance, update_dir):
        """Marker exists but elapsed > crash window → clears marker, no rollback."""
        marker = update_dir / "update_started"
        marker.write_text(str(time.time() - 120))  # 2 minutes ago
        app_instance._startup_rollback_check()
        assert not marker.exists()

    def test_crash_within_window_rolls_back(self, app_instance, update_dir):
        """Marker within crash window → rolls back to pre-update hash."""
        marker = update_dir / "update_started"
        marker.write_text(str(time.time() - 5))  # 5 seconds ago
        pre_hash = update_dir / "update_pre_hash"
        pre_hash.write_text("oldcommit123")

        with patch("app.menubar.subprocess.check_output", return_value="badcommit456\n"), \
             patch("app.menubar.subprocess.run", return_value=_mock_run()) as mock_run, \
             patch("app.menubar.get_version", return_value="v1.0.0"):
            app_instance._startup_rollback_check()

        # Should have checked out the old commit
        checkout_calls = [c for c in mock_run.call_args_list
                          if "checkout" in str(c)]
        assert len(checkout_calls) == 1
        assert "oldcommit123" in str(checkout_calls[0])

        # Should have written the bad hash to skip file
        skip_file = update_dir / "update_skipped"
        assert skip_file.exists()
        assert skip_file.read_text() == "badcommit456"

        # Marker should be cleaned up
        assert not marker.exists()

    def test_crash_no_pre_hash(self, app_instance, update_dir):
        """Crash detected but no pre-hash → logs error, does not crash."""
        marker = update_dir / "update_started"
        marker.write_text(str(time.time() - 5))

        with patch("app.menubar.subprocess.check_output", return_value="bad123\n"), \
             patch("app.menubar.subprocess.run") as mock_run:
            app_instance._startup_rollback_check()

        # Should NOT have attempted checkout (no rollback target)
        checkout_calls = [c for c in mock_run.call_args_list
                          if "checkout" in str(c)]
        assert len(checkout_calls) == 0


# ---------------------------------------------------------------------------
# _check_for_updates (integration of skip + idle + dispatch)
# ---------------------------------------------------------------------------

class TestCheckForUpdates:
    def test_skips_rolled_back_release(self, app_instance, update_dir):
        """A previously rolled-back hash is skipped."""
        skip_file = update_dir / "update_skipped"
        skip_file.write_text("badrelease123")

        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "badrelease123")), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_not_called()

    def test_defers_when_mcp_active(self, app_instance, update_dir):
        """Update deferred when MCP was recently active."""
        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "newrelease")), \
             patch.object(app_instance, "_mcp_recently_active", return_value=True), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_not_called()

    def test_triggers_auto_update(self, app_instance, update_dir):
        """Normal update triggers _auto_update."""
        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "newrelease")), \
             patch.object(app_instance, "_mcp_recently_active", return_value=False), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_called_once_with("v2.0.0")

    def test_no_update_available(self, app_instance, update_dir):
        with patch("app.menubar.check_repo_update_available",
                    return_value=(0, "", "")), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_not_called()
        assert app_instance.update_available == 0


# ---------------------------------------------------------------------------
# _auto_update
# ---------------------------------------------------------------------------

class TestAutoUpdate:
    def test_runs_post_update_hook(self, app_instance, update_dir):
        """post-update.sh is called after successful checkout."""
        post_update_ran = []

        def mock_run(cmd, **kw):
            if isinstance(cmd, list) and "post-update.sh" in str(cmd[-1]):
                post_update_ran.append(cmd)
            return _mock_run(stdout="[post-update] Done")

        with patch("app.menubar.apply_repo_update", return_value=(True, "ok")), \
             patch("app.menubar.subprocess.run", side_effect=mock_run), \
             patch("app.menubar.subprocess.check_output", return_value="abc123\n"), \
             patch("app.menubar.get_version", return_value="v1.0.0"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False), \
             patch.object(app_instance, "_stop_mcp_server", create=True), \
             patch("os.path.isfile", return_value=True), \
             patch("os._exit") as mock_exit:
            app_instance._auto_update("v2.0.0")

        assert len(post_update_ran) == 1
        mock_exit.assert_called_once_with(1)

    def test_failed_checkout_rolls_back(self, app_instance, update_dir):
        """Failed checkout resets to pre-update hash."""
        with patch("app.menubar.apply_repo_update",
                    return_value=(False, "checkout failed")), \
             patch("app.menubar.subprocess.check_output", return_value="pre123\n"), \
             patch("app.menubar.subprocess.run") as mock_run, \
             patch("app.menubar.get_version", return_value="v1.0.0"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False):
            app_instance._auto_update("v2.0.0")

        # Should have reset --hard to pre-update hash
        reset_calls = [c for c in mock_run.call_args_list
                       if "reset" in str(c)]
        assert len(reset_calls) == 1
        assert "pre123" in str(reset_calls[0])

    def test_writes_crash_marker(self, app_instance, update_dir):
        """update_started file is written after successful checkout."""
        with patch("app.menubar.apply_repo_update", return_value=(True, "ok")), \
             patch("app.menubar.subprocess.run", return_value=_mock_run()), \
             patch("app.menubar.subprocess.check_output", return_value="abc\n"), \
             patch("app.menubar.get_version", return_value="v1.0.0"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False), \
             patch.object(app_instance, "_stop_mcp_server", create=True), \
             patch("os.path.isfile", return_value=False), \
             patch("os._exit"):
            app_instance._auto_update("v2.0.0")

        marker = update_dir / "update_started"
        assert marker.exists()
        ts = float(marker.read_text())
        assert time.time() - ts < 5


# ---------------------------------------------------------------------------
# _mcp_code_changed
# ---------------------------------------------------------------------------

class TestMcpCodeChanged:
    def test_detects_changes(self, app_instance):
        with patch("app.menubar.subprocess.run",
                    return_value=_mock_run(stdout="mcp/server.py\n")):
            assert app_instance._mcp_code_changed("v2.0.0") is True

    def test_no_changes(self, app_instance):
        with patch("app.menubar.subprocess.run",
                    return_value=_mock_run(stdout="")):
            assert app_instance._mcp_code_changed("v2.0.0") is False

    def test_error_assumes_changed(self, app_instance):
        with patch("app.menubar.subprocess.run", side_effect=OSError("fail")):
            assert app_instance._mcp_code_changed("v2.0.0") is True


# ---------------------------------------------------------------------------
# _mark_startup_healthy
# ---------------------------------------------------------------------------

class TestMarkStartupHealthy:
    def test_clears_update_files(self, app_instance, update_dir):
        (update_dir / "update_started").write_text("123")
        (update_dir / "update_pre_hash").write_text("abc")
        app_instance._mark_startup_healthy()
        assert not (update_dir / "update_started").exists()
        assert not (update_dir / "update_pre_hash").exists()

    def test_unblocks_skipped_on_new_commit(self, app_instance, update_dir):
        """If current HEAD differs from skipped hash, clear the skip file."""
        skip = update_dir / "update_skipped"
        skip.write_text("oldbadhash")
        with patch("app.menubar.subprocess.check_output", return_value="newgoodhash\n"):
            app_instance._mark_startup_healthy()
        assert not skip.exists()

    def test_keeps_skip_on_same_commit(self, app_instance, update_dir):
        """If HEAD is still the skipped commit, keep the skip file."""
        skip = update_dir / "update_skipped"
        skip.write_text("samehash")
        with patch("app.menubar.subprocess.check_output", return_value="samehash\n"):
            app_instance._mark_startup_healthy()
        assert skip.exists()


# ---------------------------------------------------------------------------
# _post_update_health_check
# ---------------------------------------------------------------------------

class TestPostUpdateHealthCheck:
    def test_no_health_file_is_noop(self, app_instance, update_dir):
        app_instance._post_update_health_check()
        assert app_instance._health_checked is True

    def test_no_regressions(self, app_instance, update_dir):
        health = update_dir / "pre_update_health.json"
        health.write_text(json.dumps({
            "ollama": True, "mlx": True, "mcp": True, "version": "v1.0.0"}))
        app_instance._post_update_health_check()
        assert not health.exists()  # cleaned up

    def test_detects_ollama_regression(self, app_instance, update_dir):
        health = update_dir / "pre_update_health.json"
        health.write_text(json.dumps({
            "ollama": True, "mlx": True, "mcp": True, "version": "v1.0.0"}))
        app_instance.ollama_ok = False
        app_instance._post_update_health_check()
        assert not health.exists()

    def test_only_runs_once(self, app_instance, update_dir):
        health = update_dir / "pre_update_health.json"
        health.write_text(json.dumps({
            "ollama": True, "mlx": True, "mcp": True, "version": "v1.0.0"}))
        app_instance._post_update_health_check()
        app_instance._health_checked = True
        # Second call should be a noop (file already deleted)
        app_instance._post_update_health_check()


# ---------------------------------------------------------------------------
# _mcp_recently_active
# ---------------------------------------------------------------------------

class TestMcpRecentlyActive:
    def test_recent_log_returns_true(self, app_instance, tmp_path):
        log = tmp_path / "mcp.log"
        log.write_text("some log")
        with patch.object(menubar, "MCP_LOG_FILE", str(log)):
            assert app_instance._mcp_recently_active() is True

    def test_stale_log_returns_false(self, app_instance, tmp_path):
        log = tmp_path / "mcp.log"
        log.write_text("some log")
        old_time = time.time() - 300
        os.utime(str(log), (old_time, old_time))
        with patch.object(menubar, "MCP_LOG_FILE", str(log)):
            assert app_instance._mcp_recently_active() is False

    def test_missing_log_returns_false(self, app_instance, tmp_path):
        with patch.object(menubar, "MCP_LOG_FILE", str(tmp_path / "nope.log")):
            assert app_instance._mcp_recently_active() is False
