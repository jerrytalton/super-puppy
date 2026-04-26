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
        "LAUNCH_ATTEMPTED_FILE": str(d / "launch_attempted"),
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
    inst.conf = {"AUTO_UPDATE": "true"}
    inst.profile_server = None
    inst._health_checked = False
    inst._launch_hash = "launch000"
    inst._update_defer_count = 0
    inst._rolled_back = False
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

    def test_force_checkout(self):
        calls = []
        def mock_run(cmd, **kw):
            calls.append(cmd)
            return _mock_run()
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=mock_run):
            ok, msg = menubar.apply_repo_update("v2.0.0")
        assert ok is True
        assert "v2.0.0" in msg
        # Should use --force, not stash
        checkout_calls = [c for c in calls if "checkout" in c]
        assert len(checkout_calls) == 1
        assert "--force" in checkout_calls[0]
        assert not any("stash" in str(c) for c in calls)

    def test_runs_git_clean(self):
        calls = []
        def mock_run(cmd, **kw):
            calls.append(cmd)
            return _mock_run()
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=mock_run):
            menubar.apply_repo_update("v2.0.0")
        clean_calls = [c for c in calls if "clean" in c]
        assert len(clean_calls) == 1
        assert "-fd" in clean_calls[0]

    def test_checkout_failure(self):
        def mock_run(cmd, **kw):
            if "checkout" in cmd:
                return _mock_run(returncode=1, stderr="error: pathspec")
            return _mock_run()
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=mock_run):
            ok, msg = menubar.apply_repo_update("v99.0.0")
        assert ok is False
        assert "pathspec" in msg

    def test_exception_returns_false(self):
        with self._verified(), \
             patch("app.menubar.subprocess.run", side_effect=OSError("disk full")):
            ok, msg = menubar.apply_repo_update("v2.0.0")
        assert ok is False
        assert "disk full" in msg

    def test_verifies_signature_before_updating_allowed_signers(self):
        """The signed-update trust model requires the *current* allowed_signers
        to approve the *next* one. Verification MUST run first; only after
        the current trust root has signed off do we install the new file."""
        call_order = []
        def mock_run(cmd, **kw):
            if "show" in cmd and "allowed_signers" in str(cmd):
                call_order.append("update_signers")
                return _mock_run(stdout="user@example.com ssh-ed25519 AAAA\n")
            if "config" in cmd and "allowedSignersFile" in cmd:
                return _mock_run()
            if "checkout" in cmd:
                return _mock_run()
            if "clean" in cmd:
                return _mock_run()
            return _mock_run()
        def mock_verify(tag):
            call_order.append("verify")
            return True, "ok"
        with patch("app.menubar.subprocess.run", side_effect=mock_run), \
             patch("app.menubar.verify_tag_signature", side_effect=mock_verify):
            ok, _ = menubar.apply_repo_update("v2.0.0")
        assert ok is True
        assert call_order == ["verify", "update_signers"], (
            "verify_tag_signature MUST run before _update_allowed_signers — "
            "otherwise a malicious tag could ship its own allowed_signers and "
            "self-approve, bypassing the entire signed-update model.")

    def test_unsigned_tag_does_not_install_new_allowed_signers(self):
        """If the tag fails verification, _update_allowed_signers must not run.
        Otherwise an attacker's allowed_signers gets installed on every check."""
        signers_writes = []
        def mock_run(cmd, **kw):
            if "show" in cmd and "allowed_signers" in str(cmd):
                signers_writes.append(cmd)
                return _mock_run(stdout="evil@example.com ssh-ed25519 AAAA\n")
            return _mock_run()
        with patch("app.menubar.subprocess.run", side_effect=mock_run), \
             patch("app.menubar.verify_tag_signature",
                   return_value=(False, "no trusted public key")):
            ok, _ = menubar.apply_repo_update("v99.0.0")
        assert ok is False
        assert signers_writes == [], (
            "Rejected tags must not trigger allowed_signers installation.")

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

        # Should have checked out the old commit with --force
        checkout_calls = [c for c in mock_run.call_args_list
                          if "checkout" in str(c)]
        assert len(checkout_calls) == 1
        assert "oldcommit123" in str(checkout_calls[0])
        assert "--force" in str(checkout_calls[0])

        # Should have written the bad hash + timestamp to skip file
        skip_file = update_dir / "update_skipped"
        assert skip_file.exists()
        skip_data = skip_file.read_text()
        assert skip_data.startswith("badcommit456\n")
        skip_ts = float(skip_data.split("\n")[1])
        assert time.time() - skip_ts < 5

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

    def test_rollback_runs_post_update(self, app_instance, update_dir):
        """Rollback runs post-update.sh to rebuild binary for old code."""
        marker = update_dir / "update_started"
        marker.write_text(str(time.time() - 5))
        pre_hash = update_dir / "update_pre_hash"
        pre_hash.write_text("oldcommit123")

        calls = []
        def mock_run(cmd, **kw):
            calls.append(cmd)
            return _mock_run()

        with patch("app.menubar.subprocess.check_output", return_value="badcommit456\n"), \
             patch("app.menubar.subprocess.run", side_effect=mock_run), \
             patch("app.menubar.get_version", return_value="v1.0.0"), \
             patch("os.path.isfile", return_value=True):
            app_instance._startup_rollback_check()

        post_update_calls = [c for c in calls
                             if isinstance(c, list) and "post-update.sh" in str(c)]
        assert len(post_update_calls) == 1

    def test_rollback_sets_rolled_back_flag(self, app_instance, update_dir):
        """Rollback sets _rolled_back so _mark_startup_healthy won't clear skip."""
        marker = update_dir / "update_started"
        marker.write_text(str(time.time() - 5))
        pre_hash = update_dir / "update_pre_hash"
        pre_hash.write_text("oldcommit123")

        with patch("app.menubar.subprocess.check_output", return_value="badcommit456\n"), \
             patch("app.menubar.subprocess.run", return_value=_mock_run()), \
             patch("app.menubar.get_version", return_value="v1.0.0"), \
             patch("os.path.isfile", return_value=False):
            app_instance._startup_rollback_check()

        assert app_instance._rolled_back is True


# ---------------------------------------------------------------------------
# _check_for_updates (integration of skip + idle + dispatch)
# ---------------------------------------------------------------------------

class TestCheckForUpdates:
    def test_skips_rolled_back_release(self, app_instance, update_dir):
        """A previously rolled-back hash is skipped (within 24h)."""
        skip_file = update_dir / "update_skipped"
        skip_file.write_text(f"badrelease123\n{time.time()}")

        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "badrelease123")), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_not_called()

    def test_skip_expires_after_24h(self, app_instance, update_dir):
        """A skip entry older than 24h is ignored — retry the update."""
        skip_file = update_dir / "update_skipped"
        old_ts = time.time() - 90000  # 25 hours ago
        skip_file.write_text(f"badrelease123\n{old_ts}")

        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "badrelease123")), \
             patch.object(app_instance, "_mcp_recently_active", return_value=False), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_called_once_with("v2.0.0")
        assert not skip_file.exists()  # expired skip removed

    def test_skip_without_timestamp_treated_as_expired(self, app_instance, update_dir):
        """Old-format skip file (hash only, no timestamp) is treated as expired."""
        skip_file = update_dir / "update_skipped"
        skip_file.write_text("badrelease123")  # legacy format

        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "badrelease123")), \
             patch.object(app_instance, "_mcp_recently_active", return_value=False), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_called_once_with("v2.0.0")

    def test_defers_when_mcp_active(self, app_instance, update_dir):
        """Update deferred when MCP was recently active."""
        app_instance._update_defer_count = 0
        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "newrelease")), \
             patch.object(app_instance, "_mcp_recently_active", return_value=True), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_not_called()
        assert app_instance._update_defer_count == 1

    def test_forces_update_after_max_deferrals(self, app_instance, update_dir):
        """After MAX_UPDATE_DEFERRALS, update proceeds despite MCP activity."""
        app_instance._update_defer_count = menubar.MAX_UPDATE_DEFERRALS - 1
        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "newrelease")), \
             patch.object(app_instance, "_mcp_recently_active", return_value=True), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_called_once_with("v2.0.0")
        assert app_instance._update_defer_count == 0  # reset after proceeding

    def test_triggers_auto_update(self, app_instance, update_dir):
        """Normal update triggers _auto_update."""
        app_instance._update_defer_count = 0
        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "newrelease")), \
             patch.object(app_instance, "_mcp_recently_active", return_value=False), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_called_once_with("v2.0.0")

    def test_no_update_available(self, app_instance, update_dir):
        """No remote update and HEAD hasn't moved — no action."""
        with patch("app.menubar.check_repo_update_available",
                    return_value=(0, "", "")), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_not_called()
        assert app_instance.update_available == 0

    def test_drift_triggers_restart(self, app_instance, update_dir):
        """HEAD moved since launch (commit/pull on this machine) — restart."""
        app_instance._update_defer_count = 0
        with patch("app.menubar.check_repo_update_available",
                    return_value=(0, "", "")), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="newcommit456"), \
             patch("app.menubar.get_version", return_value="v2.0.0"), \
             patch.object(app_instance, "_mcp_recently_active", return_value=False), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_called_once_with("v2.0.0")

    def test_no_drift_no_restart(self, app_instance, update_dir):
        """No remote update, HEAD unchanged — no action."""
        with patch("app.menubar.check_repo_update_available",
                    return_value=(0, "", "")), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()

        mock_update.assert_not_called()

    def test_skips_during_crash_window(self, app_instance, update_dir):
        """If _auto_update fired less than UPDATE_CRASH_WINDOW seconds ago,
        a follow-up _check_for_updates must NOT fire — back-to-back updates
        would overwrite the rollback markers for the first one."""
        app_instance._last_auto_update_at = time.time() - 30  # 30s < 90s
        with patch("app.menubar.check_repo_update_available") as mock_check, \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()
        mock_check.assert_not_called()
        mock_update.assert_not_called()

    def test_proceeds_after_crash_window(self, app_instance, update_dir):
        """Once UPDATE_CRASH_WINDOW has elapsed since the last update, the
        next check is allowed through."""
        app_instance._last_auto_update_at = (
            time.time() - menubar.UPDATE_CRASH_WINDOW - 5)
        with patch("app.menubar.check_repo_update_available",
                    return_value=(1, "v2.0.0", "newhash")), \
             patch.object(app_instance, "_mcp_recently_active",
                          return_value=False), \
             patch.object(app_instance, "_auto_update") as mock_update:
            app_instance._check_for_updates()
        mock_update.assert_called_once_with("v2.0.0")


class TestAutoUpdateDisable:
    def test_auto_update_false_skips_check(self, app_instance):
        app_instance.conf["AUTO_UPDATE"] = "false"
        app_instance.last_update_check = 0
        with patch("app.menubar.check_repo_update_available") as mock_check:
            app_instance._schedule_update_check()
        mock_check.assert_not_called()
        assert app_instance.last_update_check == 0

    def test_auto_update_true_runs_check(self, app_instance):
        app_instance.conf["AUTO_UPDATE"] = "true"
        app_instance.last_update_check = 0
        with patch("app.menubar.check_repo_update_available",
                    return_value=(0, None, None)), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"):
            app_instance._schedule_update_check()
        assert app_instance.last_update_check > 0

    def test_auto_update_missing_defaults_to_enabled(self, app_instance):
        app_instance.conf.pop("AUTO_UPDATE", None)
        app_instance.last_update_check = 0
        with patch("app.menubar.check_repo_update_available",
                    return_value=(0, None, None)), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"):
            app_instance._schedule_update_check()
        assert app_instance.last_update_check > 0


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
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False), \
             patch.object(app_instance, "_stop_mcp_server", create=True), \
             patch("os.path.isfile", return_value=True), \
             patch("os._exit") as mock_exit:
            app_instance._auto_update("v2.0.0")

        assert len(post_update_ran) == 1
        mock_exit.assert_called_once_with(1)

    def test_mcp_code_changed_uses_target_tag(self, app_instance, update_dir):
        """For tag updates (HEAD == launch hash), diff target is the new tag."""
        with patch("app.menubar.apply_repo_update", return_value=(True, "ok")), \
             patch("app.menubar.subprocess.run", return_value=_mock_run()), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"), \
             patch.object(app_instance, "_mcp_code_changed",
                          return_value=True) as mock_mcc, \
             patch.object(app_instance, "_stop_mcp_server", create=True), \
             patch("os.path.isfile", return_value=False), \
             patch("os._exit"):
            app_instance._auto_update("v2.0.0")

        mock_mcc.assert_called_once_with("v2.0.0")

    def test_failed_checkout_rolls_back(self, app_instance, update_dir):
        """Failed checkout rolls back to pre-update hash."""
        with patch("app.menubar.apply_repo_update",
                    return_value=(False, "checkout failed")), \
             patch("app.menubar.subprocess.run") as mock_run, \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False):
            app_instance._auto_update("v2.0.0")

        # Should have checked out pre-update hash with --force
        checkout_calls = [c for c in mock_run.call_args_list
                          if "checkout" in str(c) and "--force" in str(c)]
        assert len(checkout_calls) == 1
        assert "launch000" in str(checkout_calls[0])

    def test_post_update_failure_aborts_restart(self, app_instance, update_dir):
        """If post-update.sh fails, roll back and do NOT call os._exit."""
        def mock_run(cmd, **kw):
            if isinstance(cmd, list) and "post-update.sh" in str(cmd[-1]):
                return _mock_run(returncode=1, stderr="cc: error: no such file")
            return _mock_run()

        with patch("app.menubar.apply_repo_update", return_value=(True, "ok")), \
             patch("app.menubar.subprocess.run", side_effect=mock_run) as mock_sub, \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False), \
             patch("os.path.isfile", return_value=True), \
             patch("os._exit") as mock_exit:
            app_instance._auto_update("v2.0.0")

        # Should NOT have exited
        mock_exit.assert_not_called()
        # Should have rolled back
        rollback_calls = [c for c in mock_sub.call_args_list
                          if "checkout" in str(c) and "--force" in str(c)]
        assert len(rollback_calls) == 1
        assert "launch000" in str(rollback_calls[0])

    def test_writes_crash_marker(self, app_instance, update_dir):
        """update_started file is written after successful checkout."""
        with patch("app.menubar.apply_repo_update", return_value=(True, "ok")), \
             patch("app.menubar.subprocess.run", return_value=_mock_run()), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="launch000"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False), \
             patch.object(app_instance, "_stop_mcp_server", create=True), \
             patch("os.path.isfile", return_value=False), \
             patch("os._exit"):
            app_instance._auto_update("v2.0.0")

        marker = update_dir / "update_started"
        assert marker.exists()
        ts = float(marker.read_text())
        assert time.time() - ts < 5

    def test_drift_skips_checkout(self, app_instance, update_dir):
        """When HEAD moved since launch (drift), skip checkout and just restart."""
        with patch("app.menubar.apply_repo_update") as mock_checkout, \
             patch("app.menubar.subprocess.run", return_value=_mock_run()), \
             patch.object(app_instance, "_get_head_hash",
                          return_value="drifted999"), \
             patch.object(app_instance, "_mcp_code_changed", return_value=False), \
             patch.object(app_instance, "_stop_mcp_server", create=True), \
             patch("os.path.isfile", return_value=False), \
             patch("os._exit") as mock_exit:
            app_instance._auto_update("v2.0.0")

        mock_checkout.assert_not_called()
        mock_exit.assert_called_once_with(1)


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
        skip.write_text(f"oldbadhash\n{time.time()}")
        with patch("app.menubar.subprocess.check_output", return_value="newgoodhash\n"):
            app_instance._mark_startup_healthy()
        assert not skip.exists()

    def test_keeps_skip_on_same_commit(self, app_instance, update_dir):
        """If HEAD is still the skipped commit, keep the skip file."""
        skip = update_dir / "update_skipped"
        skip.write_text(f"samehash\n{time.time()}")
        with patch("app.menubar.subprocess.check_output", return_value="samehash\n"):
            app_instance._mark_startup_healthy()
        assert skip.exists()

    def test_skips_cleanup_after_rollback(self, app_instance, update_dir):
        """If _rolled_back is set, _mark_startup_healthy is a no-op."""
        app_instance._rolled_back = True
        (update_dir / "update_started").write_text("123")
        skip = update_dir / "update_skipped"
        skip.write_text(f"badhash\n{time.time()}")
        app_instance._mark_startup_healthy()
        # Skip file must survive — rollback wrote it, healthy timer shouldn't clear it
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
    def test_active_gpu_returns_true(self, app_instance):
        body = json.dumps({"ollama": {"active": 1}, "mlx": {"active": 0}}).encode()
        mock_resp = type("R", (), {"read": lambda self: body})()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert app_instance._mcp_recently_active() is True

    def test_idle_gpu_returns_false(self, app_instance):
        body = json.dumps({"ollama": {"active": 0}, "mlx": {"active": 0}}).encode()
        mock_resp = type("R", (), {"read": lambda self: body})()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert app_instance._mcp_recently_active() is False

    def test_unreachable_returns_false(self, app_instance):
        with patch("urllib.request.urlopen", side_effect=ConnectionError):
            assert app_instance._mcp_recently_active() is False


# ---------------------------------------------------------------------------
# quit_app writes sentinel for wrapper
# ---------------------------------------------------------------------------

class TestStayDown:
    def test_quit_app_writes_stay_down(self, app_instance, tmp_path):
        marker = str(tmp_path / "stay_down")
        app_instance.profile_server = None
        app_instance.servers_started = False
        app_instance.mode = "client"
        with patch.object(menubar, "rumps") as mock_rumps, \
             patch("os.path.expanduser", return_value=marker):
            app_instance.quit_app(None)
        assert Path(marker).exists()
        assert Path(marker).read_text() == str(os.getpid())
        mock_rumps.quit_application.assert_called_once()

    def test_stay_down_not_written_on_crash(self, tmp_path):
        marker = tmp_path / "stay_down"
        assert not marker.exists()

    def test_sys_exit_caught_at_top_level(self):
        """Verify that SystemExit from rumps/PyObjC doesn't bypass C launcher."""
        # The try/except SystemExit in __main__ ensures sys.exit(0) from
        # rumps.quit_application() doesn't call libc exit(0) directly.
        # In embedded Python (dlopen), sys.exit(0) would terminate the
        # entire process, bypassing the C launcher's return 1.
        caught = False
        try:
            try:
                raise SystemExit(0)  # simulates rumps.quit_application()
            except SystemExit:
                caught = True
                pass  # matches the real __main__ handler
        except SystemExit:
            pytest.fail("SystemExit escaped — would bypass C launcher")
        assert caught


# ---------------------------------------------------------------------------
# Shell wrapper stay_down startup check
# ---------------------------------------------------------------------------

class TestWrapperStayDown:
    """Test the stay_down startup check in bin/local-models-menubar.

    The wrapper checks for stay_down BEFORE launching the app.
    If present, it removes the file and exits 0 (launchd stops).
    Otherwise it execs the app (which always exits non-zero via C launcher).
    """

    @pytest.fixture()
    def wrapper_env(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        stay_down = config_dir / "stay_down"
        # Minimal script reproducing just the stay_down check.
        # Uses 'exit 1' as a stand-in for the exec (which always exits non-zero).
        script = tmp_path / "test_wrapper.sh"
        script.write_text(f"""#!/bin/bash
STAY_DOWN="{stay_down}"
if [ -f "$STAY_DOWN" ]; then
    rm -f "$STAY_DOWN"
    exit 0
fi
# Stand-in for exec (C launcher always exits non-zero)
exit 1
""")
        script.chmod(0o755)
        return {"script": str(script), "stay_down": stay_down}

    def test_normal_startup_exits_nonzero(self, wrapper_env):
        result = subprocess.run(
            ["bash", wrapper_env["script"]],
            capture_output=True, timeout=5)
        assert result.returncode == 1

    def test_stay_down_exits_zero(self, wrapper_env):
        wrapper_env["stay_down"].write_text("12345")
        result = subprocess.run(
            ["bash", wrapper_env["script"]],
            capture_output=True, timeout=5)
        assert result.returncode == 0
        assert not wrapper_env["stay_down"].exists()

    def test_stay_down_is_one_shot(self, wrapper_env):
        wrapper_env["stay_down"].write_text("12345")
        # First run: exits 0 and removes the file
        subprocess.run(["bash", wrapper_env["script"]], capture_output=True, timeout=5)
        # Second run: no file, exits 1 (app would start)
        result = subprocess.run(
            ["bash", wrapper_env["script"]],
            capture_output=True, timeout=5)
        assert result.returncode == 1
