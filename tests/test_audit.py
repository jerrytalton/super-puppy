import json
import os
import stat

import pytest

from lib import audit


def test_atomic_write_makes_bak(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("old")
    audit.atomic_write(p, "new")
    assert p.read_text() == "new"
    assert (tmp_path / "f.txt.bak").read_text() == "old"


def test_atomic_write_bak_matches_source_mode(tmp_path):
    p = tmp_path / "secret.json"
    p.write_text("secret")
    p.chmod(0o600)
    audit.atomic_write(p, "new")
    bak = tmp_path / "secret.json.bak"
    assert bak.exists()
    mode = stat.S_IMODE(os.stat(bak).st_mode)
    assert mode == 0o600
    assert mode & (stat.S_IRGRP | stat.S_IROTH) == 0


def test_merge_json_key_preserves_siblings(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"mcpServers": {"other": {"x": 1}}, "keep": True}))
    audit.merge_json_key(p, "mcpServers.local-models", {"type": "http"})
    data = json.loads(p.read_text())
    assert data["mcpServers"]["other"] == {"x": 1}
    assert data["mcpServers"]["local-models"] == {"type": "http"}
    assert data["keep"] is True


def test_merge_json_raises_on_malformed(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    with pytest.raises(Exception):
        audit.merge_json_key(p, "a.b", 1)


def test_append_hook_no_duplicate(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{}")
    entry = {"matcher": "*", "hooks": [{"type": "command", "command": "sp-session-ping"}]}
    audit.append_hook(p, entry)
    audit.append_hook(p, entry)
    data = json.loads(p.read_text())
    assert len(data["hooks"]["SessionStart"]) == 1


def test_append_hook_preserves_other_hook_event(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": "existing-pretooluse"}]}]
        }
    }))
    entry = {"matcher": "*", "hooks": [{"type": "command", "command": "sp-session-ping"}]}
    audit.append_hook(p, entry)
    data = json.loads(p.read_text())
    assert data["hooks"]["PreToolUse"] == [
        {"matcher": "*", "hooks": [{"type": "command", "command": "existing-pretooluse"}]}
    ]
    assert len(data["hooks"]["SessionStart"]) == 1


def test_upsert_appends_when_absent():
    out = audit.upsert_guidance("# My CLAUDE.md\n\nSome rules.\n")
    assert audit.GUIDANCE_MARKERS[0] in out
    assert audit.GUIDANCE_MARKERS[1] in out
    assert "Some rules." in out  # user content preserved


def test_upsert_is_idempotent():
    once = audit.upsert_guidance("# doc\n")
    twice = audit.upsert_guidance(once)
    assert once == twice


def test_upsert_replaces_stale_block_only_between_markers():
    doc = "# doc\n\nkeep me\n" + audit.GUIDANCE_MARKERS[0] + "\nOLD JUNK\n" + audit.GUIDANCE_MARKERS[1] + "\n\nkeep me too\n"
    out = audit.upsert_guidance(doc)
    assert "OLD JUNK" not in out
    assert "keep me" in out and "keep me too" in out


def test_guidance_text_has_no_hardware_names():
    lowered = audit.GUIDANCE_TEXT.lower()
    for banned in ("m3 ultra", "512gb", "m5 max", "holo3", "wan2.2", "voxtral"):
        assert banned not in lowered


# ── Check registry: Claude Code ──────────────────────────────────────────

def _fake_home(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude.json").write_text("{}")
    (tmp_path / ".claude" / "CLAUDE.md").write_text("# rules\n")
    (tmp_path / ".claude" / "settings.json").write_text("{}")
    return tmp_path


def test_run_all_flags_missing_mcp(tmp_path):
    home = _fake_home(tmp_path)
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-mcp"]["status"] == "fail"
    assert results["claude-guidance"]["status"] == "fail"
    assert results["claude-hook"]["status"] == "fail"
    assert results["token-present"]["status"] == "fail"


def test_fix_mcp_then_passes(tmp_path):
    home = _fake_home(tmp_path)
    audit.fix("claude-mcp", home=home, token="secret")
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-mcp"]["status"] == "pass"


def test_fix_mcp_inlines_token_when_safe(tmp_path):
    # Explicitly private (0600): not group/other-readable and not inside a
    # git tree. Deterministic regardless of the ambient umask — a freshly
    # `write_text`-created file is typically 0644 (world-readable) under a
    # standard umask, which the guard correctly treats as unsafe.
    home = _fake_home(tmp_path)
    cj = home / ".claude.json"
    cj.chmod(0o600)
    audit.fix("claude-mcp", home=home, token="secret")
    entry = json.loads(cj.read_text())["mcpServers"]["local-models"]
    assert entry["headers"]["Authorization"] == "Bearer secret"
    assert entry["headers"]["X-SP-Client"]


def test_fix_guidance_is_idempotent_and_passes(tmp_path):
    home = _fake_home(tmp_path)
    audit.fix("claude-guidance", home=home)
    first = (home / ".claude" / "CLAUDE.md").read_text()
    audit.fix("claude-guidance", home=home)
    second = (home / ".claude" / "CLAUDE.md").read_text()
    assert first == second
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-guidance"]["status"] == "pass"


def test_fix_hook_then_passes(tmp_path):
    home = _fake_home(tmp_path)
    audit.fix("claude-hook", home=home)
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-hook"]["status"] == "pass"
    data = json.loads((home / ".claude" / "settings.json").read_text())
    assert len(data["hooks"]["SessionStart"]) == 1
    # idempotent
    audit.fix("claude-hook", home=home)
    data = json.loads((home / ".claude" / "settings.json").read_text())
    assert len(data["hooks"]["SessionStart"]) == 1


def test_mcp_fix_refuses_token_in_world_readable(tmp_path):
    import stat
    home = _fake_home(tmp_path)
    cj = home / ".claude.json"
    cj.chmod(cj.stat().st_mode | stat.S_IROTH)  # world-readable
    audit.fix("claude-mcp", home=home, token="secret")
    entry = json.loads(cj.read_text())["mcpServers"]["local-models"]
    # token must NOT be inlined into a world-readable file
    assert "secret" not in json.dumps(entry)
    # but the entry is still written (headers/url present) so the check still passes
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-mcp"]["status"] == "pass"


def test_mcp_fix_absent_file_creates_private_token_file(tmp_path):
    # Fresh-machine case: ~/.claude.json does not exist yet. Before the fix,
    # _unsafe_to_inline_token() returned False for a nonexistent path, the
    # token was inlined, and atomic_write created the file under the ambient
    # umask (typically 0644 — world-readable). The chmod(0o600) after the
    # write is what must close that leak.
    home = tmp_path
    (home / ".claude").mkdir()
    cj = home / ".claude.json"
    assert not cj.exists()
    audit.fix("claude-mcp", home=home, token="secret")
    assert cj.exists()
    mode = stat.S_IMODE(os.stat(cj).st_mode)
    assert mode == 0o600
    assert mode & (stat.S_IRGRP | stat.S_IROTH) == 0
    entry = json.loads(cj.read_text())["mcpServers"]["local-models"]
    assert entry["headers"]["Authorization"] == "Bearer secret"
    assert entry["headers"]["X-SP-Client"]


def test_mcp_fix_absent_file_no_world_readable_intermediate(tmp_path):
    # Same fresh-machine case, but asserting on atomic_write's own
    # intermediates rather than just the final chmod: no leftover .tmp,
    # and if a .bak exists (it shouldn't, since there was no prior file)
    # it must not be group/other readable either.
    home = tmp_path
    (home / ".claude").mkdir()
    cj = home / ".claude.json"
    assert not cj.exists()
    audit.fix("claude-mcp", home=home, token="secret")
    mode = stat.S_IMODE(os.stat(cj).st_mode)
    assert mode == 0o600
    assert not (home / ".claude.json.tmp").exists()
    bak = home / ".claude.json.bak"
    if bak.exists():
        bak_mode = stat.S_IMODE(os.stat(bak).st_mode)
        assert bak_mode & (stat.S_IRGRP | stat.S_IROTH) == 0


def test_mcp_fix_refuses_token_inside_git_worktree(tmp_path):
    import subprocess
    home = _fake_home(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=home, check=True)
    cj = home / ".claude.json"
    cj.chmod(0o600)  # private permissions, but tracked inside a git work tree
    audit.fix("claude-mcp", home=home, token="secret")
    entry = json.loads(cj.read_text())["mcpServers"]["local-models"]
    assert "secret" not in json.dumps(entry)


def test_token_present_check(tmp_path):
    home = _fake_home(tmp_path)
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["token-present"]["status"] == "fail"
    token_dir = home / ".config" / "local-models"
    token_dir.mkdir(parents=True)
    (token_dir / "mcp_auth_token").write_text("abc123")
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["token-present"]["status"] == "pass"


def test_fix_unknown_check_raises(tmp_path):
    home = _fake_home(tmp_path)
    with pytest.raises(ValueError):
        audit.fix("does-not-exist", home=home)


def test_fix_non_fixable_check_raises(tmp_path):
    home = _fake_home(tmp_path)
    with pytest.raises(ValueError):
        audit.fix("token-present", home=home)


def test_fix_all_fixes_only_failing_and_fixable(tmp_path):
    home = _fake_home(tmp_path)
    summaries = audit.fix_all(home=home, token="secret")
    ids_mentioned = " ".join(summaries)
    assert "claude-mcp" in ids_mentioned
    assert "claude-guidance" in ids_mentioned
    assert "claude-hook" in ids_mentioned
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["claude-mcp"]["status"] == "pass"
    assert results["claude-guidance"]["status"] == "pass"
    assert results["claude-hook"]["status"] == "pass"


# ── Check registry: Codex / Gemini (n/a-vs-fail gating) + other-agents ───

def test_codex_gemini_are_na_when_absent(tmp_path):
    home = _fake_home(tmp_path)
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["codex-mcp"]["status"] == "n/a"
    assert results["codex-guidance"]["status"] == "n/a"
    assert results["gemini-mcp"]["status"] == "n/a"
    assert results["gemini-guidance"]["status"] == "n/a"
    assert results["codex-mcp"]["fixable"] is False
    assert results["gemini-mcp"]["fixable"] is False


def test_codex_gemini_fail_not_na_when_dirs_present(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".codex").mkdir()
    (home / ".gemini").mkdir()
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["codex-mcp"]["status"] == "fail"
    assert results["codex-guidance"]["status"] == "fail"
    assert results["gemini-mcp"]["status"] == "fail"
    assert results["gemini-guidance"]["status"] == "fail"
    assert results["codex-mcp"]["fixable"] is True
    assert results["gemini-mcp"]["fixable"] is True


def test_fix_codex_mcp_then_passes_and_preserves_user_toml(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text('# my own comment\nmodel = "o1"\n')
    audit.fix("codex-mcp", home=home, token="secret")
    text = (home / ".codex" / "config.toml").read_text()
    assert "# my own comment" in text
    assert 'model = "o1"' in text
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["codex-mcp"]["status"] == "pass"


def test_fix_codex_mcp_is_idempotent(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text('model = "o1"\n')
    audit.fix("codex-mcp", home=home, token="secret")
    first = (home / ".codex" / "config.toml").read_text()
    audit.fix("codex-mcp", home=home, token="secret")
    second = (home / ".codex" / "config.toml").read_text()
    assert first == second


def test_fix_codex_mcp_raises_on_unparseable_toml(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text("not [ valid toml =")
    with pytest.raises(Exception):
        audit.fix("codex-mcp", home=home, token="secret")
    # untouched
    assert (home / ".codex" / "config.toml").read_text() == "not [ valid toml ="


def test_fix_codex_mcp_refuses_token_in_world_readable(tmp_path):
    import stat
    home = _fake_home(tmp_path)
    (home / ".codex").mkdir()
    cfg = home / ".codex" / "config.toml"
    cfg.write_text('model = "o1"\n')
    cfg.chmod(cfg.stat().st_mode | stat.S_IROTH)
    audit.fix("codex-mcp", home=home, token="secret")
    assert "secret" not in cfg.read_text()


def test_fix_codex_guidance_upserts_agents_md(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".codex").mkdir()
    audit.fix("codex-guidance", home=home)
    text = (home / ".codex" / "AGENTS.md").read_text()
    assert audit.render_block() in text
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["codex-guidance"]["status"] == "pass"


def test_fix_codex_raises_when_not_installed(tmp_path):
    home = _fake_home(tmp_path)
    with pytest.raises(Exception):
        audit.fix("codex-mcp", home=home, token="secret")
    with pytest.raises(Exception):
        audit.fix("codex-guidance", home=home)


def test_fix_gemini_mcp_then_passes(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".gemini").mkdir()
    audit.fix("gemini-mcp", home=home, token="secret")
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["gemini-mcp"]["status"] == "pass"


def test_fix_gemini_mcp_refuses_token_in_world_readable(tmp_path):
    import stat
    home = _fake_home(tmp_path)
    (home / ".gemini").mkdir()
    settings = home / ".gemini" / "settings.json"
    settings.write_text("{}")
    settings.chmod(settings.stat().st_mode | stat.S_IROTH)
    audit.fix("gemini-mcp", home=home, token="secret")
    entry = json.loads(settings.read_text())["mcpServers"]["local-models"]
    assert "secret" not in json.dumps(entry)


def test_fix_gemini_mcp_absent_file_creates_private_token_file(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".gemini").mkdir()
    settings = home / ".gemini" / "settings.json"
    assert not settings.exists()
    audit.fix("gemini-mcp", home=home, token="secret")
    assert settings.exists()
    mode = stat.S_IMODE(os.stat(settings).st_mode)
    assert mode == 0o600
    assert mode & (stat.S_IRGRP | stat.S_IROTH) == 0
    entry = json.loads(settings.read_text())["mcpServers"]["local-models"]
    assert entry["headers"]["Authorization"] == "Bearer secret"
    assert entry["headers"]["X-SP-Client"]


def test_fix_gemini_guidance_is_idempotent_and_passes(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".gemini").mkdir()
    audit.fix("gemini-guidance", home=home)
    first = (home / ".gemini" / "GEMINI.md").read_text()
    audit.fix("gemini-guidance", home=home)
    second = (home / ".gemini" / "GEMINI.md").read_text()
    assert first == second
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["gemini-guidance"]["status"] == "pass"


def test_fix_gemini_raises_when_not_installed(tmp_path):
    home = _fake_home(tmp_path)
    with pytest.raises(Exception):
        audit.fix("gemini-mcp", home=home, token="secret")


def test_other_agents_na_when_absent(tmp_path):
    home = _fake_home(tmp_path)
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["other-agents"]["status"] == "n/a"
    assert results["other-agents"]["fixable"] is False


def test_other_agents_warns_and_never_fixable(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".cursor").mkdir()
    results = {c["id"]: c for c in audit.run_all(home=home)}
    assert results["other-agents"]["status"] == "warn"
    assert results["other-agents"]["fixable"] is False
    with pytest.raises(Exception):
        audit.fix("other-agents", home=home)


def test_fix_all_skips_other_agents_and_token_present(tmp_path):
    home = _fake_home(tmp_path)
    (home / ".cursor").mkdir()
    # Should not raise despite other-agents (warn, unfixable) and
    # token-present (fail, unfixable) both being present.
    audit.fix_all(home=home, token="secret")


# ── sp-doctor CLI ───────────────────────────────────────────────────────────

def test_sp_doctor_json_runs(tmp_path):
    import subprocess
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude.json").write_text("{}")
    (tmp_path / ".claude" / "CLAUDE.md").write_text("# x\n")
    (tmp_path / ".claude" / "settings.json").write_text("{}")
    env = {**os.environ, "SP_AUDIT_HOME": str(tmp_path)}
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOCTOR = os.path.join(REPO, "bin", "sp-doctor")
    r = subprocess.run([DOCTOR, "--json"], env=env, capture_output=True, text=True)
    data = json.loads(r.stdout)
    assert any(c["id"] == "claude-mcp" for c in data)


def test_run_all_survives_ascii_locale_with_non_ascii_config(tmp_path):
    """Regression: the menu bar app runs under launchd with an ASCII/C locale
    (no LANG). `read_text()`/`write_text()` without an explicit encoding then
    default to ASCII and raise UnicodeDecodeError on a config file containing
    UTF-8 bytes (e.g. an em-dash in ~/.claude.json). Shipped in v1.3.0; the
    live "Audit…" menu item crashed with:
        'ascii' codec can't decode byte 0xe2 in position ...
    Drive audit.run_all in a subprocess under a stripped ASCII env against a
    fake home whose configs contain non-ASCII, and assert it does not crash.
    """
    import subprocess
    import sys
    from pathlib import Path

    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    # non-ASCII (em-dash, arrow) in both a JSON config and a markdown guidance file
    (home / ".claude.json").write_text('{"note": "cost — value → ok"}', encoding="utf-8")
    (home / ".claude" / "CLAUDE.md").write_text("# rules — see below → done\n", encoding="utf-8")
    (home / ".claude" / "settings.json").write_text("{}", encoding="utf-8")

    script = (
        "import sys; sys.path.insert(0, %r);"
        "from lib import audit; from pathlib import Path;"
        "r = audit.run_all(home=Path(%r));"
        "assert isinstance(r, list) and r;"
        % (str(Path(__file__).resolve().parent.parent), str(home))
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "LANG": "",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
    }
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"run_all crashed under ASCII locale: {proc.stderr}")
    assert "UnicodeDecodeError" not in proc.stderr
    assert "UnicodeEncodeError" not in proc.stderr


def test_fix_group_only_touches_its_group(tmp_path):
    from lib import audit
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude.json").write_text("{}", encoding="utf-8")
    (home / ".claude" / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
    (home / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
    summaries = audit.fix_group("claude", home=home, token="tok")
    assert len(summaries) == 3  # mcp, guidance, hook
    after = {c["id"]: c["status"] for c in audit.run_all(home=home)}
    assert after["claude-mcp"] == "pass"
    assert after["claude-guidance"] == "pass"
    assert after["claude-hook"] == "pass"
    # codex/gemini absent → still n/a, untouched by a claude fix
    assert after["codex-mcp"] == "n/a"


def test_fix_group_unknown_is_noop(tmp_path):
    from lib import audit
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude.json").write_text("{}", encoding="utf-8")
    assert audit.fix_group("nonesuch", home=home) == []
