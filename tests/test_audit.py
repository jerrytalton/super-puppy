import json

import pytest

from lib import audit


def test_atomic_write_makes_bak(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("old")
    audit.atomic_write(p, "new")
    assert p.read_text() == "new"
    assert (tmp_path / "f.txt.bak").read_text() == "old"


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
