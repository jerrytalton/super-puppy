from lib import audit


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
