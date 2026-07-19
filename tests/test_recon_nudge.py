"""Tests for bin/sp-recon-nudge — the UserPromptSubmit offload nudge hook.

The hook must: nudge on reconnaissance-shaped prompts, stay silent on
edit/action-shaped ones (those need exact bytes on the frontier model), and
fail open (exit 0, no output) on anything malformed — it must never block a
prompt.
"""
import json
import os
import subprocess

HOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "bin", "sp-recon-nudge")


def _run(stdin: str):
    p = subprocess.run([HOOK], input=stdin, capture_output=True, text=True)
    return p.returncode, p.stdout


def _nudged(prompt: str) -> bool:
    code, out = _run(json.dumps({"prompt": prompt}))
    assert code == 0, f"hook must always exit 0, got {code}"
    if not out.strip():
        return False
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    return "recon-local" in ctx or "local_summarize" in ctx


RECON = [
    "how does the audit system work?",
    "where is the token validated?",
    "summarize this 5000-line log",
    "explain the fleet heartbeat flow",
    "which files handle authentication?",
    "give me an overview of the mcp server",
    "walk me through the release process",
    "explore the lib directory",
]
EDIT = [
    "fix the bug in audit.py",
    "refactor local_summarize to stream",
    "implement a new dispatch tool",
    "add a test for the hook",
    "rename the recon agent",
    "commit these changes and push",
    "update the version to 1.6.0",
]


def test_nudges_on_reconnaissance():
    for p in RECON:
        assert _nudged(p), f"expected nudge for recon prompt: {p!r}"


def test_silent_on_edit_and_action():
    for p in EDIT:
        assert not _nudged(p), f"expected silence for edit prompt: {p!r}"


def test_fail_open_on_malformed_input():
    for junk in ["not json", "", "{}", '{"prompt": ""}', '[1,2,3]']:
        code, out = _run(junk)
        assert code == 0, f"must exit 0 on {junk!r}, got {code}"
        assert out.strip() == "", f"must stay silent on {junk!r}, got {out!r}"
