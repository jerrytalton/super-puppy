"""Super Puppy configuration audit: verify (and optionally fix) that installed
agent tools are wired to use SP — MCP registration, guidance blocks, session hook.

Design note (spec §S2): guidance blocks are text an AI agent treats as
instructions. Fixes here are NEVER automatic — a caller (sp-doctor / menubar /
install.sh opt-in) applies them explicitly, with a diff. The managed block is
minimal and mechanical; changing GUIDANCE_TEXT is a reviewed code diff.
"""

import dataclasses
import json
import os
import shutil
import socket
import stat
import tomllib
from pathlib import Path
from typing import Optional

GUIDANCE_MARKERS = ("<!-- >>> super-puppy >>> -->", "<!-- <<< super-puppy <<< -->")

# The streamable-http endpoint the menu bar app's MCP server listens on
# (see mcp/local-models-server.py, install.sh's claude-mcp registration).
SP_MCP_URL = "http://127.0.0.1:8100/mcp"

GUIDANCE_TEXT = """\
## Local Models (super-puppy)

Reach for the `local-models` MCP server whenever a task needs capabilities you \
lack (vision, audio, image/video) or cheap parallel compute (bulk transforms, \
second opinions, delegated reasoning while you keep working). Call \
`local_models_status` for what's live.

**Do things in parallel on the server and with the local cluster whenever it \
makes sense.** While you work, the local GPUs should be busy too — fire off \
`local_dispatch` before you start reasoning, not after you finish.

| Need | Tool |
|------|------|
| Look at an image or screenshot | `local_vision` |
| Plan a GUI interaction | `local_computer_use` |
| Generate / edit an image | `local_image`, `local_image_edit` |
| Generate video | `local_video` |
| Translate / transcribe / speak | `local_translate`, `local_transcribe`, `local_speak` |
| Bulk boilerplate / repetitive transforms | `local_generate` |
| Second opinion on code or designs | `local_review`, `local_candidates` |
| Summarize before reading a huge file | `local_summarize` |
| Delegate reasoning while you keep working | `local_dispatch` then `local_collect` |
| Find files by concept | `local_similarity_search`, `local_embed` |

Model/hardware specifics live in each tool's runtime description and \
`local_models_status` — trust those over anything static.
"""


def render_block() -> str:
    return f"{GUIDANCE_MARKERS[0]}\n{GUIDANCE_TEXT}\n{GUIDANCE_MARKERS[1]}"


def upsert_guidance(text: str) -> str:
    block = render_block()
    start, end = GUIDANCE_MARKERS
    if start in text and end in text:
        pre = text[: text.index(start)]
        post = text[text.index(end) + len(end):]
        return f"{pre}{block}{post}"
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return f"{text}{sep}{block}\n"


def atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically.

    Writes a `.bak` of the prior content (if any), then writes to a temp
    file in the SAME directory and `os.replace`s it into place — same
    filesystem, so the rename is atomic. A crash mid-write leaves either
    the old file or the temp file, never a half-written target.

    Permission invariant: the `.bak` always matches the mode of the file
    it backs up (a 0600 secret-bearing config yields a 0600 `.bak`, never
    the ambient-umask default). The `.tmp` is created owner-only (0600)
    so a secret written mid-fix is never briefly world/group-readable; if
    `path` already exists, the tmp is then bumped to `path`'s real mode
    before the replace (so an ordinary 0644 config stays 0644). A
    brand-new file is left at 0600 — the safe default for a config that
    may have just had a token inlined into it.
    """
    path = Path(path)
    if path.exists():
        bak = path.parent / (path.name + ".bak")
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        shutil.copymode(path, bak)
    tmp = path.parent / (path.name + ".tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    if path.exists():
        shutil.copymode(path, tmp)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    """Load JSON from `path`, or {} if absent. Lets JSONDecodeError
    propagate — a file we can't parse must never be silently overwritten."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_json_key(path, dotted_key: str, value) -> None:
    """Read-modify-write `path`, setting `dotted_key` (e.g. "a.b.c") to
    `value` without disturbing sibling keys at any level. Creates the
    file (and intermediate dicts) if absent. Raises on malformed JSON."""
    path = Path(path)
    data = _load_json(path)
    node = data
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    atomic_write(path, json.dumps(data, indent=2))


def append_hook(settings_path, hook_entry: dict) -> None:
    """Merge `hook_entry` into hooks.SessionStart, preserving existing
    hooks (including other hook events like PreToolUse) and skipping the
    append if an identical `command` is already present."""
    path = Path(settings_path)
    data = _load_json(path)
    hooks = data.setdefault("hooks", {})
    arr = hooks.setdefault("SessionStart", [])
    existing_commands = {h.get("command") for e in arr for h in e.get("hooks", [])}
    new_commands = {h.get("command") for h in hook_entry.get("hooks", [])}
    if not (new_commands & existing_commands):
        arr.append(hook_entry)
    atomic_write(path, json.dumps(data, indent=2))


# ── Check registry ──────────────────────────────────────────────────────
#
# Each check reads-only and reports {id, tool, status, detail, fixable};
# `status` is one of pass|fail|warn|n/a. Checks never raise on a malformed
# config file — a syntax error is reported as a `fail` with the parser's
# message so one broken file doesn't blank out the rest of the audit table.
# Fixes are the opposite (spec §S2/S6): they raise loudly on an unparseable
# target rather than silently overwriting something they can't understand.


@dataclasses.dataclass
class Check:
    id: str
    tool: str
    status: str  # pass | fail | warn | n/a
    detail: str
    fixable: bool


def _client_hostname() -> str:
    """Short hostname stamped into the X-SP-Client header so the server can
    attribute requests to the machine that made them (see mcp/local-models-server.py)."""
    return (socket.gethostname() or "unknown").split(".")[0] or "unknown"


def _in_git_worktree(path: Path) -> bool:
    return any((parent / ".git").exists() for parent in (path.parent, *path.parent.parents))


def _unsafe_to_inline_token(path: Path) -> bool:
    """True if writing a secret into `path` risks leaking it off this
    machine: the file is readable by group/other, or it lives inside a git
    work tree (spec §S4 — `~/.claude.json` and friends are often synced or
    committed). Generalized beyond claude-mcp because the same file-leak
    risk applies to every config file a fix might inline a token into
    (~/.gemini/settings.json, ~/.codex/config.toml).

    A file that does not exist yet is NOT unsafe by this check alone — the
    fresh-machine case (no pre-existing file, ambient umask would otherwise
    leave a world-readable file after write) is made safe by each caller's
    `os.chmod(path, 0o600)` immediately after inlining a token, not here."""
    if path.exists():
        mode = os.stat(path).st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            return True
    return _in_git_worktree(path)


def _http_mcp_entry(token: Optional[str], target_path: Path) -> tuple[dict, bool]:
    """Build a streamable-http MCP entry with X-SP-Client attribution.
    Returns (entry, token_was_inlined)."""
    headers = {"X-SP-Client": _client_hostname()}
    inlined = False
    if token and not _unsafe_to_inline_token(target_path):
        headers = {"Authorization": f"Bearer {token}", **headers}
        inlined = True
    return {"type": "http", "url": SP_MCP_URL, "headers": headers}, inlined


def _mcp_fix_summary(check_id: str, path: Path, token: Optional[str], inlined: bool) -> str:
    if token and not inlined:
        return (
            f"{check_id}: wrote mcpServers.local-models to {path} "
            "(token referenced, not inlined — target is world-readable or under a git work tree)"
        )
    if inlined:
        return f"{check_id}: wrote mcpServers.local-models to {path} (token inlined)"
    return f"{check_id}: wrote mcpServers.local-models to {path} (no token provided)"


# ── Claude Code ──────────────────────────────────────────────────────────

def _check_claude_mcp(home: Path) -> Check:
    path = home / ".claude.json"
    try:
        data = _load_json(path)
    except json.JSONDecodeError as e:
        return Check("claude-mcp", "claude", "fail", f"{path} is not valid JSON: {e}", True)
    entry = data.get("mcpServers", {}).get("local-models")
    if isinstance(entry, dict) and entry.get("url") and entry.get("headers", {}).get("X-SP-Client"):
        return Check("claude-mcp", "claude", "pass", f"registered in {path} with X-SP-Client attribution", True)
    return Check("claude-mcp", "claude", "fail", f"mcpServers.local-models missing url/X-SP-Client header in {path}", True)


def _fix_claude_mcp(home: Path, token: Optional[str]) -> str:
    path = home / ".claude.json"
    entry, inlined = _http_mcp_entry(token, path)
    merge_json_key(path, "mcpServers.local-models", entry)
    if inlined:
        os.chmod(path, 0o600)
    return _mcp_fix_summary("claude-mcp", path, token, inlined)


def _check_claude_guidance(home: Path) -> Check:
    path = home / ".claude" / "CLAUDE.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if render_block() in text:
        return Check("claude-guidance", "claude", "pass", f"guidance block current in {path}", True)
    return Check("claude-guidance", "claude", "fail", f"guidance block missing or stale in {path}", True)


def _fix_claude_guidance(home: Path, token: Optional[str]) -> str:
    path = home / ".claude" / "CLAUDE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, upsert_guidance(path.read_text(encoding="utf-8") if path.exists() else ""))
    return f"claude-guidance: upserted guidance block in {path}"


SESSION_HOOK_ENTRY = {
    "matcher": "*",
    "hooks": [{"type": "command", "command": "sp-session-ping claude-code"}],
}


def _has_session_ping_hook(data: dict) -> bool:
    for entry in data.get("hooks", {}).get("SessionStart", []):
        for h in entry.get("hooks", []):
            if "sp-session-ping" in (h.get("command") or ""):
                return True
    return False


def _check_claude_hook(home: Path) -> Check:
    path = home / ".claude" / "settings.json"
    try:
        data = _load_json(path)
    except json.JSONDecodeError as e:
        return Check("claude-hook", "claude", "fail", f"{path} is not valid JSON: {e}", True)
    if _has_session_ping_hook(data):
        return Check("claude-hook", "claude", "pass", f"SessionStart hook present in {path}", True)
    return Check("claude-hook", "claude", "fail", f"SessionStart hook missing in {path}", True)


def _fix_claude_hook(home: Path, token: Optional[str]) -> str:
    path = home / ".claude" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    append_hook(path, SESSION_HOOK_ENTRY)
    return f"claude-hook: added SessionStart hook to {path}"


def _require_tool_dir(home: Path, dirname: str, tool_label: str) -> Path:
    """A fix must never invent a config directory for a tool that isn't
    installed — raise instead (same 'don't touch a target you don't
    understand' spirit as the unparseable-file guard)."""
    d = home / dirname
    if not d.exists():
        raise ValueError(f"{tool_label} not installed ({d} absent) — nothing to fix")
    return d


# ── Codex ─────────────────────────────────────────────────────────────────
#
# Codex's config.toml is user-owned and may carry comments/formatting we
# must not disturb, so the fix never round-trips it through tomllib's
# writer. It only reads with tomllib (to validate + to check pass/fail) and
# otherwise treats the file as opaque text, appending/replacing a single
# marker-delimited managed section.

CODEX_MARKERS = ("# >>> super-puppy >>>", "# <<< super-puppy <<<")


def _upsert_marked_text(text: str, markers: tuple, block: str) -> str:
    start, end = markers
    if start in text and end in text:
        pre = text[: text.index(start)]
        post = text[text.index(end) + len(end):]
        return f"{pre}{block}{post}"
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return f"{text}{sep}{block}\n"


def _codex_managed_block(token: Optional[str], unsafe: bool) -> str:
    lines = [
        CODEX_MARKERS[0],
        "[mcp_servers.local-models]",
        f'url = "{SP_MCP_URL}"',
        "",
        "[mcp_servers.local-models.headers]",
        f'X-SP-Client = "{_client_hostname()}"',
    ]
    if token and not unsafe:
        lines.append(f'Authorization = "Bearer {token}"')
    lines.append(CODEX_MARKERS[1])
    return "\n".join(lines)


def _check_codex_mcp(home: Path) -> Check:
    codex_dir = home / ".codex"
    if not codex_dir.exists():
        return Check("codex-mcp", "codex", "n/a", "Codex not installed (~/.codex absent)", False)
    path = codex_dir / "config.toml"
    if not path.exists():
        return Check("codex-mcp", "codex", "fail", f"{path} missing", True)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        return Check("codex-mcp", "codex", "fail", f"{path} is not valid TOML: {e}", True)
    entry = data.get("mcp_servers", {}).get("local-models")
    if isinstance(entry, dict) and entry.get("url") and entry.get("headers", {}).get("X-SP-Client"):
        return Check("codex-mcp", "codex", "pass", f"registered in {path} with X-SP-Client attribution", True)
    return Check("codex-mcp", "codex", "fail", f"mcp_servers.local-models missing url/X-SP-Client header in {path}", True)


def _fix_codex_mcp(home: Path, token: Optional[str]) -> str:
    codex_dir = _require_tool_dir(home, ".codex", "Codex")
    path = codex_dir / "config.toml"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing.strip():
        try:
            tomllib.loads(existing)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"refusing to touch unparseable {path}: {e}") from e
    unsafe = _unsafe_to_inline_token(path)
    new_text = _upsert_marked_text(existing, CODEX_MARKERS, _codex_managed_block(token, unsafe))
    tomllib.loads(new_text)  # sanity: our own managed section must itself parse
    atomic_write(path, new_text)
    if token and not unsafe:
        os.chmod(path, 0o600)
    if token and unsafe:
        return (
            f"codex-mcp: appended managed [mcp_servers.local-models] section to {path} "
            "(token referenced, not inlined — target is world-readable or under a git work tree)"
        )
    if token:
        return f"codex-mcp: appended managed [mcp_servers.local-models] section to {path} (token inlined)"
    return f"codex-mcp: appended managed [mcp_servers.local-models] section to {path} (no token provided)"


def _check_codex_guidance(home: Path) -> Check:
    codex_dir = home / ".codex"
    if not codex_dir.exists():
        return Check("codex-guidance", "codex", "n/a", "Codex not installed (~/.codex absent)", False)
    path = codex_dir / "AGENTS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if render_block() in text:
        return Check("codex-guidance", "codex", "pass", f"guidance block current in {path}", True)
    return Check("codex-guidance", "codex", "fail", f"guidance block missing or stale in {path}", True)


def _fix_codex_guidance(home: Path, token: Optional[str]) -> str:
    codex_dir = _require_tool_dir(home, ".codex", "Codex")
    path = codex_dir / "AGENTS.md"
    atomic_write(path, upsert_guidance(path.read_text(encoding="utf-8") if path.exists() else ""))
    return f"codex-guidance: upserted guidance block in {path}"


# ── Gemini ────────────────────────────────────────────────────────────────
#
# Gemini's settings.json is plain JSON, so the shape mirrors Claude exactly
# (merge_json_key for the MCP entry, upsert_guidance for the doc block).

def _check_gemini_mcp(home: Path) -> Check:
    gemini_dir = home / ".gemini"
    if not gemini_dir.exists():
        return Check("gemini-mcp", "gemini", "n/a", "Gemini CLI not installed (~/.gemini absent)", False)
    path = gemini_dir / "settings.json"
    try:
        data = _load_json(path)
    except json.JSONDecodeError as e:
        return Check("gemini-mcp", "gemini", "fail", f"{path} is not valid JSON: {e}", True)
    entry = data.get("mcpServers", {}).get("local-models")
    if isinstance(entry, dict) and entry.get("url") and entry.get("headers", {}).get("X-SP-Client"):
        return Check("gemini-mcp", "gemini", "pass", f"registered in {path} with X-SP-Client attribution", True)
    return Check("gemini-mcp", "gemini", "fail", f"mcpServers.local-models missing url/X-SP-Client header in {path}", True)


def _fix_gemini_mcp(home: Path, token: Optional[str]) -> str:
    gemini_dir = _require_tool_dir(home, ".gemini", "Gemini CLI")
    path = gemini_dir / "settings.json"
    entry, inlined = _http_mcp_entry(token, path)
    merge_json_key(path, "mcpServers.local-models", entry)
    if inlined:
        os.chmod(path, 0o600)
    return _mcp_fix_summary("gemini-mcp", path, token, inlined)


def _check_gemini_guidance(home: Path) -> Check:
    gemini_dir = home / ".gemini"
    if not gemini_dir.exists():
        return Check("gemini-guidance", "gemini", "n/a", "Gemini CLI not installed (~/.gemini absent)", False)
    path = gemini_dir / "GEMINI.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if render_block() in text:
        return Check("gemini-guidance", "gemini", "pass", f"guidance block current in {path}", True)
    return Check("gemini-guidance", "gemini", "fail", f"guidance block missing or stale in {path}", True)


def _fix_gemini_guidance(home: Path, token: Optional[str]) -> str:
    gemini_dir = _require_tool_dir(home, ".gemini", "Gemini CLI")
    path = gemini_dir / "GEMINI.md"
    atomic_write(path, upsert_guidance(path.read_text(encoding="utf-8") if path.exists() else ""))
    return f"gemini-guidance: upserted guidance block in {path}"


# ── Other agents (detect-only) ───────────────────────────────────────────

_OTHER_AGENT_DIRS = {
    "Cursor": ".cursor",
    "opencode": ".config/opencode",
    "Windsurf": "Library/Application Support/Windsurf",
}
_OTHER_AGENTS_DOC = "docs/troubleshooting.md"


def _check_other_agents(home: Path) -> Check:
    found = [name for name, rel in _OTHER_AGENT_DIRS.items() if (home / rel).exists()]
    if not found:
        return Check("other-agents", "other", "n/a", "no other agent tools detected", False)
    return Check(
        "other-agents",
        "other",
        "warn",
        f"{', '.join(found)} detected; Super Puppy not configured (detect-only) — see {_OTHER_AGENTS_DOC}",
        False,
    )


# ── Shared / registry ────────────────────────────────────────────────────

def _check_token_present(home: Path) -> Check:
    path = home / ".config" / "local-models" / "mcp_auth_token"
    if path.exists() and path.stat().st_size > 0:
        return Check("token-present", "shared", "pass", f"token present at {path}", False)
    return Check("token-present", "shared", "fail", f"token missing or empty at {path} — run install.sh", False)


_REGISTRY: dict[str, tuple] = {
    "token-present": (_check_token_present, None),
    "claude-mcp": (_check_claude_mcp, _fix_claude_mcp),
    "claude-guidance": (_check_claude_guidance, _fix_claude_guidance),
    "claude-hook": (_check_claude_hook, _fix_claude_hook),
    "codex-mcp": (_check_codex_mcp, _fix_codex_mcp),
    "codex-guidance": (_check_codex_guidance, _fix_codex_guidance),
    "gemini-mcp": (_check_gemini_mcp, _fix_gemini_mcp),
    "gemini-guidance": (_check_gemini_guidance, _fix_gemini_guidance),
    "other-agents": (_check_other_agents, None),
}


def run_all(home: Optional[Path] = None) -> list[dict]:
    home = Path(home) if home is not None else Path.home()
    return [dataclasses.asdict(check_fn(home)) for check_fn, _ in _REGISTRY.values()]


def fix(check_id: str, home: Optional[Path] = None, token: Optional[str] = None) -> str:
    home = Path(home) if home is not None else Path.home()
    if check_id not in _REGISTRY:
        raise ValueError(f"unknown check id: {check_id!r}")
    _, fix_fn = _REGISTRY[check_id]
    if fix_fn is None:
        raise ValueError(f"check {check_id!r} has no fix (report-only)")
    return fix_fn(home, token)


def fix_all(home: Optional[Path] = None, token: Optional[str] = None) -> list[str]:
    home = Path(home) if home is not None else Path.home()
    summaries = []
    for check_fn, fix_fn in _REGISTRY.values():
        if fix_fn is None:
            continue
        result = check_fn(home)
        if result.status == "fail":
            summaries.append(fix_fn(home, token))
    return summaries


def fix_group(group: str, home: Optional[Path] = None,
              token: Optional[str] = None) -> list[str]:
    """Apply every fixable failing check whose `tool` equals `group`
    (e.g. "claude", "codex", "gemini"). Returns one summary per fix.
    Lets a fix's own error propagate — a config we can't parse must not
    be silently overwritten (spec §S2/S6)."""
    home = Path(home) if home is not None else Path.home()
    summaries = []
    for check_fn, fix_fn in _REGISTRY.values():
        if fix_fn is None:
            continue
        result = check_fn(home)
        if result.tool == group and result.status == "fail":
            summaries.append(fix_fn(home, token))
    return summaries
