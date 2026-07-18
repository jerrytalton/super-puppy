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
import re
import shutil
import socket
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

    Permission invariant: the `.bak` matches the mode of the file it backs
    up, and the `.tmp` is created owner-only (0600) then bumped to `path`'s
    real mode before the replace (an ordinary 0644 config stays 0644; a new
    file is left at 0600). No secret is written into these files anymore —
    the token is referenced by env var — but keeping intermediates private
    is cheap defense in depth.
    """
    # Follow symlinks to the REAL target and write through it. A plain
    # os.replace() on a symlink swaps the link itself for a regular file —
    # which silently detaches a CLAUDE.md symlinked into a dotfiles repo
    # (edits stop propagating in both directions). Writing the resolved
    # target preserves the link.
    path = Path(path)
    real = Path(os.path.realpath(path))
    if real.exists():
        bak = real.parent / (real.name + ".bak")
        bak.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
        shutil.copymode(real, bak)
    tmp = real.parent / (real.name + ".tmp")
    fd = os.open(str(tmp), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    if real.exists():
        shutil.copymode(real, tmp)
    os.replace(tmp, real)


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


SP_TOKEN_ENV = "SP_MCP_TOKEN"
# The auth header references the token by env var. Claude Code expands
# ${SP_MCP_TOKEN} from the environment at load time (verified: env-var
# interpolation works in http MCP headers), so the literal secret never
# lands in ~/.claude.json — the file is safe to sync or commit, and there's
# no token-leak surface to guard. install.sh provisions SP_MCP_TOKEN from
# the untracked ~/.config/local-models/mcp_auth_token file.
_AUTH_HEADER_VALUE = f"Bearer ${{{SP_TOKEN_ENV}}}"


def _http_mcp_entry() -> dict:
    """Build a streamable-http MCP entry: X-SP-Client attribution + an
    env-var-referenced bearer token (never the literal secret)."""
    return {
        "type": "http",
        "url": SP_MCP_URL,
        "headers": {
            "Authorization": _AUTH_HEADER_VALUE,
            "X-SP-Client": _client_hostname(),
        },
    }


def _mcp_fix_summary(check_id: str, path: Path) -> str:
    return (f"{check_id}: registered mcpServers.local-models in {path} "
            f"(token via ${{{SP_TOKEN_ENV}}}, not inlined)")


# ── Claude Code accounts ─────────────────────────────────────────────────
#
# Jerry (and anyone who juggles work/personal logins) runs several Claude
# Code accounts on one machine by pointing CLAUDE_CONFIG_DIR at a different
# directory per login (a zsh wrapper swaps it by $PWD). Each such directory
# is a full config home — its own .claude.json, settings.json, and CLAUDE.md.
# The default login is the exception: its big config lives at ~/.claude.json
# while settings.json/CLAUDE.md live under ~/.claude.
#
# Auditing only the default login silently graded one of N accounts "good"
# while the others had no SP wiring at all. We discover the extra config dirs
# the same way the machine actually selects them — by reading the
# CLAUDE_CONFIG_DIR assignments out of the user's shell rc files — plus an
# optional declarative list for non-shell setups.

@dataclasses.dataclass
class ClaudeAccount:
    """One Claude Code login on this machine. `config_dir` holds settings.json
    and CLAUDE.md; `claude_json` is where that login's .claude.json lives."""
    label: str          # id/display slug: "default", "Blacklake", "dddg"
    config_dir: Path
    claude_json: Path
    is_default: bool

    def cid(self, base: str) -> str:
        """Per-account check id. The default login keeps the bare id
        (`claude-mcp`) for backward compatibility with the UI/CLI; extra
        logins are suffixed (`claude-mcp@Blacklake`)."""
        return base if self.is_default else f"{base}@{self.label}"


_SHELL_RC_FILES = (".zshrc", ".zshenv", ".zprofile", ".bashrc", ".bash_profile", ".profile")
_CLAUDE_CONFIG_DIR_RE = re.compile(
    r"(?<![A-Za-z0-9_])CLAUDE_CONFIG_DIR\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s;]+))"
)
EXTRA_CONFIG_DIRS_FILE = (".config", "local-models", "claude_config_dirs")


def _expand_home(raw: str, home: Path) -> Optional[Path]:
    """Resolve a config-dir path captured from shell rc / list file, expanding
    $HOME/${HOME}/~ against `home` (not the OS home) so audits stay hermetic
    under a fake home. Returns None for paths we can't anchor to `home`."""
    s = raw.strip()
    if not s:
        return None
    s = s.replace("${HOME}", str(home)).replace("$HOME", str(home))
    if s == "~" or s.startswith("~/"):
        s = str(home) + s[1:]
    p = Path(s)
    return p if p.is_absolute() else (home / p)


def _discover_config_dirs(home: Path) -> list[Path]:
    """Every non-default CLAUDE_CONFIG_DIR this machine uses, discovered from
    the user's shell rc files (the mechanism that actually selects them) and an
    optional declarative list at ~/.config/local-models/claude_config_dirs.
    Only existing directories are returned; the default (~/.claude) is dropped."""
    raws: list[str] = []
    for rc in _SHELL_RC_FILES:
        f = home / rc
        if not f.exists():
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _CLAUDE_CONFIG_DIR_RE.finditer(text):
            raws.append(next(g for g in m.groups() if g is not None))
    list_file = home.joinpath(*EXTRA_CONFIG_DIRS_FILE)
    if list_file.exists():
        for line in list_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                raws.append(line)

    default = os.path.normpath(str(home / ".claude"))
    seen: set[str] = set()
    out: list[Path] = []
    for raw in raws:
        p = _expand_home(raw, home)
        if p is None:
            continue
        key = os.path.normpath(str(p))
        if key == default or key in seen:
            continue
        seen.add(key)
        if p.is_dir():
            out.append(p)
    return out


def _account_label(config_dir: Path, used: set) -> str:
    """A short, alnum, unique display slug for a config dir. Prefers the
    parent dir name when the dir itself is a dotted `.claude-*` (so
    ~/Blacklake/.claude-work → "Blacklake")."""
    name = config_dir.name
    base = config_dir.parent.name if name.startswith(".claude") else name
    slug = re.sub(r"[^A-Za-z0-9]", "", base) or "alt"
    label, i = slug, 2
    while label in used:
        label, i = f"{slug}{i}", i + 1
    used.add(label)
    return label


def _claude_accounts(home: Path) -> list[ClaudeAccount]:
    accounts = [ClaudeAccount("default", home / ".claude", home / ".claude.json", True)]
    used = {"default"}
    for d in _discover_config_dirs(home):
        accounts.append(ClaudeAccount(_account_label(d, used), d, d / ".claude.json", False))
    return accounts


# ── Claude Code ──────────────────────────────────────────────────────────

def _check_claude_mcp(account: ClaudeAccount, home: Path) -> Check:
    cid, path = account.cid("claude-mcp"), account.claude_json
    try:
        data = _load_json(path)
    except json.JSONDecodeError as e:
        return Check(cid, "claude", "fail", f"{path} is not valid JSON: {e}", True)
    entry = data.get("mcpServers", {}).get("local-models")
    if isinstance(entry, dict) and entry.get("url") and entry.get("headers", {}).get("X-SP-Client"):
        return Check(cid, "claude", "pass", f"registered in {path} with X-SP-Client attribution", True)
    return Check(cid, "claude", "fail", f"mcpServers.local-models missing url/X-SP-Client header in {path}", True)


def _fix_claude_mcp(account: ClaudeAccount, token: Optional[str]) -> str:
    merge_json_key(account.claude_json, "mcpServers.local-models", _http_mcp_entry())
    return _mcp_fix_summary(account.cid("claude-mcp"), account.claude_json)


def _has_local_models_guidance(text: str) -> bool:
    """True if this agent-guidance file already tells the agent about the
    local-models tools — our managed block OR the user's own hand-written
    guidance. We never duplicate or overwrite guidance a user maintains
    themselves; a distinctive tool reference is enough to call it present."""
    if GUIDANCE_MARKERS[0] in text:
        return True
    low = text.lower()
    return ("local_models_status" in low
            or "local_dispatch" in low
            or ("local-models" in low and "mcp" in low))


_IMPORT_RE = re.compile(r"(?:^|\s)@(\S+)")


def _resolve_guidance_text(path: Path, home: Path, depth: int = 4,
                           _seen: Optional[set] = None) -> str:
    """Read a CLAUDE.md and inline its `@path` imports the way Claude Code
    does at load time (relative-to-file or absolute/~ paths, max 4 hops,
    cycle-guarded). An account whose CLAUDE.md is just `@~/.claude/CLAUDE.md`
    inherits the global guidance, so the audit must follow the import to see
    it — otherwise it false-fails an account that IS correctly wired."""
    if _seen is None:
        _seen = set()
    real = os.path.realpath(path)
    if real in _seen or not path.exists():
        return ""
    _seen.add(real)
    text = path.read_text(encoding="utf-8", errors="replace")
    if depth <= 0:
        return text
    parts = [text]
    for m in _IMPORT_RE.finditer(text):
        raw = m.group(1)
        if raw.startswith(("~", "$")):
            target = _expand_home(raw, home)          # home-relative (~ / $HOME)
        else:
            p = Path(raw)
            target = p if p.is_absolute() else (path.parent / raw)  # file-relative
        if target is not None and target.is_file():
            parts.append(_resolve_guidance_text(target, home, depth - 1, _seen))
    return "\n".join(parts)


def _check_claude_guidance(account: ClaudeAccount, home: Path) -> Check:
    cid, path = account.cid("claude-guidance"), account.config_dir / "CLAUDE.md"
    if _has_local_models_guidance(_resolve_guidance_text(path, home)):
        return Check(cid, "claude", "pass",
                     f"local-models guidance present in {path}", True)
    return Check(cid, "claude", "fail",
                 f"no local-models guidance in {path}", True)


def _fix_claude_guidance(account: ClaudeAccount, token: Optional[str]) -> str:
    path = account.config_dir / "CLAUDE.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, upsert_guidance(path.read_text(encoding="utf-8") if path.exists() else ""))
    return f"{account.cid('claude-guidance')}: upserted guidance block in {path}"


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


def _check_claude_hook(account: ClaudeAccount, home: Path) -> Check:
    cid, path = account.cid("claude-hook"), account.config_dir / "settings.json"
    try:
        data = _load_json(path)
    except json.JSONDecodeError as e:
        return Check(cid, "claude", "fail", f"{path} is not valid JSON: {e}", True)
    if _has_session_ping_hook(data):
        return Check(cid, "claude", "pass", f"SessionStart hook present in {path}", True)
    return Check(cid, "claude", "fail", f"SessionStart hook missing in {path}", True)


def _fix_claude_hook(account: ClaudeAccount, token: Optional[str]) -> str:
    path = account.config_dir / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    append_hook(path, SESSION_HOOK_ENTRY)
    return f"{account.cid('claude-hook')}: added SessionStart hook to {path}"


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


def _codex_managed_block() -> str:
    return "\n".join([
        CODEX_MARKERS[0],
        "[mcp_servers.local-models]",
        f'url = "{SP_MCP_URL}"',
        "",
        "[mcp_servers.local-models.headers]",
        f'Authorization = "{_AUTH_HEADER_VALUE}"',
        f'X-SP-Client = "{_client_hostname()}"',
        CODEX_MARKERS[1],
    ])


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
    new_text = _upsert_marked_text(existing, CODEX_MARKERS, _codex_managed_block())
    tomllib.loads(new_text)  # sanity: our own managed section must itself parse
    atomic_write(path, new_text)
    return (f"codex-mcp: appended managed [mcp_servers.local-models] section to {path} "
            f"(token via ${{{SP_TOKEN_ENV}}}, not inlined)")


def _check_codex_guidance(home: Path) -> Check:
    codex_dir = home / ".codex"
    if not codex_dir.exists():
        return Check("codex-guidance", "codex", "n/a", "Codex not installed (~/.codex absent)", False)
    path = codex_dir / "AGENTS.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if _has_local_models_guidance(text):
        return Check("codex-guidance", "codex", "pass", f"local-models guidance present in {path}", True)
    return Check("codex-guidance", "codex", "fail", f"no local-models guidance in {path}", True)


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
    merge_json_key(path, "mcpServers.local-models", _http_mcp_entry())
    return _mcp_fix_summary("gemini-mcp", path)


def _check_gemini_guidance(home: Path) -> Check:
    gemini_dir = home / ".gemini"
    if not gemini_dir.exists():
        return Check("gemini-guidance", "gemini", "n/a", "Gemini CLI not installed (~/.gemini absent)", False)
    path = gemini_dir / "GEMINI.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if _has_local_models_guidance(text):
        return Check("gemini-guidance", "gemini", "pass", f"local-models guidance present in {path}", True)
    return Check("gemini-guidance", "gemini", "fail", f"no local-models guidance in {path}", True)


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


# Claude checks are expanded once per discovered account (see _enumerate);
# their check/fix functions take (account, home)/(account, token). Everything
# else is single-context and keyed on `home`.
_CLAUDE_CHECKS: dict[str, tuple] = {
    "claude-mcp": (_check_claude_mcp, _fix_claude_mcp),
    "claude-guidance": (_check_claude_guidance, _fix_claude_guidance),
    "claude-hook": (_check_claude_hook, _fix_claude_hook),
}

_REGISTRY: dict[str, tuple] = {
    "token-present": (_check_token_present, None),
    "codex-mcp": (_check_codex_mcp, _fix_codex_mcp),
    "codex-guidance": (_check_codex_guidance, _fix_codex_guidance),
    "gemini-mcp": (_check_gemini_mcp, _fix_gemini_mcp),
    "gemini-guidance": (_check_gemini_guidance, _fix_gemini_guidance),
    "other-agents": (_check_other_agents, None),
}


def _enumerate(home: Path):
    """Yield (Check, fix_callable_or_None) for every check on this machine —
    the Claude checks expanded across every discovered account, then the
    single-context registry checks. The fix callable closes over its context
    (account or home) and takes only the token, so callers never re-derive it."""
    for account in _claude_accounts(home):
        for check_fn, fix_fn in _CLAUDE_CHECKS.values():
            check = check_fn(account, home)
            cb = (lambda tok, _f=fix_fn, _a=account: _f(_a, tok))
            yield check, cb
    for check_fn, fix_fn in _REGISTRY.values():
        cb = None if fix_fn is None else (lambda tok, _f=fix_fn: _f(home, tok))
        yield check_fn(home), cb


def run_all(home: Optional[Path] = None) -> list[dict]:
    home = Path(home) if home is not None else Path.home()
    return [dataclasses.asdict(check) for check, _ in _enumerate(home)]


def fix(check_id: str, home: Optional[Path] = None, token: Optional[str] = None) -> str:
    home = Path(home) if home is not None else Path.home()
    for check, cb in _enumerate(home):
        if check.id == check_id:
            if cb is None:
                raise ValueError(f"check {check_id!r} has no fix (report-only)")
            return cb(token)
    raise ValueError(f"unknown check id: {check_id!r}")


def fix_all(home: Optional[Path] = None, token: Optional[str] = None) -> list[str]:
    home = Path(home) if home is not None else Path.home()
    return [cb(token) for check, cb in _enumerate(home)
            if cb is not None and check.status == "fail"]


def fix_group(group: str, home: Optional[Path] = None,
              token: Optional[str] = None) -> list[str]:
    """Apply every fixable failing check whose `tool` equals `group`
    (e.g. "claude", "codex", "gemini") — across ALL accounts for that group.
    Returns one summary per fix. Lets a fix's own error propagate — a config
    we can't parse must not be silently overwritten (spec §S2/S6)."""
    home = Path(home) if home is not None else Path.home()
    return [cb(token) for check, cb in _enumerate(home)
            if cb is not None and check.tool == group and check.status == "fail"]
