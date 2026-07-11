"""Super Puppy configuration audit: verify (and optionally fix) that installed
agent tools are wired to use SP — MCP registration, guidance blocks, session hook.

Design note (spec §S2): guidance blocks are text an AI agent treats as
instructions. Fixes here are NEVER automatic — a caller (sp-doctor / menubar /
install.sh opt-in) applies them explicitly, with a diff. The managed block is
minimal and mechanical; changing GUIDANCE_TEXT is a reviewed code diff.
"""

import json
import os
from pathlib import Path

GUIDANCE_MARKERS = ("<!-- >>> super-puppy >>> -->", "<!-- <<< super-puppy <<< -->")

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
    """
    path = Path(path)
    if path.exists():
        (path.parent / (path.name + ".bak")).write_text(path.read_text())
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    """Load JSON from `path`, or {} if absent. Lets JSONDecodeError
    propagate — a file we can't parse must never be silently overwritten."""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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
