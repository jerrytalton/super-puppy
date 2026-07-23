#!/usr/bin/env python3
"""Remove a served model's entry from an mlx-openai-server config.yaml.

Usage: migrate-mlx-config.py <config.yaml> <served_model_name>

One-shot migration helper called by post-update.sh: its MLX config merge is
append-only, so retiring a model (glm-5.2 moved to the ds4 backend,
2026-07-22) needs an explicit removal or updated 512GB boxes double-serve
it — MLX claims the name first in discovery order and keeps 418GB of dead
weights pullable. Stdlib-only and text-based (the system python3 that
post-update.sh uses has no pyyaml): drops the matching model block — its
leading comment lines through its last body line. Idempotent: exits 0 and
leaves the file untouched when no entry matches.

Before the FIRST actual modification, writes a timestamped backup of the
pre-migration file (`config.yaml.pre-ds4-<YYYYmmddTHHMMSS>`) next to it —
a manual rollback path (see docs/troubleshooting.md) if a user needs the
original entry back. Skipped once any such backup already exists, since
post-update.sh re-runs this on every update and only the first run's
"before" state is worth keeping.
"""

import re
import sys
from datetime import datetime
from pathlib import Path

_ENTRY_RE = re.compile(r"^\s*-\s*model_path:")
_SERVED_RE = re.compile(r"^\s*served_model_name:\s*(\S+)\s*$")


def remove_served_model(text: str, served_name: str) -> tuple[str, bool]:
    """Return (new_text, removed). Block boundaries mirror post-update.sh's
    merge: an entry starts at `- model_path:` (plus the run of comment lines
    directly above it) and ends at the next entry or EOF."""
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if _ENTRY_RE.match(l)]
    if not starts:
        return text, False

    for n, s in enumerate(starts):
        begin = s
        while begin > 0 and lines[begin - 1].strip().startswith("#"):
            begin -= 1
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        # A following entry's leading comments belong to IT, not to us.
        if n + 1 < len(starts):
            while end - 1 > s and lines[end - 1].strip().startswith("#"):
                end -= 1
        block = lines[s:end]
        matches = any(
            (m := _SERVED_RE.match(l)) and m.group(1) == served_name
            for l in block
        )
        if matches:
            new_text = "\n".join(lines[:begin] + lines[end:])
            new_text = re.sub(r"\n{3,}", "\n\n", new_text)
            return new_text, True
    return text, False


def _write_backup_if_absent(cfg_path: Path, original_text: str) -> None:
    """Write a timestamped pre-migration backup unless one already exists."""
    if any(cfg_path.parent.glob(f"{cfg_path.name}.pre-ds4-*")):
        return
    backup_path = cfg_path.with_name(
        f"{cfg_path.name}.pre-ds4-{datetime.now():%Y%m%dT%H%M%S}")
    backup_path.write_text(original_text, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: migrate-mlx-config.py <config.yaml> <served_model_name>",
              file=sys.stderr)
        return 2
    path, served = sys.argv[1], sys.argv[2]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new_text, removed = remove_served_model(text, served)
    if removed:
        _write_backup_if_absent(Path(path), text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        print(f"removed MLX entry for served model {served!r}")
    else:
        print(f"no MLX entry for {served!r} — nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
