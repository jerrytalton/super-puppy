#!/usr/bin/env python3
"""Fail a release when the live test layer didn't actually run.

`pytest` exits 0 whether the correctness/smoke suites verified a live stack
or skipped every case. Those two outcomes look identical in the summary line
and mean opposite things, so the release gate has to tell them apart itself.

It matters because the failure is silent and self-concealing. v1.5.0 shipped
with `local_image` broken: the tests that would have caught it were skipping
(and, before the sys.modules stub leak was fixed, running against mocks), so
the gate went green on a suite that had never once exercised a real backend.
A gate that reports success when it verified nothing is worse than no gate —
it converts "we didn't check" into "we checked and it's fine".

Usage:
    check-live-coverage.py <junit.xml> [--min N]

Exits 0 when at least N live cases executed, 1 otherwise (or on a malformed
report — an unreadable gate result is not a pass).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Suites that drive real Ollama/MLX/mflux calls. Anything here skipping
# wholesale means the machine cutting the release verified nothing live.
LIVE_MODULES = (
    "test_tools_correctness",
    "test_tools_smoke_laptop",
    "test_tools_smoke_everyday",
)


def live_case_counts(report: Path) -> tuple[int, int]:
    """Return (executed, skipped) for live-module cases in a junit report."""
    root = ET.parse(report).getroot()
    executed = skipped = 0
    for case in root.iter("testcase"):
        origin = f"{case.get('classname', '')}.{case.get('file', '')}"
        if not any(mod in origin for mod in LIVE_MODULES):
            continue
        if case.find("skipped") is not None:
            skipped += 1
        else:
            executed += 1
    return executed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--min", type=int, default=1,
                        help="minimum live cases that must have executed")
    args = parser.parse_args()

    if not args.report.is_file():
        print(f"live-coverage: no junit report at {args.report}", file=sys.stderr)
        return 1
    try:
        executed, skipped = live_case_counts(args.report)
    except ET.ParseError as e:
        print(f"live-coverage: unreadable junit report: {e}", file=sys.stderr)
        return 1

    if executed < args.min:
        print(
            f"live-coverage: {executed} live test(s) ran, {skipped} skipped — "
            f"need at least {args.min}.\n"
            "  The suite passed without exercising a real backend, which is "
            "how a broken tool ships under a green tag.\n"
            "  Start Ollama/MLX (start-local-models), pull the active "
            "profile's models, then re-run.",
            file=sys.stderr,
        )
        return 1

    print(f"live-coverage: {executed} live test(s) executed, {skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
