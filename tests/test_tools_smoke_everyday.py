"""512 GB tier smoke tests — gated `slow`.

Same structure as test_tools_smoke_laptop.py but exercises the bigger
models in the shipped `512gb` preset. Run with:

    pytest -m slow tests/test_tools_smoke_everyday.py

Skips the whole module if Ollama or MLX isn't reachable, or if this
machine is below the 512GB tier (ds4 not installed) — but FAILS loudly,
rather than skipping, when ds4 is installed and not answering: on the box
that serves glm-5.2, a ds4 outage must never hide as a module skip. Also
skips individual cases when the target model isn't pulled locally.
"""

from __future__ import annotations

import pytest

from lib.models import DEFAULT_PROFILES
from tests._smoke_helpers import (
    CHAT_CASES, FIXTURE_CASES,
    client, require_ds4, require_local_services, run_chat_case,
    run_fixture_case, run_stream_case, smoke_tmp,
)

# Skip the module at collection time if local services aren't up.
require_local_services()
# The 512gb tier's chat tasks route to ds4-served glm-5.2. Skip below the
# 512GB tier; FAIL (not skip) when ds4 is installed but down.
require_ds4()


# Derived from the shipped preset so the map can't drift when the tier
# changes (it did, twice — vision and tts both went stale after 0e3fefb).
TIER_512GB = dict(DEFAULT_PROFILES["profiles"]["512gb"]["tasks"])


pytestmark = [pytest.mark.smoke, pytest.mark.slow]


@pytest.mark.parametrize(
    "tool,profile_task,build_body",
    CHAT_CASES,
    ids=[c[0] for c in CHAT_CASES],
)
def test_everyday_chat(client, smoke_tmp, tool, profile_task, build_body):
    run_chat_case(client, TIER_512GB, tool, profile_task, build_body, smoke_tmp)


@pytest.mark.parametrize(
    "tool,profile_task,build_body,expect_key",
    FIXTURE_CASES,
    ids=[c[0] for c in FIXTURE_CASES],
)
def test_everyday_tool(client, smoke_tmp, tool, profile_task, build_body, expect_key):
    run_fixture_case(
        client, TIER_512GB, tool, profile_task, build_body, expect_key, smoke_tmp)


def test_everyday_unfiltered_stream(client):
    """unfiltered is stream-only (no /api/test handler), so the chat cases
    can't cover it — and it's the one task served from a local-dir MLX
    entry, the path most likely to break discovery."""
    run_stream_case(client, TIER_512GB, "unfiltered", "unfiltered",
                    "Reply with the single word OK.")
