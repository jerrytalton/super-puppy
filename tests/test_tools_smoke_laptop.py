"""32 GB tier smoke tests — run on every `pytest` invocation.

Walks every task in the shipped `32gb` preset and drives the profile-server
/api/test route with the tier's default model, producing a real output
artifact for each one. Skips cleanly when:

  - Ollama or MLX isn't running locally (whole module skips)
  - a specific model isn't pulled (that test skips)

Companion slow suite: test_tools_smoke_everyday.py (512 GB tier).
"""

from __future__ import annotations

import pytest

from lib.models import DEFAULT_PROFILES
from tests._smoke_helpers import (
    CHAT_CASES, FIXTURE_CASES,
    client, require_local_services, run_chat_case, run_fixture_case, smoke_tmp,
)

# Skip the module at collection time if local services aren't up.
require_local_services()

# `smoke` marker: live end-to-end — excluded from the pre-commit hook
# because cold model loads are occasionally flaky and the hook must be
# deterministic. Users still get it by running `pytest tests/`.
pytestmark = pytest.mark.smoke


# Derived from the shipped preset so the map can't drift when the tier
# changes (it did, twice). Tasks absent from the tier skip cleanly via
# run_chat_case/run_fixture_case.
TIER_32GB = dict(DEFAULT_PROFILES["profiles"]["32gb"]["tasks"])


@pytest.mark.parametrize(
    "tool,profile_task,build_body",
    CHAT_CASES,
    ids=[c[0] for c in CHAT_CASES],
)
def test_laptop_chat(client, smoke_tmp, tool, profile_task, build_body):
    run_chat_case(client, TIER_32GB, tool, profile_task, build_body, smoke_tmp)


@pytest.mark.parametrize(
    "tool,profile_task,build_body,expect_key",
    FIXTURE_CASES,
    ids=[c[0] for c in FIXTURE_CASES],
)
def test_laptop_tool(client, smoke_tmp, tool, profile_task, build_body, expect_key):
    run_fixture_case(
        client, TIER_32GB, tool, profile_task, build_body, expect_key, smoke_tmp)


@pytest.mark.smoke
def test_fleet_report_roundtrip(client):
    """Fleet heartbeat round-trip: POST usage report, then GET fleet and verify."""
    from lib import activity
    activity.init_db()

    payload = {
        "machine": "smoke-fleet-rt",
        "version": "v0.0.0",
        "mode": "server",
        "sent_at": 1,
        "audit": [],
        "usage": [
            {
                "day": "2026-07-11",
                "tool": "vision",
                "source": "mcp",
                "count": 1,
                "errors": 0,
                "avg_ms": 50,
            }
        ],
    }

    # POST the report; accept 200 or 429 (rate-limited from prior run)
    r = client.post("/api/fleet/report", json=payload)
    assert r.status_code in (200, 429), f"POST /api/fleet/report failed: {r.get_json()}"

    # GET the fleet and verify the machine appears
    got = client.get("/api/fleet").get_json()
    assert got is not None, "GET /api/fleet returned None"
    machines = got.get("machines", [])
    assert any(
        m["machine"] == "smoke-fleet-rt" for m in machines
    ), f"Machine 'smoke-fleet-rt' not found in fleet. Got: {machines}"
