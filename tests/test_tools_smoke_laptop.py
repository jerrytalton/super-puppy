"""Laptop-profile smoke tests — run on every `pytest` invocation.

Walks every task in the `laptop` profile and drives the profile-server
/api/test route with the profile's default model, producing a real output
artifact for each one. Skips cleanly when:

  - Ollama or MLX isn't running locally (whole module skips)
  - a specific model isn't pulled (that test skips)

Companion slow suite: test_tools_smoke_everyday.py.
"""

from __future__ import annotations

import pytest

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


LAPTOP = {
    "code":          "qwen3.6:27b",
    "general":       "qwen3.6:27b",
    "reasoning":     "qwen3.6:27b",
    "long_context":  "qwen3.6:27b",
    "translation":   "qwen3.6:27b",
    "vision":        "qwen3.6:27b",
    "image_gen":     "x/flux2-klein:latest",
    "transcription": "whisper-v3",
    "tts":           "mlx-community/Kokoro-82M-bf16",
    "embedding":     "nomic-embed-text:latest",
    # unfiltered and computer_use intentionally absent — see DEFAULT_PROFILES.
    # Cases parametrized against missing keys skip cleanly via run_fixture_case.
}


@pytest.mark.parametrize(
    "tool,profile_task,build_body",
    CHAT_CASES,
    ids=[c[0] for c in CHAT_CASES],
)
def test_laptop_chat(client, smoke_tmp, tool, profile_task, build_body):
    run_chat_case(client, LAPTOP, tool, profile_task, build_body, smoke_tmp)


@pytest.mark.parametrize(
    "tool,profile_task,build_body,expect_key",
    FIXTURE_CASES,
    ids=[c[0] for c in FIXTURE_CASES],
)
def test_laptop_tool(client, smoke_tmp, tool, profile_task, build_body, expect_key):
    run_fixture_case(
        client, LAPTOP, tool, profile_task, build_body, expect_key, smoke_tmp)
