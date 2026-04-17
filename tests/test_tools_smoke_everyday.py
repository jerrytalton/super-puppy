"""Everyday-profile smoke tests — gated `slow`.

Same structure as test_tools_smoke_laptop.py but exercises the bigger
models in the `everyday` profile. Run with:

    pytest -m slow tests/test_tools_smoke_everyday.py

Skips the whole module if Ollama or MLX isn't reachable, and skips
individual cases when the target model isn't pulled locally.
"""

from __future__ import annotations

import pytest

from tests._smoke_helpers import (
    CHAT_CASES, FIXTURE_CASES,
    client, require_local_services, run_chat_case, run_fixture_case, smoke_tmp,
)

# Skip the module at collection time if local services aren't up.
require_local_services()


EVERYDAY = {
    "code":          "qwen3.6-35b-bf16",
    "general":       "qwen3.6-35b-bf16",
    "reasoning":     "qwen3.6-35b-bf16",
    "long_context":  "qwen3.6-35b-bf16",
    "translation":   "qwen3.6-35b-bf16",
    "vision":        "qwen3.5:122b",
    "image_gen":     "x/z-image-turbo:bf16",
    "image_edit":    "black-forest-labs/FLUX.1-Kontext-dev",
    "transcription": "whisper-v3",
    "tts":           "mlx-community/Voxtral-4B-TTS-2603-mlx-bf16",
    "embedding":     "qwen3-embedding:8b",
    "unfiltered":    "dolphin3:8b",
    "computer_use":  "ui-tars-72b",
    "video":         "AITRADER/Wan2.2-T2V-A14B-mlx-bf16",
}


pytestmark = [pytest.mark.smoke, pytest.mark.slow]


@pytest.mark.parametrize(
    "tool,profile_task,build_body",
    CHAT_CASES,
    ids=[c[0] for c in CHAT_CASES],
)
def test_everyday_chat(client, smoke_tmp, tool, profile_task, build_body):
    run_chat_case(client, EVERYDAY, tool, profile_task, build_body, smoke_tmp)


@pytest.mark.parametrize(
    "tool,profile_task,build_body,expect_key",
    FIXTURE_CASES,
    ids=[c[0] for c in FIXTURE_CASES],
)
def test_everyday_tool(client, smoke_tmp, tool, profile_task, build_body, expect_key):
    run_fixture_case(
        client, EVERYDAY, tool, profile_task, build_body, expect_key, smoke_tmp)
