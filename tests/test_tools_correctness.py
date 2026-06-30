"""Tool correctness tests — gated on releases, NOT every commit.

The smoke suites (`test_tools_smoke_*.py`) prove each tool *runs* against
the profile's chosen model. They do NOT prove the answer is *right*: a
model that silently ignores its input returns nonempty output and passes
a smoke test while being completely wrong. That is exactly how a vision
regression slipped through — Ollama's `-mlx` tags advertise vision, never
see the image, and hallucinate plausible answers.

These tests feed a KNOWN input and assert the output reflects it, for the
model each profile actually picks. They are heavyweight (real model loads)
and so carry the `correctness` marker, which the default `pytest tests/`
run excludes (see pyproject `addopts`). `bin/release.sh` runs them as part
of its version-bump gate. Add a case here whenever a tool's correctness
depends on the chosen model honoring its input.

Skip behavior mirrors the smoke suites: services down → module skips;
model not pulled → that case skips; wrong answer → loud failure.
"""

from __future__ import annotations

import pytest

from tests._smoke_helpers import (
    assert_tool_output_contains, client, ps, require_local_services,
    smoke_tmp, write_png,
)

# Skip the whole module at collection if local services aren't up.
require_local_services()

pytestmark = pytest.mark.correctness


def _model_for(task: str) -> str:
    """The model the active profile picks for `task`, or skip."""
    name, _backend, _warn = ps()._pick_model_for_task(task)
    if not name:
        pytest.skip(f"active profile has no usable model for {task!r}")
    return name


# Saturated, unambiguous colors with tolerated synonyms. A model that
# actually sees the image names the color; a blind backend guesses and
# fails at least one of these.
_COLOR_CASES = [
    ("red", (220, 20, 20), ["red", "crimson", "maroon"]),
    ("green", (20, 170, 20), ["green"]),
    ("blue", (20, 20, 220), ["blue", "navy"]),
]


@pytest.mark.parametrize("label,rgb,accept", _COLOR_CASES)
def test_vision_reads_dominant_color(client, smoke_tmp, label, rgb, accept):
    """The vision model must report the actual dominant color of the image.

    This is the test that would have caught the `-mlx` vision regression:
    a tower-less model returns a color uncorrelated with the input."""
    model = _model_for("vision")
    img = write_png(smoke_tmp / f"color_{label}.png", size=96, rgb=rgb)
    assert_tool_output_contains(
        client, tool="vision", model=model, expect_any=accept,
        image_path=str(img),
        prompt="What is the single dominant color of this image? "
               "Answer with just the color name, one word.",
    )


def test_translation_actually_translates(client):
    """A translation to French of 'Hello' must contain a French greeting,
    not echo the English or produce unrelated text."""
    model = _model_for("translation")
    assert_tool_output_contains(
        client, tool="translate", model=model,
        expect_any=["bonjour", "salut"],
        target="French", text="Hello",
    )
