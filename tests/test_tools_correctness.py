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

from lib.models import DS4_MODEL_NAME
from tests._smoke_helpers import (
    assert_media_output, assert_tool_output_contains, call_api_test, client,
    ps, require_ds4, require_local_services, smoke_tmp, write_png,
    write_speech_wav, write_text,
)


def _avg_rgb(path):
    """Average (r,g,b) of an image, downscaled — for asserting a
    generated/edited image is the color it was asked to be."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((8, 8))
    px = list(im.getdata())
    n = len(px)
    return tuple(sum(p[i] for p in px) // n for i in range(3))

# Skip the whole module at collection if local services aren't up.
require_local_services()

pytestmark = pytest.mark.correctness


def _model_for(task: str) -> str:
    """The model the active profile picks for `task`.

    Skips only when the active profile doesn't define the task at all.
    If the profile DOES define it and the picker still returns nothing,
    that's a capability-gate or discovery regression silently dropping a
    tool from the release gate — fail loudly, never skip (2026-08-14
    red-team: a skip here made the gate vacuous for that tool)."""
    name, _backend, _warn = ps()._pick_model_for_task(task)
    if not name:
        profiles = ps().load_profiles()
        active = (profiles.get("profiles") or {}).get(profiles.get("active"), {})
        tasks = active.get("tasks", {})
        if task in tasks:
            pytest.fail(
                f"active profile defines {task!r} -> {tasks[task]!r} but "
                f"_pick_model_for_task returned nothing — picker/capability "
                f"regression, not a missing model")
        pytest.skip(f"active profile has no model for {task!r}")
    return name


# Saturated, unambiguous colors with tolerated synonyms. A model that
# actually sees the image names the color; a blind backend guesses and
# fails at least one of these.
_COLOR_CASES = [
    ("red", (220, 20, 20), ["red", "crimson", "maroon", "scarlet"]),
    ("green", (20, 170, 20), ["green", "lime", "emerald"]),
    ("blue", (20, 20, 220), ["blue", "navy", "azure"]),
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


def test_transcription_reads_speech(client, smoke_tmp):
    """Whisper must transcribe spoken words, not return empty/garbage.
    A backend that silently drops the audio fails this."""
    model = _model_for("transcription")
    wav = write_speech_wav(smoke_tmp / "speech.wav",
                           "The quick brown fox jumps over the lazy dog")
    assert_tool_output_contains(
        client, tool="transcribe", model=model,
        expect_any=["quick brown fox", "brown fox", "lazy dog"],
        audio_path=str(wav),
    )


def test_chat_follows_a_basic_instruction(client):
    """The general/code text model must actually follow a trivial
    instruction — catches a model that loads but generates garbage."""
    model = _model_for("general")
    assert_tool_output_contains(
        client, tool="general", model=model, expect_any=["banana"],
        prompt="Reply with exactly one word: banana",
    )


def test_ds4_glm52_chat_correctness(client):
    """glm-5.2 pinned explicitly (not via _model_for): the release gate for
    the whole ds4 dispatch chain — profile-server → :8002 → strict-tolerant
    JSON parse → content/reasoning_content extraction. glm-5.2 always
    thinks on ds4, so a real response exercises exactly the reasoning-heavy
    payloads where the control-char encoder bug bites. Skips below the
    512GB tier; FAILS if ds4 is installed but down (require_ds4)."""
    require_ds4()
    assert_tool_output_contains(
        client, tool="general", model=DS4_MODEL_NAME,
        expect_any=["kumquat"],
        prompt="Reply with exactly one word: kumquat")


def test_summarize_reflects_source(client, smoke_tmp):
    """A summary must mention the source's actual subject."""
    model = _model_for("long_context")
    src = write_text(
        smoke_tmp / "src.txt",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, "
        "France. It was completed in 1889 and is named after Gustave "
        "Eiffel, whose company designed and built it.\n")
    assert_tool_output_contains(
        client, tool="summarize", model=model,
        expect_any=["eiffel", "paris", "tower"],
        file_path=str(src),
    )


def test_embedding_is_a_real_vector(client):
    """The embedder must return a numeric vector of real dimension, not
    an empty or degenerate result."""
    model = _model_for("embedding")
    status, data = call_api_test(
        client, "embed", model, text="The quick brown fox.")
    err = str(data.get("error", "")) if isinstance(data, dict) else ""
    if err and any(s in err.lower() for s in ("not downloaded", "no model", "cannot connect")):
        pytest.skip(f"embed({model}): {err}")
    assert status == 200, f"embed({model}) HTTP {status}: {data}"
    vec = data.get("embeddings")
    if vec and isinstance(vec[0], list):
        vec = vec[0]
    assert vec and len(vec) >= 64, f"degenerate embedding: len={len(vec) if vec else 0}"
    assert all(isinstance(x, (int, float)) for x in vec[:8]), "non-numeric embedding"
    assert any(abs(x) > 1e-6 for x in vec), "all-zero embedding vector"


def test_computer_use_grounds_on_screenshot(client, smoke_tmp):
    """Computer_use must return a grounding action, not hang. MLX models
    now dispatch via a one-shot mlx_vlm subprocess (the :8000 VLM path
    hangs on the mlx 0.31.2 stream bug); a correct dispatch yields a
    click action, a broken one errors or times out."""
    model = _model_for("computer_use")
    img = write_png(smoke_tmp / "screen.png", size=400, rgb=(240, 240, 240))
    assert_tool_output_contains(
        client, tool="computer_use", model=model,
        expect_any=["click", "box", '"x"'],
        image_path=str(img), intent="Click the center of the screen.")


# ── media generation tools ──────────────────────────────────────────

def test_image_gen_produces_requested_color(client):
    """The image model must generate an image that is the color it was
    asked for — a model that ignores the prompt fails the color check."""
    model = _model_for("image_gen")
    path = assert_media_output(
        client, tool="image_gen", model=model, expect_key="image_path",
        min_bytes=1000,
        prompt="a solid pure blue rectangle filling the entire frame, "
               "flat blue color, no other colors, no text")
    r, g, b = _avg_rgb(path)
    assert b > r and b > g, f"expected a blue image, got avg rgb ({r},{g},{b})"


def test_image_edit_recolors_the_input(client, smoke_tmp):
    """Editing a solid-red image to blue must shift the output toward
    blue — catches an edit model that ignores its input."""
    model = _model_for("image_edit")
    red = write_png(smoke_tmp / "red_in.png", size=256, rgb=(220, 20, 20))
    path = assert_media_output(
        client, tool="image_edit", model=model, expect_key="image_path",
        min_bytes=1000, image_path=str(red),
        prompt="change the color to solid blue, keep it a flat solid color")
    r, g, b = _avg_rgb(path)
    assert b > r, f"edit did not move red->blue: avg rgb ({r},{g},{b})"


def test_tts_produces_audio(client):
    """local_speak must produce a real, non-empty audio file (regression
    guard for the 'mlx-audio' GPU-tracking KeyError, now fixed)."""
    model = _model_for("tts")
    assert_media_output(
        client, tool="speak", model=model, expect_key="audio_path",
        min_bytes=2000, text="Testing the local text to speech pipeline.")


@pytest.mark.slow
def test_video_produces_mp4(client):
    """Video generation must produce a real, non-empty MP4. Marked slow;
    skips cleanly if the model isn't pulled. Tiny resolution/frame count
    keep it to ~90s — at Wan2.2's default 1280x704 it's ~100 min/clip
    (per-step time scales with the latent size)."""
    model = _model_for("video")
    # dims/frames go straight into the subprocess argv, so they must be
    # strings (the Playground sends strings; JSON ints raise TypeError).
    assert_media_output(
        client, tool="video", model=model, expect_key="video_path",
        min_bytes=10000, prompt="a red ball bouncing",
        num_frames="9", width="320", height="320")
