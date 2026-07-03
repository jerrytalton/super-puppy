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
    assert_tool_output_contains, call_api_test, client, ps,
    require_local_services, smoke_tmp, write_png, write_speech_wav, write_text,
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


# ── Known-broken tools (tracked, not run) ───────────────────────────
# These are documented failures with root causes outside our control or
# pending a separate fix. `run=False` keeps the suite from hanging on
# them while still flagging (xpass) the day they start working again.

@pytest.mark.xfail(
    run=False,
    reason="mlx-openai-server multimodal (VLM) generation hangs on the mlx "
           "0.31.2 thread-local-stream bug (mlx-lm #1256); RPC-times-out "
           "with no output. Upstream. Remove run=False when fixed.")
def test_computer_use_grounds_on_screenshot(client, smoke_tmp):
    model = _model_for("computer_use")
    img = write_png(smoke_tmp / "screen.png", size=200, rgb=(240, 240, 240))
    assert_tool_output_contains(
        client, tool="computer_use", model=model, expect_any=["click", "x", "y"],
        image_path=str(img), intent="Click the center of the screen.")


@pytest.mark.xfail(
    run=False,
    reason="local_speak errors with 'mlx-audio' (dispatch/env bug in our "
           "code, not the mlx stream bug). Remove run=False once fixed.")
def test_tts_produces_audio(client, smoke_tmp):
    model = _model_for("tts")
    assert_tool_output_contains(
        client, tool="speak", model=model, expect_key="audio_path",
        expect_any=[".wav", ".mp3", "/tmp"], text="Hello world.")
