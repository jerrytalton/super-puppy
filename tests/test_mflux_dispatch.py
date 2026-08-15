"""Unit tests for `lib.models.mflux_command` + `mflux_is_turbo`.

Pure dispatch table — no subprocess, no network. Catches dispatch bugs
(e.g. "FLUX.2-klein routed to the FLUX.1 generator") at unit-test speed.
Pair this with `test_tools_smoke_laptop.py` for the real invocation check.
"""

from __future__ import annotations

import shutil

import pytest

from lib.models import (
    _MFLUX_DISPATCH,
    _MFLUX_EDIT_DISPATCH,
    mflux_command,
    mflux_edit_command,
    mflux_is_turbo,
)


DISPATCH_CASES = [
    # (input,                                         binary,                          base-model,           has_model_flag)
    ("black-forest-labs/FLUX.2-klein-9B",             "mflux-generate-flux2",          "flux2-klein-9b",      True),
    ("black-forest-labs/FLUX.2-klein-4B",             "mflux-generate-flux2",          "flux2-klein-4b",      True),
    ("black-forest-labs/FLUX.2-klein-base-9B",        "mflux-generate-flux2",          "flux2-klein-base-9b", True),
    ("black-forest-labs/FLUX.2-klein-base-4B",        "mflux-generate-flux2",          "flux2-klein-base-4b", True),
    ("mlx-community/Z-Image-Turbo",                   "mflux-generate-z-image-turbo",  "z-image-turbo",       True),
    ("x/z-image-turbo:bf16",                          "mflux-generate-z-image-turbo",  "z-image-turbo",       False),
    ("mlx-community/Z-Image",                         "mflux-generate-z-image",        "z-image",             True),
    ("some-org/Qwen-Image-diffusion",                 "mflux-generate-qwen",           "qwen",                True),
    ("black-forest-labs/FLUX.1-dev",                  "mflux-generate",                "dev",                 True),
    ("black-forest-labs/FLUX.1-schnell",              "mflux-generate",                "schnell",             True),
    ("black-forest-labs/FLUX.1-Krea-dev",             "mflux-generate",                "krea-dev",            True),
    ("dev",                                           "mflux-generate",                None,                  True),   # fallback w/ --model
    ("schnell",                                       "mflux-generate",                None,                  True),
    ("some/completely-unknown-model",                 "mflux-generate",                None,                  True),
]


@pytest.mark.parametrize("model_id,binary,base,has_model", DISPATCH_CASES,
                         ids=[c[0] for c in DISPATCH_CASES])
def test_mflux_command(model_id, binary, base, has_model):
    got_binary, args = mflux_command(model_id)
    assert got_binary == binary, f"{model_id!r}: wrong binary"
    if base is not None:
        assert "--base-model" in args, f"{model_id!r}: missing --base-model"
        idx = args.index("--base-model")
        assert args[idx + 1] == base, f"{model_id!r}: wrong base-model"
    else:
        assert "--base-model" not in args, f"{model_id!r}: unexpected --base-model"
    if has_model:
        # Either --model carries the HF path (recognized family + path-ish id),
        # or --model carries the id verbatim (fallback path).
        assert "--model" in args, f"{model_id!r}: missing --model"


@pytest.mark.parametrize("model_id,is_turbo", [
    ("FLUX.1-schnell",       True),
    ("z-image-turbo",        True),
    ("FLUX.1-dev",           False),
    ("FLUX.2-klein-9B",      False),    # "klein" is not turbo; German for "small"
    ("nemotron-super",       False),
])
def test_mflux_is_turbo(model_id, is_turbo):
    assert mflux_is_turbo(model_id) is is_turbo


def test_mflux_dispatch_binaries_installed():
    """Every binary named in the dispatch table must actually exist on PATH.

    A silent rename in an mflux version bump would break image gen for the
    whole family using that binary; this catches it on the next test run.
    Skips if mflux-generate itself isn't installed (dev-machine-less CI).
    """
    if shutil.which("mflux-generate") is None:
        pytest.skip("mflux not installed on this machine")

    named = {binary for _, binary, _ in _MFLUX_DISPATCH}
    named.add("mflux-generate")  # table-less fallback binary
    missing = sorted(b for b in named if shutil.which(b) is None)
    assert not missing, (
        f"mflux dispatch names binaries not on PATH: {missing}. "
        "Either mflux was upgraded/downgraded, or the dispatch table is stale.")


# ── edit dispatch ────────────────────────────────────────────────────

EDIT_CASES = [
    # (input,                                binary,                      base-model,       image flag,      strength)
    ("black-forest-labs/FLUX.1-Kontext-dev", "mflux-generate-kontext",    "",               "--image-path",  True),
    ("black-forest-labs/FLUX.2-klein-9B",    "mflux-generate-flux2-edit", "flux2-klein-9b", "--image-paths", False),
    ("black-forest-labs/FLUX.2-klein-4B",    "mflux-generate-flux2-edit", "flux2-klein-4b", "--image-paths", False),
    ("Qwen/Qwen-Image-Edit-2509",            "mflux-generate-qwen-edit",  "qwen",           "--image-paths", False),
    ("briaai/Fibo-Edit",                     "mflux-generate-fibo-edit",  "fibo-edit",      "--image-paths", False),
]


@pytest.mark.parametrize("model_id,binary,base,image_flag,strength", EDIT_CASES)
def test_mflux_edit_command(model_id, binary, base, image_flag, strength):
    """Editing has its own binaries AND its own argument conventions.

    kontext takes a single --image-path and honours --image-strength; the
    flux2/qwen/fibo edit binaries take a variadic --image-paths and have no
    strength knob. Passing the wrong flag is an immediate argparse failure,
    and passing the wrong binary silently edits with the wrong weights.
    """
    cmd = mflux_edit_command(model_id)
    assert cmd.binary == binary
    assert cmd.image_flag == image_flag
    assert cmd.supports_strength is strength
    if base:
        assert "--base-model" in cmd.extra_args
        assert cmd.extra_args[cmd.extra_args.index("--base-model") + 1] == base
    else:
        assert "--base-model" not in cmd.extra_args
    assert "--model" in cmd.extra_args, "resolved model must reach the binary"


def test_preset_image_edit_picks_route_to_flux2_edit_4b():
    """The 2026-08-15 bake-off retired Kontext as the preset, and the
    corrected rerun (the first round's klein-9B runs were silently the
    4B — see the --base-model-without---model trap in the playbook)
    settled on klein-4B: quality indistinguishable from true 9B across
    recolor/remove/add-sign at half the time (56s vs 106s) and RAM
    (13.6GB vs 24GB), sharing weights with the 64gb gen pick."""
    from lib.models import DEFAULT_PROFILES
    picks = {name: prof["tasks"]["image_edit"]
             for name, prof in DEFAULT_PROFILES["profiles"].items()
             if "image_edit" in prof.get("tasks", {})}
    assert picks, "expected at least one tier with an image_edit preset"
    for tier, pick in picks.items():
        cmd = mflux_edit_command(pick)
        assert cmd.binary == "mflux-generate-flux2-edit", \
            f"{tier}: {pick!r} routes to {cmd.binary}"
        assert "flux2-klein-4b" in cmd.extra_args, \
            f"{tier}: {pick!r} resolves base {cmd.extra_args}"


def test_edit_dispatch_does_not_reuse_the_generate_table():
    """Regression guard for the bug this replaced.

    mflux_command() maps Kontext to a *generate* binary that has no image
    input at all, so image_edit cannot route through it.
    """
    gen_binary, _ = mflux_command("black-forest-labs/FLUX.1-Kontext-dev")
    edit = mflux_edit_command("black-forest-labs/FLUX.1-Kontext-dev")
    assert gen_binary != edit.binary
    assert edit.binary == "mflux-generate-kontext"


def test_unknown_edit_model_keeps_the_caller_model():
    """An unrecognized id must not be silently swapped for Kontext's own
    weights — it falls back to the kontext binary but carries --model, so a
    genuinely incompatible choice fails loudly in mflux instead."""
    cmd = mflux_edit_command("some-org/mystery-editor")
    assert cmd.extra_args == ["--model", "some-org/mystery-editor"]


def test_mflux_edit_binaries_installed():
    if shutil.which("mflux-generate") is None:
        pytest.skip("mflux not installed on this machine")
    named = {b for _, b, _, _, _ in _MFLUX_EDIT_DISPATCH}
    missing = sorted(b for b in named if shutil.which(b) is None)
    assert not missing, f"mflux edit dispatch names binaries not on PATH: {missing}"
