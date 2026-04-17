"""Unit tests for `lib.models.mflux_command` + `mflux_is_turbo`.

Pure dispatch table — no subprocess, no network. Catches dispatch bugs
(e.g. "FLUX.2-klein routed to the FLUX.1 generator") at unit-test speed.
Pair this with `test_tools_smoke_laptop.py` for the real invocation check.
"""

from __future__ import annotations

import shutil

import pytest

from lib.models import _MFLUX_DISPATCH, mflux_command, mflux_is_turbo


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
