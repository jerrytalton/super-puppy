"""Shared machinery for profile-level smoke tests.

These tests hit live local services (Ollama, MLX-OpenAI-server, mflux
subprocesses) via the profile-server Flask app. They catch dispatch bugs,
missing binaries, and wire-format regressions that mocked unit tests
intentionally don't.

Skip behavior:
  - Services unreachable → entire suite skips (fast, ~1s).
  - Model not pulled locally → individual test skips with a clear reason.
  - Dispatch/invocation bug → test fails loudly.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
import time
import wave
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

REPO = Path(__file__).resolve().parent.parent


# ── profile-server import dance ─────────────────────────────────────

def _import_profile_server():
    """Import app/profile-server.py as a module. Unlike the fast unit suite,
    smoke tests need the real mlx_audio — we're exercising actual TTS."""
    try:
        import mlx_audio  # noqa: F401
        import mlx_audio.tts  # noqa: F401
    except ImportError as e:
        pytest.skip(
            f"mlx_audio unavailable — smoke suite needs real deps: {e}",
            allow_module_level=True,
        )

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "app"))

    os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
    os.environ.setdefault("MLX_URL", "http://localhost:8000")
    os.environ["PROFILE_IDLE_TIMEOUT"] = "0"
    # In-process Flask testing has no bearer token. Set the documented
    # escape hatch here so these suites don't depend on another test
    # module having been collected first (e.g. `pytest -m correctness`).
    os.environ.setdefault("SP_ALLOW_NO_AUTH", "1")

    # Use the REAL hf_scanner but stub only the slow full-cache scan.
    # The snapshot helpers (resolve_hf_snapshot, check_wan_snapshot_ready)
    # must stay real, or the video handler sees a MagicMock and reports the
    # model as "still downloading" — silently skipping the video test.
    if "lib.hf_scanner" not in sys.modules:
        try:
            import lib.hf_scanner as _hfs
            _hfs.scan_hf_cache = MagicMock(return_value=[])
        except Exception:
            stub = MagicMock()
            stub.scan_hf_cache = MagicMock(return_value=[])
            sys.modules["lib.hf_scanner"] = stub

    ps_path = REPO / "app" / "profile-server.py"
    spec = importlib.util.spec_from_file_location(
        "profile_server_smoke", str(ps_path))
    ps = importlib.util.module_from_spec(spec)
    sys.modules["profile_server_smoke"] = ps
    spec.loader.exec_module(ps)
    return ps


# Lazy singleton so each test module shares one import.
_ps = None


def ps():
    global _ps
    if _ps is None:
        _ps = _import_profile_server()
    return _ps


# ── service reachability ────────────────────────────────────────────

def _reachable(url: str) -> bool:
    try:
        return requests.get(url, timeout=1).ok
    except Exception:
        return False


def require_local_services():
    """Skip the current test module if Ollama or MLX isn't reachable."""
    if not _reachable("http://localhost:11434/api/tags"):
        pytest.skip("Ollama not reachable at localhost:11434", allow_module_level=True)
    if not _reachable("http://localhost:8000/v1/models"):
        pytest.skip("MLX-OpenAI-server not reachable at localhost:8000", allow_module_level=True)


# ── minimal test fixtures ───────────────────────────────────────────

def write_png(path: Path, size: int = 64, rgb: tuple[int, int, int] = (128, 128, 128)) -> Path:
    """Write a valid solid-color 8-bit RGB PNG using only stdlib (zlib/struct).

    Hand-rolled because the stdlib has no PNG writer and we don't want to
    pull in Pillow just for test fixtures. CRC32s are computed correctly so
    decoders accept the file.
    """
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    # IHDR: width, height, bit-depth=8, color-type=2 (RGB), compression=0,
    # filter=0, interlace=0
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * size      # filter byte + RGB per pixel
    raw = row * size
    idat = zlib.compress(raw, 6)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", idat)
           + chunk(b"IEND", b""))
    path.write_bytes(png)
    return path


def write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    """Write a mono 16-bit silent WAV."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return path


def write_text(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def write_speech_wav(path: Path, text: str) -> Path:
    """Generate a mono 16kHz WAV of `text` spoken aloud via macOS `say`.

    Unlike `write_wav` (silence), this produces real speech so a
    transcription correctness test has ground truth to assert against.
    Skips the test if `say`/`afconvert` aren't available (non-macOS/CI).
    """
    import shutil
    import subprocess
    # macOS built-ins live in /usr/bin, which the uv test env's PATH may
    # omit — resolve absolutely before falling back to PATH lookup.
    say = "/usr/bin/say" if os.path.exists("/usr/bin/say") else shutil.which("say")
    afconvert = ("/usr/bin/afconvert" if os.path.exists("/usr/bin/afconvert")
                 else shutil.which("afconvert"))
    if not say or not afconvert:
        pytest.skip("macOS `say`/`afconvert` unavailable — no speech fixture")
    aiff = path.with_suffix(".aiff")
    subprocess.run([say, "-o", str(aiff), text], check=True, timeout=30)
    subprocess.run(
        [afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
         str(aiff), str(path)],
        check=True, timeout=30)
    return path


# ── invocation ──────────────────────────────────────────────────────

# Error substrings that mean "the model isn't pulled" or "the binary isn't
# installed" — environmental, not a Super Puppy regression.
SKIP_SUBSTRINGS = (
    "not downloaded",
    "pull it first",
    "no such model",
    "no model available",
    "cannot connect to backend",
    "connection refused",
    "connection error",
    "connectionerror",
    "is not installed",
    # Media models that aren't pulled yet — skip, don't fail.
    "still downloading",
    "pull it from",
    "not downloaded",
    # Backend Python module not importable in this env — e.g. the test installs
    # PyPI `mlx-audio` while the servers pin a git commit with a different module
    # layout. The backend is unavailable here, so skip rather than hard-fail.
    "no module named",
    "is not a package",
)


def _is_skippable(err: str) -> bool:
    low = err.lower()
    return any(s in low for s in SKIP_SUBSTRINGS)


def call_api_test(client, tool: str, model: str, **body) -> tuple[int, dict]:
    payload = {"tool": tool, "model": model, **body}
    resp = client.post("/api/test", json=payload)
    try:
        data = resp.get_json() or {}
    except Exception:
        data = {"error": f"non-JSON response: {resp.data!r}"}
    return resp.status_code, data


def assert_tool_produces_output(
    client, *, tool: str, model: str, expect_key: str = "result", **body,
):
    """Invoke /api/test and assert the handler produced nonempty output.

    Skips (not fails) when the error indicates the model isn't available.
    """
    status, data = call_api_test(client, tool, model, **body)
    err = str(data.get("error", "")) if isinstance(data, dict) else ""
    if err and _is_skippable(err):
        pytest.skip(f"{tool}({model}): {err}")
    assert status == 200, f"{tool}({model}) HTTP {status}: {data}"
    assert not err, f"{tool}({model}) error: {err}"
    value = data.get(expect_key)
    assert value, f"{tool}({model}) returned empty {expect_key!r}: {data}"
    return data


def assert_media_output(client, *, tool: str, model: str, expect_key: str,
                        min_bytes: int = 100, **body):
    """Invoke /api/test for a media tool and assert it produced a real,
    non-empty output file on disk.

    Skips when the model/tool isn't available (not pulled, not installed).
    Fails when the tool ran but produced nothing/an error — that's the
    signal a media tool is actually broken. Returns the output path.
    """
    status, data = call_api_test(client, tool, model, **body)
    err = str(data.get("error", "")) if isinstance(data, dict) else ""
    if err and _is_skippable(err):
        pytest.skip(f"{tool}({model}): {err}")
    assert status == 200, f"{tool}({model}) HTTP {status}: {data}"
    assert not err, f"{tool}({model}) error: {err}"
    path = data.get(expect_key)
    assert path and Path(path).exists(), \
        f"{tool}({model}) produced no file at {expect_key!r}: {data}"
    size = Path(path).stat().st_size
    assert size >= min_bytes, \
        f"{tool}({model}) output is only {size}B (min {min_bytes}) — likely broken"
    return path


def assert_tool_output_contains(
    client, *, tool: str, model: str, expect_any: list[str],
    expect_key: str = "result", **body,
):
    """Invoke /api/test and assert the output CONTAINS expected ground truth.

    This is the correctness check `assert_tool_produces_output` can't make:
    a model that hallucinates returns nonempty output and passes the
    "produces output" smoke test while being completely wrong. Here we
    feed a known input and assert the answer reflects it — the only way
    to catch a backend that silently ignores its input (e.g. Ollama's
    `-mlx` tags that advertise vision but never see the image).

    Skips when the model isn't available; FAILS when output is present
    but wrong. `expect_any` passes if any substring matches (synonyms).
    """
    status, data = call_api_test(client, tool, model, **body)
    err = str(data.get("error", "")) if isinstance(data, dict) else ""
    if err and _is_skippable(err):
        pytest.skip(f"{tool}({model}): {err}")
    assert status == 200, f"{tool}({model}) HTTP {status}: {data}"
    assert not err, f"{tool}({model}) error: {err}"
    value = str(data.get(expect_key, ""))
    low = value.lower()
    if not any(s.lower() in low for s in expect_any):
        raise AssertionError(
            f"{tool}({model}) output did not contain any of {expect_any!r} "
            f"— the model likely ignored its input. Got: {value[:200]!r}")
    return data


# ── shared test-body builders ───────────────────────────────────────

def chat_body(prompt: str) -> dict:
    return {"prompt": prompt}


def translate_body(target: str = "French", text: str = "Hello, world.") -> dict:
    return {"target": target, "text": text}


def summarize_body(tmp: Path) -> dict:
    fp = tmp / "summarize_input.txt"
    write_text(fp, "The cat sat on the mat. The mat was red. The cat was happy.\n")
    return {"file_path": str(fp)}


def image_gen_body() -> dict:
    return {"prompt": "a small red circle, minimalist"}


def vision_body(tmp: Path) -> dict:
    return {"image_path": str(write_png(tmp / "vision.png")),
            "prompt": "Describe this image in one word."}


def computer_use_body(tmp: Path) -> dict:
    return {"image_path": str(write_png(tmp / "screenshot.png")),
            "intent": "Click any button."}


def transcribe_body(tmp: Path) -> dict:
    return {"audio_path": str(write_wav(tmp / "audio.wav"))}


def speak_body() -> dict:
    return {"text": "Hello."}


def embed_body() -> dict:
    return {"text": "The quick brown fox."}


# ── /tmp scratch dir fixture ────────────────────────────────────────

@pytest.fixture
def smoke_tmp(tmp_path_factory):
    """Provide a scratch directory under /tmp — required because
    profile-server's _is_safe_test_path gates inputs to /tmp/ only."""
    base = Path("/tmp") / f"super_puppy_smoke_{int(time.time()*1000)}"
    base.mkdir(exist_ok=True)
    yield base
    # Leave artifacts for post-mortem; /tmp is ephemeral anyway.


@pytest.fixture
def client():
    app = ps().app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── profile-based parametrization ──────────────────────────────────

# Standard tasks that boil down to a single chat roundtrip. These use the
# same handler shape; the only variation is the body field.
CHAT_CASES = (
    # (tool-key,  profile-task,  body-builder)
    ("code",       "code",         lambda tmp: chat_body("Say OK.")),
    ("general",    "general",      lambda tmp: chat_body("Reply with: OK.")),
    ("translate",  "translation",  lambda tmp: translate_body()),
    ("summarize",  "long_context", summarize_body),
    ("review",     "reasoning",    lambda tmp: {"code": "def f(): return 1"}),
)

# Tasks that exercise specialized backends (subprocess / file output).
FIXTURE_CASES = (
    # (tool-key,   profile-task,   body-builder,     expect-key)
    ("image_gen",    "image_gen",     lambda tmp: image_gen_body(),        "image_path"),
    ("vision",       "vision",        vision_body,                         "result"),
    ("computer_use", "computer_use",  computer_use_body,                   "result"),
    ("transcribe",   "transcription", transcribe_body,                     "result"),
    ("speak",        "tts",           lambda tmp: speak_body(),            "audio_path"),
    ("embed",        "embedding",     lambda tmp: embed_body(),            "embeddings"),
)


def run_chat_case(client, profile: dict, tool: str, profile_task: str, build_body, tmp: Path):
    model = profile.get(profile_task)
    if not model:
        pytest.skip(f"profile has no {profile_task!r} entry")
    assert_tool_produces_output(client, tool=tool, model=model, **build_body(tmp))


def run_fixture_case(
    client, profile: dict, tool: str, profile_task: str, build_body, expect_key: str, tmp: Path,
):
    model = profile.get(profile_task)
    if not model:
        pytest.skip(f"profile has no {profile_task!r} entry")
    assert_tool_produces_output(
        client, tool=tool, model=model, expect_key=expect_key, **build_body(tmp))
