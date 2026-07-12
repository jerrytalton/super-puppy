"""Over-the-wire test that X-SP-Client attribution survives FastMCP's
streamable-HTTP task boundary.

The real transport deps (mcp, httpx, starlette, anyio) must NOT be the mocks
that tests/test_mcp_server.py installs into sys.modules at import time. So the
actual driver lives in tests/_attribution_probe.py and runs here in a clean
subprocess via `uv run` with the real deps. See that file for the mechanism and
a note on how faithfully it reproduces production.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

_PROBE = Path(__file__).resolve().parent / "_attribution_probe.py"


@pytest.mark.skipif(shutil.which("uv") is None, reason="requires uv")
def test_attribution_survives_streamable_http_task_boundary():
    result = subprocess.run(
        ["uv", "run", str(_PROBE)],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"probe failed (rc={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "PASS:" in result.stdout, result.stdout
