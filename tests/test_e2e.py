"""End-to-end tests against live running Super Puppy services.

Hits the real MCP server (port 8100), profile server (dynamic port),
and verifies config file integrity and process health. Not mocked.

Usage:
    pytest tests/test_e2e.py -v
    pytest tests/test_e2e.py -v -m e2e
"""

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MCP_PORT = 8100
MCP_BASE = f"http://127.0.0.1:{MCP_PORT}"
AUTH_TOKEN_FILE = Path("~/.config/local-models/mcp_auth_token").expanduser()
CLAUDE_CONFIG = Path("~/.claude.json").expanduser()
CONFIG_DIR = Path("~/.config/local-models").expanduser()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _read_auth_token() -> str:
    if AUTH_TOKEN_FILE.exists():
        return AUTH_TOKEN_FILE.read_text().strip()
    return ""


def _discover_profile_port() -> int | None:
    """Find the profile server port by scanning listening TCP sockets.

    The profile-server.py runs as a Python child of a ``uv`` wrapper, so
    pgrep may return the uv PID whose lsof output is misleading. Instead,
    list *all* listening TCP sockets and match on the python process whose
    command line contains ``profile-server``.
    """
    try:
        lsof = subprocess.check_output(
            ["lsof", "-iTCP", "-sTCP:LISTEN", "-nP", "-Fp", "-Fn"],
            text=True, stderr=subprocess.DEVNULL,
        )
        # lsof -F output: lines starting with 'p' = PID, 'n' = name (addr:port)
        current_pid = None
        for line in lsof.splitlines():
            if line.startswith("p"):
                current_pid = line[1:]
            elif line.startswith("n") and current_pid:
                port_match = re.search(r":(\d+)$", line)
                if not port_match:
                    continue
                port = int(port_match.group(1))
                # Skip well-known ports that are definitely not the profile server
                if port in (8100, 11434, 8000, 80, 443):
                    continue
                try:
                    cmdline = subprocess.check_output(
                        ["ps", "-p", current_pid, "-o", "command="],
                        text=True, stderr=subprocess.DEVNULL,
                    ).strip()
                    if "profile-server" in cmdline:
                        return port
                except subprocess.CalledProcessError:
                    continue
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


def _service_available(host: str, port: int) -> bool:
    """Check if a TCP port is accepting connections."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def auth_token():
    token = _read_auth_token()
    if not token:
        pytest.skip("No auth token found at ~/.config/local-models/mcp_auth_token")
    return token


@pytest.fixture(scope="module")
def mcp_base():
    if not _service_available("127.0.0.1", MCP_PORT):
        pytest.skip(f"MCP server not running on port {MCP_PORT}")
    return MCP_BASE


@pytest.fixture(scope="module")
def profile_port():
    port = _discover_profile_port()
    if port is None:
        pytest.skip("Profile server not running (could not discover port)")
    return port


@pytest.fixture(scope="module")
def profile_base(profile_port):
    return f"http://127.0.0.1:{profile_port}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def http_get(url: str, headers: dict | None = None, timeout: int = 10) -> tuple[int, bytes, dict]:
    """Return (status_code, body_bytes, response_headers)."""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)
    except urllib.error.URLError as e:
        pytest.fail(f"Connection failed to {url}: {e}")


def http_post(url: str, body: dict, headers: dict | None = None, timeout: int = 10) -> tuple[int, bytes, dict]:
    """POST JSON, return (status_code, body_bytes, response_headers)."""
    data = json.dumps(body).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)
    except urllib.error.URLError as e:
        pytest.fail(f"Connection failed to {url}: {e}")


# ===========================================================================
# 1. MCP Server (port 8100)
# ===========================================================================

@pytest.mark.e2e
class TestMCPAuth:
    """Authentication and authorization on the MCP server."""

    def test_unauthenticated_request_returns_403(self, mcp_base):
        status, body, _ = http_get(f"{mcp_base}/mcp")
        assert status == 403, f"Expected 403 for unauthenticated /mcp, got {status}"

    def test_valid_bearer_token_not_rejected(self, mcp_base, auth_token):
        status, body, _ = http_get(
            f"{mcp_base}/mcp",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        # GET /mcp with valid auth may return 405 (method not allowed) but NOT 403
        assert status != 403, f"Got 403 with valid auth token on /mcp"

    def test_fake_session_id_returns_403(self, mcp_base):
        status, body, _ = http_get(
            f"{mcp_base}/messages?session_id=fake-session-does-not-exist"
        )
        assert status == 403, (
            f"Expected 403 for /messages with fake session_id, got {status}"
        )

    def test_wrong_bearer_token_returns_403(self, mcp_base):
        status, body, _ = http_get(
            f"{mcp_base}/mcp",
            headers={"Authorization": "Bearer wrong-token-value"},
        )
        assert status == 403, f"Expected 403 for wrong bearer token, got {status}"

    def test_gpu_endpoint_exempt_from_auth(self, mcp_base):
        status, body, _ = http_get(f"{mcp_base}/gpu")
        assert status == 200, f"Expected 200 for auth-exempt /gpu, got {status}"

    def test_mcp_models_endpoint_exempt_from_auth(self, mcp_base):
        status, body, _ = http_get(f"{mcp_base}/api/mcp-models")
        assert status == 200, f"Expected 200 for auth-exempt /api/mcp-models, got {status}"


@pytest.mark.e2e
class TestMCPModels:
    """Model discovery endpoints on the MCP server."""

    def test_mcp_models_returns_json_with_models_array(self, mcp_base):
        status, body, _ = http_get(f"{mcp_base}/api/mcp-models")
        assert status == 200
        data = json.loads(body)
        assert "models" in data, "Response missing 'models' key"
        assert isinstance(data["models"], list), "'models' should be a list"

    def test_gpu_returns_json_with_backend_keys(self, mcp_base):
        status, body, _ = http_get(f"{mcp_base}/gpu")
        assert status == 200
        data = json.loads(body)
        assert "ollama" in data, "GPU status missing 'ollama' key"
        assert "mlx" in data, "GPU status missing 'mlx' key"


@pytest.mark.e2e
class TestMCPInit:
    """MCP protocol initialization."""

    def test_mcp_initialize_returns_session(self, mcp_base, auth_token):
        """POST a JSON-RPC initialize to /mcp and expect a valid SSE response."""
        init_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test-e2e", "version": "0.1.0"},
            },
        }
        # MCP streamable-HTTP requires Accept with both JSON and SSE
        status, body, resp_headers = http_post(
            f"{mcp_base}/mcp",
            body=init_body,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert status == 200, f"MCP initialize failed with status {status}: {body[:200]}"
        # Response is SSE -- extract JSON from "data:" lines
        body_text = body.decode("utf-8", errors="replace")
        data_payload = None
        for line in body_text.splitlines():
            if line.startswith("data: "):
                data_payload = json.loads(line[6:])
                break
        assert data_payload is not None, (
            f"No 'data:' line found in SSE response: {body_text[:300]}"
        )
        assert "result" in data_payload, (
            f"MCP initialize response missing 'result': {data_payload}"
        )
        assert "serverInfo" in data_payload["result"], (
            "Missing serverInfo in init response"
        )
        # Session ID should be in response header
        session_id = resp_headers.get(
            "Mcp-Session-Id", resp_headers.get("mcp-session-id", "")
        )
        assert session_id, "No mcp-session-id header in initialize response"


# ===========================================================================
# 2. Profile Server (dynamic port)
# ===========================================================================

@pytest.mark.e2e
class TestProfileServerAPI:
    """Profile server REST endpoints."""

    def test_api_system_returns_mode_and_ram(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/api/system")
        assert status == 200, f"/api/system returned {status}"
        data = json.loads(body)
        assert "mode" in data, "System info missing 'mode'"
        assert "total_ram_gb" in data, "System info missing 'total_ram_gb'"
        assert data["total_ram_gb"] > 0, "RAM should be positive"

    def test_api_models_returns_json_array(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/api/models")
        assert status == 200, f"/api/models returned {status}"
        data = json.loads(body)
        assert isinstance(data, list), "/api/models should return a JSON array"
        assert len(data) > 0, "No models discovered -- is Ollama/MLX running?"

    def test_api_models_entries_have_required_fields(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/api/models")
        data = json.loads(body)
        if not data:
            pytest.skip("No models available")
        entry = data[0]
        for field in ("name", "backend", "total_params_b", "active_params_b"):
            assert field in entry, f"Model entry missing '{field}'"

    def test_api_tasks_returns_standard_tasks(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/api/tasks")
        assert status == 200, f"/api/tasks returned {status}"
        data = json.loads(body)
        assert isinstance(data, dict), "/api/tasks should return a JSON object"
        for key in ("code", "general", "reasoning", "long_context", "translation"):
            assert key in data, f"Missing standard task '{key}'"
            assert "label" in data[key], f"Task '{key}' missing 'label'"

    def test_api_tasks_includes_special_tasks(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/api/tasks")
        data = json.loads(body)
        for key in ("vision", "image_gen", "transcription", "tts", "embedding"):
            assert key in data, f"Missing special task '{key}'"

    def test_api_profiles_returns_profiles_and_active(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/api/profiles")
        assert status == 200, f"/api/profiles returned {status}"
        data = json.loads(body)
        assert "profiles" in data, "Profiles response missing 'profiles' dict"
        assert "active" in data, "Profiles response missing 'active' key"
        assert isinstance(data["profiles"], dict), "'profiles' should be a dict"

    def test_api_gpu_returns_backends(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/api/gpu")
        assert status == 200, f"/api/gpu returned {status}"
        data = json.loads(body)
        assert "ollama" in data or "mlx" in data, (
            "GPU endpoint should have backend keys"
        )


@pytest.mark.e2e
class TestProfileServerPages:
    """Static pages and PWA assets from the profile server."""

    def test_tools_returns_html(self, profile_base):
        status, body, headers = http_get(f"{profile_base}/tools")
        assert status == 200, f"/tools returned {status}"
        assert b"<html" in body.lower() or b"<!doctype" in body.lower(), (
            "/tools should return HTML"
        )

    def test_manifest_json_is_valid(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/manifest.json")
        assert status == 200, f"/manifest.json returned {status}"
        data = json.loads(body)
        assert "name" in data or "short_name" in data, (
            "manifest.json should have a name or short_name"
        )

    def test_sw_js_is_javascript(self, profile_base):
        status, body, headers = http_get(f"{profile_base}/sw.js")
        assert status == 200, f"/sw.js returned {status}"
        content_type = headers.get("Content-Type", "")
        assert "javascript" in content_type, (
            f"sw.js Content-Type should be JavaScript, got {content_type}"
        )

    def test_index_returns_html(self, profile_base):
        status, body, _ = http_get(f"{profile_base}/")
        assert status == 200, f"/ returned {status}"
        assert b"<html" in body.lower() or b"<!doctype" in body.lower(), (
            "Index should return HTML"
        )


# ===========================================================================
# 3. Claude Config Integration
# ===========================================================================

@pytest.mark.e2e
class TestClaudeConfig:
    """Verify Claude Code's MCP config points to local-models."""

    def test_claude_json_exists(self):
        assert CLAUDE_CONFIG.exists(), f"{CLAUDE_CONFIG} not found"

    def test_claude_json_has_local_models_entry(self):
        if not CLAUDE_CONFIG.exists():
            pytest.skip("~/.claude.json missing")
        data = json.loads(CLAUDE_CONFIG.read_text())
        mcp_servers = data.get("mcpServers", {})
        assert "local-models" in mcp_servers, (
            "'local-models' not found in mcpServers"
        )

    def test_local_models_entry_type_is_http(self):
        if not CLAUDE_CONFIG.exists():
            pytest.skip("~/.claude.json missing")
        data = json.loads(CLAUDE_CONFIG.read_text())
        entry = data.get("mcpServers", {}).get("local-models", {})
        if not entry:
            pytest.skip("No local-models entry")
        assert entry.get("type") == "http", (
            f"local-models type should be 'http', got {entry.get('type')}"
        )

    def test_local_models_url_is_valid(self):
        if not CLAUDE_CONFIG.exists():
            pytest.skip("~/.claude.json missing")
        data = json.loads(CLAUDE_CONFIG.read_text())
        entry = data.get("mcpServers", {}).get("local-models", {})
        url = entry.get("url", "")
        assert url, "local-models entry missing 'url'"
        assert "8100" in url or ":8100" in url or "mcp" in url, (
            f"URL should reference port 8100 or /mcp path: {url}"
        )
        assert "127.0.0.1" in url or "localhost" in url or ".ts.net" in url, (
            f"URL should point to localhost or Tailscale FQDN: {url}"
        )


# ===========================================================================
# 4. Config File Integrity
# ===========================================================================

@pytest.mark.e2e
class TestConfigFiles:
    """Verify config files exist and are well-formed."""

    def test_profiles_json_exists_and_valid(self):
        path = CONFIG_DIR / "profiles.json"
        assert path.exists(), f"{path} not found"
        data = json.loads(path.read_text())
        assert "profiles" in data, "profiles.json missing 'profiles' key"

    def test_mcp_preferences_json_exists_and_valid(self):
        path = CONFIG_DIR / "mcp_preferences.json"
        assert path.exists(), f"{path} not found"
        data = json.loads(path.read_text())
        assert isinstance(data, dict), "mcp_preferences.json should be a JSON object"

    def test_network_conf_exists_and_has_required_keys(self):
        path = CONFIG_DIR / "network.conf"
        assert path.exists(), f"{path} not found"
        content = path.read_text()
        # Should have at least a hostname or port config
        has_key = any(
            key in content
            for key in ("MODEL_SERVER_HOST", "OLLAMA_PORT", "TAILSCALE_HOSTNAME")
        )
        assert has_key, "network.conf missing expected configuration keys"

    def test_mcp_auth_token_exists_and_nonempty(self):
        assert AUTH_TOKEN_FILE.exists(), f"{AUTH_TOKEN_FILE} not found"
        token = AUTH_TOKEN_FILE.read_text().strip()
        assert len(token) > 0, "Auth token file is empty"

    def test_mode_conf_has_valid_force_local(self):
        path = CONFIG_DIR / "mode.conf"
        if not path.exists():
            pytest.skip("mode.conf not present (optional)")
        content = path.read_text()
        match = re.search(r"FORCE_LOCAL\s*=\s*[\"']?(true|false)[\"']?", content, re.IGNORECASE)
        assert match, (
            f"mode.conf should have FORCE_LOCAL=true|false, got: {content.strip()}"
        )


# ===========================================================================
# 5. Process Health
# ===========================================================================

@pytest.mark.e2e
class TestProcessHealth:
    """Verify required processes are running."""

    def _process_running(self, pattern: str) -> bool:
        try:
            subprocess.check_output(["pgrep", "-f", pattern], stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False

    def test_super_puppy_app_running(self):
        assert self._process_running("SuperPuppy"), (
            "SuperPuppy.app process not found"
        )

    def test_mcp_server_running(self):
        assert self._process_running("local-models-server"), (
            "MCP server process (local-models-server) not found"
        )

    def test_profile_server_running(self):
        assert self._process_running("profile-server"), (
            "Profile server process not found"
        )


# ===========================================================================
# 6. lib/models.py Integration
# ===========================================================================

@pytest.mark.e2e
class TestLibModels:
    """Verify shared constants and helpers from lib/models.py."""

    def test_standard_tasks_nonempty(self):
        from lib.models import STANDARD_TASKS
        assert len(STANDARD_TASKS) > 0, "STANDARD_TASKS is empty"
        for key in ("code", "general", "reasoning", "long_context", "translation"):
            assert key in STANDARD_TASKS, f"Missing standard task '{key}'"

    def test_special_tasks_nonempty(self):
        from lib.models import SPECIAL_TASKS
        assert len(SPECIAL_TASKS) > 0, "SPECIAL_TASKS is empty"
        for key in ("vision", "image_gen", "transcription", "tts", "embedding", "computer_use"):
            assert key in SPECIAL_TASKS, f"Missing special task '{key}'"

    def test_task_filters_covers_standard_tasks(self):
        from lib.models import STANDARD_TASKS, TASK_FILTERS
        for key in STANDARD_TASKS:
            assert key in TASK_FILTERS, (
                f"TASK_FILTERS missing entry for standard task '{key}'"
            )

    def test_task_filters_have_required_structure(self):
        from lib.models import TASK_FILTERS
        for key, filt in TASK_FILTERS.items():
            assert isinstance(filt, dict), f"TASK_FILTERS['{key}'] should be a dict"
            assert "exclude_names" in filt, (
                f"TASK_FILTERS['{key}'] missing 'exclude_names'"
            )

    def test_always_exclude_nonempty(self):
        from lib.models import ALWAYS_EXCLUDE
        assert len(ALWAYS_EXCLUDE) > 0, "ALWAYS_EXCLUDE is empty"

    def test_known_active_params_nonempty(self):
        from lib.models import KNOWN_ACTIVE_PARAMS
        assert len(KNOWN_ACTIVE_PARAMS) > 0, "KNOWN_ACTIVE_PARAMS is empty"

    def test_config_paths_are_absolute(self):
        from lib.models import CONFIG_DIR, PROFILES_FILE, MCP_PREFS_FILE, NETWORK_CONF
        for path in (CONFIG_DIR, PROFILES_FILE, MCP_PREFS_FILE, NETWORK_CONF):
            assert path.is_absolute(), f"{path} should be absolute"

    def test_active_params_b_non_moe_passthrough(self):
        from lib.models import active_params_b
        result = active_params_b("test-model", 7.0, "llama", None, None)
        assert result == 7.0, "Non-MoE model should return total_b unchanged"

    def test_active_params_b_moe_axb_parsing(self):
        from lib.models import active_params_b
        result = active_params_b(
            "qwen3-coder_A3B", 30.0, "qwen3", 128, 8
        )
        assert result == 3.0, "AXB pattern should parse active params from name"

    def test_active_params_b_known_hybrid(self):
        from lib.models import active_params_b
        result = active_params_b(
            "deepseek-r1:671b", 671.0, "deepseek2", 256, 8
        )
        assert result == 37.0, "Known hybrid should return table value"

    def test_active_params_b_ratio_fallback(self):
        from lib.models import active_params_b
        result = active_params_b(
            "some-moe-model", 100.0, "unknown_family", 8, 2
        )
        assert result == 25.0, "Ratio fallback: 100 * 2/8 = 25"
