# ds4 Backend Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship ds4 (antirez/ds4, glm5.2 branch, pinned commit `bd89932`) as the serving backend for glm-5.2 on the 512GB tier, replacing mlx-openai-server for that one model (MLX keeps whisper/vision/TTS/ui-venus).

**Architecture:** glm-5.2 today is a `served_model_name` inside the shared mlx-openai-server; every dispatcher is a binary branch (`backend == "ollama"` → :11434, else → MLX :8000). This plan threads a third backend value `"ds4"` (OpenAI-compatible, `ds4-server` on internal-only port 8002) through discovery, dispatch, status, service lifecycle, UI, tests, and docs. ds4's `/v1/models` returns no metadata, so discovery hardcodes glm-5.2's params/context/size from shared constants; ds4 is always-resident (no on-demand/warm/keep-alive concepts apply).

**Tech Stack:** Python 3.12 (uv, PEP 723), Flask (profile server), FastMCP/httpx (MCP server), rumps (menubar), bash (install/service scripts), pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-ds4-port-surface-audit.md`

## Global Constraints

- **Branch:** all work on `feat/ds4-backend` off `main`. Conventional Commits, one logical change per commit. The repo pre-commit hook runs the unit suite — every task must leave it green.
- **Unit suite command** (run after every implementation step): `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"` (fast subset; smoke/e2e need live services and skip cleanly anyway). Full documented run: see CLAUDE.md Testing section.
- **Pinned ds4 source:** `https://github.com/antirez/ds4`, branch `glm5.2`, commit `bd89932`. Model file: `antirez/GLM-5.2-GGUF` / `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf`, exactly **262036650048 bytes** on disk (~244 GiB).
- **DS4_DIR default:** `$HOME/.local/share/super-puppy/ds4` (checkout + build + gguf), overridable via `DS4_DIR` in `~/.config/local-models/network.conf`. `install.sh` must detect and reuse an existing `~/experiments/ds4` checkout+gguf by symlinking (never re-download 244GiB on the dev machine).
- **Port 8002** (`DS4_PORT` in network.conf + `lib/models.py` `_NETWORK_DEFAULTS`/`_NUMERIC_KEYS`), **internal-only**: NOT added to the tailscale serve tuple (`app/menubar.py` `_start_tailscale_serve`). Client-mode traffic reaches glm-5.2 via the desktop's MCP (8100) and profile server (8101).
- **ds4-server MUST launch with `cwd=$DS4_DIR`** — it resolves `metal/flash_attn.metal` relative to cwd. Launch command: `./ds4-server --metal --port $DS4_PORT --ctx 32768 -m ds4flash.gguf`, where `ds4flash.gguf` is a symlink to `gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf`. Readiness: `GET /v1/models` returns 200; allow up to **300s** (cold load ~70s — be generous; NEVER the MLX 60s watchdog).
- **ds4 JSON responses MUST be parsed with `json.loads(..., strict=False)`** — ds4 emits unescaped control characters in long `reasoning_content`; strict parsers (`resp.json()`) intermittently crash dispatch.
- **NEVER forward `chat_template_kwargs.enable_thinking` to ds4** — verified broken 2026-07-22: HTTP 200 but reasoning migrates into `content`. glm-5.2 on ds4 always thinks. Read reply text as `content` with `reasoning_content` fallback.
- **Full residency** — no `--ssd-streaming` flags anywhere (measured: 15.5 → 3.5–4.9 tok/s; rejected).
- **Hardcoded discovery metadata** for glm-5.2 on ds4: `total_params_b=380`, `active_params_b=32`, `context=131072`, `vision=False`, `vram_bytes=disk_bytes=262036650048`, `is_loaded=True`, `on_demand=False`. If ds4 is unreachable, glm-5.2 is absent from discovery (same semantics as MLX-down today).
- **Docs update in the same commit as the code they describe.**
- No new secrets/tokens. Pin exact dependency versions. Shared constants go in `lib/models.py`, never duplicated.

---

## File map (what changes where)

| File | Change |
|---|---|
| `lib/models.py` | `LLM_BACKENDS`, `DS4_*` constants, `ds4_dir()`/`ds4_installed()`, `DS4_PORT`/`DS4_DIR` in network defaults, `PROFILES_VERSION` 31→32 (Task 9) |
| `config/local-models/network.conf` | `DS4_PORT=8002`, `DS4_DIR=""` |
| `mcp/local-models-server.py` | `DS4_URL`, `chat_ds4()`, dispatch elif, `LLM_BACKENDS` fallback, ds4 discovery block, status/gpu/activity endpoints |
| `app/profile-server.py` | `DS4_URL`, `_fetch_ds4_models()`, aggregate order, `LLM_BACKENDS` import, `_chat_url`/`_chat`/`_chat_stream` ds4 branches, missing-model exclusion, `api_diagnostics` probe |
| `app/menubar.py` | `DS4_LOCAL`/`DS4_STUCK_LOADING_S`, `warm_ping_targets(mlx_served=)`, ds4 monitored service (menu item, refresh, watchdog, `_restart_ds4`, Copy Diagnostics), `DS4_URL` env injection |
| `bin/start-local-models` | ds4 launch/stop/status (presence-gated, cwd-pinned, 300s readiness) |
| `bin/local-models-mcp-detect` | `export DS4_URL=http://localhost:$DS4_PORT` (no tailscale rewrite) |
| `bin/migrate-mlx-config.py` | NEW — removes a served model's entry from a user's mlx yaml (stdlib-only, idempotent) |
| `bin/post-update.sh` | one-shot glm-5.2 MLX-entry migration (512GB-gated, failure-tolerant) |
| `bin/apply-mlx-glm52-patch.sh` | DELETED |
| `config/mlx-server/config.yaml` | glm-5.2 entry removed |
| `install.sh` | ds4 clone+make+download step replaces glm52-patch block; `MISSING_RUNTIMES`; served-name resolution skip; uninstall additions |
| `uninstall.sh` | kill ds4-server, DS4_DIR cleanup note |
| `app/activity.html` | `.backend-ds4` CSS (light+dark), active-dot color ternary |
| `app/diagnostics.html` | conditional ds4 service row |
| `tests/*` | 2 hard breakage fixes + new coverage per task; `_smoke_helpers` DS4 wiring; one `correctness` ds4 case |
| `CLAUDE.md`, `README.md`, `docs/architecture.md`, `docs/troubleshooting.md`, `docs/usage-telemetry.md` | folded into the code tasks below |

Verified no-ops (do not touch): `lib/activity.py` (backend column is free text), GPU-tracking internals (`defaultdict`), `resolve_pref_candidate`/`pick_model_from_prefs` (pure), `lib/audit.py`, `lib/mlx_vlm.py`, launchd plists, `app/audit.html`, `app/tools.html`, `super-puppy.c`, `tests/test_playground_coverage.py`, `tests/test_tools_smoke_laptop.py`, `tests/fleet/` (no port/backend assumptions), `docs/tailscale-setup.md` (8002 is not served), `docs/model-prompting.md`, `docs/RELEASING.md`.

---

### Task 1: Shared constants — `LLM_BACKENDS`, DS4 metadata, `DS4_PORT`/`DS4_DIR`

**Files:**
- Modify: `lib/models.py` (after `_NUMERIC_KEYS` at line 41; new section after `validate_network_conf`)
- Modify: `config/local-models/network.conf` (after `MLX_PORT=8000` at line 12)
- Test: `tests/test_core.py` (class `TestValidateNetworkConf` ends at line 355; add new class after it)
- Docs: `CLAUDE.md` (Shared Library section, line 118-124)

**Interfaces:**
- Consumes: nothing (first task).
- Produces (imported by every later task):
  - `LLM_BACKENDS: frozenset[str]` = `{"ollama", "mlx", "ds4"}`
  - `DS4_MODEL_NAME: str` = `"glm-5.2"`
  - `DS4_MODEL_BYTES: int` = `262_036_650_048`
  - `DS4_TOTAL_PARAMS_B: int` = `380`
  - `DS4_ACTIVE_PARAMS_B: int` = `32`
  - `DS4_CONTEXT: int` = `131072`
  - `DS4_DIR_DEFAULT: str` = `"~/.local/share/super-puppy/ds4"`
  - `ds4_dir() -> Path` — DS4_DIR from network.conf, else default, expanded
  - `ds4_installed() -> bool` — `(ds4_dir() / "ds4-server").exists()`
  - network.conf gains `DS4_PORT` (numeric, default `8002`) and `DS4_DIR` (path string, default empty)

- [ ] **Step 0: Create the branch**

```bash
cd /Users/jerry/super-puppy && git checkout -b feat/ds4-backend main
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core.py` immediately after class `TestValidateNetworkConf` (after line 355):

```python
class TestDs4Constants:
    def test_llm_backends_includes_all_three_chat_backends(self):
        from lib.models import LLM_BACKENDS
        assert LLM_BACKENDS == {"ollama", "mlx", "ds4"}

    def test_ds4_metadata_constants(self):
        # These hardcoded values feed BOTH discovery paths (MCP + profile
        # server); a typo here silently drops glm-5.2 from task lists
        # (TASK_FILTERS min_active_b/min_ctx gates) or corrupts memory math.
        from lib import models
        assert models.DS4_MODEL_NAME == "glm-5.2"
        assert models.DS4_MODEL_BYTES == 262_036_650_048
        assert models.DS4_TOTAL_PARAMS_B == 380
        assert models.DS4_ACTIVE_PARAMS_B == 32
        assert models.DS4_CONTEXT == 131072

    def test_ds4_dir_default_and_override(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text('DS4_PORT=8002\nDS4_DIR="/opt/ds4"\n')
        with patch.object(models, "NETWORK_CONF", conf):
            assert models.ds4_dir() == Path("/opt/ds4")
        conf.write_text("DS4_PORT=8002\n")
        with patch.object(models, "NETWORK_CONF", conf):
            assert models.ds4_dir() == Path(
                "~/.local/share/super-puppy/ds4").expanduser()

    def test_ds4_installed_requires_server_binary(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text(f'DS4_DIR="{tmp_path}/ds4"\n')
        with patch.object(models, "NETWORK_CONF", conf):
            assert models.ds4_installed() is False
            (tmp_path / "ds4").mkdir()
            (tmp_path / "ds4" / "ds4-server").write_bytes(b"#!/bin/sh\n")
            assert models.ds4_installed() is True

    def test_ds4_port_repaired_when_non_numeric(self, tmp_path):
        from lib import models
        conf = tmp_path / "network.conf"
        conf.write_text("DS4_PORT=8002abc\n")
        with patch.object(models, "NETWORK_CONF", conf), \
             patch.object(models, "CONFIG_DIR", tmp_path):
            warnings = models.validate_network_conf()
        assert any("non-numeric" in w for w in warnings)
        assert "DS4_PORT=8002" in conf.read_text()

    def test_network_conf_template_has_ds4_keys(self):
        # _NETWORK_DEFAULTS "must match config/local-models/network.conf"
        # (lib/models.py comment) — this catches template drift.
        template = (Path(__file__).resolve().parent.parent
                    / "config" / "local-models" / "network.conf")
        content = template.read_text()
        assert "DS4_PORT=8002" in content
        assert "DS4_DIR=" in content
```

`test_core.py` already imports `Path` and `patch` at the top — verify with `grep -n "^from\|^import" tests/test_core.py` and add `from pathlib import Path` / `from unittest.mock import patch` only if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py::TestDs4Constants -v`
Expected: FAIL — `ImportError: cannot import name 'LLM_BACKENDS'` (and AttributeErrors for the DS4 constants).

- [ ] **Step 3: Implement the constants in `lib/models.py`**

In `lib/models.py`, replace lines 29-41:

```python
_NETWORK_DEFAULTS = {
    "TAILSCALE_HOSTNAME": "super-puppy",
    "OLLAMA_PORT": "11434",
    "MLX_PORT": "8000",
    "SERVER_RAM_GB": "0",
    "PROBE_TIMEOUT": "2",
    "PROFILE_PORT": "8101",
    "OP_REF": "",
    "IS_SERVER": "false",
    "AUTO_UPDATE": "true",
}

_NUMERIC_KEYS = {"OLLAMA_PORT", "MLX_PORT", "SERVER_RAM_GB", "PROBE_TIMEOUT", "PROFILE_PORT"}
```

with:

```python
_NETWORK_DEFAULTS = {
    "TAILSCALE_HOSTNAME": "super-puppy",
    "OLLAMA_PORT": "11434",
    "MLX_PORT": "8000",
    "DS4_PORT": "8002",
    "DS4_DIR": "",
    "SERVER_RAM_GB": "0",
    "PROBE_TIMEOUT": "2",
    "PROFILE_PORT": "8101",
    "OP_REF": "",
    "IS_SERVER": "false",
    "AUTO_UPDATE": "true",
}

_NUMERIC_KEYS = {"OLLAMA_PORT", "MLX_PORT", "DS4_PORT", "SERVER_RAM_GB",
                 "PROBE_TIMEOUT", "PROFILE_PORT"}
```

Then insert a new section immediately after the `validate_network_conf` function (after line 106, before the `KNOWN_ACTIVE_PARAMS` section):

```python
# ── Chat backends & ds4 ──────────────────────────────────────────────
# Three chat-LLM backends. "ds4" is antirez/ds4 serving glm-5.2 on the
# 512GB tier (OpenAI-compatible, localhost:8002, internal-only — never
# tailscale-served; client-mode traffic is brokered by the desktop's MCP
# and profile server).

LLM_BACKENDS: frozenset[str] = frozenset({"ollama", "mlx", "ds4"})

# ds4's /v1/models returns one model with NO params/context/vision
# metadata, and its GGUF lives outside every existing sizing path (not an
# HF snapshot, not an Ollama blob). Discovery must hardcode these values;
# without them TASK_FILTERS min_active_b/min_ctx silently drop glm-5.2
# from every task list. DS4_MODEL_BYTES is the exact on-disk size of
# GLM-5.2-UD-Q2_K_RoutedQ2K.gguf.
DS4_MODEL_NAME = "glm-5.2"
DS4_MODEL_BYTES = 262_036_650_048
DS4_TOTAL_PARAMS_B = 380
DS4_ACTIVE_PARAMS_B = 32
DS4_CONTEXT = 131072

DS4_DIR_DEFAULT = "~/.local/share/super-puppy/ds4"


def ds4_dir() -> Path:
    """The ds4 checkout/build/gguf directory.

    network.conf's DS4_DIR overrides; default is DS4_DIR_DEFAULT. install.sh
    provisions this directory on 512GB machines (and symlinks an existing
    ~/experiments/ds4 checkout when present).
    """
    if NETWORK_CONF.exists():
        for line in NETWORK_CONF.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("DS4_DIR="):
                val = stripped.partition("=")[2].strip().strip('"').strip("'")
                if val:
                    return Path(val).expanduser()
    return Path(DS4_DIR_DEFAULT).expanduser()


def ds4_installed() -> bool:
    """True only where install.sh actually provisioned ds4 (512GB tier).

    Gates every ds4 surface that would otherwise show a permanently-red
    service on machines that never run it.
    """
    return (ds4_dir() / "ds4-server").exists()
```

- [ ] **Step 4: Add the keys to the network.conf template**

In `config/local-models/network.conf`, replace lines 10-12:

```
# Ports (must match the server's configs)
OLLAMA_PORT=11434
MLX_PORT=8000
```

with:

```
# Ports (must match the server's configs)
OLLAMA_PORT=11434
MLX_PORT=8000
# ds4 (glm-5.2 engine, 512GB tier). Internal-only — never tailscale-served.
DS4_PORT=8002

# ds4 checkout/build/gguf directory. Empty = ~/.local/share/super-puppy/ds4.
# install.sh sets this on 512GB machines.
DS4_DIR=""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -v`
Expected: all PASS (including the pre-existing `TestValidateNetworkConf` cases).

- [ ] **Step 6: Update CLAUDE.md Shared Library section**

In `CLAUDE.md`, the Shared Library bullet list (lines 118-124), insert after the `active_params_b()` bullet (line 122):

```markdown
- `LLM_BACKENDS` — the three chat backends (`ollama`, `mlx`, `ds4`); `DS4_MODEL_NAME`/`DS4_MODEL_BYTES`/`DS4_TOTAL_PARAMS_B`/`DS4_ACTIVE_PARAMS_B`/`DS4_CONTEXT` — hardcoded glm-5.2 metadata for ds4 discovery (ds4's `/v1/models` returns none); `ds4_dir()`/`ds4_installed()` — DS4_DIR resolution and presence gate
```

- [ ] **Step 7: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add lib/models.py config/local-models/network.conf tests/test_core.py CLAUDE.md
git commit -m "feat(lib): add ds4 backend constants, DS4_PORT/DS4_DIR config, and shared LLM_BACKENDS"
```

---

### Task 2: MCP server — `chat_ds4()` dispatch + `LLM_BACKENDS` fallback

**Files:**
- Modify: `mcp/local-models-server.py` (constants ~line 60; `pick_model` ~lines 583-591; `chat_mlx`/`chat` ~lines 634-664; `lib.models` import block lines 38-48)
- Test: `tests/test_mcp_server.py` (fixtures end ~line 113; `TestPickModel` at ~line 149)
- Docs: `docs/usage-telemetry.md` (line 9), `CLAUDE.md` (line 128)

**Interfaces:**
- Consumes: `LLM_BACKENDS`, from Task 1.
- Produces: `DS4_URL: str` (env `DS4_URL`, default `http://localhost:8002`); `async chat_ds4(model: str, messages: list[dict], max_tokens: int = 4096, think: bool = True) -> str`; `chat(model, backend, messages, ...)` routes `backend == "ds4"` to `chat_ds4`. Task 3 and Task 12 rely on `DS4_URL` and the error string `"is ds4-server running?"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_server.py`. First check the module's imports (`grep -n "^import\|^from" tests/test_mcp_server.py`) — add `import asyncio` and `import json` at the top if not present. Then append after class `TestPickModel`:

```python
# ── ds4 chat dispatch ───────────────────────────────────────────────

class _FakeDs4Response:
    """Mimics httpx.Response for ds4: .text carries raw control chars, so
    strict .json() raises exactly like the real failure mode."""

    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.text)  # strict — raises on control chars


def _fake_async_client(response_text, captured):
    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json=None):
            captured["url"] = url
            captured["body"] = json
            return _FakeDs4Response(response_text)

    return _FakeAsyncClient


class TestChatDs4:
    def test_chat_routes_ds4_and_parses_unescaped_control_chars(self):
        """The real 2026-07-22 failure: ds4's encoder emits raw control
        chars inside long reasoning_content; strict JSON parsing (resp.json /
        plain json.loads) raises and crashes dispatch. chat() must route
        backend='ds4' to chat_ds4 (the old else-branch misrouted it to MLX
        :8000) and parse with strict=False."""
        raw = ('{"choices":[{"message":{"content":"OK then",'
               '"reasoning_content":"thinking\x01hard"}}]}')
        captured = {}
        with patch.object(server.httpx, "AsyncClient",
                          _fake_async_client(raw, captured)):
            result = asyncio.run(server.chat(
                "glm-5.2", "ds4",
                [{"role": "user", "content": "hi"}]))
        assert "OK then" in result
        assert captured["url"].endswith(":8002/v1/chat/completions")

    def test_chat_ds4_never_forwards_think_toggle(self):
        """enable_thinking is VERIFIED BROKEN on ds4 (200 OK but reasoning
        migrates into content, no think-block markers to strip). think=False
        must NOT put chat_template_kwargs on the wire."""
        raw = '{"choices":[{"message":{"content":"hi"}}]}'
        captured = {}
        with patch.object(server.httpx, "AsyncClient",
                          _fake_async_client(raw, captured)):
            asyncio.run(server.chat_ds4(
                "glm-5.2", [{"role": "user", "content": "hi"}], think=False))
        assert "chat_template_kwargs" not in captured["body"]
        assert "think" not in captured["body"]

    def test_chat_ds4_falls_back_to_reasoning_content(self):
        """glm-5.2 on ds4 always thinks and can burn its whole token budget
        thinking — content comes back empty. Surface reasoning_content
        instead of returning an empty string."""
        raw = ('{"choices":[{"message":{"content":"",'
               '"reasoning_content":"the answer is 42"}}]}')
        with patch.object(server.httpx, "AsyncClient",
                          _fake_async_client(raw, {})):
            result = asyncio.run(server.chat_ds4(
                "glm-5.2", [{"role": "user", "content": "hi"}]))
        assert result == "the answer is 42"

    def test_gpu_tracking_accepts_ds4_key(self):
        """_gpu_active is a defaultdict for exactly this reason (a plain
        dict KeyError'd on mlx-audio and broke local_speak) — pin the
        guarantee for the new backend."""
        with server._gpu_request("ds4", "chat:glm-5.2"):
            assert server._gpu_active["ds4"] == 1
        assert server._gpu_active["ds4"] == 0

    def test_pick_model_any_llm_fallback_includes_ds4(self):
        """With no prefs and no task match, pick_model falls back to 'any
        LLM'. The old ('ollama', 'mlx') tuple silently excluded a
        ds4-backed registry — glm-5.2 would be invisible to fallback."""
        server._models["glm-5.2"] = {
            "backend": "ds4", "total_params_b": 380,
            "active_params_b": 32, "context": 131072, "vision": False,
        }
        with patch.object(server, "load_mcp_prefs", return_value={}):
            assert server.pick_model("general") == ("glm-5.2", "ds4")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_mcp_server.py::TestChatDs4 -v`
Expected: FAIL — `AttributeError: module has no attribute 'chat_ds4'`; the dispatch test fails because `chat()` posts to `:8000` (MLX branch); the fallback test fails with `ValueError: No model available`.

- [ ] **Step 3: Implement in `mcp/local-models-server.py`**

3a. Add `LLM_BACKENDS` to the `lib.models` import block (lines 38-48) — it currently reads:

```python
from lib.models import (
    HF_TASK_BACKENDS,
    MCP_PREFS_FILE,
    MLX_SERVER_CONFIG,
    NETWORK_CONF,
    active_params_b,
    mflux_command,
    mflux_is_turbo,
    model_has_vision,
    pick_model_from_prefs,
)
```

Add `LLM_BACKENDS,` after `HF_TASK_BACKENDS,` (alphabetical order is not enforced here, but keep it tidy: insert after `HF_TASK_BACKENDS,`).

3b. Add the URL constant. Replace lines 60-61:

```python
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
```

with:

```python
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
# ds4 is internal-only (never tailscale-served) so this is always localhost;
# bin/local-models-mcp-detect exports it with the configured DS4_PORT.
DS4_URL = os.environ.get("DS4_URL", "http://localhost:8002")
```

3c. In `pick_model` (lines 583-591), replace both `("ollama", "mlx")` tuples:

```python
        for name, info in _models.items():
            if info["backend"] in ("ollama", "mlx") and (not is_eligible or is_eligible(name, info["backend"])):
                return name, info["backend"]

    # Build actionable error message.
    available = [n for n, m in _models.items() if m["backend"] in ("ollama", "mlx")]
```

becomes:

```python
        for name, info in _models.items():
            if info["backend"] in LLM_BACKENDS and (not is_eligible or is_eligible(name, info["backend"])):
                return name, info["backend"]

    # Build actionable error message.
    available = [n for n, m in _models.items() if m["backend"] in LLM_BACKENDS]
```

3d. Add `chat_ds4` after `chat_mlx` (after line 653) and rewrite `chat`. The current `chat` (lines 656-664):

```python
async def chat(model: str, backend: str, messages: list[dict],
               max_tokens: int = 4096, think: bool = True) -> str:
    with _gpu_request(backend, f"chat:{model}"):
        warning = _gpu_contention_warning(backend)
        if backend == "ollama":
            result = await chat_ollama(model, messages, max_tokens, think)
        else:
            result = await chat_mlx(model, messages, max_tokens, think)
        return warning + result
```

becomes:

```python
async def chat_ds4(model: str, messages: list[dict],
                   max_tokens: int = 4096, think: bool = True) -> str:
    # think is accepted for signature parity but deliberately NOT forwarded:
    # ds4's chat_template_kwargs.enable_thinking is verified broken
    # (2026-07-22) — HTTP 200, but the reasoning migrates into `content`
    # with no think-block markers to strip. glm-5.2 on ds4 always thinks.
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": False}
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{DS4_URL}/v1/chat/completions", json=body)
            resp.raise_for_status()
            # ds4's JSON encoder can emit unescaped control characters in
            # long reasoning_content (observed 2026-07-22); strict parsing
            # (resp.json()) intermittently crashes dispatch.
            data = json.loads(resp.text, strict=False)
            msg = data["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
    except httpx.HTTPStatusError as e:
        raise RuntimeError(_http_error_detail(e, f"ds4 chat ({model})")) from e
    except httpx.ConnectError:
        raise RuntimeError(f"ds4 chat ({model}): cannot connect to {DS4_URL} — is ds4-server running?")
    except httpx.TimeoutException:
        raise RuntimeError(f"ds4 chat ({model}): request timed out after 300s")


async def chat(model: str, backend: str, messages: list[dict],
               max_tokens: int = 4096, think: bool = True) -> str:
    with _gpu_request(backend, f"chat:{model}"):
        warning = _gpu_contention_warning(backend)
        if backend == "ollama":
            result = await chat_ollama(model, messages, max_tokens, think)
        elif backend == "ds4":
            result = await chat_ds4(model, messages, max_tokens, think)
        else:
            result = await chat_mlx(model, messages, max_tokens, think)
        return warning + result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_mcp_server.py -v`
Expected: all PASS (new + existing).

- [ ] **Step 5: Docs (same commit)**

`docs/usage-telemetry.md` line 9 currently ends: `Each row records tool, model, backend, status, duration, and the machine that made the request.` Append one sentence:

```markdown
The `backend` column is free text; values today are `ollama`, `mlx`, `ds4` (glm-5.2 on the 512GB tier), and the subprocess backends (`mflux`, `mlx-audio`, `mlx-video`) — no schema change is needed for new backends.
```

`CLAUDE.md` line 128 currently begins: `The \`mcp/local-models-server.py\` MCP server runs as a persistent streamable-HTTP service on port 8100, managed by the menu bar app. It exposes Ollama, MLX, and local tool models (TTS via mlx-audio, image editing via mflux) as MCP tools.` Change `It exposes Ollama, MLX, and local tool models` to `It exposes Ollama, MLX, ds4 (glm-5.2 on the 512GB tier), and local tool models`.

- [ ] **Step 6: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add mcp/local-models-server.py tests/test_mcp_server.py docs/usage-telemetry.md CLAUDE.md
git commit -m "feat(mcp): dispatch chat to ds4 with control-char-tolerant JSON parsing"
```

---

### Task 3: MCP server — ds4 discovery + status/GPU/activity endpoints

**Files:**
- Modify: `mcp/local-models-server.py` (`discover_models` lines 356-492; `local_models_status` lines 669-700; `_startup` lines 1876-1882; `_gpu_status` lines 1885-1917; `_activity_status` line 1926)
- Test: `tests/test_mcp_server.py`, `tests/test_e2e.py` (line 66 port skip-list; line 234 `/gpu` keys)
- Docs: `CLAUDE.md` (line 18), `docs/architecture.md` (line 28)

**Interfaces:**
- Consumes: `DS4_URL` (Task 2); `DS4_MODEL_NAME`, `DS4_TOTAL_PARAMS_B`, `DS4_ACTIVE_PARAMS_B`, `DS4_CONTEXT` (Task 1 — add them to the `lib.models` import block).
- Produces: `_models[DS4_MODEL_NAME] == {"backend": "ds4", "total_params_b": 380, "active_params_b": 32, "context": 131072, "vision": False}` when ds4 answers; `/gpu` JSON gains a `"ds4"` key with `active`/`tasks`/`responsive`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_server.py`:

```python
# ── ds4 discovery ───────────────────────────────────────────────────

def _fake_discovery_client(ds4_up):
    """AsyncClient stub: Ollama and MLX unreachable; ds4 configurable.
    Exercises the real discover_models control flow, not a mock of it."""
    class _Resp:
        status_code = 200
        def json(self):
            return {}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            if url.startswith(server.DS4_URL):
                if ds4_up:
                    return _Resp()
                raise ConnectionError("ds4 down")
            raise ConnectionError("backend down")

        async def post(self, url, **kwargs):
            raise ConnectionError("backend down")

    return _Client


class TestDs4Discovery:
    def _discover(self, ds4_up, tmp_path):
        # MLX_SERVER_CONFIG must point away from the real user yaml: an
        # unmigrated ~/.config/mlx-server/config.yaml still lists glm-5.2
        # as an MLX served-name and would leak into the registry.
        from unittest.mock import MagicMock
        from pathlib import Path
        import lib.hf_scanner as hf_scanner
        with patch.object(server.httpx, "AsyncClient",
                          _fake_discovery_client(ds4_up)), \
             patch.object(server, "MLX_SERVER_CONFIG",
                          Path(tmp_path) / "absent.yaml"), \
             patch.object(hf_scanner, "scan_hf_cache",
                          MagicMock(return_value=[])):
            return asyncio.run(server.discover_models())

    def test_ds4_up_inserts_glm52_with_hardcoded_metadata(self, tmp_path):
        """ds4's /v1/models returns NO metadata. Without these hardcoded
        values, TASK_FILTERS min_active_b (reasoning: 10) and min_ctx
        (long_context: 64000) silently drop glm-5.2 from every task list."""
        models = self._discover(True, tmp_path)
        assert models["glm-5.2"] == {
            "backend": "ds4",
            "total_params_b": 380,
            "active_params_b": 32,
            "context": 131072,
            "vision": False,
        }

    def test_ds4_down_means_glm52_absent(self, tmp_path):
        """Same semantics as MLX-down today: unreachable backend, no model."""
        models = self._discover(False, tmp_path)
        assert "glm-5.2" not in models
```

And in `tests/test_e2e.py`:

1. Line 66, the port skip-list in the profile-server port finder:

```python
                if port in (8100, 11434, 8000, 80, 443):
```

becomes:

```python
                if port in (8100, 11434, 8000, 8002, 80, 443):
```

2. `test_gpu_returns_json_with_backend_keys` (lines 234-239):

```python
    def test_gpu_returns_json_with_backend_keys(self, mcp_base):
        status, body, _ = http_get(f"{mcp_base}/gpu")
        assert status == 200
        data = json.loads(body)
        assert "ollama" in data, "GPU status missing 'ollama' key"
        assert "mlx" in data, "GPU status missing 'mlx' key"
```

becomes:

```python
    def test_gpu_returns_json_with_backend_keys(self, mcp_base):
        status, body, _ = http_get(f"{mcp_base}/gpu")
        assert status == 200
        data = json.loads(body)
        assert "ollama" in data, "GPU status missing 'ollama' key"
        assert "mlx" in data, "GPU status missing 'mlx' key"
        assert "ds4" in data, "GPU status missing 'ds4' key"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_mcp_server.py::TestDs4Discovery -v`
Expected: FAIL — `KeyError: 'glm-5.2'` (no ds4 block in discovery). (`test_e2e.py` needs live services; it skips locally — that's fine, it runs on the server.)

- [ ] **Step 3: Implement discovery + endpoints**

3a. Extend the `lib.models` import block with the DS4 constants (after `LLM_BACKENDS,` added in Task 2):

```python
    DS4_ACTIVE_PARAMS_B,
    DS4_CONTEXT,
    DS4_MODEL_NAME,
    DS4_TOTAL_PARAMS_B,
```

3b. In `discover_models`, insert a ds4 block after the on-demand MLX registration loop (after line 476, `models[sn] = {...}` for-loop end) and before the HF-cache scan:

```python
        # ds4 (glm-5.2, 512GB tier). Its /v1/models returns no metadata, so
        # the entry is hardcoded from lib.models — see the audit spec.
        # Placed after MLX and overwriting any stale claim: an unmigrated
        # user yaml may still list glm-5.2 as an MLX served-name, but when
        # ds4 answers, ds4 is the backend that actually serves it.
        # If ds4 is unreachable, glm-5.2 is simply absent (same semantics
        # as MLX-down today).
        try:
            resp = await client.get(f"{DS4_URL}/v1/models", timeout=5)
            if resp.status_code == 200:
                models[DS4_MODEL_NAME] = {
                    "backend": "ds4",
                    "total_params_b": DS4_TOTAL_PARAMS_B,
                    "active_params_b": DS4_ACTIVE_PARAMS_B,
                    "context": DS4_CONTEXT,
                    "vision": False,
                }
        except Exception as e:
            logging.info("ds4 discovery skipped: %s", e)
```

3c. `local_models_status` (lines 680-684) — replace:

```python
    ollama = {k: v for k, v in _models.items() if v["backend"] == "ollama"}
    mlx = {k: v for k, v in _models.items() if v["backend"] == "mlx"}

    lines = [f"Ollama ({OLLAMA_URL}): {len(ollama)} models",
             f"MLX ({MLX_URL}): {len(mlx)} models", ""]
```

with:

```python
    ollama = {k: v for k, v in _models.items() if v["backend"] == "ollama"}
    mlx = {k: v for k, v in _models.items() if v["backend"] == "mlx"}
    ds4 = {k: v for k, v in _models.items() if v["backend"] == "ds4"}

    lines = [f"Ollama ({OLLAMA_URL}): {len(ollama)} models",
             f"MLX ({MLX_URL}): {len(mlx)} models"]
    if ds4:
        lines.append(f"ds4 ({DS4_URL}): {len(ds4)} models")
    lines.append("")
```

(The `if ds4:` guard keeps laptop output unchanged — ds4 only runs on the 512GB tier.)

3d. `_startup` (lines 1876-1882) — replace the two count lines:

```python
    ollama_count = sum(1 for v in _models.values() if v["backend"] == "ollama")
    mlx_count = sum(1 for v in _models.values() if v["backend"] == "mlx")
    logging.info("local-models MCP: %d Ollama + %d MLX models", ollama_count, mlx_count)
```

with:

```python
    ollama_count = sum(1 for v in _models.values() if v["backend"] == "ollama")
    mlx_count = sum(1 for v in _models.values() if v["backend"] == "mlx")
    ds4_count = sum(1 for v in _models.values() if v["backend"] == "ds4")
    logging.info("local-models MCP: %d Ollama + %d MLX + %d ds4 models",
                 ollama_count, mlx_count, ds4_count)
```

3e. `_gpu_status` (lines 1885-1917) — inside the `with _gpu_lock:` block, add a `"ds4"` entry to `data` (after the `"mlx"` entry):

```python
            "ds4": {
                "active": _gpu_active["ds4"],
                "tasks": [
                    {**e, "elapsed_ms": int((now - e["started"]) * 1000)}
                    for e in _gpu_active_details["ds4"]
                ],
            },
```

and after the Ollama responsiveness probe, add:

```python
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            await client.get(f"{DS4_URL}/v1/models")
            data["ds4"]["responsive"] = True
    except Exception:
        data["ds4"]["responsive"] = False
```

3f. `_activity_status` (line 1926) — replace:

```python
        for backend in ("ollama", "mlx"):
```

with:

```python
        for backend in ("ollama", "mlx", "ds4"):
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_mcp_server.py -v`
Expected: all PASS.

- [ ] **Step 5: Docs (same commit)**

`CLAUDE.md` line 18 currently:

```markdown
- The MCP server discovers models live from Ollama and MLX at startup (parallel `/api/show` calls). Any new `ollama pull` is immediately available as a tool.
```

becomes:

```markdown
- The MCP server discovers models live from Ollama, MLX, and ds4 at startup (parallel `/api/show` calls). Any new `ollama pull` is immediately available as a tool. ds4's `/v1/models` carries no metadata, so glm-5.2's params/context/size are hardcoded in `lib/models.py` (`DS4_*` constants).
```

`docs/architecture.md` line 28 currently: `Persistent streamable-HTTP service on port 8100. Discovers models from Ollama and MLX at startup.` Change to `Persistent streamable-HTTP service on port 8100. Discovers models from Ollama, MLX, and ds4 at startup.`

- [ ] **Step 6: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add mcp/local-models-server.py tests/test_mcp_server.py tests/test_e2e.py CLAUDE.md docs/architecture.md
git commit -m "feat(mcp): discover ds4-served glm-5.2 and surface ds4 in status endpoints"
```

---

### Task 4: Profile server — ds4 discovery, eligibility, missing-model exclusion

**Files:**
- Modify: `app/profile-server.py` (URL constants lines 81-82; `_fetch_all_models` lines 1202-1216; `_LLM_BACKENDS` line 1228; `_check_missing_models` lines 1524-1556; `lib.models` import block lines 44-70)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Consumes: `LLM_BACKENDS`, `DS4_MODEL_NAME`, `DS4_MODEL_BYTES`, `DS4_TOTAL_PARAMS_B`, `DS4_ACTIVE_PARAMS_B`, `DS4_CONTEXT` (Task 1).
- Produces: `DS4_URL: str` module constant; `_fetch_ds4_models(existing) -> dict` returning the full profile-server model-dict shape (keys: `name, backend, disk_bytes, vram_bytes, total_params_b, active_params_b, context, has_vision, family, quant, is_loaded, expires_at, on_demand`). Task 5 uses `DS4_URL`; Task 11 uses the same import list.
- Note: **no separate memory-bar change is needed** — glm-5.2 is the 512gb preset's warm `general` model, so `api_profiles_memory` picks up `vram_bytes=DS4_MODEL_BYTES` from this discovery entry and counts the 244GiB as warm/fixed residency automatically. The warm loop in `api_profiles_warm` only preloads `ollama`/`mlx` backends, so ds4 (always resident) is correctly skipped with no change.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_profile_server.py` (module imports `ps`, `patch`; verify `from unittest.mock import patch, MagicMock` is present at top). Append near the other model-fetch tests:

```python
# ── ds4 discovery ───────────────────────────────────────────────────

class TestFetchDs4Models:
    def test_ds4_up_inserts_hardcoded_entry(self):
        """ds4 serves one pinned model with no metadata; the entry must be
        fully hardcoded (sizes included — the GGUF is invisible to every
        existing sizing path) and marked always-resident, or the memory bar
        undercounts 244GiB and warm logic tries to keep-alive it."""
        resp = MagicMock()
        resp.ok = True
        with patch.object(ps.requests, "get", return_value=resp):
            out = ps._fetch_ds4_models(existing={})
        entry = out["glm-5.2"]
        assert entry["backend"] == "ds4"
        assert entry["disk_bytes"] == 262_036_650_048
        assert entry["vram_bytes"] == 262_036_650_048
        assert entry["total_params_b"] == 380
        assert entry["active_params_b"] == 32
        assert entry["context"] == 131072
        assert entry["has_vision"] is False
        assert entry["is_loaded"] is True
        assert entry["on_demand"] is False

    def test_ds4_down_returns_empty(self):
        with patch.object(ps.requests, "get",
                          side_effect=ps.requests.ConnectionError("down")):
            assert ps._fetch_ds4_models(existing={}) == {}

    def test_existing_name_not_overwritten(self):
        resp = MagicMock()
        resp.ok = True
        with patch.object(ps.requests, "get", return_value=resp):
            out = ps._fetch_ds4_models(existing={"glm-5.2": {}})
        assert out == {}

    def test_ds4_model_is_eligible_for_llm_tasks(self):
        """The one-line bug this guards: _LLM_BACKENDS without 'ds4' gives
        glm-5.2 zero eligible tasks — invisible in every dropdown."""
        resp = MagicMock()
        resp.ok = True
        with patch.object(ps.requests, "get", return_value=resp):
            entry = ps._fetch_ds4_models(existing={})["glm-5.2"]
        tasks = ps.get_eligible_tasks("glm-5.2", entry)
        for task in ("code", "general", "reasoning", "long_context",
                     "translation"):
            assert task in tasks, f"glm-5.2 missing eligible task {task!r}"

    def test_missing_models_check_skips_ds4_served_name(self):
        """glm-5.2 is pre-provisioned by install.sh, not pullable. With the
        MLX yaml entry gone, an unpatched _check_missing_models would prompt
        the user to pull it, and the pull would 404 (`ollama pull glm-5.2`)."""
        with patch.object(ps, "get_all_models",
                          return_value={"qwen3.6:27b": {"backend": "ollama"}}):
            missing, _ = ps._check_missing_models({"general": ["glm-5.2"]})
        assert missing == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py::TestFetchDs4Models -v`
Expected: FAIL — `AttributeError: ... has no attribute '_fetch_ds4_models'`; the missing-models test fails with `missing == ["glm-5.2"]`.

- [ ] **Step 3: Implement**

3a. Extend the `lib.models` import block (lines 44-70) with (keeping alphabetical position):

```python
    DS4_ACTIVE_PARAMS_B,
    DS4_CONTEXT,
    DS4_MODEL_BYTES,
    DS4_MODEL_NAME,
    DS4_TOTAL_PARAMS_B,
    LLM_BACKENDS,
    ds4_installed,
```

(`ds4_installed` gates `_fetch_mlx_models`'s glm-5.2 skip below, so it's a real dependency of this task, not just importing ahead for Task 11's diagnostics probe.)

3b. Replace lines 81-82:

```python
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
```

with:

```python
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MLX_URL = os.environ.get("MLX_URL", "http://localhost:8000")
# ds4 is internal-only (port 8002 is never tailscale-served); the menubar
# injects DS4_URL with the configured DS4_PORT.
DS4_URL = os.environ.get("DS4_URL", "http://localhost:8002")
```

3c. Add `_fetch_ds4_models` immediately before `_fetch_hf_cache_models` (line 1172):

```python
def _fetch_ds4_models(existing):
    """ds4-served glm-5.2 (512GB tier). ds4's /v1/models returns no
    metadata, so every field is hardcoded from lib.models: the GGUF is not
    an HF snapshot and none of the existing sizing paths can see it.
    Always-resident (loads at startup, never unloads): is_loaded=True,
    on_demand=False, and vram==disk counts the full 244GiB as fixed
    residency in the memory bar. Unreachable ds4 ⇒ empty dict — glm-5.2
    absent, same semantics as MLX-down."""
    if DS4_MODEL_NAME in existing:
        return {}
    try:
        resp = requests.get(f"{DS4_URL}/v1/models", timeout=2)
        if not resp.ok:
            return {}
    except Exception:
        return {}
    return {DS4_MODEL_NAME: {
        "name": DS4_MODEL_NAME,
        "backend": "ds4",
        "disk_bytes": DS4_MODEL_BYTES,
        "vram_bytes": DS4_MODEL_BYTES,
        "total_params_b": DS4_TOTAL_PARAMS_B,
        "active_params_b": DS4_ACTIVE_PARAMS_B,
        "context": DS4_CONTEXT,
        "has_vision": False,
        "family": "ds4",
        "quant": "q2k",
        "is_loaded": True,
        "expires_at": None,
        "on_demand": False,
    }}
```

3d. Wire it into `_fetch_all_models` (lines 1202-1216). Replace:

```python
    models = _fetch_ollama_models()
    models.update(_fetch_mlx_models(existing=models))
    if not remote_mode:
        models.update(_fetch_hf_cache_models(existing=models))
    return models
```

with:

```python
    models = _fetch_ollama_models()
    if not remote_mode:
        # ds4 before MLX: an unmigrated user yaml may still list glm-5.2 as
        # an MLX served-name; when ds4 answers, ds4 is the backend that
        # actually serves it. Never scanned in remote mode — DS4_URL is
        # localhost and the desktop's registry already includes it.
        models.update(_fetch_ds4_models(existing=models))
    models.update(_fetch_mlx_models(existing=models))
    if not remote_mode:
        models.update(_fetch_hf_cache_models(existing=models))
    return models
```

3e. Replace line 1228:

```python
_LLM_BACKENDS = {"ollama", "mlx"}
```

with:

```python
_LLM_BACKENDS = LLM_BACKENDS
```

(Keep the local alias name — `get_eligible_tasks` at line 1237 references `_LLM_BACKENDS` and this is the minimal diff; the alias now points at the shared frozenset.)

3f. In `_check_missing_models` (lines 1550-1555), replace:

```python
        for c in candidates:
            if on_demand and "/" in c:
                continue
            if not _model_exists(c) and c not in seen:
```

with:

```python
        for c in candidates:
            if on_demand and "/" in c:
                continue
            # ds4-served models are pre-provisioned by install.sh, not
            # pullable — prompting would dead-end in `ollama pull glm-5.2`.
            if c == DS4_MODEL_NAME:
                continue
            if not _model_exists(c) and c not in seen:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -v`
Expected: all PASS (new + all 56 existing).

- [ ] **Step 5: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "feat(profile-server): discover ds4-served glm-5.2 with hardcoded metadata"
```

---

### Task 5: Profile server — chat dispatch (`_chat_url`, `_chat`, `_chat_stream`)

**Files:**
- Modify: `app/profile-server.py` (`_chat_url` lines 2067-2071; `_chat` lines 2133-2168; `_chat_stream` lines 2171-2247)
- Test: `tests/test_profile_server.py` (class `TestChatUrl` at lines 110-115)

**Interfaces:**
- Consumes: `DS4_URL` (Task 4).
- Produces: `_chat_url("ds4") == f"{DS4_URL}/v1/chat/completions"`; `_chat(model, "ds4", messages, ...)` and `_chat_stream(model, "ds4", ...)` hit ds4 with no think toggle and strict-tolerant parsing. Task 12's correctness test exercises this path live.

- [ ] **Step 1: Write the failing tests**

In `tests/test_profile_server.py`, extend class `TestChatUrl` (after line 115):

```python
    def test_ds4_backend(self):
        assert ps._chat_url("ds4") == "http://localhost:8002/v1/chat/completions"
```

And append a new class:

```python
class TestChatDs4Dispatch:
    def _resp(self, text):
        r = MagicMock()
        r.text = text
        r.raise_for_status = MagicMock()
        return r

    def test_chat_ds4_posts_to_ds4_without_think_toggle(self):
        """think=False must NOT forward chat_template_kwargs to ds4
        (verified broken: reasoning migrates into content), and the reply
        must survive raw control chars in reasoning_content (ds4 encoder
        bug — strict resp.json() raises)."""
        raw = ('{"choices":[{"message":{"content":"pong",'
               '"reasoning_content":"pondering\x01deeply"}}]}')
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["body"] = json
            return self._resp(raw)

        with patch.object(ps.requests, "post", side_effect=fake_post):
            out = ps._chat("glm-5.2", "ds4",
                           [{"role": "user", "content": "ping"}], think=False)
        assert out == "pong"
        assert captured["url"] == "http://localhost:8002/v1/chat/completions"
        assert "chat_template_kwargs" not in captured["body"]
        assert "think" not in captured["body"]

    def test_chat_ds4_falls_back_to_reasoning_content(self):
        raw = ('{"choices":[{"message":{"content":"",'
               '"reasoning_content":"all reasoning, no answer"}}]}')
        with patch.object(ps.requests, "post",
                          return_value=self._resp(raw)):
            out = ps._chat("glm-5.2", "ds4",
                           [{"role": "user", "content": "hi"}])
        assert out == "all reasoning, no answer"

    def test_chat_stream_ds4_yields_tokens_with_tolerant_parse(self):
        """The streaming branch parses each SSE chunk; a chunk with a raw
        control char must yield its token, not be dropped by a strict
        parser (long thinking answers stream reasoning first)."""
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hel\x01lo"}}]}',
            b"data: [DONE]",
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = iter(lines)
        with patch.object(ps.requests, "post", return_value=resp):
            events = list(ps._chat_stream(
                "glm-5.2", "ds4", [{"role": "user", "content": "hi"}]))
        joined = "".join(events)
        assert "Hel\\u0001lo" in joined or "Hello" in joined.replace("\\u0001", "")
        assert '"done": true' in joined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py::TestChatUrl tests/test_profile_server.py::TestChatDs4Dispatch -v`
Expected: FAIL — `_chat_url("ds4")` returns the Ollama URL (the current else-branch); `_chat`/`_chat_stream` post to the Ollama URL.

- [ ] **Step 3: Implement**

3a. Replace `_chat_url` (lines 2067-2071):

```python
def _chat_url(backend):
    """Return the chat endpoint URL for a backend."""
    if backend == "mlx":
        return f"{MLX_URL}/v1/chat/completions"
    return f"{OLLAMA_URL}/api/chat"
```

with:

```python
def _chat_url(backend):
    """Return the chat endpoint URL for a backend."""
    if backend == "mlx":
        return f"{MLX_URL}/v1/chat/completions"
    if backend == "ds4":
        return f"{DS4_URL}/v1/chat/completions"
    return f"{OLLAMA_URL}/api/chat"
```

3b. In `_chat` (lines 2148-2168), the dispatch body currently reads:

```python
    with _track_playground(tool, model, backend):
        try:
            if backend == "mlx":
                body = {"model": model, "messages": messages, "stream": False}
                if not think:
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                resp = requests.post(f"{MLX_URL}/v1/chat/completions",
                                     json=body, timeout=timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            else:
```

Insert a ds4 branch before the mlx branch:

```python
    with _track_playground(tool, model, backend):
        try:
            if backend == "ds4":
                # No think toggle: ds4's enable_thinking is verified broken
                # (reasoning migrates into `content`) — never forward
                # chat_template_kwargs. glm-5.2 on ds4 always thinks.
                body = {"model": model, "messages": messages, "stream": False}
                resp = requests.post(f"{DS4_URL}/v1/chat/completions",
                                     json=body, timeout=timeout)
                resp.raise_for_status()
                # strict=False: ds4 emits unescaped control chars in long
                # reasoning_content; resp.json() intermittently raises.
                msg = json.loads(resp.text,
                                 strict=False)["choices"][0]["message"]
                return msg.get("content") or msg.get("reasoning_content") or ""
            elif backend == "mlx":
                body = {"model": model, "messages": messages, "stream": False}
                if not think:
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                resp = requests.post(f"{MLX_URL}/v1/chat/completions",
                                     json=body, timeout=timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            else:
```

(The existing `else:` Ollama branch is unchanged.)

3c. In `_chat_stream` (lines 2184-2213), the body currently begins:

```python
    try:
        if backend == "mlx":
            body = {"model": model, "messages": messages, "stream": True}
```

Insert a ds4 branch first:

```python
    try:
        if backend == "ds4":
            # OpenAI-style SSE like MLX, but: never forward the think
            # toggle (broken on ds4), and parse chunks with strict=False
            # (unescaped control chars in streamed reasoning/content).
            body = {"model": model, "messages": messages, "stream": True}
            try:
                resp = requests.post(f"{DS4_URL}/v1/chat/completions",
                                     json=body, stream=True, timeout=300)
                resp.raise_for_status()
                yield f"data: {json.dumps({'model': model})}\n\n"
                for line in resp.iter_lines():
                    if not line:
                        continue
                    text = line.decode("utf-8", errors="replace")
                    if text.startswith("data: "):
                        text = text[6:]
                    if text.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(text, strict=False)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
            except requests.RequestException as e:
                raise RuntimeError(
                    f"Stream ({model} via {backend}): "
                    f"{_requests_error_detail(e)}") from e
        elif backend == "mlx":
            body = {"model": model, "messages": messages, "stream": True}
```

(The rest of the mlx branch and the ollama `else:` are unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "feat(profile-server): dispatch chat and streaming to the ds4 backend"
```

---

### Task 6: Service scripts — `start-local-models` ds4 lifecycle + `local-models-mcp-detect` DS4_URL

**Files:**
- Modify: `bin/start-local-models` (config block lines 27-36; `stop_services` lines 92-108; `show_status` lines 110-145; a new ds4 launch section after the MLX section, line 205; summary lines 207-216)
- Modify: `bin/local-models-mcp-detect` (exports at lines 68-69)
- Docs: `CLAUDE.md` (Runtime Architecture lines 29-31 and Remote access table lines 45-52), `docs/architecture.md` (components diagram lines 7-20, ports table lines 42-47), `README.md` (Commands lines 148-158, OpenAI-API section lines 55-87)

**Interfaces:**
- Consumes: `DS4_PORT`/`DS4_DIR` from network.conf (Task 1). Presence gate: `$DS4_DIR/ds4-server` executable (provisioned by Task 10; on the dev machine it can be exercised early by pointing `DS4_DIR` at `~/experiments/ds4`).
- Produces: `start-local-models` launches/stops/reports ds4-server; `local-models-mcp-detect` exports `DS4_URL` so the MCP server (spawned by the menubar through this wrapper) sees it. Task 8's menubar auto-restart relies on `start-local-models` handling ds4.

This task is bash glue — per repo standards, script tests are judgment-call; the branchy logic here is exercised by the live verification steps below and by the e2e/smoke suites once ds4 is provisioned. No mocked bash tests (they'd assert the script contains strings — theater).

- [ ] **Step 1: Add ds4 config defaults to `bin/start-local-models`**

Replace lines 27-36:

```bash
# Load network config
OLLAMA_PORT=11434
MLX_PORT=8000
PROBE_TIMEOUT=2
TAILSCALE_HOSTNAME=""

if [ -f "$NETWORK_CONF" ]; then
    # shellcheck source=/dev/null
    source "$NETWORK_CONF"
fi
```

with:

```bash
# Load network config
OLLAMA_PORT=11434
MLX_PORT=8000
DS4_PORT=8002
DS4_DIR=""
PROBE_TIMEOUT=2
TAILSCALE_HOSTNAME=""

if [ -f "$NETWORK_CONF" ]; then
    # shellcheck source=/dev/null
    source "$NETWORK_CONF"
fi
DS4_DIR="${DS4_DIR:-$HOME/.local/share/super-puppy/ds4}"
```

- [ ] **Step 2: Add ds4 to `stop_services`**

In `stop_services` (lines 92-108), insert after the MLX pkill block (after line 101):

```bash
    if pkill -f "ds4-server" 2>/dev/null; then
        echo "  ds4-server stopped."
    else
        echo "  ds4-server not running."
    fi
```

- [ ] **Step 3: Add ds4 to `show_status`**

In `show_status`, insert after the MLX status block (after line 143, before `exit 0`):

```bash
    if [ -x "$DS4_DIR/ds4-server" ]; then
        if curl -sf "http://localhost:${DS4_PORT}/v1/models" > /dev/null 2>&1; then
            echo "  ds4-server:        running (http://localhost:${DS4_PORT}, glm-5.2)"
        else
            echo "  ds4-server:        not running"
        fi
    fi
```

- [ ] **Step 4: Add the ds4 launch section**

Insert after the MLX-OpenAI-Server section (after line 205, before the `# --- Summary ---` block):

```bash
# --- ds4 (glm-5.2 frontier engine, 512GB tier) ---
# Presence-gated: install.sh only provisions $DS4_DIR/ds4-server on 512GB
# machines. MUST run with cwd=$DS4_DIR — ds4-server resolves
# metal/flash_attn.metal relative to the working directory (do NOT copy the
# MLX launch's `cd $HOME`). Cold load is ~70s; readiness allows 300s.
if [ -x "$DS4_DIR/ds4-server" ]; then
    if curl -sf "http://localhost:${DS4_PORT}/v1/models" > /dev/null 2>&1; then
        echo "ds4-server already running."
    else
        echo "Starting ds4-server (glm-5.2, ~70s cold load)..."
        DS4_LOG="/tmp/local-models-ds4.log"
        (cd "$DS4_DIR" && exec ./ds4-server --metal --port "$DS4_PORT" \
            --ctx 32768 -m ds4flash.gguf) > "$DS4_LOG" 2>&1 &
        disown
        if ! wait_for_service "http://localhost:${DS4_PORT}/v1/models" "ds4-server" 300; then
            echo "  ERROR: ds4-server failed to start. Log: $DS4_LOG" >&2
        fi
    fi
fi
```

- [ ] **Step 5: Add ds4 to the summary block**

In the summary (lines 213-215), after the `MLX-OpenAI:` line, add:

```bash
if [ -x "$DS4_DIR/ds4-server" ]; then
    echo "  ds4 (glm-5.2): http://localhost:${DS4_PORT}"
fi
```

- [ ] **Step 6: Export DS4_URL in `bin/local-models-mcp-detect`**

Replace lines 68-69:

```bash
export OLLAMA_URL
export MLX_URL
```

with:

```bash
export OLLAMA_URL
export MLX_URL
# ds4 never gets a Tailscale FQDN rewrite: port 8002 is internal-only.
# In client mode glm-5.2 is reached through the desktop's MCP (8100), so a
# localhost DS4_URL on a laptop is correct — it simply discovers nothing.
export DS4_URL="http://localhost:${DS4_PORT:-8002}"
```

- [ ] **Step 7: Verify**

```bash
bash -n bin/start-local-models && bash -n bin/local-models-mcp-detect && echo SYNTAX-OK
~/.local/bin/start-local-models --status
```

Expected: `SYNTAX-OK`; status output shows the existing services and — only if `$DS4_DIR/ds4-server` exists on this machine — a ds4 line. On the dev 512GB box, additionally run `start-local-models` and confirm `curl -s http://localhost:8002/v1/models` returns 200 within 300s, then `start-local-models --stop` reports `ds4-server stopped.`.

Run the unit suite (unchanged code paths, must stay green): `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`

- [ ] **Step 8: Docs (same commit)**

`CLAUDE.md`:
1. After line 31 (`- **MLX-OpenAI-Server** — http://localhost:8000, config at ~/.config/mlx-server/config.yaml`), add:

```markdown
- **ds4-server** — `http://localhost:8002` (512GB tier only), serving glm-5.2 from a Q2K GGUF under `DS4_DIR` (default `~/.local/share/super-puppy/ds4`, override in network.conf). Launched by `start-local-models` with `cwd=$DS4_DIR` (it resolves `metal/flash_attn.metal` relative to cwd). Always-resident: loads once (~70s), never unloads. Port 8002 is **internal-only** — never added to `tailscale serve`; client-mode traffic reaches glm-5.2 through the desktop's MCP (8100) and profile server (8101).
```

2. After the Remote access table (line 50, after the MLX row), add below the table (line 52 area):

```markdown
ds4 (8002) is deliberately absent from this table — internal-only, never served.
```

3. Modes table (line 37), Server row: change `Runs Ollama, MLX, MCP locally.` to `Runs Ollama, MLX, ds4 (512GB tier), MCP locally.`

`docs/architecture.md`:
1. In the components diagram (lines 13-15), add a line after `│      ├── MLX server      localhost:8000              │`:

```
│      ├── ds4-server      localhost:8002 (512GB tier)  │
```

2. After the ports table (line 47), add:

```markdown
ds4-server (8002, glm-5.2 on the 512GB tier) is internal-only and never added to `tailscale serve`; remote clients reach glm-5.2 through the MCP server (8100) and profile server (8101).
```

`README.md`:
1. In Commands (line 149), the comment `# start Ollama + MLX servers` becomes `# start Ollama + MLX (+ ds4 on the 512GB tier)`.
2. After the OpenAI-compatible API section (after line 87), add:

```markdown
On the 512GB tier, glm-5.2 is served by [ds4](https://github.com/antirez/ds4) with the same OpenAI-compatible API on port 8002 (localhost only):

​```bash
curl http://localhost:8002/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "glm-5.2",
  "messages": [{"role":"user","content":"hello"}]
}'
​```
```

(Remove the zero-width escapes around the code fence when writing — they mark nesting here only.)

- [ ] **Step 9: Commit**

```bash
git add bin/start-local-models bin/local-models-mcp-detect CLAUDE.md docs/architecture.md README.md
git commit -m "feat(bin): launch ds4-server from start-local-models and export DS4_URL"
```

---

### Task 7: Menubar — warm-ping classifier excludes ds4-served bare names

**Files:**
- Modify: `app/menubar.py` (`warm_ping_targets` lines 726-739; `_on_warm_tick` line 1471)
- Test: `tests/test_core.py` (after `test_warm_ping_targets_classifies_backend` at line 614)

**Interfaces:**
- Consumes: nothing new (pure-logic change; `self.mlx_config_info` already exists at menubar init, line 1357).
- Produces: `warm_ping_targets(data, mlx_served=None) -> list[tuple[str, str]]` — bare names not in `mlx_served` are excluded (ds4-served, always resident, no ping needed). `mlx_served=None` preserves legacy behavior. Task 9's `test_512gb_warm_ping_skips_glm52` relies on this signature.

- [ ] **Step 1: Write the failing test**

In `tests/test_core.py`, after `test_warm_ping_targets_classifies_backend` (line 621):

```python
    def test_warm_ping_targets_excludes_ds4_served_bare_names(self):
        """A bare warm name that is NOT an MLX served-name is ds4-served:
        always resident, no idle unload, so a keep-warm ping is useless —
        and before ds4 existed the old heuristic would have pinged MLX for
        a model MLX doesn't serve (404 every 240s)."""
        data = {"active": "t", "profiles": {"t": {
            "warm": ["general", "embedding"],
            "tasks": {"general": "glm-5.2", "embedding": "qwen3.5-fast"}}}}
        targets = dict(menubar.warm_ping_targets(
            data, mlx_served={"qwen3.5-fast"}))
        assert targets == {"qwen3.5-fast": "mlx"}

    def test_warm_ping_targets_legacy_without_served_set(self):
        """mlx_served=None keeps the old classify-bare-as-mlx behavior."""
        data = {"active": "t", "profiles": {"t": {
            "warm": ["general"], "tasks": {"general": "qwen3.5-fast"}}}}
        assert menubar.warm_ping_targets(data) == [("qwen3.5-fast", "mlx")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -k warm_ping -v`
Expected: the new exclusion test FAILS (`glm-5.2` present, classified `mlx`); the legacy test passes; `TypeError` if positional signature mismatch — confirm the failure message before proceeding.

- [ ] **Step 3: Implement**

Replace `warm_ping_targets` (lines 726-739):

```python
def warm_ping_targets(data):
    """(model, backend) for the active profile's warm models worth keep-warming.

    backend: 'ollama' for ':' tags, 'mlx' for bare served-names. HuggingFace
    repos ('/') are excluded — they're invoked per-call (mflux / mlx-audio),
    not long-lived server models to keep resident.
    """
    targets = []
    for model in sorted(warm_model_names(data)):
        if "/" in model and ":" not in model:
            continue
        backend = "ollama" if ":" in model else "mlx"
        targets.append((model, backend))
    return targets
```

with:

```python
def warm_ping_targets(data, mlx_served=None):
    """(model, backend) for the active profile's warm models worth keep-warming.

    backend: 'ollama' for ':' tags, 'mlx' for bare served-names present in
    the MLX server config. HuggingFace repos ('/') are excluded — they're
    invoked per-call (mflux / mlx-audio), not long-lived server models to
    keep resident. Bare names NOT in `mlx_served` are ds4-served: always
    resident with no idle unload, so they need no keep-warm ping.
    mlx_served=None preserves the legacy classify-bare-as-mlx behavior for
    callers without config access.
    """
    targets = []
    for model in sorted(warm_model_names(data)):
        if "/" in model and ":" not in model:
            continue
        if ":" in model:
            targets.append((model, "ollama"))
        elif mlx_served is None or model in mlx_served:
            targets.append((model, "mlx"))
    return targets
```

And in `_on_warm_tick` (line 1471), replace:

```python
        targets = warm_ping_targets(load_profiles())
```

with:

```python
        targets = warm_ping_targets(load_profiles(),
                                    mlx_served=set(self.mlx_config_info))
```

(`self.mlx_config_info` is `{served_name: {...}}` from `query_mlx_model_info_from_config()`, set at init line 1357 — its key set IS the MLX served-name list.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -v`
Expected: all PASS (including the untouched `test_warm_ping_targets_classifies_backend` and — still — `test_warm_models_bare_names_are_mlx_served`, since glm-5.2 is still in the yaml until Task 9).

- [ ] **Step 5: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add app/menubar.py tests/test_core.py
git commit -m "feat(menubar): skip warm pings for ds4-served bare model names"
```

---

### Task 8: Menubar — ds4 as a monitored service

**Files:**
- Modify: `app/menubar.py` (constants after line 235; `lib.models` import line 229; init lines 1325-1353 and menu construction lines 1384-1439; `_refresh_server_mode` lines 2247-2286; `_restart_mlx` neighborhood — add `_restart_ds4` after line 2149; `_update_menu` lines 2462-2497; `_copy_diagnostics` lines 1894-1927; profile-server env injection lines 2583-2586)
- Test: `tests/test_core.py`
- Docs: `CLAUDE.md` (Menu Bar Features lines 151-157; Key files table lines 83-100)

**Interfaces:**
- Consumes: `ds4_dir()`, `ds4_installed()` from `lib.models` (Task 1); `start-local-models` ds4 handling (Task 6) for the shared auto-restart path.
- Produces: `DS4_STUCK_LOADING_S = 300`; instance state `self.ds4_present`, `self.ds4_ok`, `self.ds4_loading`, `self.ds4_port`, `self.ds4_url`; `_restart_ds4(sender)` menu callback; `DS4_URL` in the profile server's env. Design note: ds4 is deliberately NOT added to the pre-update health snapshot (`_auto_update`, lines 3120-3132) — a ~70s ds4 cold load overlapping the post-update health check would false-flag healthy updates, and the snapshot only feeds a notification. The tailscale serve tuple (lines 1870-1871) is NOT touched: 8002 stays internal.

- [ ] **Step 1: Write the failing test**

Menubar service monitoring is thread/rumps-bound; the unit-testable seam is the constant and the classifier already covered. Add a guard that the watchdog threshold is the ds4-specific one (a copy-paste of the MLX 60s would kill every legitimate ~70s load — the exact bug the spec calls out):

```python
class TestDs4Menubar:
    def test_ds4_watchdog_threshold_is_generous(self):
        """ds4's cold load is ~70s. The MLX watchdog (60s) copied here
        would SIGKILL every legitimate load, forever. 300s matches the
        readiness deadline in start-local-models and _restart_ds4."""
        assert menubar.DS4_STUCK_LOADING_S == 300
        assert menubar.DS4_STUCK_LOADING_S > 70
```

Add to `tests/test_core.py` after `TestDs4Constants`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py::TestDs4Menubar -v`
Expected: FAIL — `AttributeError: module 'menubar' has no attribute 'DS4_STUCK_LOADING_S'`.

- [ ] **Step 3: Implement in `app/menubar.py`**

3a. Import: line 229 currently `from lib.models import CLAUDE_CONFIG_FILE  # early import for MCP_TOOLS_FILE` — change to:

```python
from lib.models import CLAUDE_CONFIG_FILE, ds4_dir, ds4_installed  # early import for MCP_TOOLS_FILE
```

3b. Constants — after line 235 (`MLX_LOCAL = "http://localhost:8000"`), add:

```python
DS4_LOCAL = "http://localhost:8002"
DS4_STUCK_LOADING_S = 300   # ds4 cold load ~70s; NOT the MLX 60s watchdog
```

3c. Init state — after line 1326 (`self.mlx_port = self.conf["MLX_PORT"]`), add:

```python
        self.ds4_port = self.conf.get("DS4_PORT", "8002")
        self.ds4_url = f"http://localhost:{self.ds4_port}"
        self.ds4_present = ds4_installed()
```

and after line 1350 (`self.mlx_loading = False`), add:

```python
        self.ds4_ok = False
        self.ds4_loading = False
```

3d. Menu items — after the MLX menu block (lines 1388-1391), add:

```python
        self.menu_ds4 = rumps.MenuItem("ds4 …")
        self.menu_ds4_restart = rumps.MenuItem(
            "Restart ds4", callback=self._restart_ds4)
        self.menu_ds4.add(self.menu_ds4_restart)
```

and in the `menu_items` list (lines 1424-1438), change:

```python
        menu_items += [
            None,
            self.menu_ollama,
            self.menu_mlx,
            self.menu_mcp,
```

to:

```python
        menu_items += [
            None,
            self.menu_ollama,
            self.menu_mlx,
        ]
        if self.ds4_present:
            menu_items.append(self.menu_ds4)
        menu_items += [
            self.menu_mcp,
```

(The menu row exists ONLY on machines where install.sh provisioned ds4 — presence can't change mid-run without a reinstall, which restarts the app.)

3e. `_refresh_server_mode` — after the MLX stuck-loading watchdog block (after line 2266), add:

```python
        if self.ds4_present:
            self.ds4_ok = probe_service(self.ds4_url, 2)
            self.ds4_loading = (not self.ds4_ok
                                and process_is_running("ds4-server"))
            # ds4's cold load is ~70s — the MLX 60s threshold would kill
            # every legitimate load. 300s matches the readiness deadline.
            if self.ds4_loading:
                if not hasattr(self, '_ds4_loading_since'):
                    self._ds4_loading_since = time.time()
                elif time.time() - self._ds4_loading_since > DS4_STUCK_LOADING_S:
                    subprocess.run(["pkill", "-9", "-f", "ds4-server"],
                                   capture_output=True, timeout=3)
                    self.ds4_loading = False
                    del self._ds4_loading_since
            else:
                if hasattr(self, '_ds4_loading_since'):
                    del self._ds4_loading_since
```

and extend the auto-restart trigger (lines 2269-2275). Replace:

```python
        if (self.servers_started
                and ((not self.ollama_ok and not self.ollama_loading)
                     or (not self.mlx_ok and not self.mlx_loading))):
```

with:

```python
        if (self.servers_started
                and ((not self.ollama_ok and not self.ollama_loading)
                     or (not self.mlx_ok and not self.mlx_loading)
                     or (self.ds4_present and not self.ds4_ok
                         and not self.ds4_loading))):
```

(`_start_local_servers` runs `start-local-models`, which now relaunches ds4 — Task 6.)

3f. `_restart_ds4` — add after `_restart_mlx` (after line 2149):

```python
    def _restart_ds4(self, _):
        """Restart just ds4-server (glm-5.2). cwd-pinned to DS4_DIR — the
        binary resolves metal/flash_attn.metal relative to cwd. Cold load
        is ~70s, so the readiness poll runs up to 300s, not the 15s the
        other services use."""
        self.ds4_ok = False
        self.ds4_loading = True
        self._update_menu()
        def _do():
            try:
                subprocess.run(["pkill", "-f", "ds4-server"],
                               capture_output=True, timeout=5)
                time.sleep(2)
                subprocess.run(["pkill", "-9", "-f", "ds4-server"],
                               capture_output=True, timeout=3)
                if hasattr(self, '_ds4_log') and self._ds4_log and not self._ds4_log.closed:
                    self._ds4_log.close()
                self._ds4_log = open("/tmp/local-models-ds4-restart.log", "w")
                subprocess.Popen(
                    ["./ds4-server", "--metal", "--port", str(self.ds4_port),
                     "--ctx", "32768", "-m", "ds4flash.gguf"],
                    stdout=self._ds4_log, stderr=self._ds4_log,
                    cwd=str(ds4_dir()),
                    start_new_session=True)
                for _ in range(300):
                    time.sleep(1)
                    if probe_service(self.ds4_url, 2):
                        break
            except Exception as e:
                rumps.notification("Local Models", "ds4 restart failed", str(e))
            self.refresh(None)
        threading.Thread(target=_do, daemon=True).start()
```

3g. `_update_menu` — in the client-mode branch (lines 2467-2471), after `self.menu_mlx.hide()` add `self.menu_ds4.hide()`. In the server branch (after the MLX status block ending line 2497), add:

```python
            if self.ds4_present:
                self.menu_ds4.show()
                if self.ds4_ok:
                    self._styled_menu(self.menu_ds4, GRN, "ds4", "1 model")
                elif getattr(self, 'ds4_loading', False):
                    self._styled_menu(self.menu_ds4, YEL, "ds4", "loading…")
                else:
                    self._styled_menu(self.menu_ds4, RED, "ds4", down_detail)
                self.menu_ds4_restart.set_callback(self._restart_ds4)
```

("1 model" is correct and static: ds4 serves exactly the one pinned GGUF; there is no dynamic list to pluralize.)

3h. `_copy_diagnostics` (lines 1898-1914) — replace the whole `lines = [...]` literal (splitting it so ds4 appears after the MLX line, without brittle index arithmetic):

```python
        lines = [
            f"Super Puppy {self.app_version}",
            f"Mode: {self.mode}",
            f"Desktop: {self.desktop}",
            f"Force local: {self.force_local}",
            f"Remote reachable: {self.remote_reachable}",
            f"Desktop IP: {self.desktop_ip}",
            f"Desktop FQDN: {getattr(self, 'desktop_fqdn', '')}",
            f"Ollama: {'up' if self.ollama_ok else 'down'}",
            f"MLX: {'up' if self.mlx_ok else 'down'}",
        ]
        if self.ds4_present:
            lines.append(f"ds4: {'up' if self.ds4_ok else 'down'}")
        lines += [
            f"MCP process: {'alive' if mcp_alive else 'dead'}",
            f"MCP models: {len(self.mcp_models)}",
            f"Ollama models: {len(self.ollama_models)}",
            f"MLX models: {len(self.mlx_models)}",
            f"RAM: {self.ram_gb} GB",
            f"TS hostname: {self.ts_hostname}",
        ]
```

3i. Profile-server env injection — after line 2586 (`env["MLX_URL"] = ...`), add:

```python
        # localhost always: 8002 is never tailscale-served; in client mode
        # the profile server proxies glm-5.2 requests to the desktop anyway.
        env["DS4_URL"] = self.ds4_url
```

Apply the same one-line addition inside the orphan-respawn `Popen` env path only if it builds a separate env (it reuses `env` — no extra change needed; verify by reading lines 2620-2625).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py tests/test_deployment.py tests/test_menu_refresh.py -v`
Expected: all PASS (`test_menu_refresh.py` exercises `_update_menu` paths — if it constructs instances without `ds4_present`, add `inst.ds4_present = False` to its fixture the same way `test_deployment.py`'s `app_instance` sets `ollama_ok`; check with the failure output).

- [ ] **Step 5: Live verification (dev machine)**

Restart the menubar app (`launchctl kickstart -k gui/$(id -u)/com.local-models.menubar`). On the 512GB box: the ds4 row appears with a green dot and "1 model" once loaded; "Copy Diagnostics" contains a `ds4: up` line. On the laptop: no ds4 row. Take a screenshot of the open menu and inspect it with `local_vision` (per the Visual Changes rule) — confirm the ds4 row renders with a status dot, not raw text.

- [ ] **Step 6: Docs (same commit)**

`CLAUDE.md` Menu Bar Features (line 154): change `- **Service status** — green/yellow/red dots for Ollama, MLX, MCP.` to `- **Service status** — green/yellow/red dots for Ollama, MLX, ds4 (512GB tier only), MCP.` Key files table (after the Menu bar log row, line 99): add

```markdown
| ds4 logs | `/tmp/local-models-ds4.log` (launch), `/tmp/local-models-ds4-restart.log` (menu restart) |
```

`docs/architecture.md` line 24 (menu bar app description): change `Manages the lifecycle of Ollama, MLX, MCP, and profile server processes.` to `Manages the lifecycle of Ollama, MLX, ds4 (512GB tier), MCP, and profile server processes.`

`README.md` line 138 (Menu Bar App bullets): change `- **Status** — Ollama/MLX running or down, MCP configured or not` to `- **Status** — Ollama/MLX/ds4 running or down, MCP configured or not`.

- [ ] **Step 7: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add app/menubar.py tests/test_core.py tests/test_menu_refresh.py CLAUDE.md
git commit -m "feat(menubar): monitor ds4-server with restart and a 300s load watchdog"
```

---

### Task 9: Config cut-over — remove glm-5.2 from the MLX config, migrate user configs, bump PROFILES_VERSION, fix the two hard-breaking tests

**Files:**
- Modify: `config/mlx-server/config.yaml` (lines 57-63)
- Create: `bin/migrate-mlx-config.py`
- Modify: `bin/post-update.sh` (after the MLX config merge loop, line 113)
- Modify: `lib/models.py` (line 429, `PROFILES_VERSION`)
- Test: `tests/test_core.py` (line 623 `test_warm_models_bare_names_are_mlx_served`), `tests/test_profile_server.py` (lines 510-570 glm-5.2 trio), `tests/test_deployment.py` (new migration tests)
- Docs: `README.md` (MLX Models section line 254), `CLAUDE.md` (line 20)

**Interfaces:**
- Consumes: `DS4_MODEL_NAME` (Task 1); warm-ping `mlx_served` parameter (Task 7).
- Produces: `bin/migrate-mlx-config.py <config.yaml> <served_model_name>` — exit 0 whether or not an entry was removed (idempotent), exit 2 on usage error; function `remove_served_model(text: str, served_name: str) -> tuple[str, bool]` for tests. `PROFILES_VERSION == 32`.

**This is the task that breaks tests if split** — the yaml edit, the migration, and the test fixes land in ONE commit so the suite never goes red.

- [ ] **Step 1: Write the failing tests for the migration script**

Add to `tests/test_deployment.py` (it already imports `subprocess`, `json`, `Path`; add `import importlib.util` at the top if absent):

```python
# ---------------------------------------------------------------------------
# bin/migrate-mlx-config.py — glm-5.2 → ds4 one-shot migration
# ---------------------------------------------------------------------------

def _load_migrate_module():
    path = Path(__file__).parent.parent / "bin" / "migrate-mlx-config.py"
    spec = importlib.util.spec_from_file_location("migrate_mlx_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SAMPLE_YAML = """\
server:
  host: "127.0.0.1"
  port: 8000

models:
  # Tiny model to verify server works
  - model_path: mlx-community/Llama-3.2-3B-Instruct-4bit
    model_type: lm
    served_model_name: llama-3b
    context_length: 8192

  # Frontier (512GB tier) — GLM-5.2, ~418GB at 4-bit
  - model_path: mlx-community/GLM-5.2-4bit
    model_type: lm
    served_model_name: glm-5.2
    context_length: 131072
    on_demand: true
    on_demand_idle_timeout: 300

  # User-added custom model — must survive untouched
  - model_path: mlx-community/My-Custom-7B
    model_type: lm
    served_model_name: my-custom
    context_length: 4096
"""


class TestMigrateMlxConfig:
    def test_removes_glm52_block_and_its_comment(self):
        mod = _load_migrate_module()
        out, removed = mod.remove_served_model(SAMPLE_YAML, "glm-5.2")
        assert removed is True
        assert "glm-5.2" not in out
        assert "GLM-5.2-4bit" not in out
        assert "Frontier (512GB tier)" not in out          # comment gone too

    def test_preserves_other_entries_including_user_custom(self):
        mod = _load_migrate_module()
        out, _ = mod.remove_served_model(SAMPLE_YAML, "glm-5.2")
        assert "served_model_name: llama-3b" in out
        assert "served_model_name: my-custom" in out
        assert "User-added custom model" in out

    def test_idempotent_second_run(self):
        """post-update.sh runs on every update; the second run must be a
        no-op, not a crash or a mangled file."""
        mod = _load_migrate_module()
        once, _ = mod.remove_served_model(SAMPLE_YAML, "glm-5.2")
        twice, removed = mod.remove_served_model(once, "glm-5.2")
        assert removed is False
        assert twice == once

    def test_cli_rewrites_file_and_exits_zero(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(SAMPLE_YAML)
        script = Path(__file__).parent.parent / "bin" / "migrate-mlx-config.py"
        result = subprocess.run(
            ["python3", str(script), str(cfg), "glm-5.2"],
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert "glm-5.2" not in cfg.read_text()
        # And again — idempotent at the CLI level too.
        result = subprocess.run(
            ["python3", str(script), str(cfg), "glm-5.2"],
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_deployment.py::TestMigrateMlxConfig -v`
Expected: FAIL — `FileNotFoundError` loading `bin/migrate-mlx-config.py`.

- [ ] **Step 3: Create `bin/migrate-mlx-config.py`**

```python
#!/usr/bin/env python3
"""Remove a served model's entry from an mlx-openai-server config.yaml.

Usage: migrate-mlx-config.py <config.yaml> <served_model_name>

One-shot migration helper called by post-update.sh: its MLX config merge is
append-only, so retiring a model (glm-5.2 moved to the ds4 backend,
2026-07-22) needs an explicit removal or updated 512GB boxes double-serve
it — MLX claims the name first in discovery order and keeps 418GB of dead
weights pullable. Stdlib-only and text-based (the system python3 that
post-update.sh uses has no pyyaml): drops the matching model block — its
leading comment lines through its last body line. Idempotent: exits 0 and
leaves the file untouched when no entry matches.
"""

import re
import sys

_ENTRY_RE = re.compile(r"^\s*-\s*model_path:")
_SERVED_RE = re.compile(r"^\s*served_model_name:\s*(\S+)\s*$")


def remove_served_model(text: str, served_name: str) -> tuple[str, bool]:
    """Return (new_text, removed). Block boundaries mirror post-update.sh's
    merge: an entry starts at `- model_path:` (plus the run of comment lines
    directly above it) and ends at the next entry or EOF."""
    lines = text.split("\n")
    starts = [i for i, l in enumerate(lines) if _ENTRY_RE.match(l)]
    if not starts:
        return text, False

    for n, s in enumerate(starts):
        begin = s
        while begin > 0 and lines[begin - 1].strip().startswith("#"):
            begin -= 1
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        # A following entry's leading comments belong to IT, not to us.
        if n + 1 < len(starts):
            while end - 1 > s and lines[end - 1].strip().startswith("#"):
                end -= 1
        block = lines[s:end]
        matches = any(
            (m := _SERVED_RE.match(l)) and m.group(1) == served_name
            for l in block
        )
        if matches:
            new_text = "\n".join(lines[:begin] + lines[end:])
            new_text = re.sub(r"\n{3,}", "\n\n", new_text)
            return new_text, True
    return text, False


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: migrate-mlx-config.py <config.yaml> <served_model_name>",
              file=sys.stderr)
        return 2
    path, served = sys.argv[1], sys.argv[2]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new_text, removed = remove_served_model(text, served)
    if removed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        print(f"removed MLX entry for served model {served!r}")
    else:
        print(f"no MLX entry for {served!r} — nothing to do")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Then: `chmod +x bin/migrate-mlx-config.py`

- [ ] **Step 4: Run migration tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_deployment.py::TestMigrateMlxConfig -v`
Expected: PASS.

- [ ] **Step 5: Remove the glm-5.2 entry from the repo config and bump PROFILES_VERSION**

In `config/mlx-server/config.yaml`, delete lines 56-63 entirely (the blank line before the comment, through the end of the entry):

```yaml

  # Frontier (512GB tier) — GLM-5.2, ~418GB at 4-bit
  - model_path: mlx-community/GLM-5.2-4bit
    model_type: lm
    served_model_name: glm-5.2
    context_length: 131072
    on_demand: true
    on_demand_idle_timeout: 300
```

In `lib/models.py` line 429, replace:

```python
PROFILES_VERSION = 31  # bump to force-refresh preset profiles on all machines
```

with:

```python
PROFILES_VERSION = 32  # bump to force-refresh preset profiles on all machines
```

(DEFAULT_PROFILES task names are deliberately unchanged — `glm-5.2` stays the value for the 512gb text tasks; only its serving backend moved.)

- [ ] **Step 6: Wire the migration into `bin/post-update.sh`**

After the MLX config merge loop (after line 113, `done`), insert:

```bash
# One-shot migration (2026-07 ds4 ship): glm-5.2 is served by ds4 on the
# 512GB tier now. The merge above is append-only and will never remove the
# user's old glm-5.2 MLX entry, which would double-serve the name (MLX
# claims it first in discovery order) and keep 418GB of dead weights.
# Failure-tolerant on purpose: post-update.sh failing rolls back the whole
# update (menubar._auto_update), and a cosmetic yaml migration must never
# do that.
if [ "$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')" -ge 512 ] \
   && [ -f "$MLX_DIR/config.yaml" ]; then
    python3 "$REPO_DIR/bin/migrate-mlx-config.py" "$MLX_DIR/config.yaml" glm-5.2 \
        && log "MLX config migration checked (glm-5.2 → ds4)" \
        || log "WARNING: glm-5.2 MLX-entry migration failed (non-fatal)"
fi
```

- [ ] **Step 7: Fix the two hard-breaking tests + repoint stale glm-5.2 examples (same commit)**

7a. `tests/test_core.py` lines 623-642 — replace `test_warm_models_bare_names_are_mlx_served` wholesale:

```python
    def test_warm_models_bare_names_are_mlx_or_ds4_served(self):
        """Every bare-name (no ':' and no '/') warm model in a shipped preset
        must be either a served_model_name in config/mlx-server/config.yaml
        or the ds4-served model. This guards warm_ping_targets' string-shape
        heuristic: a bare name in neither set would silently ping the wrong
        backend (or no backend at all)."""
        import yaml
        from lib.models import DS4_MODEL_NAME
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "mlx-server" / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        served = {m["served_model_name"] for m in cfg["models"]} | {DS4_MODEL_NAME}
        for name, prof in menubar.DEFAULT_PROFILES["profiles"].items():
            tasks = prof["tasks"]
            for key in prof.get("warm", []):
                model = tasks[key]
                if ":" in model or "/" in model:
                    continue  # ollama tag or HF repo — fine, warm_ping_targets handles them
                assert model in served, (
                    f"profile {name!r} warm bare-name {model!r} "
                    f"is neither an MLX served_model_name nor ds4-served"
                )

    def test_512gb_warm_ping_skips_glm52(self):
        """The concrete ship assertion: with the real repo yaml (glm-5.2
        removed) and the real 512gb preset, warm pings no longer target
        glm-5.2 — it dropped out naturally by leaving the served set."""
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "mlx-server" / "config.yaml"
        served = {m["served_model_name"]
                  for m in yaml.safe_load(cfg_path.read_text())["models"]}
        data = {"active": "512gb", "profiles": menubar.DEFAULT_PROFILES["profiles"]}
        targets = dict(menubar.warm_ping_targets(data, mlx_served=served))
        assert "glm-5.2" not in targets
        assert targets.get("qwen3-embedding:8b") == "ollama"
```

7b. `tests/test_profile_server.py` lines 510-570 — the trio uses glm-5.2 as its *example* of an MLX-config-resolved model, which is now counterfactual (they pass only because they mock `_load_mlx_config`). Repoint them to a model that IS in the shipped yaml so the examples stay honest. In `test_memory_estimates_undownloaded_warm_model` (lines 510-526), `test_mlx_downloaded_model_sized_from_disk_not_name` (lines 528-541), and `test_pull_resolves_bare_mlx_served_name_to_hf_repo` (lines 543-570), replace every occurrence of:
- `"glm-5.2"` → `"qwen3.5-fast"`
- `"mlx-community/GLM-5.2-4bit"` → `"mlx-community/Qwen3.5-35B-A3B-4bit"`

and update the comment in `test_mlx_downloaded_model_sized_from_disk_not_name` from `# GLM-5.2-4bit has no parseable param count in its name and isn't in the` to `# A model path without a parseable param count that isn't in the` — keep the mocked `_hf_cache_bytes`/size values as they are (the sizes are arbitrary test fixtures). In `test_mlx_downloaded_model_sized_from_disk_not_name`, note `Qwen3.5-35B-A3B-4bit` DOES have a parseable count — so instead use the fictional path `"mlx-community/BigModel-4bit"` for that one test:

```python
    def test_mlx_downloaded_model_sized_from_disk_not_name(self):
        # A downloaded model whose name lacks a parseable param count and
        # isn't in the known-params table would size to 0 from the name.
        # It must be sized from its real on-disk bytes instead (else the
        # memory bar shows "0 bit" for it).
        with patch.object(ps, "_load_mlx_config",
                          return_value={"big-model": {"model_path": "mlx-community/BigModel-4bit"}}), \
             patch.object(ps, "_mlx_loaded_ids", return_value=set()), \
             patch.object(ps, "_hf_model_downloaded", return_value=True), \
             patch.object(ps, "_hf_cache_bytes", return_value=180 * 10**9), \
             patch.object(ps, "_mlx_model_has_vision", return_value=False):
            out = ps._fetch_mlx_models(existing=set())
        assert out["big-model"]["disk_bytes"] == 180 * 10**9
        assert out["big-model"]["vram_bytes"] == 180 * 10**9
```

For the other two tests, the mocked `_load_mlx_config` / model-name swap to `qwen3.5-fast` + `mlx-community/Qwen3.5-35B-A3B-4bit` is a pure find-replace within those test bodies.

- [ ] **Step 8: Run the whole unit suite**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS — in particular `tests/test_core.py -k warm_models` and the repointed profile-server trio. Also confirm `TestPostUpdateConfigRefs::test_mlx_server_config_refs_exist` still passes (config.yaml still exists; only one entry left).

- [ ] **Step 9: Docs (same commit)**

`CLAUDE.md` line 20: `- MLX models marked \`on_demand: true\` download on first use and unload after idle timeout.` — append: `glm-5.2 is NOT one of them anymore: it's served by ds4 (always resident) on the 512GB tier.`

`CLAUDE.md` line 91 (Key files table, MLX server config row): change `(user-writable, survives updates)` to `(user-writable, survives updates; one-shot exception — the ds4 migration removes a leftover glm-5.2 entry on 512GB machines)`.

`README.md` line 254 (MLX Models section) — after `Your edits persist across auto-updates.` append: `One exception: the retired glm-5.2 MLX entry is removed once on 512GB machines by the ds4 migration (it moved to the ds4 backend).`

- [ ] **Step 10: Commit**

```bash
git add config/mlx-server/config.yaml bin/migrate-mlx-config.py bin/post-update.sh lib/models.py \
        tests/test_core.py tests/test_profile_server.py tests/test_deployment.py CLAUDE.md README.md
git commit -m "feat(config): move glm-5.2 serving from MLX to ds4 with user-config migration"
```

---

### Task 10: install.sh — ds4 provisioning, patch retirement, uninstallers

**Files:**
- Modify: `install.sh` (glm52-patch block lines 642-650; `MISSING_RUNTIMES` lines 667-691; served-name resolution heredoc lines 812-834; uninstall section lines 46-132)
- Delete: `bin/apply-mlx-glm52-patch.sh`
- Modify: `uninstall.sh` (stop-services lines 22-29; summary lines 104-120)
- Test: `tests/test_deployment.py`
- Docs: `CLAUDE.md` (line 23), `README.md` (lines 235-239, 282), `docs/troubleshooting.md` (replace lines 151-205)

**Interfaces:**
- Consumes: `set_conf` (install.sh line 136), `RAM_CHECK` (line 625), Task 1's network.conf keys, Task 6's launch contract (`$DS4_DIR/ds4-server`, `ds4flash.gguf` symlink).
- Produces: a provisioned `$DS4_DIR` on 512GB machines: pinned checkout (`bd89932`), built `ds4-server`, `gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf`, `ds4flash.gguf` symlink, and `DS4_DIR` written to network.conf.

- [ ] **Step 1: Write the failing retirement-guard test**

Add to `tests/test_deployment.py`:

```python
class TestGlm52PatchRetired:
    """The pinned mlx-lm patch is retired by the ds4 ship. A dangling
    invocation of the deleted script would hard-fail install.sh on 512GB
    machines; a dangling doc reference sends users to a runbook that no
    longer exists."""

    def test_patch_script_deleted_and_unreferenced(self):
        repo = Path(__file__).parent.parent
        assert not (repo / "bin" / "apply-mlx-glm52-patch.sh").exists(), \
            "bin/apply-mlx-glm52-patch.sh should be deleted (retired by ds4)"
        offenders = []
        for sub in ("install.sh", "uninstall.sh", "CLAUDE.md", "README.md"):
            if "apply-mlx-glm52-patch" in (repo / sub).read_text():
                offenders.append(sub)
        for md in (repo / "docs").glob("*.md"):
            if "apply-mlx-glm52-patch" in md.read_text():
                offenders.append(str(md.relative_to(repo)))
        assert not offenders, f"stale glm52-patch references: {offenders}"

    def test_install_provisions_ds4_pinned(self):
        """install.sh must clone the PINNED commit (engine is weeks old;
        an unpinned clone ships whatever antirez pushed last night)."""
        text = (Path(__file__).parent.parent / "install.sh").read_text()
        assert "bd89932" in text
        assert "antirez/ds4" in text
        assert "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf" in text
        assert "ds4flash.gguf" in text
```

(Deliberately a text-level guard: install.sh is interactive bash that can't run in CI; this pins the retirement + pin invariants that a refactor could silently drop. The provisioning itself is verified live in Step 6.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_deployment.py::TestGlm52PatchRetired -v`
Expected: FAIL — the patch script still exists and install.sh doesn't mention ds4.

- [ ] **Step 3: Replace the patch block with ds4 provisioning in `install.sh`**

Replace lines 642-650:

```bash
# GLM-5.2 (512gb tier) needs two mlx-lm model files from unmerged upstream
# PR ml-explore/mlx-lm#1463 — released mlx-lm can't load its shared-indexer
# layout. The script is idempotent and self-disables once upstream ships
# support. See docs/troubleshooting.md ("glm-5.2 fails to load").
if [ "$RAM_CHECK" -ge 512 ] && command -v mlx-openai-server > /dev/null; then
    echo "  Applying GLM-5.2 mlx-lm patch..."
    "$SCRIPT_DIR/bin/apply-mlx-glm52-patch.sh" \
        || echo "  Warning: GLM-5.2 patch failed (glm-5.2 won't load; other models unaffected)"
fi
```

with:

```bash
# ds4 (512gb tier): serves glm-5.2 from a Q2K GGUF — 244GiB resident vs the
# retired 390GB mlx-openai-server path, and no more pinned mlx-lm patch.
# Pinned commit on the glm5.2 branch: the engine is weeks old, so we ship
# exactly what was verified (tool-calling round-trip, 15.5 tok/s, strict
# JSON quirk documented in docs/troubleshooting.md).
DS4_COMMIT="bd89932"
DS4_GGUF_REPO="antirez/GLM-5.2-GGUF"
DS4_GGUF_FILE="GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"
if [ "$RAM_CHECK" -ge 512 ]; then
    DS4_DIR=$(grep '^DS4_DIR=' "$HOME/.config/local-models/network.conf" 2>/dev/null \
        | head -1 | cut -d= -f2- | tr -d '"' || true)
    DS4_DIR="${DS4_DIR:-$HOME/.local/share/super-puppy/ds4}"

    # Reuse a dev checkout: if ~/experiments/ds4 already has the binary and
    # the 244GiB GGUF, symlink it — never re-download 244GiB.
    if [ ! -e "$DS4_DIR" ] && [ -x "$HOME/experiments/ds4/ds4-server" ] \
       && [ -f "$HOME/experiments/ds4/gguf/$DS4_GGUF_FILE" ]; then
        mkdir -p "$(dirname "$DS4_DIR")"
        ln -sfn "$HOME/experiments/ds4" "$DS4_DIR"
        echo "  ds4: reusing existing checkout at ~/experiments/ds4"
    fi

    if [ ! -x "$DS4_DIR/ds4-server" ]; then
        echo "  Building ds4 (glm-5.2 engine, pinned $DS4_COMMIT)..."
        if [ ! -d "$DS4_DIR/.git" ]; then
            mkdir -p "$DS4_DIR"
            git clone --branch glm5.2 https://github.com/antirez/ds4 "$DS4_DIR" \
                || echo "  Warning: ds4 clone failed (glm-5.2 will be unavailable)"
        fi
        if [ -d "$DS4_DIR/.git" ]; then
            git -C "$DS4_DIR" fetch --quiet origin glm5.2 || true
            git -C "$DS4_DIR" checkout --quiet "$DS4_COMMIT" \
                || echo "  Warning: pinned ds4 commit $DS4_COMMIT not found"
            (cd "$DS4_DIR" && make ds4-server) \
                || echo "  Warning: ds4 build failed (glm-5.2 will be unavailable)"
        fi
    fi

    # Weights: 244GiB GGUF, with a disk-space precheck (none of the existing
    # pull paths can see this file — it's not an HF snapshot layout we scan).
    if [ -x "$DS4_DIR/ds4-server" ] && [ ! -f "$DS4_DIR/gguf/$DS4_GGUF_FILE" ]; then
        if ! command -v hf > /dev/null; then
            brew install hf 2>/dev/null || true
        fi
        FREE_GB=$(df -g "$HOME" | awk 'NR==2 {print $4}')
        if [ "${FREE_GB:-0}" -lt 260 ]; then
            echo "  Warning: only ${FREE_GB}GB free — the glm-5.2 GGUF needs ~250GB."
            echo "           Free space and re-run install.sh to download it."
        elif command -v hf > /dev/null; then
            echo "  Downloading glm-5.2 GGUF (~244GiB — this takes a while)..."
            hf download "$DS4_GGUF_REPO" "$DS4_GGUF_FILE" --local-dir "$DS4_DIR/gguf" \
                || echo "  Warning: glm-5.2 GGUF download failed — re-run install.sh to retry."
        else
            echo "  Warning: hf CLI unavailable — cannot download the glm-5.2 GGUF."
        fi
    fi

    if [ -f "$DS4_DIR/gguf/$DS4_GGUF_FILE" ]; then
        ln -sfn "gguf/$DS4_GGUF_FILE" "$DS4_DIR/ds4flash.gguf"
    fi
    set_conf "DS4_DIR" "\"$DS4_DIR\""
fi
```

- [ ] **Step 4: MISSING_RUNTIMES + served-name resolution + uninstall sections**

4a. In the `MISSING_RUNTIMES` block (line 670), after `command -v mlx-openai-server > /dev/null || MISSING_RUNTIMES+=("mlx-openai-server")`, add:

```bash
if [ "$RAM_CHECK" -ge 512 ]; then
    DS4_DIR_CHECK=$(grep '^DS4_DIR=' "$HOME/.config/local-models/network.conf" 2>/dev/null \
        | head -1 | cut -d= -f2- | tr -d '"' || true)
    [ -x "${DS4_DIR_CHECK:-$HOME/.local/share/super-puppy/ds4}/ds4-server" ] \
        || MISSING_RUNTIMES+=("ds4-server")
fi
```

and in `remediation_for` (lines 672-679), add a case before `*)`:

```bash
        ds4-server)        echo "re-run install.sh (clones antirez/ds4@bd89932 and runs make ds4-server)" ;;
```

4b. Served-name resolution (the third python heredoc, lines 812-834): glm-5.2 is a bare served-name in the 512gb profile but no longer resolves through the MLX config — without an exclusion the loop prints a misleading `NOTE: served-name 'glm-5.2' has no model_path` on every 512GB install. In that heredoc, replace:

```python
served = {m for m in profile.get('tasks', {}).values()
          if m and ':' not in m and '/' not in m}
```

with:

```python
served = {m for m in profile.get('tasks', {}).values()
          if m and ':' not in m and '/' not in m}
served.discard('glm-5.2')  # ds4-served: provisioned by the ds4 install step, not the MLX config
```

4c. Uninstall (`install.sh` lines 46-132): after line 54 (`sleep 1` in the stop block), add:

```bash
    pkill -f "ds4-server" 2>/dev/null || true
```

and after the config-dir removal block (line 120), add:

```bash
    # ds4 checkout + 244GiB GGUF
    DS4_DIR=$(grep '^DS4_DIR=' "$HOME/.config/local-models/network.conf" 2>/dev/null \
        | head -1 | cut -d= -f2- | tr -d '"' || true)
    DS4_DIR="${DS4_DIR:-$HOME/.local/share/super-puppy/ds4}"
    if [ -e "$DS4_DIR" ]; then
        echo ""
        echo "ds4 directory: $DS4_DIR (checkout + glm-5.2 GGUF, ~244GiB)"
        echo "  Not removed automatically. Delete it manually if you want:"
        echo "  rm -rf \"$DS4_DIR\""
    fi
```

(Note: the config-dir prompt at lines 107-120 may already have deleted network.conf; that's why this reads the conf BEFORE falling back to the default — move this block ABOVE the config-dir removal prompt, i.e. insert after line 105, before the `CONFIG_DIR` block.)

4d. `uninstall.sh`: in the stop-services block (after line 28, `pkill -f "local-models-server.py"`), add:

```bash
pkill -f "ds4-server" 2>/dev/null || true
```

and in the summary "NOT removed" list (line 110, after the HuggingFace line), add:

```bash
echo "  - ds4 checkout + glm-5.2 GGUF (~244GiB) at \${DS4_DIR:-~/.local/share/super-puppy/ds4}"
```

and to the "To remove downloaded models" block (line 120), add:

```bash
echo "  rm -rf ~/.local/share/super-puppy/ds4   # ds4 checkout + 244GiB GGUF"
```

- [ ] **Step 5: Delete the patch script and scrub references**

```bash
git rm bin/apply-mlx-glm52-patch.sh
```

Scrub references (each is edited in Step 7's doc pass; this step is the code-level scrub):
- `install.sh` — done in Step 3.
- Verify: `grep -rn "apply-mlx-glm52-patch" --include="*.sh" --include="*.py" .` returns nothing.

- [ ] **Step 6: Run tests + live verification**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_deployment.py -v`
Expected: `TestGlm52PatchRetired` FAILS still on doc references (CLAUDE.md/README/troubleshooting edited next step) — proceed to Step 7, then re-run.

Also: `bash -n install.sh && bash -n uninstall.sh && echo SYNTAX-OK`.

Live (dev 512GB box): run `./install.sh` far enough to hit the ds4 step (or extract and run the block with `RAM_CHECK=512`): it must symlink `~/experiments/ds4` → `$DS4_DIR` without downloading, write `DS4_DIR` into network.conf, and leave `$DS4_DIR/ds4flash.gguf` resolving to the GGUF.

- [ ] **Step 7: Docs (same commit)**

7a. `CLAUDE.md` line 23 — replace the whole bullet:

```markdown
- **glm-5.2 (512gb tier) requires patched mlx-lm/mlx-openai-server.** Released mlx-lm can't load GLM-5.2's shared-indexer layout, and stock mlx-openai-server can't keep a 390GB on-demand model resident. `bin/apply-mlx-glm52-patch.sh` (run by install.sh on 512GB machines, idempotent, pinned to upstream PR ml-explore/mlx-lm#1463) fixes both; re-run it after any `uv tool upgrade mlx-openai-server`. Details: docs/troubleshooting.md.
```

with:

```markdown
- **glm-5.2 (512gb tier) is served by ds4, not MLX.** antirez/ds4 (glm5.2 branch, pinned `bd89932`, built by install.sh into `DS4_DIR`) serves the Q2K-routed GGUF (~244GiB resident vs 390GB on MLX — the co-residency OOM class is gone, and the old pinned mlx-lm patch is retired). Quirks encoded in code and docs/troubleshooting.md: launch needs `cwd=$DS4_DIR` (metal shader paths), responses need `json.loads(..., strict=False)` (unescaped control chars in reasoning_content), and the `enable_thinking` toggle must never be forwarded (broken: reasoning migrates into content). ds4 serializes requests — parallel fan-outs to glm-5.2 queue.
```

7b. `README.md` lines 235-239 — replace the footnote:

```markdown
\* glm-5.2 needs two mlx model files from an unmerged upstream PR plus two
mlx-openai-server fixes; `install.sh` applies them automatically on 512GB
machines via `bin/apply-mlx-glm52-patch.sh` (idempotent — re-run it after
any `uv tool upgrade mlx-openai-server`). See
[docs/troubleshooting.md](docs/troubleshooting.md) for details.
```

with:

```markdown
\* glm-5.2 is served by [ds4](https://github.com/antirez/ds4) (pinned
build, provisioned by `install.sh` on 512GB machines) from a ~244GiB Q2K
GGUF — always resident, OpenAI-compatible on localhost:8002. See
[docs/troubleshooting.md](docs/troubleshooting.md) for build/run details.
```

7c. `README.md` line 282 — replace the structure line:

```
│   ├── apply-mlx-glm52-patch.sh # Pinned mlx patches for glm-5.2 (512GB tier)
```

with:

```
│   ├── migrate-mlx-config.py    # One-shot user-config migration (glm-5.2 → ds4)
```

7d. `docs/troubleshooting.md` — replace the whole section at lines 151-205 (heading `## glm-5.2 fails to load: "Missing 285 parameters" (or unloads every 5 minutes)` through the line ending `...survives updates.`) with:

```markdown
## glm-5.2 (ds4) — build, run, and quirks

**Since 2026-07, glm-5.2 on the 512GB tier is served by [antirez/ds4](https://github.com/antirez/ds4)** (glm5.2 branch, pinned commit `bd89932`), not mlx-openai-server. The old pinned mlx-lm patch script is retired and deleted; if a doc tells you to run it, you're reading an old release.

**Layout** (default `~/.local/share/super-puppy/ds4`, override with `DS4_DIR` in `~/.config/local-models/network.conf`):

​```
$DS4_DIR/
├── ds4-server                 # built by `make ds4-server` at bd89932
├── metal/                     # metal shaders — resolved RELATIVE TO CWD
├── ds4flash.gguf              # symlink → gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
└── gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf   # 244GiB, hf download antirez/GLM-5.2-GGUF
​```

**Manual run** (what `start-local-models` does):

​```bash
cd "$DS4_DIR" && ./ds4-server --metal --port 8002 --ctx 32768 -m ds4flash.gguf
​```

### ds4-server exits instantly / "cannot open metal/flash_attn.metal"

You launched it from the wrong directory. ds4-server resolves its metal
shader sources **relative to the current working directory** — always `cd
"$DS4_DIR"` first (the service scripts and the menu bar restart do this).

### Chat requests intermittently crash with a JSON decode error

ds4's JSON encoder can emit **unescaped control characters** inside long
`reasoning_content`. Python's strict default parser raises on them. Super
Puppy's dispatchers parse ds4 responses with `json.loads(..., strict=False)`;
any new consumer of `:8002` must do the same (or sanitize). Worth reporting
upstream if you can reproduce a minimal case.

### "I disabled thinking but glm-5.2 still thinks" / reasoning shows up in the answer

`chat_template_kwargs.enable_thinking: false` is **broken on ds4**: the
server returns 200 but the model's reasoning simply moves from
`reasoning_content` into `content` (no think-block markers to strip).
Super Puppy never forwards the toggle to ds4 — glm-5.2 always thinks there.
Budget tokens accordingly (it can burn a small `max_tokens` entirely on
thinking; dispatchers fall back to surfacing `reasoning_content`).

### glm-5.2 is missing from model lists

glm-5.2 appears in discovery only while ds4 answers on `:8002` (same
semantics as MLX being down). Check `start-local-models --status`, the menu
bar's ds4 row, and `/tmp/local-models-ds4.log`. Cold load takes ~70s —
readiness probes allow up to 300s.

### Requests to glm-5.2 queue up

ds4 serializes requests (single live session). glm-5.2 is the 512GB tier's
general/reasoning/long-context workhorse, so parallel `local_dispatch`
fan-outs will queue — that's the accepted tradeoff for the ~146GB of memory
headroom vs the MLX path.

### Rolling back to the MLX path (one release grace)

The MLX path still works if you need it: re-add the glm-5.2 entry to
`~/.config/mlx-server/config.yaml` (`model_path: mlx-community/GLM-5.2-4bit`,
`served_model_name: glm-5.2`, `on_demand: true`), download the 418GB 4-bit
weights, and note you'd also need the retired mlx-lm patch from a pre-ds4
release. Practical only as a stopgap; the entry will not be re-removed
(the migration is one-shot per name).
```

(Remove the zero-width escapes around the inner code fences when writing.)

- [ ] **Step 8: Re-run tests, full suite, commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS, including `TestGlm52PatchRetired`.

```bash
git add install.sh uninstall.sh tests/test_deployment.py CLAUDE.md README.md docs/troubleshooting.md
git commit -m "feat(install): provision ds4 on the 512GB tier, retiring the mlx glm-5.2 patch"
```

---

### Task 11: UI — activity log backend tag + diagnostics service row

**Files:**
- Modify: `app/activity.html` (backend-tag CSS lines 131-141; active-dot ternary line 379)
- Modify: `app/diagnostics.html` (Services section lines 126-131)
- Modify: `app/profile-server.py` (`api_diagnostics` lines 3143-3213)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Consumes: `DS4_URL`, `ds4_installed` import (Task 4).
- Produces: `/api/diagnostics` JSON `services` gains `"ds4": true|false|null` (`null` = not installed on this machine; UI hides the row).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_profile_server.py`:

```python
class TestDiagnosticsDs4:
    def test_diagnostics_reports_ds4_when_installed(self, client):
        resp_ok = MagicMock()
        resp_ok.ok = True
        with patch.object(ps, "ds4_installed", return_value=True), \
             patch.object(ps.requests, "get", return_value=resp_ok), \
             patch.object(ps, "ollama_get", return_value=None), \
             patch.object(ps, "get_all_models", return_value={}):
            d = client.get("/api/diagnostics").get_json()
        assert d["services"]["ds4"] is True

    def test_diagnostics_ds4_null_when_not_installed(self, client):
        """Laptops never run ds4 — a permanently-red 'Down' row would be
        noise. null tells the UI to omit the row entirely."""
        with patch.object(ps, "ds4_installed", return_value=False), \
             patch.object(ps.requests, "get",
                          side_effect=ps.requests.ConnectionError("x")), \
             patch.object(ps, "ollama_get", return_value=None), \
             patch.object(ps, "get_all_models", return_value={}):
            d = client.get("/api/diagnostics").get_json()
        assert d["services"]["ds4"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py::TestDiagnosticsDs4 -v`
Expected: FAIL — `KeyError: 'ds4'`.

- [ ] **Step 3: Implement `api_diagnostics` probe**

In `app/profile-server.py`, after the `mcp_up` probe (lines 3169-3174), add:

```python
    ds4_up = None
    if ds4_installed():
        ds4_up = False
        try:
            resp = requests.get(f"{DS4_URL}/v1/models", timeout=2)
            ds4_up = resp.ok
        except Exception:
            pass
```

and in the returned `"services"` dict (lines 3199-3203), replace:

```python
        "services": {
            "ollama": ollama_up,
            "mlx": mlx_up,
            "mcp": mcp_up,
        },
```

with:

```python
        "services": {
            "ollama": ollama_up,
            "mlx": mlx_up,
            "ds4": ds4_up,
            "mcp": mcp_up,
        },
```

(`ds4_installed` was added to the import block in Task 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -v`
Expected: all PASS.

- [ ] **Step 5: diagnostics.html row**

In `app/diagnostics.html`, the Services section (lines 126-131) currently:

```html
      <div class="section">
        <div class="section-title">Services</div>
        <div class="row"><span class="label">Ollama</span><span class="value">${dot(svc.ollama)}</span></div>
        <div class="row"><span class="label">MLX Server</span><span class="value">${dot(svc.mlx)}</span></div>
        <div class="row"><span class="label">MCP Server</span><span class="value">${dot(svc.mcp)}</span></div>
      </div>
```

becomes:

```html
      <div class="section">
        <div class="section-title">Services</div>
        <div class="row"><span class="label">Ollama</span><span class="value">${dot(svc.ollama)}</span></div>
        <div class="row"><span class="label">MLX Server</span><span class="value">${dot(svc.mlx)}</span></div>
        ${svc.ds4 == null ? '' : `<div class="row"><span class="label">ds4 (glm-5.2)</span><span class="value">${dot(svc.ds4)}</span></div>`}
        <div class="row"><span class="label">MCP Server</span><span class="value">${dot(svc.mcp)}</span></div>
      </div>
```

(`svc.ds4 == null` intentionally uses loose equality — it matches both `null` and `undefined`, so old cached API responses degrade to "row hidden".)

- [ ] **Step 6: activity.html backend tag + dot color**

6a. Replace lines 136-141:

```css
.backend-ollama { background: #e8f5e9; color: #2e7d32; }
.backend-mlx { background: #e3f2fd; color: #1565c0; }
@media (prefers-color-scheme: dark) {
  .backend-ollama { background: #1b3a1e; color: #66bb6a; }
  .backend-mlx { background: #0d2744; color: #42a5f5; }
}
```

with:

```css
.backend-ollama { background: #e8f5e9; color: #2e7d32; }
.backend-mlx { background: #e3f2fd; color: #1565c0; }
.backend-ds4 { background: #f6ebf4; color: #8e4585; }
@media (prefers-color-scheme: dark) {
  .backend-ollama { background: #1b3a1e; color: #66bb6a; }
  .backend-mlx { background: #0d2744; color: #42a5f5; }
  .backend-ds4 { background: #3a2136; color: #ce93c4; }
}
```

6b. Replace line 379:

```javascript
      const color = r.backend === 'ollama' ? 'var(--green)' : 'var(--accent)';
```

with:

```javascript
      const color = r.backend === 'ollama' ? 'var(--green)'
                  : r.backend === 'ds4' ? '#b07aa1'
                  : 'var(--accent)';
```

(The history table at line 445 and the active-meta at line 385 already emit `backend-${r.backend}` — they pick up `.backend-ds4` with no change.)

- [ ] **Step 7: Visual verification**

On a machine with recent ds4 activity (or by temporarily inserting a fake `ds4` row into the activity DB), open the Activity Log and Diagnostics pages in both light and dark mode, screenshot each, and inspect with `local_vision`: the ds4 tag must render as a purple-family pill (readable contrast in both themes) and the diagnostics ds4 row must show a dot, not raw template text. On a laptop, confirm the diagnostics ds4 row is absent.

- [ ] **Step 8: Run the full unit suite, then commit**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"`
Expected: PASS.

```bash
git add app/activity.html app/diagnostics.html app/profile-server.py tests/test_profile_server.py
git commit -m "feat(ui): show the ds4 backend in activity and diagnostics views"
```

---

### Task 12: Smoke/correctness wiring — release-gate the live ds4 path

**Files:**
- Modify: `tests/_smoke_helpers.py` (env setup lines 49-55; `require_local_services` lines 99-104; `SKIP_SUBSTRINGS` lines 179-198)
- Modify: `tests/test_tools_smoke_everyday.py` (module setup line 23)
- Modify: `tests/test_tools_correctness.py` (new test)
- Docs: `CLAUDE.md` (Testing list lines 143-146)

**Interfaces:**
- Consumes: `lib.models.ds4_installed` (Task 1); the profile-server ds4 dispatch (Tasks 4-5); error string `"is ds4-server running"` (Tasks 2/5 — the profile-server path surfaces requests' `"Cannot connect to backend"`, already in SKIP_SUBSTRINGS; the MCP path surfaces the ds4-specific text).
- Produces: `DS4_URL` module constant and `require_ds4()` in `_smoke_helpers`; one `correctness`-marked live ds4 case so `bin/release.sh` (which runs `pytest -m "not slow and not e2e"`, INCLUDING `correctness`) gates releases on the ds4 path.

- [ ] **Step 1: Add the smoke-helper wiring**

1a. In `tests/_smoke_helpers.py`, after line 50 (`os.environ.setdefault("MLX_URL", ...)`), add:

```python
    os.environ.setdefault("DS4_URL", "http://localhost:8002")
```

and at module level after the `_reachable` helper (line 96), add:

```python
DS4_URL = os.environ.get("DS4_URL", "http://localhost:8002")


def require_ds4():
    """For suites exercising the ds4-served glm-5.2 path.

    Machines without a ds4 install (below the 512GB tier) skip. Machines
    WITH an install FAIL loudly when the server doesn't answer: on the box
    that serves glm-5.2, a ds4 outage must never hide as a model-not-pulled
    skip — that's exactly how backend outages went unnoticed before.
    """
    from lib.models import ds4_installed
    if not ds4_installed():
        # allow_module_level: the everyday suite calls this at import time,
        # like require_local_services above.
        pytest.skip("ds4 not installed on this machine (512GB tier only)",
                    allow_module_level=True)
    if not _reachable(f"{DS4_URL}/v1/models"):
        # At module level this surfaces as a collection error — loud on
        # purpose: the box that serves glm-5.2 must not hide a ds4 outage.
        pytest.fail(
            f"ds4-server is installed but not answering at {DS4_URL} — "
            "start it (start-local-models) or check /tmp/local-models-ds4.log")
```

1b. Extend `SKIP_SUBSTRINGS` (lines 179-198): after the line `"connectionerror",`, add:

```python
    "is ds4-server running",
```

- [ ] **Step 2: Wire the everyday suite**

In `tests/test_tools_smoke_everyday.py`, replace lines 17-23:

```python
from tests._smoke_helpers import (
    CHAT_CASES, FIXTURE_CASES,
    client, require_local_services, run_chat_case, run_fixture_case, smoke_tmp,
)

# Skip the module at collection time if local services aren't up.
require_local_services()
```

with:

```python
from tests._smoke_helpers import (
    CHAT_CASES, FIXTURE_CASES,
    client, require_ds4, require_local_services, run_chat_case,
    run_fixture_case, smoke_tmp,
)

# Skip the module at collection time if local services aren't up.
require_local_services()
# The 512gb tier's chat tasks route to ds4-served glm-5.2. Skip below the
# 512GB tier; FAIL (not skip) when ds4 is installed but down.
require_ds4()
```

- [ ] **Step 3: Add the correctness case (release gate)**

In `tests/test_tools_correctness.py`, extend the `_smoke_helpers` import (lines 25-29) with `require_ds4`, then add after `test_chat_follows_a_basic_instruction` (line 113):

```python
def test_ds4_glm52_chat_correctness(client):
    """glm-5.2 pinned explicitly (not via _model_for): the release gate for
    the whole ds4 dispatch chain — profile-server → :8002 → strict-tolerant
    JSON parse → content/reasoning_content extraction. glm-5.2 always
    thinks on ds4, so a real response exercises exactly the reasoning-heavy
    payloads where the control-char encoder bug bites. Skips below the
    512GB tier; FAILS if ds4 is installed but down (require_ds4)."""
    require_ds4()
    assert_tool_output_contains(
        client, tool="general", model="glm-5.2",
        expect_any=["kumquat"],
        prompt="Reply with exactly one word: kumquat")
```

- [ ] **Step 4: Verify**

On the laptop: `uv run --with pytest --with flask --with pyyaml --with requests --with pillow pytest tests/test_tools_correctness.py::test_ds4_glm52_chat_correctness -m correctness -v` → SKIP (`ds4 not installed`).
On the 512GB box with ds4 running: same command → PASS with a real glm-5.2 round-trip; with ds4 stopped → FAIL with the "installed but not answering" message.
Then the unit suite: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"` → PASS.

- [ ] **Step 5: Docs (same commit)**

`CLAUDE.md` Testing list — line 144 (`test_tools_smoke_everyday.py` bullet): append `Requires ds4 on the 512GB tier (fails loud if ds4-server is installed but down).` After line 146's `test_error_handling.py` bullet, no change; instead extend the `test_tools_correctness`-adjacent knowledge: the Testing intro already documents `correctness` via pyproject — add to line 144's bullet or add a bullet after it:

```markdown
- `tests/test_tools_correctness.py` — release-gated output-correctness cases (`correctness` marker, run by `bin/release.sh`), including one pinned ds4 glm-5.2 chat case so releases gate on the live ds4 path.
```

- [ ] **Step 6: Commit**

```bash
git add tests/_smoke_helpers.py tests/test_tools_smoke_everyday.py tests/test_tools_correctness.py CLAUDE.md
git commit -m "test(smoke): gate releases on the live ds4 glm-5.2 path"
```

---

### Task 13: Final verification + checkpoint

**Files:**
- Modify: `tasks.md` (checkpoint)

No new code — this is the ship gate.

- [ ] **Step 1: Full unit + deployment suite**

```bash
uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -q -m "not smoke and not e2e"
```
Expected: PASS, zero failures.

- [ ] **Step 2: Live end-to-end on the 512GB box**

```bash
start-local-models                       # ds4 section launches, ready ≤300s
curl -s http://localhost:8002/v1/models  # 200
start-local-models --status              # ds4-server: running (glm-5.2)
uv run --with pytest --with flask --with pyyaml --with requests --with pillow \
    pytest tests/test_tools_correctness.py::test_ds4_glm52_chat_correctness -m correctness -v
```
Expected: correctness PASS. Then in a Claude Code session: `local_models_status` lists `glm-5.2 | 380B (32B active) | 128K ctx | [ds4]`; `local_generate` with `model="glm-5.2"` returns coherent output. Menu bar shows the ds4 row green. Activity Log shows the purple ds4 tag on those requests.

- [ ] **Step 3: Migration rehearsal**

On the 512GB box (whose user yaml still has glm-5.2): run `bin/post-update.sh` manually; verify `~/.config/mlx-server/config.yaml` no longer contains `served_model_name: glm-5.2`, whisper/ui-venus entries intact, and a second run logs `nothing to do`. Restart services; confirm the profile server's `/api/models` shows glm-5.2 with `"backend": "ds4"` (not mlx).

- [ ] **Step 4: Client-mode spot check (laptop)**

With the desktop serving: laptop Claude Code calls `local_generate(model="glm-5.2")` through the desktop's MCP (8100) and gets a response — proving 8002-internal-only routing works. `tailscale serve status` on the desktop must NOT list 8002.

- [ ] **Step 5: Checkpoint + wrap up**

Update `tasks.md` with Last Known Good State (branch `feat/ds4-backend`, all 12 commits, live gates passed) and Next Step (merge via PR; cut a release with `bin/release.sh` — remember a shipped tag goes live fleet-wide within ~2 min, and `bin/release.sh` now exercises the ds4 correctness case). Commit:

```bash
git add tasks.md
git commit -m "chore: checkpoint ds4 backend port"
```

Then follow superpowers:finishing-a-development-branch to merge/PR.

---

## Self-review notes (spec coverage)

- Audit §Design decisions 1-6: discovery/metadata seam (Tasks 1/3/4), port exposure internal-only (Tasks 6/8 non-changes, Task 13 step 4), residency model + memory math + watchdog (Tasks 4/8), build ownership (Task 10; post-update deliberately does NOT rebuild ds4 — the pin is fixed for this ship and a build failure there would roll back whole updates; migration only), weights location/download + disk precheck + served-name resolution (Task 10), migration + PROFILES_VERSION (Task 9).
- Audit §Ship gates: tool-calling verified (pre-cleared); think-toggle encoded (Tasks 2/5 + tests + troubleshooting doc); concurrency documented (CLAUDE.md line 23 rewrite, troubleshooting); stability → pinned commit + MLX fallback documented for one release (troubleshooting rollback section); release gating → Task 12 correctness case; strict-JSON → Tasks 2/5 with regression tests.
- Audit §Touchpoints: every table row is mapped to a task above except verified no-ops (listed under the file map). `app/profiles.html` pull/missing prompts are handled server-side (Task 4's `_check_missing_models` exclusion — the UI only renders what the API reports missing). `app/menubar.py:3120-3132` health snapshot: deliberate non-change, documented in Task 8.
- Audit §Docs line refs reviewed with no change needed: `CLAUDE.md:112` (vision section — ds4 is not vision-capable, nothing to add), `CLAUDE.md:121` (DEFAULT_PROFILES bullet — task names unchanged), `README.md:285` (config/ structure line — still accurate), `docs/architecture.md:75` (modes table — client routing unchanged) and `:125` (key-files — unchanged paths).
- Hard test breakages fixed in the same commit as the yaml change (Task 9). New coverage: `_chat_url("ds4")`, chat dispatch (both servers), discovery metadata (both servers), pick_model fallback, GPU key, network.conf template/repair, e2e 8002 + `/gpu` ds4 key, `_smoke_helpers` DS4 wiring, correctness gate.
