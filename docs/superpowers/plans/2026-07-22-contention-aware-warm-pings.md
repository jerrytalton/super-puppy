# Contention-Aware Warm Pings Implementation Plan (rev 2, post red-team)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Warm-model keep-alive pings never force a model load into a contended machine; refreshing already-resident models stays cheap and unconditional, and residency is allowed to lapse under pressure.

**Architecture:** The menubar's 240s warm tick (`app/menubar.py:_on_warm_tick` → `_ping_warm`) today force-loads every warm model unconditionally; on 2026-07-22 that contributed to a hard-reset OOM. Rev 1 of this plan gated rounds on `vm_stat` free pages — red-teamed and rejected: free+speculative is ~1GB on a *healthy* loaded 512GB box (reclaimable memory lives in inactive/purgeable), so the gate would trip permanently and create a 390GB unload/reload thrash loop, and it never compared free memory to what a ping would load. Rev 2 is residency-first: **refresh pings to already-resident models are always sent** (they can't cause the OOM class); **loading an evicted warm model is a gated commitment** requiring — no SP inference in flight (MCP `/gpu`), no foreign models resident on either backend (`/api/ps` for Ollama, `/v1/models` for MLX — evidence of third-party or Playground work), kernel pressure normal (`kern.memorystatus_vm_pressure_level` == 1), and available memory (`vm_stat` free+speculative+inactive+purgeable) ≥ model size + 16GB headroom. Model sizes come from `/api/ps`//`api/tags` (Ollama) and the HF cache dir (MLX). All decision logic is pure module-level functions covered by `tests/test_core.py`.

**Tech Stack:** Python 3.12 (menubar, PEP 723), stdlib only (`urllib.request`, `subprocess`, `json`), pytest.

## Global Constraints

- Do NOT gate anything on `vm_stat` free+speculative alone (red-team finding 1: ~1GB on a healthy loaded box) or on `memory_pressure -Q` free% (model-playbook 2026-07-22: reported 97% on a dying machine). Availability = free + speculative + inactive + purgeable pages; distress = `sysctl -n kern.memorystatus_vm_pressure_level` > 1.
- Refresh pings (model already resident) are never blocked — they are the cheap path and cannot trigger a load.
- Cold-load gates re-probe immediately before each load (a round can run minutes; finding 7).
- Probe failures on the refresh path fail OPEN; a failed *size* probe fails CLOSED for that cold load only (loading an unknown-size model is exactly the unbounded commitment this plan exists to prevent) with the reason logged.
- MCP `/gpu` is auth-exempt (`_AUTH_EXEMPT_PATHS`, `mcp/local-models-server.py:195`) — no bearer header needed. Client timeout 5s, because the endpoint internally probes MLX with a 2s timeout (finding 6).
- `vm_stat` page size is parsed from its header line, never hardcoded; unparseable output returns `None` (fail open), not 0 (finding 4).
- Conventional commits; the pre-commit hook runs the unit suite and must stay green.
- Match existing test mechanics in `tests/test_core.py` (macOS modules stubbed at import; see its header). Note: the existing warm tests live in class `TestWarmProfiles`.

---

### Task 1: Pure decision + parsing functions

**Files:**
- Modify: `app/menubar.py` (module level, below `WARM_KEEP_ALIVE = "30m"` at line 249)
- Test: `tests/test_core.py` (new class `TestWarmGate`, after `TestWarmProfiles`)

**Interfaces:**
- Produces: `parse_vm_stat_available_gb(text: str) -> float | None` (None when nothing parseable);
  `cold_load_skip_reason(active_counts: dict[str, int], foreign: set[str], pressure_level: int, available_gb: float, size_gb: float | None, headroom_gb: float = 16.0) -> str | None` (None = load allowed).
  Constant `WARM_HEADROOM_GB = 16.0`.

- [ ] **Step 1: Write the failing tests**

```python
class TestWarmGate:
    VM_STAT = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                97134.\n"
        "Pages active:                           15234131.\n"
        "Pages inactive:                         15856694.\n"
        "Pages speculative:                         12618.\n"
        "Pages purgeable:                           72004.\n"
    )

    def test_parse_vm_stat_available_sums_reclaimable(self):
        # (97134 + 12618 + 15856694 + 72004) pages * 16384 = 244.72GB
        got = menubar.parse_vm_stat_available_gb(self.VM_STAT)
        assert got == pytest.approx(244.7, abs=0.1)

    def test_parse_vm_stat_respects_header_page_size(self):
        text = self.VM_STAT.replace("16384", "4096")
        got = menubar.parse_vm_stat_available_gb(text)
        assert got == pytest.approx(244.7 / 4, abs=0.1)

    def test_parse_vm_stat_garbage_is_none(self):
        assert menubar.parse_vm_stat_available_gb("no pages here") is None

    def test_cold_load_blocked_by_inflight(self):
        r = menubar.cold_load_skip_reason({"ollama": 2}, set(), 1, 400.0, 30.0)
        assert r is not None and "ollama=2" in r

    def test_cold_load_blocked_by_foreign_residents(self):
        r = menubar.cold_load_skip_reason({}, {"someone-elses:70b"}, 1, 400.0, 30.0)
        assert r is not None and "someone-elses:70b" in r

    def test_cold_load_blocked_by_pressure(self):
        r = menubar.cold_load_skip_reason({}, set(), 2, 400.0, 30.0)
        assert r is not None and "pressure" in r

    def test_cold_load_blocked_by_insufficient_available(self):
        r = menubar.cold_load_skip_reason({}, set(), 1, 100.0, 390.0)
        assert r is not None and "390" in r and "100" in r

    def test_cold_load_blocked_by_unknown_size(self):
        r = menubar.cold_load_skip_reason({}, set(), 1, 400.0, None)
        assert r is not None and "unknown" in r

    def test_cold_load_allowed_when_quiet_and_fits(self):
        assert menubar.cold_load_skip_reason({}, set(), 1, 400.0, 380.0) is None

    def test_cold_load_headroom_enforced(self):
        assert menubar.cold_load_skip_reason({}, set(), 1, 390.0, 380.0) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with pyyaml --with requests pytest tests/test_core.py::TestWarmGate -v`
Expected: FAIL with `AttributeError: module 'app.menubar' has no attribute 'parse_vm_stat_available_gb'`

- [ ] **Step 3: Implement**

Below `WARM_KEEP_ALIVE = "30m"` in `app/menubar.py`:

```python
WARM_HEADROOM_GB = 16.0   # margin a cold warm-load must leave on top of its own size

_VM_STAT_AVAILABLE_FIELDS = ("Pages free:", "Pages speculative:",
                             "Pages inactive:", "Pages purgeable:")


def parse_vm_stat_available_gb(text):
    """Reclaimable-memory GB from `vm_stat` output, or None if unparseable.

    free+speculative alone is ~1GB on a healthy loaded box (macOS keeps free
    pages near zero; reclaimable memory lives in inactive/purgeable), so
    availability must sum all four. Page size comes from the header line.
    """
    page_size, pages, matched = 16384, 0, False
    for line in text.splitlines():
        if "page size of" in line:
            page_size = int(line.split("page size of")[1].split()[0])
        elif line.startswith(_VM_STAT_AVAILABLE_FIELDS):
            pages += int(line.split(":")[1].strip().rstrip("."))
            matched = True
    return pages * page_size / 1024 ** 3 if matched else None


def cold_load_skip_reason(active_counts, foreign, pressure_level,
                          available_gb, size_gb, headroom_gb=WARM_HEADROOM_GB):
    """Why an evicted warm model must NOT be re-loaded right now (None = go).

    Loading is a commitment of the model's full footprint; every gate here
    is evidence the machine is doing something else with that memory. A
    blocked reload is retried on the next quiet 240s tick — refreshes of
    already-resident models never pass through this gate.
    """
    busy = {k: v for k, v in sorted(active_counts.items()) if v}
    if busy:
        return "inference in flight (%s)" % ", ".join(f"{k}={v}" for k, v in busy.items())
    if foreign:
        return "foreign models resident (%s)" % ", ".join(sorted(foreign))
    if pressure_level > 1:
        return f"memory pressure level {pressure_level}"
    if size_gb is None:
        return "model size unknown — refusing an unbounded load"
    if available_gb < size_gb + headroom_gb:
        return (f"insufficient memory (need {size_gb:.0f}GB + {headroom_gb:.0f}GB headroom, "
                f"{available_gb:.0f}GB available)")
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with pyyaml --with requests pytest tests/test_core.py::TestWarmGate -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add app/menubar.py tests/test_core.py
git commit -m "feat(menubar): decision logic for gated warm-model reloads"
```

---

### Task 2: Probes — activity, pressure, availability, residency, sizes

**Files:**
- Modify: `app/menubar.py` (module level, below Task 1)
- Test: `tests/test_core.py` (extend `TestWarmGate`), `tests/test_mcp_server.py` (one shape-tie test)

**Interfaces:**
- Consumes: `parse_vm_stat_available_gb` (Task 1); `query_mlx_model_info_from_config`'s config-reading pattern (`app/menubar.py:983`).
- Produces: `gpu_active_counts(timeout: float = 5.0) -> dict[str, int]` ({} on failure);
  `memory_pressure_level() -> int` (1 on failure); `vm_available_gb() -> float` (inf on failure);
  `ollama_resident_models(port: str) -> dict[str, float]` (name → size GB, {} on failure);
  `mlx_loaded_ids(port: str) -> set[str]` (empty on failure);
  `mlx_model_size_gb(bare_name: str) -> float | None` (HF-cache bytes for the yaml's model_path).

- [ ] **Step 1: Write the failing tests**

In `tests/test_core.py::TestWarmGate` (note `_fake_response` helper):

```python
    @staticmethod
    def _fake_response(payload: bytes):
        resp = MagicMock()
        resp.read.return_value = payload
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_gpu_active_counts_parses_mcp_shape(self):
        payload = json.dumps({"ollama": {"active": 2, "tasks": []},
                              "mlx": {"active": 0, "tasks": [], "responsive": True}}).encode()
        with patch.object(menubar.urllib.request, "urlopen",
                          return_value=self._fake_response(payload)) as mock_open:
            assert menubar.gpu_active_counts() == {"ollama": 2, "mlx": 0}
        req = mock_open.call_args[0][0]
        assert req.get_full_url() == "http://127.0.0.1:8100/gpu"

    def test_gpu_active_counts_fails_open(self):
        with patch.object(menubar.urllib.request, "urlopen", side_effect=OSError("down")):
            assert menubar.gpu_active_counts() == {}

    def test_memory_pressure_level_fails_open_to_normal(self):
        with patch.object(menubar.subprocess, "check_output", side_effect=OSError("no sysctl")):
            assert menubar.memory_pressure_level() == 1

    def test_vm_available_gb_fails_open(self):
        with patch.object(menubar.subprocess, "check_output", side_effect=OSError("boom")):
            assert menubar.vm_available_gb() == float("inf")
        with patch.object(menubar.subprocess, "check_output", return_value="garbage"):
            assert menubar.vm_available_gb() == float("inf")

    def test_ollama_resident_models_parses_api_ps(self):
        payload = json.dumps({"models": [
            {"name": "qwen3.6:27b", "size": 35045790186},
            {"name": "dolphin3:8b", "size": 5000000000}]}).encode()
        with patch.object(menubar.urllib.request, "urlopen",
                          return_value=self._fake_response(payload)):
            got = menubar.ollama_resident_models("11434")
        assert got["qwen3.6:27b"] == pytest.approx(32.6, abs=0.1)
        assert set(got) == {"qwen3.6:27b", "dolphin3:8b"}

    def test_mlx_loaded_ids_parses_v1_models(self):
        payload = json.dumps({"data": [{"id": "whisper-large"}, {"id": "glm-5.2"}]}).encode()
        with patch.object(menubar.urllib.request, "urlopen",
                          return_value=self._fake_response(payload)):
            assert menubar.mlx_loaded_ids("8000") == {"whisper-large", "glm-5.2"}

    def test_mlx_model_size_gb_sums_hf_snapshot(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("models:\n  - model_path: org/My-Model\n    served_model_name: my-model\n")
        blob_dir = tmp_path / "hub" / "models--org--My-Model" / "snapshots" / "abc"
        blob_dir.mkdir(parents=True)
        (blob_dir / "weights.safetensors").write_bytes(b"x" * 4096)
        monkeypatch.setattr(menubar, "MLX_SERVER_CONFIG_PATH", str(cfg))
        monkeypatch.setattr(menubar, "HF_HUB_CACHE_PATH", str(tmp_path / "hub"))
        got = menubar.mlx_model_size_gb("my-model")
        assert got == pytest.approx(4096 / 1024 ** 3)
        assert menubar.mlx_model_size_gb("not-served") is None
```

In `tests/test_mcp_server.py`, tie the probe's expectations to the real endpoint (place next to the existing `/gpu` tests):

```python
def test_gpu_status_shape_matches_menubar_probe():
    """app/menubar.py::gpu_active_counts consumes {backend: {"active": int}} —
    guard the real endpoint's shape so a server change can't silently break
    the warm-ping busy gate."""
    import asyncio
    resp = asyncio.get_event_loop().run_until_complete(server._gpu_status(None))
    data = json.loads(resp.body)
    for backend in ("ollama", "mlx"):
        assert isinstance(data[backend]["active"], int)
```

(Adapt the invocation to however the neighboring `/gpu` tests in that file call the endpoint — reuse their client/fixture mechanics verbatim.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with pyyaml --with requests pytest tests/test_core.py::TestWarmGate -v`
Expected: FAIL with `AttributeError: ... 'gpu_active_counts'`

- [ ] **Step 3: Implement**

```python
MLX_SERVER_CONFIG_PATH = os.path.expanduser("~/.config/mlx-server/config.yaml")
HF_HUB_CACHE_PATH = os.path.expanduser("~/.cache/huggingface/hub")


def gpu_active_counts(timeout=5.0):
    """Per-backend in-flight MCP request counts from /gpu (auth-exempt).

    Fails open to {} — no MCP server means no MCP-dispatched inference.
    Timeout must exceed the endpoint's own internal 2s MLX probe.
    """
    try:
        with urllib.request.urlopen(
                urllib.request.Request("http://127.0.0.1:8100/gpu"),
                timeout=timeout) as r:
            data = json.loads(r.read())
        return {k: int(v.get("active", 0))
                for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def memory_pressure_level():
    """Kernel memory pressure: 1 normal, 2 warn, 4 critical. 1 on failure."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True)
        return int(out.strip())
    except Exception:
        return 1


def vm_available_gb():
    """Reclaimable GB right now; inf if unmeasurable (fail open)."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return float("inf")
    got = parse_vm_stat_available_gb(out)
    return float("inf") if got is None else got


def ollama_resident_models(port, timeout=5.0):
    """{model name: size GB} currently loaded in Ollama; {} on failure."""
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/api/ps", timeout=timeout) as r:
            data = json.loads(r.read())
        return {m["name"]: m.get("size", 0) / 1024 ** 3
                for m in data.get("models", [])}
    except Exception:
        return {}


def mlx_loaded_ids(port, timeout=5.0):
    """Model ids the MLX server is currently serving; empty set on failure."""
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/v1/models", timeout=timeout) as r:
            data = json.loads(r.read())
        return {m["id"] for m in data.get("data", [])}
    except Exception:
        return set()


def mlx_model_size_gb(bare_name):
    """On-disk GB of an MLX served-name's HF snapshot, or None if unknown."""
    try:
        import yaml
        with open(MLX_SERVER_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        path = next((m["model_path"] for m in cfg.get("models", [])
                     if m.get("served_model_name") == bare_name), None)
        if not path:
            return None
        snap_root = os.path.join(HF_HUB_CACHE_PATH,
                                 "models--" + path.replace("/", "--"), "snapshots")
        total = 0
        for root, _dirs, files in os.walk(snap_root):
            for name in files:
                total += os.path.getsize(os.path.join(root, name))
        return total / 1024 ** 3 if total else None
    except Exception:
        return None
```

Note: `query_mlx_model_info_from_config` (line 983) builds the same config/cache paths inline; leave it as is (scope discipline) but the new constants make the paths patchable for tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with pyyaml --with requests pytest tests/test_core.py::TestWarmGate tests/test_mcp_server.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add app/menubar.py tests/test_core.py tests/test_mcp_server.py
git commit -m "feat(menubar): residency, pressure, and size probes for warm gating"
```

---

### Task 3: Residency-first ping round + wiring + docs

**Files:**
- Modify: `app/menubar.py:1476-1499` (`_ping_warm` method → module function `ping_warm`), `app/menubar.py:1467-1474` (`_on_warm_tick`)
- Modify: `CLAUDE.md` (Menu Bar Features), `README.md` (~241-243 warm paragraph)
- Test: `tests/test_core.py` (extend `TestWarmGate`)

**Interfaces:**
- Consumes: everything from Tasks 1-2, `WARM_KEEP_ALIVE`.
- Produces: `ping_warm(targets: list[tuple[str, str]], ollama_port: str, mlx_port: str) -> dict[str, int]` returning `{"refreshed": n, "loaded": n, "skipped": n}` for observability. `_ping_warm` method deleted; `_on_warm_tick` threads `ping_warm(targets, self.ollama_port, self.mlx_port)`.

- [ ] **Step 1: Write the failing tests**

```python
    def _machine(self, resident_ollama=None, mlx_ids=None, counts=None,
                 pressure=1, available=400.0):
        return [patch.object(menubar, "ollama_resident_models",
                             return_value=resident_ollama or {}),
                patch.object(menubar, "mlx_loaded_ids", return_value=mlx_ids or set()),
                patch.object(menubar, "gpu_active_counts", return_value=counts or {}),
                patch.object(menubar, "memory_pressure_level", return_value=pressure),
                patch.object(menubar, "vm_available_gb", return_value=available)]

    def test_ping_warm_refreshes_resident_models_unconditionally(self):
        patches = self._machine(resident_ollama={"m:1b": 1.0}, mlx_ids={"bare"},
                                counts={"ollama": 3}, pressure=4, available=1.0)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch.object(menubar.urllib.request, "urlopen",
                          return_value=self._fake_response(b"{}")) as mock_open:
            stats = menubar.ping_warm([("m:1b", "ollama"), ("bare", "mlx")],
                                      "11434", "8000")
        assert stats == {"refreshed": 2, "loaded": 0, "skipped": 0}
        assert mock_open.call_count == 2

    def test_ping_warm_blocks_reload_when_foreign_resident(self, caplog):
        patches = self._machine(resident_ollama={"third-party:70b": 40.0})
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch.object(menubar, "mlx_model_size_gb", return_value=30.0), \
             patch.object(menubar.urllib.request, "urlopen") as mock_open, \
             caplog.at_level("INFO"):
            stats = menubar.ping_warm([("bare", "mlx")], "11434", "8000")
        assert stats == {"refreshed": 0, "loaded": 0, "skipped": 1}
        mock_open.assert_not_called()
        assert "third-party:70b" in caplog.text

    def test_ping_warm_reloads_evicted_model_on_quiet_machine(self):
        patches = self._machine()
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch.object(menubar, "ollama_model_size_gb", return_value=20.0), \
             patch.object(menubar.urllib.request, "urlopen",
                          return_value=self._fake_response(b"{}")) as mock_open:
            stats = menubar.ping_warm([("m:1b", "ollama")], "11434", "8000")
        assert stats == {"refreshed": 0, "loaded": 1, "skipped": 0}
        assert mock_open.call_count == 1

    def test_ping_warm_blocks_reload_of_unknown_size_model(self, caplog):
        patches = self._machine()
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patch.object(menubar, "mlx_model_size_gb", return_value=None), \
             patch.object(menubar.urllib.request, "urlopen") as mock_open, \
             caplog.at_level("INFO"):
            stats = menubar.ping_warm([("bare", "mlx")], "11434", "8000")
        assert stats["skipped"] == 1
        mock_open.assert_not_called()

    def test_on_warm_tick_threads_ping_warm_with_ports(self):
        app = MagicMock()
        app.mode = "server"
        app.servers_started = True
        app.ollama_port, app.mlx_port = "11434", "8000"
        with patch.object(menubar, "load_profiles", return_value={}), \
             patch.object(menubar, "warm_ping_targets",
                          return_value=[("m:1b", "ollama")]), \
             patch.object(menubar.threading, "Thread") as mock_thread:
            menubar.LocalModelsApp._on_warm_tick(app, None)
        assert mock_thread.call_args.kwargs["target"] is menubar.ping_warm
        assert mock_thread.call_args.kwargs["args"] == (
            [("m:1b", "ollama")], "11434", "8000")
```

Also add `ollama_model_size_gb` to Task 2's probes (test):

```python
    def test_ollama_model_size_gb_reads_api_tags(self):
        payload = json.dumps({"models": [{"name": "m:1b", "size": 2147483648}]}).encode()
        with patch.object(menubar.urllib.request, "urlopen",
                          return_value=self._fake_response(payload)):
            assert menubar.ollama_model_size_gb("m:1b", "11434") == pytest.approx(2.0)
            assert menubar.ollama_model_size_gb("missing:1b", "11434") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with pyyaml --with requests pytest tests/test_core.py::TestWarmGate -v`
Expected: FAIL with `AttributeError: ... 'ping_warm'`

- [ ] **Step 3: Implement**

Module-level, near `warm_ping_targets`:

```python
def ollama_model_size_gb(model, port, timeout=5.0):
    """On-disk GB of an Ollama tag from /api/tags, or None if unknown."""
    try:
        with urllib.request.urlopen(
                f"http://localhost:{port}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read())
        for m in data.get("models", []):
            if m.get("name") == model:
                return m.get("size", 0) / 1024 ** 3
    except Exception:
        pass
    return None


def _send_warm_ping(model, backend, ollama_port, mlx_port):
    if backend == "ollama":
        body = json.dumps({"model": model, "prompt": "",
                           "keep_alive": WARM_KEEP_ALIVE}).encode()
        url = f"http://localhost:{ollama_port}/api/generate"
    else:
        body = json.dumps({"model": model, "max_tokens": 1,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        url = f"http://localhost:{mlx_port}/v1/chat/completions"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=600):
        pass


def ping_warm(targets, ollama_port, mlx_port):
    """One warm round, residency-first. Returns counts for observability.

    Refreshing a resident model is always safe — it can't trigger a load.
    Re-loading an evicted model commits its full footprint, so each reload
    re-probes the gates immediately before sending (a round can run minutes,
    and the first reload changes the machine the second one sees).
    """
    stats = {"refreshed": 0, "loaded": 0, "skipped": 0}
    resident_ollama = ollama_resident_models(ollama_port)
    resident_mlx = mlx_loaded_ids(mlx_port)
    warm_names = {m for m, _ in targets}
    for model, backend in targets:
        resident = model in (resident_ollama if backend == "ollama" else resident_mlx)
        try:
            if resident:
                _send_warm_ping(model, backend, ollama_port, mlx_port)
                stats["refreshed"] += 1
                continue
            foreign = (set(resident_ollama) | resident_mlx) - warm_names
            size_gb = (ollama_model_size_gb(model, ollama_port)
                       if backend == "ollama" else mlx_model_size_gb(model))
            reason = cold_load_skip_reason(
                gpu_active_counts(), foreign, memory_pressure_level(),
                vm_available_gb(), size_gb)
            if reason:
                logging.info("keep-warm: not reloading %s: %s", model, reason)
                stats["skipped"] += 1
                continue
            logging.info("keep-warm: reloading evicted %s (%s)", model, backend)
            _send_warm_ping(model, backend, ollama_port, mlx_port)
            stats["loaded"] += 1
            resident_ollama = ollama_resident_models(ollama_port)
            resident_mlx = mlx_loaded_ids(mlx_port)
        except Exception as e:
            logging.debug("keep-warm ping failed for %s: %s", model, e)
    return stats
```

In `_on_warm_tick` (line 1467-1474):

```python
        threading.Thread(target=ping_warm,
                         args=(targets, self.ollama_port, self.mlx_port),
                         daemon=True).start()
```

Delete the `_ping_warm` method. (Its refresh timeout moves from 120s to 600s inside `_send_warm_ping` — a gated reload of a large model legitimately takes minutes.)

Docs, same commit — `CLAUDE.md` Menu Bar Features bullet:

```markdown
- **Contention-aware keep-warm** — warm pings refresh models that are already resident, but re-loading an evicted warm model is gated: skipped (and logged) while any MCP request is in flight, any non-warm model is resident on Ollama/MLX, kernel memory pressure is above normal, or available memory can't fit the model plus 16GB headroom. Residency lapses under contention and restores on the next quiet 4-minute tick.
```

`README.md` (~line 241-243):

```markdown
Warm models are re-pinged every 4 minutes. Refreshes only touch models that
are already loaded; if a warm model was evicted, SP re-loads it only when the
machine is quiet — nothing in flight, no foreign models resident, normal
memory pressure, and enough available memory for the model plus headroom.
```

- [ ] **Step 4: Run the full unit suite**

First: `grep -rn "_ping_warm" app/ tests/` — expect only the definition being deleted and the call site being updated.
Run: `uv run --with pytest --with pyyaml --with requests --with flask pytest tests/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add app/menubar.py tests/test_core.py CLAUDE.md README.md
git commit -m "feat(menubar): residency-first warm pings that never compete for memory"
```

---

## Self-Review

- Red-team findings addressed: 1 (availability = 4-field sum + pressure sysctl; refreshes never memory-gated → no thrash loop), 2 (residency-first; reload requires size+headroom fit), 3 (foreign-resident detection on both backends covers Playground and third-party Ollama/MLX use; pressure covers non-HTTP memory hogs), 4 (parse → None → inf fail-open; page size from header), 5 (no bearer header), 6 (5s probe timeout), 7 (gates re-probed per reload; residency re-read after each load), 8 (shape-tie test in test_mcp_server.py; `_on_warm_tick` wiring test; class anchor corrected to `TestWarmProfiles`), 9 (no per-round sysctl for constants; the only sysctl left is the pressure probe, which must be fresh).
- Placeholders: none — every step carries code. The one "adapt to neighboring fixture" note in Task 2's shape-tie test is deliberate: it must reuse that file's real client mechanics rather than invent parallel ones.
- Type consistency: `cold_load_skip_reason(dict, set, int, float, float|None)` matches all call sites; `ping_warm(list[tuple[str,str]], str, str) -> dict` matches the wiring test and `_on_warm_tick`.
