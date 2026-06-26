# Warm-set Model Residency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each profile an explicit warm (resident) vs on-demand (streamed) model split, keep the warm set hot via a ticker, evict non-warm models promptly, and replace the misleading "sum-all-models" memory bar with a warm-set-vs-budget bar fed by a testable endpoint.

**Architecture:** `lib/models.py` gains a `warm` task-key list per profile, a `WARM_BUDGET_FRACTION`, and a shared `warm_model_names(profiles_data)` resolver. The profile server uses it for warm-aware Ollama `keep_alive` and a new `/api/profiles/<name>/memory` endpoint (all residency math, unit-tested). The menu-bar app runs a keep-warm ticker that re-pings the active profile's warm models. `profiles.html` renders the bar from the endpoint.

**Tech Stack:** Python 3.12 (`lib/`, `app/`), Flask (profile server), rumps (menu-bar), vanilla JS (profiles.html), pytest.

## Global Constraints

- `PROFILES_VERSION` becomes **27** (currently 26).
- Every preset gains `"warm": ["general", "embedding"]`. `WARM_BUDGET_FRACTION = 0.65`.
- Absent `warm` key ⇒ treat as `[]` (all on-demand). Consumers read `profile.get("warm", [])`.
- Warm-state thresholds (cap = `max_ram_gb << 30`, budget = `int(cap * 0.65)`): `thrash` if `warm_bytes > cap`; else `tight` if `warm_bytes > budget` **or** `peak_bytes > cap`; else `ok`. `peak_bytes = warm_bytes + largest_on_demand_bytes`.
- On-demand model set excludes any model already in the warm set (a model backing both a warm and a non-warm task is warm, counted once).
- bash/Python run on macOS; commits are SSH-signed automatically via 1Password (no signing flags). End commit bodies with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Test command (run the full trio before each Python commit; the v26→v27 bump ripples into profile-server tests): `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py tests/test_profile_server.py tests/test_error_handling.py -q`
- **Git safety:** stay on branch `feat/warm-set-residency`. NEVER run `git checkout/switch/reset/restore/stash/rebase` or check out any commit/branch. Only `git add` + `git commit`. (A prior session suffered a detached-HEAD incident; the menu-bar auto-updater is currently unloaded to prevent recurrence.)

---

## File Structure

- `lib/models.py` — **modify**: `warm` on presets; `WARM_BUDGET_FRACTION`; `PROFILES_VERSION=27`; new `warm_model_names(data)`. Single source of truth.
- `app/profile-server.py` — **modify**: `keep_alive_for(model)` + warm-aware keep_alive at the 5 sites; `/warm` loads only the warm set; new `GET /api/profiles/<name>/memory`.
- `app/menubar.py` — **modify**: keep-warm ticker (rumps.Timer, ~240s) → daemon thread pinging active warm models.
- `app/profiles.html` — **modify**: `renderMemory` reads `/api/profiles/<name>/memory`; single bar + hatched transient-peak + budget/cap markers + state colors.
- `tests/test_core.py` — **modify**: warm contract, migration, `warm_model_names`.
- `tests/test_profile_server.py` — **modify**: `/warm` scope, `keep_alive_for`, `/memory` state math.

---

## Task 1: Schema — warm lists, budget, version, resolver (`lib/models.py`)

**Files:**
- Modify: `lib/models.py` (`PROFILES_VERSION` ~line 396; each profile in `DEFAULT_PROFILES` ~398–470; add `WARM_BUDGET_FRACTION` and `warm_model_names` after `migrate_profiles`)
- Test: `tests/test_core.py` (`TestDefaultProfilesSeeding`)

**Interfaces:**
- Produces: `WARM_BUDGET_FRACTION: float = 0.65`; `PROFILES_VERSION = 27`; each preset has `warm: list[str]`; `warm_model_names(data: dict) -> set[str]` (model names for the active profile's warm task keys; `{}` if no active profile).

- [ ] **Step 1: Write failing tests** — append to `class TestDefaultProfilesSeeding` in `tests/test_core.py`:

```python
    def test_every_preset_has_valid_warm_keys(self):
        for name, prof in menubar.DEFAULT_PROFILES["profiles"].items():
            warm = prof.get("warm")
            assert warm == ["general", "embedding"], f"{name} warm={warm}"
            for key in warm:
                assert key in prof["tasks"], f"{name} warm key {key} not in tasks"

    def test_warm_model_names_resolves_active(self):
        from lib.models import warm_model_names, DEFAULT_PROFILES
        data = {"active": "128gb", "profiles": DEFAULT_PROFILES["profiles"]}
        names = warm_model_names(data)
        tasks = DEFAULT_PROFILES["profiles"]["128gb"]["tasks"]
        assert names == {tasks["general"], tasks["embedding"]}

    def test_warm_model_names_no_active(self):
        from lib.models import warm_model_names
        assert warm_model_names({"active": None, "profiles": {}}) == set()

    def test_migrate_adds_warm_to_presets_custom_absent(self):
        from lib.models import migrate_profiles
        out = migrate_profiles({"version": 26, "active": "64gb",
                                "profiles": {"mine": {"max_ram_gb": 8, "tasks": {"code": "c:1b"}}}})
        assert out["profiles"]["64gb"]["warm"] == ["general", "embedding"]
        assert "warm" not in out["profiles"]["mine"]   # custom untouched; absent ⇒ on-demand
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -k "warm" -q`
Expected: FAIL (`warm` key missing / `ImportError: warm_model_names`).

- [ ] **Step 3: Bump version + add `warm` to every preset.** In `lib/models.py` set `PROFILES_VERSION = 27`. Add `"warm": ["general", "embedding"],` to each of the four profile dicts (`32gb`, `64gb`, `128gb`, `512gb`) — place it right after the `"max_ram_gb": ...,` line in each, before `"tasks": {`.

- [ ] **Step 4: Add the budget constant and resolver.** Immediately after the `migrate_profiles` function add:

```python
WARM_BUDGET_FRACTION = 0.65  # warm set should fit under this fraction of tier RAM


def warm_model_names(data: dict) -> set[str]:
    """Model names kept warm for the active profile (its `warm` task keys).

    Returns an empty set when there is no active profile or it has no warm list.
    A task key that isn't present in the profile's tasks is skipped.
    """
    prof = (data.get("profiles") or {}).get(data.get("active"))
    if not prof:
        return set()
    tasks = prof.get("tasks", {})
    return {tasks[k] for k in prof.get("warm", []) if k in tasks}
```

- [ ] **Step 5: Run the full trio**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py tests/test_profile_server.py tests/test_error_handling.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add lib/models.py tests/test_core.py
git commit -m "feat(profiles): warm lists + budget + warm_model_names; PROFILES_VERSION 27"
```

---

## Task 2: Warm endpoint loads only the warm set (`app/profile-server.py`)

**Files:**
- Modify: `app/profile-server.py` (`api_profiles_warm`, ~line 1752, the `candidates = ...` line)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Consumes: each profile's `warm` list (Task 1).

- [ ] **Step 1: Write the failing test** in `tests/test_profile_server.py` (follow the file's existing Flask test-client + mock patterns; patch `load_profiles`, `get_all_models`, `ollama_get`, and `requests.post`):

```python
    def test_warm_loads_only_warm_set(self, client):
        prof = {"max_ram_gb": 128, "warm": ["general", "embedding"],
                "tasks": {"general": "work:bf16", "code": "coder:bf16",
                          "embedding": "embed:8b", "image_gen": "x/img:latest"}}
        with patch.object(ps, "load_profiles", return_value={"active": "t", "profiles": {"t": prof}}), \
             patch.object(ps, "get_all_models", return_value={
                 "work:bf16": {"backend": "ollama"}, "coder:bf16": {"backend": "ollama"},
                 "embed:8b": {"backend": "ollama"}, "x/img:latest": {"backend": "ollama"}}), \
             patch.object(ps, "ollama_get", return_value={"models": []}), \
             patch("requests.post") as post:
            post.return_value.status_code = 200
            r = client.post("/api/profiles/t/warm")
        assert r.status_code == 200
        warmed = {c.kwargs["json"]["model"] for c in post.call_args_list}
        assert warmed == {"work:bf16", "embed:8b"}      # general + embedding only
        assert "coder:bf16" not in warmed and "x/img:latest" not in warmed
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -k warm_loads_only -q`
Expected: FAIL (current code warms all four models).

- [ ] **Step 3: Scope the candidates to the warm set.** In `api_profiles_warm`, replace:

```python
    candidates = list(dict.fromkeys(tasks.values()))
```
with:
```python
    warm_keys = profile.get("warm", [])
    candidates = list(dict.fromkeys(tasks[k] for k in warm_keys if k in tasks))
```

- [ ] **Step 4: Run the profile-server suite**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "feat(profile-server): warm only the profile's warm set, not every model"
```

---

## Task 3: Memory endpoint with state math (`app/profile-server.py`)

**Files:**
- Modify: `app/profile-server.py` (add route near the other `/api/profiles/...` routes ~line 1752; add `WARM_BUDGET_FRACTION` to the `from lib.models import (...)` block ~line 42)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Produces: `GET /api/profiles/<name>/memory` → JSON `{cap_bytes, budget_bytes, warm:[{name,task,bytes}], warm_bytes, on_demand:[{name,task,bytes}], largest_on_demand_bytes, peak_bytes, state}` with `state ∈ {"ok","tight","thrash"}`.

- [ ] **Step 1: Write failing tests** in `tests/test_profile_server.py`:

```python
    def _mem(self, client, max_ram_gb, tasks, warm, sizes):
        prof = {"max_ram_gb": max_ram_gb, "warm": warm, "tasks": tasks}
        models = {n: {"backend": "ollama", "vram_bytes": b, "disk_bytes": b}
                  for n, b in sizes.items()}
        with patch.object(ps, "load_profiles", return_value={"active": "t", "profiles": {"t": prof}}), \
             patch.object(ps, "get_all_models", return_value=models):
            return client.get("/api/profiles/t/memory").get_json()

    def test_memory_ok(self, client):
        GB = 1 << 30
        d = self._mem(client, 128,
                      {"general": "w", "embedding": "e", "code": "c"},
                      ["general", "embedding"],
                      {"w": 55 * GB, "e": 8 * GB, "c": 52 * GB})
        assert d["warm_bytes"] == 63 * GB
        assert d["largest_on_demand_bytes"] == 52 * GB
        assert d["peak_bytes"] == 115 * GB           # 63 + 52 ≤ 128
        assert d["state"] == "ok"

    def test_memory_tight_dominant_over_budget(self, client):
        GB = 1 << 30
        d = self._mem(client, 512,
                      {"general": "glm", "embedding": "e", "vision": "v"},
                      ["general", "embedding"],
                      {"glm": 418 * GB, "e": 8 * GB, "v": 95 * GB})
        assert d["warm_bytes"] == 426 * GB           # > 333 budget, ≤ 512 cap
        assert d["peak_bytes"] == 521 * GB           # 426 + 95 > 512 cap
        assert d["state"] == "tight"

    def test_memory_thrash_warm_exceeds_cap(self, client):
        GB = 1 << 30
        d = self._mem(client, 64,
                      {"general": "a", "embedding": "b"},
                      ["general", "embedding"],
                      {"a": 50 * GB, "b": 30 * GB})   # warm 80 > 64 cap
        assert d["state"] == "thrash"

    def test_memory_dedup_shared_model_not_double_counted(self, client):
        GB = 1 << 30
        d = self._mem(client, 32,
                      {"general": "s", "code": "s", "embedding": "e", "image_gen": "img"},
                      ["general", "embedding"],
                      {"s": 5 * GB, "e": 1 * GB, "img": 6 * GB})
        # 's' backs warm general AND non-warm code → counted warm only, not on-demand
        assert {m["name"] for m in d["on_demand"]} == {"img"}
        assert d["warm_bytes"] == 6 * GB
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -k memory -q`
Expected: FAIL (404 / route missing).

- [ ] **Step 3: Add `WARM_BUDGET_FRACTION` to the lib import** (insert in the `from lib.models import (...)` block, alphabetical): add the line `WARM_BUDGET_FRACTION,`.

- [ ] **Step 4: Implement the route.** Add near `api_profiles_warm`:

```python
@app.route("/api/profiles/<name>/memory", methods=["GET"])
def api_profiles_memory(name):
    """Residency math for the memory bar: warm set vs budget, transient peak, state."""
    proxied = _proxy_to_desktop(f"/api/profiles/{name}/memory")
    if proxied is not None:
        return proxied
    data = load_profiles()
    profile = data.get("profiles", {}).get(name)
    if not profile:
        return jsonify({"error": f"Profile '{name}' not found"}), 404

    models = get_all_models()
    tasks = profile.get("tasks", {})
    warm_keys = profile.get("warm", [])

    def size_of(model):
        info = models.get(model) or {}
        return int(info.get("vram_bytes") or info.get("disk_bytes") or 0)

    warm_names = list(dict.fromkeys(tasks[k] for k in warm_keys if k in tasks))
    warm_set = set(warm_names)
    on_names = list(dict.fromkeys(
        m for k, m in tasks.items() if k not in warm_keys and m not in warm_set))

    def task_for(model):
        return next((k for k, m in tasks.items() if m == model), None)

    warm = [{"name": m, "task": task_for(m), "bytes": size_of(m)} for m in warm_names]
    on_demand = [{"name": m, "task": task_for(m), "bytes": size_of(m)} for m in on_names]

    cap_bytes = int(profile.get("max_ram_gb", 0)) << 30
    budget_bytes = int(cap_bytes * WARM_BUDGET_FRACTION)
    warm_bytes = sum(x["bytes"] for x in warm)
    largest_on_demand = max((x["bytes"] for x in on_demand), default=0)
    peak_bytes = warm_bytes + largest_on_demand

    if warm_bytes > cap_bytes:
        state = "thrash"
    elif warm_bytes > budget_bytes or peak_bytes > cap_bytes:
        state = "tight"
    else:
        state = "ok"

    return jsonify({
        "cap_bytes": cap_bytes, "budget_bytes": budget_bytes,
        "warm": warm, "warm_bytes": warm_bytes,
        "on_demand": on_demand, "largest_on_demand_bytes": largest_on_demand,
        "peak_bytes": peak_bytes, "state": state,
    })
```

- [ ] **Step 5: Run the memory tests then the full trio**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -k memory -q` → PASS
Then: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py tests/test_profile_server.py tests/test_error_handling.py -q` → PASS

- [ ] **Step 6: Commit**

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "feat(profile-server): /api/profiles/<name>/memory warm-vs-budget state"
```

---

## Task 4: Warm-aware Ollama keep_alive (`app/profile-server.py`)

**Files:**
- Modify: `app/profile-server.py` (add `keep_alive_for`; replace the `OLLAMA_KEEP_ALIVE` argument at the 4 inference sites — lines ~1966, ~2022, ~2319, ~2520; the `/warm` preload at ~1792 stays as-is since it only ever loads warm models)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Consumes: `warm_model_names` (Task 1).
- Produces: `keep_alive_for(model: str) -> str` ("30m" if model is in the active profile's warm set, else "30s").

- [ ] **Step 1: Write the failing test** in `tests/test_profile_server.py`:

```python
    def test_keep_alive_for_warm_vs_on_demand(self):
        prof = {"warm": ["general"], "tasks": {"general": "w:bf16", "code": "c:bf16"}}
        with patch.object(ps, "load_profiles", return_value={"active": "t", "profiles": {"t": prof}}):
            assert ps.keep_alive_for("w:bf16") == "30m"
            assert ps.keep_alive_for("c:bf16") == "30s"
            assert ps.keep_alive_for("unknown:1b") == "30s"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -k keep_alive_for -q`
Expected: FAIL (`AttributeError: keep_alive_for`).

- [ ] **Step 3: Add the helper + import.** Add `warm_model_names` to the `from lib.models import (...)` block. Near the `OLLAMA_KEEP_ALIVE = "30m"` definition (~line 83) add:

```python
OLLAMA_KEEP_ALIVE_ONDEMAND = "30s"  # non-warm models evict promptly after use


def keep_alive_for(model: str) -> str:
    """Long keep_alive for the active profile's warm models, short otherwise."""
    try:
        warm = warm_model_names(load_profiles())
    except Exception:
        warm = set()
    return OLLAMA_KEEP_ALIVE if model in warm else OLLAMA_KEEP_ALIVE_ONDEMAND
```

- [ ] **Step 4: Use it at the four inference sites.** At each of the ~4 request sites that currently pass `"keep_alive": OLLAMA_KEEP_ALIVE` for a *task* inference (lines ~1966, ~2022, ~2319, ~2520 — the chat/generate/embed paths that have the target `model` in scope), change the value to `keep_alive_for(model)`. Leave the `/warm` preload (~1792) on `OLLAMA_KEEP_ALIVE`. Verify none were missed:

Run: `grep -n "OLLAMA_KEEP_ALIVE\b" app/profile-server.py`
Expected: only the constant definition and the `/warm` preload line remain; the four inference sites now call `keep_alive_for(...)`.

- [ ] **Step 5: Run the full trio**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py tests/test_profile_server.py tests/test_error_handling.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "feat(profile-server): warm-aware keep_alive so non-warm models evict"
```

---

## Task 5: Keep-warm ticker (`app/menubar.py`)

**Files:**
- Modify: `app/menubar.py` (add `warm_model_names` to the `from lib.models import (...)` block ~line 477; add a `rumps.Timer` in the same place `self.timer` is created ~line 1401; add the ticker method + a pure selector)
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: `warm_model_names` (Task 1); `load_profiles()` (menubar module).
- Produces: `warm_ping_targets(data: dict) -> list[tuple[str, str]]` — `(model, backend)` for the active profile's warm models, backend ∈ {"ollama","mlx"} by string shape (`:` → ollama, no `:`/`/` → mlx; HF `/` repos excluded — not server-resident). Used to drive best-effort pings.

- [ ] **Step 1: Write the failing test** in `tests/test_core.py` `TestDefaultProfilesSeeding` (or a new class):

```python
    def test_warm_ping_targets_classifies_backend(self):
        data = {"active": "t", "profiles": {"t": {
            "warm": ["general", "embedding", "tts"],
            "tasks": {"general": "qwen3.6:27b-mlx", "embedding": "embed:8b",
                      "tts": "mlx-community/Some-TTS", "code": "coder:1b"}}}}
        targets = dict(menubar.warm_ping_targets(data))
        assert targets == {"qwen3.6:27b-mlx": "ollama", "embed:8b": "ollama"}
        # HF-repo TTS excluded (not a keep-warm server target); non-warm 'code' absent
```

(Note: a bare MLX served-name like `qwen3.5-small` would map to `"mlx"`; this profile has none in its warm set, so the test uses the ollama-tag + hf cases. If a tier's warm set ever includes a bare MLX name, it pings MLX.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -k warm_ping_targets -q`
Expected: FAIL (`AttributeError: warm_ping_targets`).

- [ ] **Step 3: Add the import + pure selector.** Add `warm_model_names` to the menubar `from lib.models import (...)` block. Add at module scope (near `load_profiles`):

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

- [ ] **Step 4: Add the ticker.** Where `self.timer = rumps.Timer(self._on_tick, POLL_INTERVAL)` is created (~line 1401), add below it:

```python
        self.warm_timer = rumps.Timer(self._on_warm_tick, 240)
        self.warm_timer.start()
```

Add the method (place near `_on_tick`):

```python
    def _on_warm_tick(self, _):
        """Keep the active profile's warm models resident (re-ping before idle unload)."""
        if self.mode not in ("server", "offline") or not self.servers_started:
            return
        targets = warm_ping_targets(load_profiles())
        if not targets:
            return
        threading.Thread(target=self._ping_warm, args=(targets,), daemon=True).start()

    def _ping_warm(self, targets):
        ollama_port = self.conf.get("OLLAMA_PORT", "11434")
        mlx_port = self.conf.get("MLX_PORT", "8000")
        for model, backend in targets:
            try:
                if backend == "ollama":
                    requests.post(f"http://localhost:{ollama_port}/api/generate",
                                  json={"model": model, "prompt": "",
                                        "keep_alive": "30m"}, timeout=120)
                else:
                    requests.post(f"http://localhost:{mlx_port}/v1/chat/completions",
                                  json={"model": model, "max_tokens": 1,
                                        "messages": [{"role": "user", "content": "hi"}]},
                                  timeout=120)
            except Exception as e:
                logging.debug("keep-warm ping failed for %s: %s", model, e)
```

(Confirm `requests`, `threading`, `logging` are already imported in `menubar.py` — they are used elsewhere in the file; if any is missing, add it. Confirm `self.conf` holds the network config dict with `OLLAMA_PORT`/`MLX_PORT`; if the attribute differs, use the existing accessor the file already uses for those ports.)

- [ ] **Step 5: Run the trio + import-smoke the menubar**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py tests/test_profile_server.py tests/test_error_handling.py -q` → PASS
Run: `python3 -c "import sys; sys.path.insert(0,'.'); import app.menubar"` (with macOS stubs this may fail to import natively — if so, rely on the test suite, which stubs rumps/objc/AppKit/WebKit, importing menubar successfully).

- [ ] **Step 6: Commit**

```bash
git add app/menubar.py tests/test_core.py
git commit -m "feat(menubar): keep-warm ticker re-pings active profile's warm models"
```

---

## Task 6: Memory bar redesign (`app/profiles.html`)

**Files:**
- Modify: `app/profiles.html` (`renderMemory`, ~line 721; it currently builds `segs` from all task picks and sums `vram_bytes`)

**Interfaces:**
- Consumes: `GET /api/profiles/<name>/memory` (Task 3).

This task changes rendered UI. Per the project rule, it is verified by rendering the page and inspecting it with a vision model, not by a unit test (the math it displays is already covered by Task 3).

- [ ] **Step 1: Rewrite `renderMemory` to read the endpoint.** Replace the body of `renderMemory` so it fetches `/api/profiles/${S.sel}/memory` (guard when `S.sel` is unset), then renders one bar over `0…cap_bytes`:
  - Solid stacked segments for each `warm[]` model (existing `.mem-seg` styling, colored per its `task` via the `TC` map; label with `baseName`).
  - One hatched segment sized to `largest_on_demand_bytes`, drawn immediately after the warm segments, representing the transient peak. Add a `.mem-seg-ondemand` CSS class with a repeating-stripe `background` (e.g. `repeating-linear-gradient`).
  - Vertical markers at `budget_bytes` and `cap_bytes` (reuse/extend the existing `.mem-bar-limit` element; add a second marker for budget).
  - Bar/usage color by `state`: `ok` → normal/green, `tight` → amber (`var(--orange)` or similar existing token), `thrash` → red (`var(--red)`).
  - Usage line: `fmt(warm_bytes) + ' warm / ' + fmt(budget_bytes) + ' budget · ' + fmt(cap_bytes) + ' cap'`.
  - Legend: warm models (solid dot) with sizes; on-demand models (hatched/ghost dot) with sizes; a peak line: `peak ${fmt(peak)} ${peak<=cap?'≤':'>'} ${fmt(cap)} cap`.
  Keep the existing helper functions (`fmt`, `baseName`, `shortName`, `TC`, `modelMeta`). Remove the old `segs`/`totalUsed = sum of all picks` logic.

- [ ] **Step 2: Render and visually verify.** Ensure local services + the profile server are running (`start-local-models`; the menu-bar app serves the page on `:8101`). Load `http://127.0.0.1:8101/tools` (or the profiles route), select the **128gb** profile, capture a screenshot, and inspect it with the `local_vision` MCP tool. Confirm: solid warm segments (~63GB) within the budget marker, a hatched coder segment extending toward but not past the cap marker, green/ok coloring, and the peak line reading `≤ cap`.

- [ ] **Step 3: Verify the tight state.** Select the **512gb** profile (or, if its models aren't pulled, temporarily point a scratch profile at large sizes). Screenshot + `local_vision`: warm bar past the budget marker, amber coloring, peak line reading `> cap`. Record both VLM observations in the commit message or PR notes.

- [ ] **Step 4: Commit**

```bash
git add app/profiles.html
git commit -m "feat(ui): memory bar shows warm set vs budget + transient peak"
```

---

## Self-Review

**Spec coverage:** warm schema + budget + version (Task 1) ✓; per-tier warm sets via the uniform `["general","embedding"]` (Task 1 data) ✓; keep-warm ticker (Task 5) ✓; differentiated keep-alive (Task 4) ✓; warm endpoint scope (Task 2) ✓; memory endpoint + state math (Task 3) ✓; bar redesign with hatched peak + states (Task 6) ✓; migration/`warm` defaults (Task 1 tests) ✓. Out-of-scope items (MCP-path keep_alive, KV modeling, UI-editable warm sets) intentionally absent.

**Type consistency:** `warm_model_names(data)` returns `set[str]`, consumed by `keep_alive_for` (Task 4) and `warm_ping_targets` (Task 5) and the memory route (Task 3 resolves warm names inline with the same `tasks[k] for k in warm_keys` rule). `state` strings `ok|tight|thrash` consistent between Task 3 impl, its tests, and Task 6's color mapping. `WARM_BUDGET_FRACTION` defined in Task 1, imported in Tasks 3. Endpoint JSON keys identical between Task 3's `jsonify(...)`, its tests, and Task 6's consumer.

**Placeholder scan:** no TBD/TODO; every code step shows full code; the one runtime-dependent verification (Task 6 visual check) is explicit about tool (`local_vision`) and expected observation.

**Note for executor:** Tasks 1–5 are pure TDD (unit-tested). Task 6 is UI; its correctness math is in Task 3, and it's verified visually via `local_vision` on the running app (the spec's "render and look" rule). If the 512gb models aren't pulled locally, verify the `tight` state against a scratch profile with large `max_ram_gb`/sizes rather than downloading ~400GB.
