# RAM-tier Model Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four ad-hoc profiles (laptop/desktop/everyday/maximum) with four RAM-tier presets (`32gb`/`64gb`/`128gb`/`512gb`), each running the best models the tier's RAM+GPU can drive with headroom, and make install pull only the chosen profile's *missing* models.

**Architecture:** `DEFAULT_PROFILES` (the data) and a shared `migrate_profiles()` move into `lib/models.py` so menubar, profile-server, and install all agree. A version bump triggers migration that drops the retired preset names. `install.sh` and `start-local-models` collapse from two RAM-selected MLX configs to one (`config.yaml`, all `on_demand`), and install diffs the chosen profile's model set against what's already on disk before pulling.

**Tech Stack:** Python 3.12 (`lib/`, `app/`), bash (`install.sh`, `bin/start-local-models`), YAML (MLX server config), pytest.

## Global Constraints

- `PROFILES_VERSION` becomes **26** (currently 25). Bump forces refresh on every machine.
- Profile keys: `32gb`, `64gb`, `128gb`, `512gb`. Labels: `"32 GB"`…`"512 GB"`. `max_ram_gb`: 32/64/128/512. Default `active`: `64gb`.
- Retired preset names (dropped on migration): `laptop`, `desktop`, `everyday`, `maximum`.
- Model→backend by string: `:` → Ollama tag; `/` → HF repo (downloaded, invoked by mflux/mlx-audio); otherwise → MLX served-name resolved in the MLX config.
- bash scripts run under `set -euo pipefail` on macOS **bash 3.2** — guard empty-array expansions (`[ ${#arr[@]} -gt 0 ]` before `"${arr[@]}"`).
- No secrets in any file. Run `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py tests/test_profile_server.py tests/test_error_handling.py -q` before each commit touching Python.
- Conventional Commits; one logical change per commit; sign commits (1Password SSH agent must be unlocked).

---

## File Structure

- `lib/models.py` — **modify**: rewrite `DEFAULT_PROFILES`, bump `PROFILES_VERSION`, add `RETIRED_PROFILE_NAMES` + `migrate_profiles()`. Single source of truth.
- `app/profile-server.py` — **modify**: `load_profiles()` delegates migration to `lib.models.migrate_profiles`.
- `app/menubar.py` — **modify**: `seed_profiles_if_missing()`/`_ensure_active_profile()` migrate stale versions; `pick_profile_for_ram()` fallback `laptop` → `32gb`.
- `config/mlx-server/config.yaml` — **modify**: one comprehensive config; add served-names `whisper-v3-turbo`, `ui-venus`, `glm-5.2`; keep `qwen3.5-small`, `llama-3b`.
- `config/mlx-server/config-laptop.yaml` — **delete**: collapsed into `config.yaml`.
- `bin/start-local-models` — **modify**: `pick_mlx_config()` always returns `config.yaml` (above a hard floor).
- `install.sh` — **modify**: RAM→tier suggestion, profile prompt list, drop per-profile `MLX_CONFIG` case, download-only-missing logic.
- `tests/test_core.py` — **modify**: profile-contract + migrate tests.
- `tests/test_profile_server.py` — **modify**: migration drops retired presets.
- `tests/test_tools_smoke_laptop.py`, `tests/test_tools_smoke_everyday.py` — **modify**: rename model maps to the new tiers (`32gb`/`512gb`).

---

## Task 1: DEFAULT_PROFILES, version, and shared migration in `lib/models.py`

**Files:**
- Modify: `lib/models.py` (the `PROFILES_VERSION` / `DEFAULT_PROFILES` block, ~line 396; add `migrate_profiles` after `DEFAULT_PROFILES`)
- Test: `tests/test_core.py` (`TestDefaultProfilesSeeding` class)

**Interfaces:**
- Produces: `DEFAULT_PROFILES: dict` (keys `32gb/64gb/128gb/512gb`, each `{label, description, max_ram_gb, tasks}`), `PROFILES_VERSION: int = 26`, `RETIRED_PROFILE_NAMES: frozenset[str]`, `migrate_profiles(data: dict) -> dict` (returns a dict shaped like DEFAULT_PROFILES: refreshes presets, drops retired names, preserves genuinely-custom profiles, fixes `active`).

- [ ] **Step 1: Write the failing tests** in `tests/test_core.py`, appending to `class TestDefaultProfilesSeeding`:

```python
    def test_tiers_present_and_capped(self):
        profs = menubar.DEFAULT_PROFILES["profiles"]
        assert set(profs) == {"32gb", "64gb", "128gb", "512gb"}
        assert [profs[k]["max_ram_gb"] for k in ("32gb", "64gb", "128gb", "512gb")] == [32, 64, 128, 512]
        assert menubar.DEFAULT_PROFILES["active"] == "64gb"

    def test_migrate_drops_retired_presets_and_fixes_active(self):
        from lib.models import migrate_profiles, PROFILES_VERSION
        old = {"version": 25, "active": "everyday",
               "profiles": {"everyday": {"tasks": {"code": "x:1b"}},
                            "mine": {"label": "Mine", "max_ram_gb": 16, "tasks": {"code": "c:1b"}}}}
        out = migrate_profiles(old)
        assert out["version"] == PROFILES_VERSION
        assert "everyday" not in out["profiles"]          # retired preset dropped
        assert "mine" in out["profiles"]                  # real custom kept
        assert set(out["profiles"]) >= {"32gb", "64gb", "128gb", "512gb"}
        assert out["active"] in out["profiles"]           # active repaired (was "everyday")

    def test_migrate_preserves_valid_active(self):
        from lib.models import migrate_profiles
        out = migrate_profiles({"version": 25, "active": "mine",
                                "profiles": {"mine": {"max_ram_gb": 8, "tasks": {}}}})
        assert out["active"] == "mine"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -k "tiers_present or migrate" -q`
Expected: FAIL (KeyError on new tier names / `ImportError: cannot import name 'migrate_profiles'`).

- [ ] **Step 3: Rewrite `DEFAULT_PROFILES` and bump the version** in `lib/models.py`. Replace the existing `PROFILES_VERSION = 25` line with `26` and replace the whole `DEFAULT_PROFILES = { ... }` literal with:

```python
DEFAULT_PROFILES = {
    "version": PROFILES_VERSION,
    "active": "64gb",
    "profiles": {
        "32gb": {
            "label": "32 GB",
            "description": "Base M5 / M1 Max class — small, fast models",
            "max_ram_gb": 32,
            "tasks": {
                "code": "qwen3.5-small",
                "general": "qwen3.5-small",
                "reasoning": "qwen3.5-small",
                "long_context": "qwen3.5-small",
                "translation": "qwen3.5-small",
                "vision": "qwen3.5-small",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/Kokoro-82M-bf16",
                "embedding": "embeddinggemma:300m",
                "image_gen": "x/flux2-klein:latest",
            },
        },
        "64gb": {
            "label": "64 GB",
            "description": "M5 / mid GPU — dense 27B workhorse",
            "max_ram_gb": 64,
            "tasks": {
                "code": "qwen3.6:27b-coding-mxfp8",
                "general": "qwen3.6:27b-mlx",
                "reasoning": "qwen3.6:27b-mlx",
                "long_context": "qwen3.6:27b-mlx",
                "translation": "qwen3.6:27b-mlx",
                "vision": "qwen3.6:27b-mlx",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/fishaudio-s2-pro-8bit-mlx",
                "embedding": "qwen3-embedding:8b",
                "unfiltered": "dolphin3:8b",
                "computer_use": "ui-venus",
                "image_gen": "x/flux2-klein:latest",
            },
        },
        "128gb": {
            "label": "128 GB",
            "description": "M5 Max class — dense bf16 + strong vision",
            "max_ram_gb": 128,
            "tasks": {
                "code": "qwen3-coder-next:latest",
                "general": "qwen3.6:27b-mlx-bf16",
                "reasoning": "qwen3.6:27b-mlx-bf16",
                "long_context": "qwen3.6:27b-mlx-bf16",
                "translation": "qwen3.6:27b-mlx-bf16",
                "vision": "qwen3.6:27b-mlx-bf16",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/fishaudio-s2-pro-8bit-mlx",
                "embedding": "qwen3-embedding:8b",
                "unfiltered": "dolphin3:8b",
                "computer_use": "ui-venus",
                "image_gen": "x/z-image-turbo:bf16",
                "image_edit": "black-forest-labs/FLUX.1-Kontext-dev",
                "video": "AITRADER/Wan2.2-T2V-A14B-mlx-bf16",
            },
        },
        "512gb": {
            "label": "512 GB",
            "description": "M3 Ultra class — frontier",
            "max_ram_gb": 512,
            "tasks": {
                "code": "qwen3-coder-next:latest",
                "general": "glm-5.2",
                "reasoning": "glm-5.2",
                "long_context": "glm-5.2",
                "translation": "glm-5.2",
                "vision": "qwen3.5:122b",
                "transcription": "whisper-v3-turbo",
                "tts": "mlx-community/fishaudio-s2-pro-8bit-mlx",
                "embedding": "qwen3-embedding:8b",
                "unfiltered": "dolphin3:8b",
                "computer_use": "ui-venus",
                "image_gen": "x/z-image-turbo:bf16",
                "image_edit": "black-forest-labs/FLUX.1-Kontext-dev",
                "video": "AITRADER/Wan2.2-T2V-A14B-mlx-bf16",
            },
        },
    },
}
```

Update the comment block above `PROFILES_VERSION` to describe the RAM-tier scheme (replace the qwen3.6-MLX-specific note).

- [ ] **Step 4: Add the migration helper** immediately after the `DEFAULT_PROFILES` literal:

```python
RETIRED_PROFILE_NAMES = frozenset({"laptop", "desktop", "everyday", "maximum"})


def migrate_profiles(data: dict) -> dict:
    """Bring an on-disk profiles dict up to the current presets.

    Refreshes the preset profiles, drops retired preset names, preserves
    genuinely-custom profiles (anything not a current or retired preset), and
    repairs `active` if it no longer names a profile that exists.
    """
    refreshed = {**DEFAULT_PROFILES, "profiles": dict(DEFAULT_PROFILES["profiles"])}
    for name, profile in (data.get("profiles") or {}).items():
        if name in DEFAULT_PROFILES["profiles"] or name in RETIRED_PROFILE_NAMES:
            continue
        refreshed["profiles"][name] = profile
    active = data.get("active")
    refreshed["active"] = active if active in refreshed["profiles"] else DEFAULT_PROFILES["active"]
    return refreshed
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -q`
Expected: PASS (the existing `test_every_preset_resolves_pullable_models` also still passes — every tier has ≥1 `:`/`/` model).

- [ ] **Step 6: Commit**

```bash
git add lib/models.py tests/test_core.py
git commit -m "feat(profiles): RAM-tier presets + shared migrate_profiles in lib"
```

---

## Task 2: Profile-server uses the shared migration

**Files:**
- Modify: `app/profile-server.py` (`load_profiles()`, ~line 1230; imports already include `DEFAULT_PROFILES`, `PROFILES_VERSION` — add `migrate_profiles`)
- Test: `tests/test_profile_server.py`

**Interfaces:**
- Consumes: `lib.models.migrate_profiles`, `DEFAULT_PROFILES`, `PROFILES_VERSION`.

- [ ] **Step 1: Write the failing test** in `tests/test_profile_server.py` (use the module's existing `ps` import and `tmp_path` pattern):

```python
    def test_load_profiles_migration_drops_retired(self, tmp_path):
        import json
        from pathlib import Path
        path = tmp_path / "profiles.json"
        path.write_text(json.dumps({"version": 25, "active": "everyday",
                                    "profiles": {"everyday": {"tasks": {}},
                                                 "custom": {"max_ram_gb": 8, "tasks": {}}}}))
        with patch.object(ps, "PROFILES_FILE", path):
            out = ps.load_profiles()
        assert out["version"] == ps.PROFILES_VERSION
        assert "everyday" not in out["profiles"]
        assert "custom" in out["profiles"]
        assert "64gb" in out["profiles"]
        assert out["active"] in out["profiles"]
```

(If `test_profile_server.py` lacks `from unittest.mock import patch`, add it.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -k migration_drops_retired -q`
Expected: FAIL (`everyday` still present — old migration preserves it as custom).

- [ ] **Step 3: Rewrite `load_profiles()`** in `app/profile-server.py` to delegate the version-mismatch branch. Add `migrate_profiles` to the `from lib.models import (...)` block, then:

```python
def load_profiles():
    if PROFILES_FILE.exists():
        try:
            data = json.loads(PROFILES_FILE.read_text())
            if data.get("version", 0) == PROFILES_VERSION:
                return data
            refreshed = migrate_profiles(data)
            save_profiles(refreshed)
            return refreshed
        except Exception:
            pass
    save_profiles(DEFAULT_PROFILES)
    return {**DEFAULT_PROFILES}
```

- [ ] **Step 4: Run the profile-server suite**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -q`
Expected: PASS. (Existing version-bump tests should still pass — they assert `version == PROFILES_VERSION` and active validity, which `migrate_profiles` preserves.)

- [ ] **Step 5: Commit**

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "refactor(profile-server): use shared migrate_profiles, drop retired presets"
```

---

## Task 3: Menubar migrates stale versions on startup; RAM fallback

**Files:**
- Modify: `app/menubar.py` (`seed_profiles_if_missing()` ~line 729; `pick_profile_for_ram()` line 743; import `migrate_profiles` + `PROFILES_VERSION` in the `from lib.models import (...)` block ~line 477)
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: `lib.models.migrate_profiles`, `PROFILES_VERSION`.
- Produces: `seed_profiles_if_missing()` now also migrates a stale-version file in place (returns `True` if it wrote).

Rationale: the profile server only runs when remote access is on, so non-server machines need the menubar to migrate too.

- [ ] **Step 1: Write the failing tests** in `tests/test_core.py` `TestDefaultProfilesSeeding`:

```python
    def test_seed_migrates_stale_version(self, tmp_path):
        prof = tmp_path / "profiles.json"
        prof.write_text(json.dumps({"version": 1, "active": "laptop",
                                    "profiles": {"laptop": {"tasks": {}}}}))
        with patch.object(menubar, "PROFILES_FILE", str(prof)):
            assert menubar.seed_profiles_if_missing() is True
            out = json.loads(prof.read_text())
        assert out["version"] == menubar.PROFILES_VERSION
        assert "laptop" not in out["profiles"]
        assert "64gb" in out["profiles"]

    def test_pick_profile_fallback_is_32gb(self):
        # no profile fits 8GB and none named in fallback set except presets
        assert menubar.pick_profile_for_ram(8, menubar.DEFAULT_PROFILES["profiles"]) == "32gb"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -k "stale_version or fallback_is_32gb" -q`
Expected: FAIL (stale file not migrated; fallback returns `None`/`laptop`).

- [ ] **Step 3: Add `PROFILES_VERSION` and `migrate_profiles` to the menubar import** (`from lib.models import (...)`), then update `seed_profiles_if_missing()`:

```python
def seed_profiles_if_missing():
    """Seed presets if profiles.json is absent/empty, or migrate a stale version.

    The profile server normally owns this file, but it only starts when remote
    access is enabled — so on a non-server machine the menu bar must seed and
    migrate, or the installer's model-pull resolves nothing / stale models.
    Returns True if it wrote.
    """
    data = load_profiles()
    if not data.get("profiles"):
        save_profiles({**DEFAULT_PROFILES})
        return True
    if data.get("version") != PROFILES_VERSION:
        save_profiles(migrate_profiles(data))
        return True
    return False
```

- [ ] **Step 4: Change the `pick_profile_for_ram` fallback** (line 743). Replace `return "laptop" if "laptop" in profiles else next(iter(profiles), None)` with:

```python
    return "32gb" if "32gb" in profiles else next(iter(profiles), None)
```

- [ ] **Step 5: Run the tests**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_core.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/menubar.py tests/test_core.py
git commit -m "feat(menubar): migrate stale profiles on startup; 32gb RAM fallback"
```

---

## Task 4: Collapse MLX configs to one; add new served-names

**Files:**
- Modify: `config/mlx-server/config.yaml`
- Delete: `config/mlx-server/config-laptop.yaml`
- Modify: `bin/start-local-models` (`pick_mlx_config()`, ~line 64)
- Modify: `install.sh` (uninstall block removes `config-laptop.yaml`? no — `mlx-server` dir is removed wholesale already; just stop referencing it)

**Interfaces:**
- Produces: served-names `whisper-v3-turbo`, `ui-venus`, `glm-5.2` (plus existing `qwen3.5-small`, `llama-3b`) available from a single `config.yaml`.

- [ ] **Step 1: Add the new served-model entries** to `config/mlx-server/config.yaml` (mirror the existing entry shapes — `whisper` type for ASR, `multimodal` for VLMs, `lm` for text):

```yaml
  # Speech-to-text (turbo) — ~6x faster than v3, near-identical accuracy
  - model_path: mlx-community/whisper-large-v3-turbo
    model_type: whisper
    served_model_name: whisper-v3-turbo
    on_demand: true
    on_demand_idle_timeout: 300

  # Computer use (UI-Venus) — GUI grounding, ~8B
  - model_path: mlx-community/UI-Venus-1.5-8B-bf16
    model_type: multimodal
    served_model_name: ui-venus
    context_length: 32768
    on_demand: true
    on_demand_idle_timeout: 300

  # Frontier (512GB tier) — GLM-5.2, ~418GB at 4-bit
  - model_path: mlx-community/GLM-5.2-4bit
    model_type: lm
    served_model_name: glm-5.2
    context_length: 131072
    on_demand: true
    on_demand_idle_timeout: 300
```

The existing `qwen3.5-small` (Qwen3.5-9B-4bit, `model_type: lm`) is reused for the 32gb workhorse. **⚠️ Verify-flag #1 (resolved in Task 6):** if Qwen3.5-9B supports vision, change its `model_type` to `multimodal` so 32gb `vision` works; otherwise change the 32gb `vision` task to a small dedicated vision model. Leave it `lm` for now; Task 6 decides.

You may prune retired served-names (`ui-tars-72b`, `qwen3.5-397b-8bit`, `nemotron-super`, `qwen3.6-35b-bf16`, the 4-bit twin, `holo3-35b`, `whisper-v3`) since no preset references them; keep `llama-3b` (health check) and `qwen3.5-fast`.

- [ ] **Step 2: Delete the laptop config**

```bash
git rm config/mlx-server/config-laptop.yaml
```

- [ ] **Step 3: Update `pick_mlx_config()`** in `bin/start-local-models` (lines ~64-79) to always use the single config above the MLX floor:

```bash
pick_mlx_config() {
    local ram_gb
    ram_gb=$(get_ram_gb)
    if [ "$ram_gb" -ge 32 ]; then
        echo "$MLX_CONFIG_DIR/config.yaml"
        echo "  Detected ${ram_gb}GB RAM — using MLX config (on-demand models)." >&2
    else
        echo ""
        echo "  Detected ${ram_gb}GB RAM — too little for MLX server, skipping." >&2
    fi
}
```

- [ ] **Step 4: Validate the YAML parses**

Run: `uv run --with pyyaml python3 -c "import yaml; d=yaml.safe_load(open('config/mlx-server/config.yaml')); names=[m['served_model_name'] for m in d['models']]; print(names); assert {'whisper-v3-turbo','ui-venus','glm-5.2','qwen3.5-small'} <= set(names)"`
Expected: prints the served-name list and exits 0.

- [ ] **Step 5: Verify install.sh no longer references config-laptop**

Run: `grep -n "config-laptop" install.sh bin/start-local-models || echo "clean"`
Expected: `clean` (if any remain, remove them — install.sh's MLX_CONFIG selection is handled in Task 5).

- [ ] **Step 6: Commit**

```bash
git add config/mlx-server/config.yaml bin/start-local-models
git rm config/mlx-server/config-laptop.yaml
git commit -m "feat(mlx): one on-demand MLX config; add whisper-turbo, ui-venus, glm-5.2 served-names"
```

---

## Task 5: install.sh — tier prompt, RAM mapping, one config, download-only-missing

**Files:**
- Modify: `install.sh` (RAM→profile suggestion ~lines 617-643; `MLX_CONFIG` case ~lines 652-656; the pull loops ~lines 779-811)

**Interfaces:**
- Consumes: `DEFAULT_PROFILES` tiers from Task 1 (via the seeded `profiles.json`).

- [ ] **Step 1: Update the RAM→suggested-tier block** (~lines 617-635). Replace the `everyday/desktop/laptop` mapping with tier names and one config path:

```bash
RAM_GB=$(sysctl -n hw.memsize | awk '{printf "%d", $1 / 1073741824}')
MLX_CONF_DIR="$HOME/.config/mlx-server"
MLX_CONFIG="$MLX_CONF_DIR/config.yaml"
if   [ "$RAM_GB" -ge 512 ]; then SUGGESTED_PROFILE="512gb"
elif [ "$RAM_GB" -ge 128 ]; then SUGGESTED_PROFILE="128gb"
elif [ "$RAM_GB" -ge 64  ]; then SUGGESTED_PROFILE="64gb"
else                              SUGGESTED_PROFILE="32gb"
fi
SUGGESTED_LABEL="${SUGGESTED_PROFILE/gb/ GB}"
```

- [ ] **Step 2: Update the profile prompt list** (~line 640):

```bash
echo "  Available profiles: 32gb, 64gb, 128gb, 512gb, skip"
printf "  Pull models for which profile? [%s] " "$SUGGESTED_PROFILE"
```

- [ ] **Step 3: Delete the per-profile MLX_CONFIG case** (~lines 652-656) — `MLX_CONFIG` is now fixed to `config.yaml` (set in Step 1). Remove the whole `case "$PROFILE_NAME" in everyday|maximum) ... esac` block.

- [ ] **Step 4: Add download-only-missing filtering** to both pull loops (~lines 779-811). For Ollama, build the present-set once and skip; same for HF via the cache check. Replace the Ollama loop body and the HF loop body:

```bash
    # Build the set of already-present Ollama tags once.
    PRESENT_OLLAMA=""
    command -v ollama > /dev/null && PRESENT_OLLAMA=$(ollama list 2>/dev/null | awk 'NR>1{print $1}')

    if [ ${#OLLAMA_MODELS[@]} -eq 0 ]; then
        echo "  No Ollama models for the '$PROFILE_NAME' profile."
    elif ! command -v ollama > /dev/null; then
        echo "  Skipping Ollama model pulls — ollama is not installed."
    else
        total=${#OLLAMA_MODELS[@]}; current=0; pulled=0
        for model in "${OLLAMA_MODELS[@]}"; do
            current=$((current + 1))
            if printf '%s\n' "$PRESENT_OLLAMA" | grep -qx "$model"; then
                echo "  [$current/$total] ollama: $model — already present, skipping"
                continue
            fi
            echo "  [$current/$total] ollama: $model — pulling"
            ollama pull "$model" || echo "    WARNING: failed to pull $model"
            pulled=$((pulled + 1))
        done
        echo "  Ollama: $pulled pulled, $((total - pulled)) already present."
    fi
```

For HuggingFace, use `hf download`'s own cache check is implicit, but to avoid the network round-trip use the cache dir. Replace the HF loop:

```bash
    if [ ${#HF_MODELS[@]} -eq 0 ]; then
        echo "  No HuggingFace/MLX models for the '$PROFILE_NAME' profile."
    elif command -v hf > /dev/null; then
        # ... (existing hf auth block stays here) ...
        HF_CACHE="$HOME/.cache/huggingface/hub"
        total=${#HF_MODELS[@]}; current=0; pulled=0
        for model in "${HF_MODELS[@]}"; do
            current=$((current + 1))
            cache_name="models--${model//\//--}"
            if [ -d "$HF_CACHE/$cache_name/snapshots" ] && \
               [ -z "$(find "$HF_CACHE/$cache_name/blobs" -name '*.incomplete' 2>/dev/null)" ]; then
                echo "  [$current/$total] huggingface: $model — already present, skipping"
                continue
            fi
            echo "  [$current/$total] huggingface: $model — downloading"
            hf download "$model" || true
            pulled=$((pulled + 1))
        done
        echo "  HuggingFace: $pulled downloaded, $((total - pulled)) already present."
    else
        echo "  WARNING: hf install failed. HuggingFace models will download on first use."
    fi
```

(Keep the existing HF-auth block from the prior change at the top of the `elif command -v hf` branch, before the loop.)

- [ ] **Step 5: Syntax-check**

Run: `bash -n install.sh && echo OK`
Expected: `OK`.

- [ ] **Step 6: Dry-resolve each tier** (proves the seeded profile + present-skip logic resolves sane sets). With a temp HOME seeded from lib:

```bash
TMPH=$(mktemp -d)
HOME="$TMPH" python3 -c "import json,sys;sys.path.insert(0,'.');from lib.models import DEFAULT_PROFILES,PROFILES_FILE;PROFILES_FILE.parent.mkdir(parents=True,exist_ok=True);PROFILES_FILE.write_text(json.dumps(DEFAULT_PROFILES))"
for p in 32gb 64gb 128gb 512gb; do
  echo "== $p =="
  HOME="$TMPH" python3 -c "
import json,pathlib,os
d=json.loads((pathlib.Path(os.environ['HOME'])/'.config/local-models/profiles.json').read_text())
t=d['profiles']['$p']['tasks']
print('ollama:', sorted({m for m in t.values() if ':' in m}))
print('hf    :', sorted({m for m in t.values() if '/' in m and ':' not in m}))
print('mlx   :', sorted({m for m in t.values() if ':' not in m and '/' not in m}))"
done
rm -rf "$TMPH"
```
Expected: `512gb` shows `mlx: ['glm-5.2','ui-venus','whisper-v3-turbo']` etc.; no `<UNRESOLVED>`.

- [ ] **Step 7: Commit**

```bash
git add install.sh
git commit -m "feat(install): RAM-tier prompt, single MLX config, download only missing models"
```

---

## Task 6: Resolve verify-flags on real hardware (this 128GB machine)

**Files:** none (runtime verification); may modify `config/mlx-server/config.yaml` (32gb vision model_type) or `lib/models.py` (qwen3-coder-next tag, GLM REAP fallback, 32gb vision model) based on findings.

These are the spec's verify flags. Each step records a finding; only the ones that fail trigger an edit.

- [ ] **Step 1: `qwen3-coder-next` tag/size** — `ollama show qwen3-coder-next:latest 2>/dev/null | grep -i 'parameters\|quant'` and check `ollama.com/library/qwen3-coder-next/tags` for the ~52GB 4-bit tag. If `:latest` isn't the 4-bit ~52GB build, set the exact tag in `lib/models.py` (128gb + 512gb `code`). Commit if changed.

- [ ] **Step 2: `qwen3.5-9b` vision** — start the MLX server with the new config, then `curl -s localhost:8000/v1/chat/completions` with an image content part to `qwen3.5-small`. If it errors as text-only, either flip its `model_type` to `multimodal` in `config.yaml` (and re-test) or change the 32gb `vision` task to `qwen3.6:27b-mlx` (burst). Commit the chosen fix.

- [ ] **Step 3: `UI-Venus-1.5-8B` loads + grounds** — `ollama`-independent: confirm `mlx-community/UI-Venus-1.5-8B-bf16` loads via the computer_use path and returns a sane click target on a real screenshot (compare against the playbook's holo3 datapoint). If it fails to serve, fall back `computer_use` → `holo3-35b` (re-add its config entry). Record the result in `~/.claude/model-playbook.md`.

- [ ] **Step 4: `fish-s2-pro` via mlx-audio** — call `local_speak` with `mlx-community/fishaudio-s2-pro-8bit-mlx`; confirm audio comes back. If the installed `mlx-audio` can't serve it, fall back `tts` → Voxtral on 64/128/512. Record in the playbook.

- [ ] **Step 5: `GLM-5.2-4bit` footprint** — only if a 512GB machine is available; otherwise note as untested. If ~418GB resident leaves too little, switch the 512gb general/reasoning/long_context/translation to a REAP-pruned GLM repo. No change on this 128GB machine.

- [ ] **Step 6: Commit any fixes** with `fix(profiles): <what the verification changed>` and update the playbook entries.

---

## Task 7: Rename smoke-test model maps to the new tiers

**Files:**
- Modify: `tests/test_tools_smoke_laptop.py`, `tests/test_tools_smoke_everyday.py`

These hand-maintained maps mirror a profile and drive live models. Re-point them at the new tiers.

- [ ] **Step 1: Update `tests/test_tools_smoke_laptop.py`** — rename the module's `LAPTOP` map to the `32gb` tier model set (workhorse `qwen3.5-small`, `whisper-v3-turbo`, `embeddinggemma:300m`, `mlx-community/Kokoro-82M-bf16`, `x/flux2-klein:latest`); drop tasks the 32gb tier omits.

- [ ] **Step 2: Update `tests/test_tools_smoke_everyday.py`** — rename the `EVERYDAY` map to the `512gb` tier set (`glm-5.2`, `qwen3-coder-next:latest`, `qwen3.5:122b`, `ui-venus`, `whisper-v3-turbo`, `mlx-community/fishaudio-s2-pro-8bit-mlx`, etc.).

- [ ] **Step 3: Confirm collection still works** (these skip when services are down):

Run: `uv run --with pytest --with flask --with pyyaml --with requests --with mlx-audio pytest tests/test_tools_smoke_laptop.py tests/test_tools_smoke_everyday.py --collect-only -q`
Expected: collects without import errors.

- [ ] **Step 4: Full unit suite green**

Run: `uv run --with pytest --with flask --with pyyaml --with requests --with mlx-audio pytest tests/test_core.py tests/test_profile_server.py tests/test_error_handling.py tests/test_mcp_server.py tests/test_deployment.py tests/test_playground_coverage.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools_smoke_laptop.py tests/test_tools_smoke_everyday.py
git commit -m "test(smoke): re-point profile maps at 32gb/512gb tiers"
```

---

## Self-Review

**Spec coverage:** tiers + caps + labels (Task 1) ✓; MoE/dense/frontier model picks (Task 1 data) ✓; one MLX config + new served-names (Task 4) ✓; install tier prompt + RAM map + drop config-select (Task 5) ✓; download-only-missing (Task 5) ✓; migration drops retired presets, both server and non-server paths (Tasks 2+3) ✓; pick_profile_for_ram fallback (Task 3) ✓; tests incl. contract + migration (Tasks 1-3,7) ✓; seven verify flags (Task 6) ✓. Out-of-scope items (Qwen3-ASR, vision-line change) are not implemented, as intended.

**Note for executor:** Task 6 is verification-driven and may edit Task 1/4 outputs; run it on this 128GB machine before the final merge. `GLM-5.2` (512gb) can only be fully validated on a 512GB box — flag as untested elsewhere.
