# Super Puppy Troubleshooting

Operational issues observed in the wild and how to fix them. Each entry is dated so you can judge whether it's still relevant.

---

## Install finished but Ollama / MLX never got installed (no models)

**2026-06-24 — install.sh.**

After `./install.sh`, the menu bar app is running and `~/.config/local-models/` is populated, but `command -v ollama` / `mlx-openai-server` come back empty and no models were pulled.

Cause: `install.sh` runs under `set -euo pipefail`. The dependency-install block (`brew install ollama`, `uv tool install mlx-openai-server`) sits near the end, and it used to `exit 1` the instant any runtime was missing — *before* the model-pull step. A single flaky `brew`/`uv` download therefore aborted the whole installer, leaving a half-installed state. Because the menu bar app is kept alive by launchd from a prior run, everything *looked* fine.

**Diagnose:**

```bash
for b in ollama mlx-openai-server uv; do printf '%-20s' "$b:"; command -v "$b" || echo MISSING; done
```

**Fix:** the installer now reports missing runtimes loudly and finishes instead of aborting, and re-prints the punch-list as its last output. Install whatever it lists, then re-run — `install.sh` is idempotent:

```bash
brew install ollama
uv tool install --python 3.12 mlx-openai-server
./install.sh           # re-runs cleanly; pulls models for your profile
```

---

## Installer pulled zero models / `OLLAMA_MODELS[@]: unbound variable`

**2026-06-24 — install.sh.**

The installer printed `Pulling models for '<profile>' profile...`, then either hung silently for 30s or crashed with `line NNN: OLLAMA_MODELS[@]: unbound variable`.

Root cause was a cascade. `install.sh` derived its model list from `~/.config/local-models/profiles.json`, which is seeded from `DEFAULT_PROFILES`. That file was only ever written by the **profile server**, which the menu bar app starts **only when remote access is enabled** (`if self.desktop and self.remote_access_enabled` in `menubar.py`). A fresh install with remote access off therefore never got `profiles.json`. The installer waited 30s **silently** (no progress), then `OLLAMA_MODELS` stayed empty and `"${OLLAMA_MODELS[@]}"` aborted under `set -u` on bash 3.2 (macOS default), where expanding an empty array is an "unbound variable" error. A stale `menubar.lock` (dead PID) blocking the app from starting made it worse.

**Fixes:**
- `DEFAULT_PROFILES`/`PROFILES_VERSION` moved to `lib/models.py` (shared).
- Menu bar app seeds `profiles.json` on startup via `seed_profiles_if_missing()`, regardless of remote access.
- `install.sh` seeds `profiles.json` directly from `lib.models` before pulling, so it no longer races the app; the leftover wait is a fallback and now shows a countdown.
- All empty-array expansions guarded for bash 3.2 + `set -u`.

**If you hit the stale-lock symptom** (`Already running (pid NNNN). Exiting.` looping in `/tmp/local-models-menubar.log` with that PID dead):

```bash
rm -f ~/.config/local-models/menubar.lock
launchctl unload ~/Library/LaunchAgents/com.local-models.menubar.plist
launchctl load   ~/Library/LaunchAgents/com.local-models.menubar.plist
```

---

## Ollama MLX image runner: `libmlxc.dylib not found`

**2026-04-16 — Ollama 0.20.7, M3 Ultra.**

MLX-backed image-gen models (`x/z-image-turbo:bf16`, `x/flux2-klein:*`) spawn a separate `ollama mlx-runner` subprocess that dlopens `libmlxc.dylib` from `/Applications/Ollama.app/Contents/Resources/mlx_metal_v{3,4}/`. Lookup is env-driven via `OLLAMA_LIBRARY_PATH` — if the parent `ollama serve` was launched without it, every image-gen call returns:

```
HTTP 500: mlx runner failed: Error: failed to initialize MLX: libmlxc.dylib not found
```

The LLM path is unaffected because GGML lives next to the ollama binary and doesn't need the env var.

**Diagnose:**

```bash
ps eww -p $(pgrep -f "ollama serve") | tr ' ' '\n' | grep OLLAMA_LIBRARY_PATH
```

Empty output → this bug.

**Fix:** restart Ollama (the Electron launcher normally populates the var). Belt-and-suspenders before relaunch:

```bash
launchctl setenv OLLAMA_LIBRARY_PATH /Applications/Ollama.app/Contents/Resources
```

**Workaround without restart:** the `local_image` MCP tool dispatches via `mflux_command(...)` as a direct subprocess (`mcp/local-models-server.py:921`), bypassing Ollama entirely. Same path for `local_image_edit` via `mflux-generate-kontext`. Image gen continues to work even when Ollama's MLX runner is broken.

---

## HuggingFace large downloads: silent zombie hangs

**2026-04 — observed on `mlx-community/Qwen3.5-397B-A17B-8bit-gs32`, ~400 GB.**

Resume works: `.incomplete` files plus the content cache let `hf download` pick up wherever it stopped. For multi-hour downloads the minimum viable wrapper is:

```bash
caffeinate -i bash -c 'until hf download <ORG>/<REPO> --max-workers 4; do sleep 30; done'
```

`caffeinate -i` prevents idle-sleep (a real failure mode for overnight downloads). Do **not** set `HF_HUB_ENABLE_HF_TRANSFER=1` — parallel-chunk resume is fragile on disconnect.

**Known failure mode:** `hf download` on very large repos can enter a silent zombie state — process alive at 0% CPU, no ESTABLISHED sockets, never exits. `HF_HUB_DOWNLOAD_TIMEOUT` does not catch it.

No validated automated recovery yet. An external stall-kill watchdog with a tight threshold (180–600 s) thrashes via `-9`-truncated `.incomplete` files faster than bytes can catch up, producing **negative** net progress. If you need a watchdog, the kill threshold has to be long enough that post-kill re-download can exceed the old high-water mark (likely 30 min+), and the high-water counter must reset on kill.

**Until that's validated, prefer manual restart:** `kill -9` the python process, let the `until` loop resume.

---

## Wedged install: menu bar icon flickers in and out forever

**Symptom:** the menu bar icon appears and immediately disappears in a tight cycle. `launchctl list | grep local-models` shows a PID that keeps changing every couple of seconds. The Playground and MCP server are unreachable. Cmd-Q does nothing — the app is crashing before it can install the quit handler.

**What happened:** auto-update applied a new tag, the new code crashed within the 90 s crash window, the launcher rolled back to the previous commit — and *the rolled-back commit also fails to launch*. After the rollback completes, the launcher clears its update markers (`update_started`, `update_pre_hash`, `launch_attempted`), so on subsequent restarts it has no memory it just rolled back. launchd's `KeepAlive` faithfully respawns the .app every time it crashes. Forever.

Common causes for "rollback target also broken":

- An environmental change since the rollback target was current — e.g., `brew upgrade` swapped out `uv`, `python@3.12`, `mflux`, or another tool both versions depend on.
- `OLLAMA_LIBRARY_PATH` got unset (see the libmlxc.dylib entry above).
- A native dependency drifted (Python venv corruption, codesign requirements changed).
- Disk full or permissions on `~/.cache/uv` or `~/.config/local-models/`.

**Recover manually:**

```bash
# Stop launchd from restarting it while you fix things.
touch ~/.config/local-models/stay_down

# Wait a few seconds for the current crashed instance to be reaped.
sleep 5

# Roll back to a tag you trust.  `git tag --sort=-v:refname | head -10`
# lists recent tags; pick one from before the breakage.
git -C ~/super-puppy checkout v1.0.10   # ← replace with your known-good tag

# Clear update markers so the launcher doesn't try to re-rollback the
# moment it comes up.
rm -f ~/.config/local-models/update_started \
      ~/.config/local-models/update_pre_hash \
      ~/.config/local-models/launch_attempted \
      ~/.config/local-models/update_skipped

# Bring it back up.  `stay_down` is consumed once on first boot, so the
# kickstart below will succeed even though we wrote it above.
launchctl kickstart -k gui/$(id -u)/com.local-models.menubar
```

If the menu bar comes back up cleanly and stays up for a minute, you're good. The next signed tag pushed to origin will resume normal auto-updates (provided the rolled-back commit is itself signed and verifiable).

**If even an older tag won't launch:** the problem is environmental, not code. Check `/tmp/local-models-menubar.log` and `/tmp/local-models-profile-server.log` for the crash trace. Most often it's a missing CLI tool — `which uv`, `which hf`, `which mflux-generate` should all resolve.

**Why we don't auto-recover from this:** the rollback path is the least-iteratively-testable code in the repo (you can't safely simulate it without bricking your install), and recovery code that bugs out is harder to debug than the original problem. The pragmatic answer is a clear runbook plus eyeballs.

---

## glm-5.2 (ds4) — build, run, and quirks

**Since 2026-07, glm-5.2 on the 512GB tier is served by [antirez/ds4](https://github.com/antirez/ds4)** (glm5.2 branch, pinned commit `bd89932`), not mlx-openai-server. The old pinned mlx-lm patch script is retired and deleted; if a doc tells you to run it, you're reading an old release.

**Layout** (default `~/.local/share/super-puppy/ds4`, override with `DS4_DIR` in `~/.config/local-models/network.conf`):

```
$DS4_DIR/
├── ds4-server                 # built by `make ds4-server` at bd89932
├── metal/                     # metal shaders — resolved RELATIVE TO CWD
├── ds4flash.gguf              # symlink → gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
└── gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf   # 244GiB, hf download antirez/GLM-5.2-GGUF
```

**Manual run** (what `start-local-models` does):

```bash
cd "$DS4_DIR" && ./ds4-server --metal --port 8002 --ctx 32768 -m ds4flash.gguf
```

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

## Menu bar log spams `resolve_desktop_tailscale exception: No such file or directory: 'tailscale'`

Remote-access features shell out to the `tailscale` CLI by name. The macOS
**standalone ("macsys") Tailscale** — the recommended build — ships its CLI
*only inside the app bundle* (`/Applications/Tailscale.app/Contents/MacOS/Tailscale`)
and never puts a `tailscale` on `PATH`, so every ~32s probe fails with ENOENT.
The Tailscale app IS running and the tunnel works; only the CLI is unreachable.

Do **not** fix this with a bare symlink. The macsys binary derives its bundle
identity from its own executable path, so invoking it through a symlink outside
the bundle fatal-errors: `The current bundleIdentifier is unknown to the
registry`. The fix is a **wrapper** that `exec`s the full bundle path (exactly
what Tailscale's own `InstallTailscaleCLI.scpt` writes to `/usr/local/bin`).

Super Puppy ships that wrapper as `bin/tailscale` and `install.sh` /
`post-update.sh` link it to `~/.local/bin/tailscale` **only when** the app is
present and no other `tailscale` is already on `PATH` (so it never shadows a
Homebrew or official-CLI install). If you hit this on an existing install,
re-run `install.sh` or `bin/post-update.sh`, or link it by hand:

```sh
ln -sfn "$(git -C <repo> rev-parse --show-toplevel)/bin/tailscale" ~/.local/bin/tailscale
```

(`~/.local/bin` is first on the menu bar app's PATH — the C launcher sets it —
so the running app picks it up on the next probe; no restart needed.)
