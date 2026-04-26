# Super Puppy Troubleshooting

Operational issues observed in the wild and how to fix them. Each entry is dated so you can judge whether it's still relevant.

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
