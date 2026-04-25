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
