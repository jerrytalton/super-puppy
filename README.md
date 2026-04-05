# Super Puppy

> **Requires Apple Silicon Mac** (M1 or later) with 64GB+ unified memory. macOS only.

Your Mac has a GPU that can run serious AI models — probably while it's sitting idle. Super Puppy turns it into a managed local model server: LLMs, vision, image generation, transcription, translation, text-to-speech, embeddings. Controlled from the menu bar, accessible over standard APIs, available to any tool on your network.

It works as a **server** or a **client** — and every client is also a server. Install it on a beefy desktop and it serves models over Tailscale. Install it on a laptop and it auto-discovers the desktop, routing requests to the bigger machine's GPU. When the desktop is unreachable — you're on a plane, at a coffee shop, whatever — the same tools keep working against local models on the laptop itself. Your code, your scripts, your Claude Code workflows never have to care which machine is doing the work. They hit the same APIs either way; Super Puppy handles the routing.

No cloud, no per-token billing — inference is fully local. Write against local AI APIs once, get the best available hardware transparently. (Initial model downloads and auto-update checks do require network access; see [Network Transparency](#network-transparency).)

Super Puppy is **not** a training or fine-tuning platform, a cloud service, or a production deployment tool. It's for people who want to run inference on hardware they own — for development, experimentation, creative work, and daily use. You need enough unified memory for the models you care about: 64GB gets you started, 128GB+ handles most things, 256GB+ runs everything.

## Quick Start

**Prerequisites:** Xcode Command Line Tools (provides `git`), Homebrew.

```bash
xcode-select --install      # if git is not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"  # if brew is not installed

git clone https://github.com/jerrytalton/super-puppy.git ~/super-puppy
cd ~/super-puppy
./install.sh
```

The installer walks you through everything: server vs. client role, network config, auth tokens, Tailscale for remote access, and which models to pull. Then:

```bash
start-local-models
```

## What You Get

### Standard APIs

Once Super Puppy is running, any application that speaks OpenAI or Ollama can use your local models.

**Ollama API** (port 11434) — chat, generation, embeddings:

```bash
curl http://localhost:11434/api/generate -d '{"model":"qwen3.5","prompt":"hello"}'

curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5",
  "messages": [{"role":"user","content":"explain quicksort"}]
}'

curl http://localhost:11434/api/embed -d '{"model":"all-minilm","input":"search query"}'
```

**OpenAI-compatible API** (port 8000) — MLX models via the standard OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen3.5-fast",
    messages=[{"role": "user", "content": "hello"}],
)

# Vision
response = client.chat.completions.create(
    model="qwen3.5-fast",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "file:///path/to/image.png"}},
        ],
    }],
)
```

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5-fast",
  "messages": [{"role":"user","content":"hello"}]
}'

curl http://localhost:8000/v1/models
```

### Playground

The menu bar app serves a web-based Playground where you can test any capability interactively — text generation, vision, image generation, transcription, TTS, translation. Open it from the Super Puppy menu or access it from other devices on your network. With Tailscale configured, the Playground is accessible from anywhere over HTTPS.

The Playground is a PWA — on iOS or iPadOS, open it in Safari and tap "Add to Home Screen" to install it as a standalone app. On Android, Chrome will prompt you to install it. Your phone becomes a frontend to your desktop's GPU.

### MCP Tools for Claude Code

If you use Claude Code, Super Puppy exposes all of its capabilities as MCP tools that Claude can call mid-conversation. Claude keeps doing what it's best at — architecture, debugging, complex reasoning — and offloads everything else to your hardware.

| Tool | What it does |
|------|-------------|
| `local_generate` | Code and text generation — auto-selects a coder or generalist |
| `local_review` | Second opinion on code from a different model architecture |
| `local_vision` | Analyze images on disk with a local vision model |
| `local_computer_use` | Plan GUI actions from a screenshot (observe only, no execution) |
| `local_image` | Generate images locally with a diffusion model |
| `local_image_edit` | Edit an existing image with a text prompt |
| `local_transcribe` | Audio to text |
| `local_speak` | Text to speech with voice presets or voice cloning |
| `local_translate` | Translate text or files |
| `local_candidates` | Run the same prompt against multiple models in parallel |
| `local_summarize` | Condense large files before reading them in full |
| `local_embed` | Generate embeddings for semantic search or clustering |
| `local_similarity_search` | Find files most related to a concept |
| `local_dispatch` / `local_collect` | Run a model in the background, collect results later |
| `local_models_status` | List available models and their capabilities |

You control which model backs each task from the menu bar app. Pull a new model and it's immediately available.

## Server and Client

The installer asks whether this machine is the model server or a client. The server runs models locally and serves them over Tailscale. Clients auto-discover the server and route requests to it:

| Environment | What happens |
|-------------|-------------|
| **Server** | All models run locally. Tailscale exposes APIs to clients. |
| **Client (server reachable)** | Routes to server via Tailscale. |
| **Client (server unreachable)** | Falls back to local models. |

All remote access uses Tailscale — services bind to localhost and are proxied with automatic TLS. Both the MCP server and the Playground require bearer token authentication for remote requests. Re-run `./install.sh --reconfigure` to change the role or server hostname.

## Menu Bar App

A puppy icon in the menu bar provides:

- **Status** — Ollama/MLX running or down, MCP configured or not
- **Model Profiles** — preset configurations (Everyday, Desktop, Maximum, Laptop) tuned for different RAM tiers
- **Task preferences** — pick which model backs each MCP tool
- **Playground** — web UI to test any tool interactively
- **Remote Access** — toggle Tailscale-based remote access to the Playground
- **Auto-update** — pulls new tagged releases automatically

## Commands

```bash
start-local-models            # start Ollama + MLX servers
start-local-models --status   # show what's running
start-local-models --stop     # stop servers
start-local-models --local    # force local servers even if server is reachable
tailscale-status              # check Tailscale connectivity and FQDN

./install.sh --reconfigure    # re-run interactive setup
./install.sh --rotate-token   # refresh the MCP auth token
./uninstall.sh                # remove Super Puppy (keeps dependencies, see script for details)
```

## Adding a New Model

**Ollama**: Just pull it. It's immediately available as an API endpoint and MCP tool.
```bash
ollama pull some-new-model
```

**MLX**: Add an entry to `config/mlx-server/config.yaml` and restart:
```bash
pkill -f mlx-openai-server
start-local-models
```


## Optional Dependencies

The installer handles core dependencies (uv, Ollama, MLX). These optional tools enable additional capabilities:

| Dependency | For | Install |
|-----------|-----|---------|
| **mflux** | Image generation (`local_image`) and editing (`local_image_edit`) | `uv tool install --python 3.12 mflux` |
| **ffmpeg** | Audio transcription of WebM, MP3, and other formats | `brew install ffmpeg` |

The installer installs both automatically on machines with 32GB+ RAM and Homebrew available. If you skipped them or installed Super Puppy before this was added, install manually with the commands above.

## Network Transparency

All inference runs locally — model input and output never leave your machine. However, Super Puppy does make network calls in these cases:

- **Auto-update**: Polls GitHub for new git tags every 2 minutes. Disable with `AUTO_UPDATE=false` in `~/.config/local-models/network.conf`.
- **Model downloads**: First use of HuggingFace embedding models (bge-m3, e5-small-v2) downloads from huggingface.co. Subsequent uses are cached locally.
- **Tailscale**: Remote access uses Tailscale's relay network when direct connections aren't possible.

For air-gapped environments: pre-download all models, set `AUTO_UPDATE=false`, and ensure all Python dependencies are cached.

## Rough Benchmarks

Approximate times on Apple Silicon (varies by model size and quantization):

| Operation | M4 Max 128GB | M4 Ultra 512GB |
|-----------|-------------|----------------|
| Image generation (Flux2) | 2–5 min | 1–3 min |
| Audio transcription (Whisper v3) | ~0.3x realtime | ~0.15x realtime |
| TTS first run (Voxtral bf16 download) | ~4GB download | ~4GB download |
| TTS generation (short text) | 5–15 sec | 3–8 sec |
| 32B code generation | 20–40 tok/s | 40–80 tok/s |

## Configuration

All user-writable config lives in `~/.config/local-models/`. The installer sets these up interactively.

### Memory and Profiles

The installer picks a profile based on your machine's RAM. You can change it any time from the Model Profiles page.

| RAM | What you get | Default profile |
|-----|-------------|-----------------|
| 32–47GB | Ollama only (MLX needs 48GB+). Small models, no image gen. | — |
| 48–63GB | Ollama + MLX. Capable coders and chat up to 32B. | Laptop |
| 64–127GB | Everything: code, vision, TTS, image gen, embeddings. | Desktop |
| 128–255GB | Plus large MoE models (70B+) with room to spare. | Server |
| 256GB+ | Full fleet including 400B+ MoE at full context. | Maximum |

| File | What |
|------|------|
| `network.conf` | Server role, hostname, ports, auth, Tailscale |
| `mcp_preferences.json` | Which model backs each MCP task type |
| `profiles.json` | Model profiles (managed by the menu bar app) |
| `mcp_auth_token` | Cached MCP bearer token (600 permissions) |

### MLX Models

Edit `config/mlx-server/config.yaml` (high-memory) or `config-laptop.yaml` (lightweight). Set `on_demand: true` for models that should only load when requested.


## CLAUDE.md Setup

The MCP tools work automatically once installed, but Claude Code performs better when it knows what's available. The installer checks for a `## Local Model Cluster` section in `~/.claude/CLAUDE.md` and offers to add one. This tells Claude when and how to use each tool. See the installer output for the recommended snippet, or check `~/.claude/CLAUDE.md` if it's already configured.

## Structure

```
super-puppy/
├── mcp/
│   └── local-models-server.py   # MCP server (PEP 723, runs via uv)
├── app/
│   ├── menubar.py               # Menu bar app (PEP 723, rumps)
│   ├── profile-server.py        # Profiles + Playground web UI
│   ├── super-puppy.c            # Native launcher for macOS app bundle
│   ├── SuperPuppy.app/          # macOS app bundle (built by install.sh)
│   ├── tools.html               # Playground interface
│   ├── profiles.html            # Model profiles interface
│   └── activity.html            # Activity monitor interface
├── mcp/
│   └── local-models-server.py   # MCP server (PEP 723, runs via uv)
├── bin/
│   ├── start-local-models       # Service manager
│   ├── local-models-menubar     # App launcher
│   ├── local-models-mcp-detect  # MCP wrapper with Tailscale discovery
│   ├── local-models-mcp-auth    # MCP auth token management
│   ├── tailscale-status         # Tailscale connectivity check
│   └── post-update.sh           # Post-update hook for auto-update
├── config/
│   ├── mlx-server/              # MLX configs (high-memory + lightweight)
│   ├── local-models/            # Network config, preferences
│   └── launchd/                 # LaunchAgent plists
├── lib/
│   ├── models.py                # Shared model constants
│   └── hf_scanner.py            # HuggingFace model discovery
├── tests/                       # pytest unit + end-to-end tests
├── web/                         # Marketing site
├── docs/                        # Setup documentation
├── install.sh                   # Interactive installer
├── uninstall.sh                 # Clean removal (keeps deps and models)
└── LICENSE                      # GPLv3
```

## License

GPLv3. See [LICENSE](LICENSE).
