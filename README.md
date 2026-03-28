# Super Puppy

Local AI model infrastructure for Claude Code. Exposes Ollama and MLX models as MCP tools so Claude can delegate bulk work, vision, transcription, translation, and image generation to local hardware.

## How It Works

Two ways to use Claude Code with local models:

**`claude`** — Claude Max does the reasoning. Local models are available as MCP tools for bulk work, vision, transcription, translation, and image generation. Best of both worlds.

**`claude-local`** — Entirely local. All requests go to Ollama (e.g. qwen3.5). No Anthropic, no internet required. Great for offline work, sensitive code, or when you just want to use the hardware you paid for.

```
claude ──> Anthropic (reasoning) + MCP tools ──> Ollama / MLX (local heavy lifting)
claude-local ──> Ollama only (fully local, no cloud)
```

### MCP Tools

When using `claude` with Max, these tools let Claude delegate to local models:

| Tool | What | Default Model |
|------|-------|---------------|
| `local_generate` | Code & text generation | qwen3-coder (code), qwen3.5 (text) |
| `local_review` | Second opinion on code | qwen3.5-large (397B) |
| `local_vision` | Analyze images on disk | qwen3-vl |
| `local_image` | Generate images | Flux2, Z-Image |
| `local_transcribe` | Audio → text | Whisper v3 |
| `local_translate` | Translate text/files | Cogito 2.1 (30+ languages) |
| `local_candidates` | Same prompt → N models | Diverse set in parallel |
| `local_summarize` | Condense large files | qwen3.5 (128K context) |
| `local_models_status` | What's available | — |

You control which model backs each task from the menu bar app. Pull a new model and it's immediately available.

## Quick Start

```bash
git clone https://github.com/jerrytalton/super-puppy.git ~/super-puppy
cd ~/super-puppy
./install.sh

ollama pull qwen3.5
start-local-models
claude    # local-models MCP auto-connects
```

### Dependencies

```bash
brew install ollama
uv tool install --python 3.12 mlx-openai-server
```

## Three Environments

The system auto-detects where you are:

| Environment | What happens |
|-------------|-------------|
| **Desktop** (512GB M3 Ultra) | All models run locally. Serves to LAN. |
| **Laptop at home** | MCP tools route to desktop over LAN. |
| **Laptop away** | Falls back to local models. Claude does the rest itself. |

## Menu Bar App

A puppy icon in the menu bar provides:

- **Status** — Ollama/MLX running or down, MCP configured or not
- **Task preferences** — pick which model backs each MCP tool (code gen, reasoning, translation, etc.)
- **Capabilities** — vision, image generation, transcription availability
- **Model Discovery** — checks HuggingFace hourly for trending models that fit your hardware

## Commands

```bash
claude                    # Claude Max + local model MCP tools
claude-local              # fully local (Ollama only, no Anthropic)

start-local-models        # start Ollama + MLX
start-local-models --stop
```

### `claude-local`

Installed by `install.sh`. Reads the preferred "General Text" model from the menu bar preferences. Auto-detects the desktop on LAN. Falls back to the first available qwen3.5 variant if no preference is set.

## Structure

```
super-puppy/
├── mcp/
│   └── local-models-server.py   # MCP server (PEP 723, runs via uv)
├── app/
│   ├── menubar.py               # Menu bar app (PEP 723, rumps)
│   ├── icon.png
│   └── icons/
├── bin/
│   ├── local-models-mcp-detect  # MCP wrapper with LAN detection
│   ├── start-local-models       # Service manager
│   └── local-models-menubar     # App launcher
├── config/
│   ├── mlx-server/              # MLX configs (desktop + laptop)
│   ├── local-models/            # Network config (desktop hostname)
│   └── launchd/                 # LaunchAgent plists
├── install.sh
└── README.md
```

## Configuration

### Task Model Preferences

Use the menu bar app to pick which model backs each task type. Saved to `~/.config/local-models/mcp_preferences.json`. The MCP server reads these on every tool call.

### MLX Models

Edit `config/mlx-server/config.yaml` (desktop) or `config-laptop.yaml` (laptop). Set `on_demand: true` for models that should only load when requested.

### LAN Serving

The desktop hostname is in `config/local-models/network.conf`. Ollama listens on `0.0.0.0` via a LaunchAgent (desktop only). If the macOS firewall is enabled, allow Ollama through:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/ollama
```

## Adding a New Model

**Ollama**: Just pull it. It's immediately available as an MCP tool.
```bash
ollama pull some-new-model
```

**MLX**: Add an entry to `config/mlx-server/config.yaml` and restart:
```bash
pkill -f mlx-openai-server
start-local-models
```

The menu bar app also checks HuggingFace hourly for trending models and offers to install them.
