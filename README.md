# Super Puppy

> **Requires Apple Silicon Mac** (M1 or later) with 64GB+ unified memory. macOS only.

Your Mac has a GPU that can run serious AI models. Super Puppy turns it into a managed local model server — LLMs, vision, image generation, transcription, translation, text-to-speech, embeddings — controlled from the menu bar, accessible over standard APIs, and available to any tool on your network.

Pull a model, and it's immediately available: through an OpenAI-compatible API, Ollama's native API, a web-based Playground, or as MCP tools in Claude Code. No cloud, no per-token billing, no data leaving your network unless you want it to.

Super Puppy is **not** a training or fine-tuning platform, a cloud service, or a production deployment tool. It's for people who want to run inference on hardware they own — for development, experimentation, creative work, and daily use.

You need enough unified memory for the models you care about. A 64GB machine handles lightweight models; 128GB+ runs most things comfortably; 256GB+ lets you run 70B+ parameter models with full context.

## Quick Start

```bash
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

### LAN Access

If this machine is the server (`IS_SERVER=true`), Ollama is accessible to other machines on your LAN:

```bash
curl http://your-server.local:11434/api/generate -d '{"model":"qwen3.5","prompt":"hello"}'
```

MLX is localhost-only but reachable via Tailscale when remote access is enabled.

## Server and Client

The installer asks whether this machine is the model server or a client. The server runs models and serves them to the LAN. Clients auto-detect the server and route requests to it:

| Environment | What happens |
|-------------|-------------|
| **Server** | All models run locally. Serves APIs to LAN. |
| **Client at home** | Routes to server over LAN. Falls back to local models if server is down. |
| **Client away** | Uses local models only. |

Re-run `./install.sh --reconfigure` to change the role or server hostname.

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

./install.sh --reconfigure    # re-run interactive setup
./install.sh --rotate-token   # refresh the MCP auth token
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

The menu bar app also checks HuggingFace hourly for trending models and offers to install them.

## Configuration

All user-writable config lives in `~/.config/local-models/`. The installer sets these up interactively.

| File | What |
|------|------|
| `network.conf` | Server role, hostname, ports, auth, Tailscale |
| `mcp_preferences.json` | Which model backs each MCP task type |
| `profiles.json` | Model profiles (managed by the menu bar app) |
| `mcp_auth_token` | Cached MCP bearer token (600 permissions) |

### MLX Models

Edit `config/mlx-server/config.yaml` (high-memory) or `config-laptop.yaml` (lightweight). Set `on_demand: true` for models that should only load when requested.

### LAN Serving

On the server, Ollama listens on `0.0.0.0` via a LaunchAgent so other machines can reach it. If the macOS firewall is enabled, allow Ollama through:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/ollama
```

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
│   ├── tools.html               # Playground interface
│   ├── profiles.html            # Model profiles interface
│   └── icon.png
├── bin/
│   ├── local-models-mcp-detect  # MCP wrapper with LAN detection
│   ├── start-local-models       # Service manager
│   └── local-models-menubar     # App launcher
├── config/
│   ├── mlx-server/              # MLX configs (high-memory + lightweight)
│   ├── local-models/            # Network config, preferences, easter eggs
│   └── launchd/                 # LaunchAgent plists
├── lib/
│   ├── models.py                # Shared model constants
│   └── hf_scanner.py            # HuggingFace model discovery
├── install.sh                   # Interactive installer
└── LICENSE                      # GPLv3
```

## License

GPLv3. See [LICENSE](LICENSE).
