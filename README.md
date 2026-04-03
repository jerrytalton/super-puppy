# Super Puppy

Claude Code is great at reasoning, but it can't generate images, transcribe audio, speak, or run a second model for a different perspective. And every token it spends on boilerplate is a token it isn't spending on the hard problem.

Super Puppy fixes that. It turns a Mac with a decent GPU into a local model server and exposes everything — Ollama, MLX, Flux, Whisper, TTS — as MCP tools that Claude can call mid-conversation. Claude keeps doing what it's best at (architecture, debugging, complex reasoning) and offloads everything else to your hardware: bulk code generation, image generation and editing, transcription, translation, text-to-speech, embeddings, and more. Nothing leaves your network unless you want it to.

But Super Puppy isn't just for Claude. Once it's running, you get standard APIs that any application can use — an OpenAI-compatible endpoint for LLMs and vision, Ollama's native API, image generation, speech synthesis, and transcription. Your scripts, web apps, notebooks, and other tools can all hit the same local models through the same infrastructure. Claude Code is the first client, not the only one.

You need an Apple Silicon Mac with enough unified memory to run the models you care about. A 64GB laptop can handle lightweight models; a 128GB+ machine can run most things comfortably; 256GB+ lets you run everything including 70B+ parameter models with full context. This is not a cloud service — the whole point is using hardware you already own.

## Quick Start

```bash
git clone https://github.com/jerrytalton/super-puppy.git ~/super-puppy
cd ~/super-puppy
./install.sh
```

The installer walks you through everything: server vs. client role, network config, auth tokens, Tailscale for remote access, and which models to pull. Then:

```bash
start-local-models
claude    # local-models MCP auto-connects
```

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
| `local_image_edit` | Edit existing images | Flux Kontext |
| `local_transcribe` | Audio to text | Whisper v3 |
| `local_speak` | Text to speech | Voxtral (20 voices, 9 languages) |
| `local_translate` | Translate text/files | Cogito 2.1 (30+ languages) |
| `local_candidates` | Same prompt to N models | Diverse set in parallel |
| `local_summarize` | Condense large files | qwen3.5 (128K context) |
| `local_embed` | Generate embeddings | mxbai-embed-large, bge-m3 |
| `local_similarity_search` | Semantic file search | Best available embedding model |
| `local_dispatch` | Start background job | Returns immediately, model works async |
| `local_collect` | Get background result | Collect when ready |
| `local_models_status` | What's available | — |

You control which model backs each task from the menu bar app. Pull a new model and it's immediately available.

## APIs for Your Own Tools

Once Super Puppy is running, you get two standard APIs on your local network:

### Ollama API (port 11434)

The [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md) for chat, generation, and embeddings:

```bash
# Generate text
curl http://localhost:11434/api/generate -d '{"model":"qwen3.5","prompt":"hello"}'

# Chat
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5",
  "messages": [{"role":"user","content":"explain quicksort"}]
}'

# Embeddings
curl http://localhost:11434/api/embed -d '{"model":"all-minilm","input":"search query"}'
```

### OpenAI-compatible API (port 8000)

MLX models are served via an [OpenAI-compatible endpoint](https://platform.openai.com/docs/api-reference), so any library that speaks OpenAI works out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Chat completion
response = client.chat.completions.create(
    model="qwen3.5-fast",
    messages=[{"role": "user", "content": "hello"}],
)

# Vision (with MLX vision models)
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
# Works with curl too
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5-fast",
  "messages": [{"role":"user","content":"hello"}]
}'

# List available models
curl http://localhost:8000/v1/models
```

### From other machines on the LAN

If this machine is the server (`IS_SERVER=true`), other machines on your network can reach the same APIs by replacing `localhost` with the server's hostname:

```bash
curl http://your-server.local:11434/api/generate -d '{"model":"qwen3.5","prompt":"hello"}'
curl http://your-server.local:8000/v1/chat/completions -d '...'
```

### Playground

The menu bar app serves a web-based Playground UI where you can test any tool interactively — text generation, vision, image generation, transcription, TTS, translation. Open it from the Super Puppy menu or access it from other devices on your network. With Tailscale configured, the Playground is accessible from anywhere over HTTPS.

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
- **Model Profiles** — preset configurations (Everyday, Desktop, Heavyweight, Laptop) tuned for different RAM tiers
- **Task preferences** — pick which model backs each MCP tool
- **Playground** — web UI to test any tool interactively
- **Model Discovery** — checks HuggingFace hourly for trending models that fit your hardware
- **Remote Access** — toggle Tailscale-based remote access to the Playground
- **Auto-update** — pulls new tagged releases automatically

## Commands

```bash
claude                        # Claude Max + local model MCP tools
claude-local                  # fully local (Ollama only, no Anthropic)

start-local-models            # start Ollama + MLX servers
start-local-models --status   # show what's running
start-local-models --stop     # stop servers
start-local-models --local    # force local servers even if server is reachable

./install.sh --reconfigure    # re-run interactive setup
./install.sh --rotate-token   # refresh the MCP auth token
```

## Adding a New Model

**Ollama**: Just pull it. It's immediately available as an MCP tool and API endpoint.
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
| `easter_eggs.json` | Optional fun notifications (off by default) |
| `mcp_auth_token` | Cached MCP bearer token (600 permissions) |

### MLX Models

Edit `config/mlx-server/config.yaml` (high-memory) or `config-laptop.yaml` (lightweight). Set `on_demand: true` for models that should only load when requested.

### LAN Serving

On the server, Ollama listens on `0.0.0.0` via a LaunchAgent so other machines can reach it. If the macOS firewall is enabled, allow Ollama through:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/ollama
```

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
│   ├── claude-local             # Fully local Claude
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

## CLAUDE.md Setup

The installer checks for a `## Local Model Cluster` section in `~/.claude/CLAUDE.md`. Without it, Claude won't know when to use local model tools. Add this to your global CLAUDE.md:

```markdown
## Local Model Cluster

There is a local model cluster available as MCP tools via the `local-models` server.

When the `local-models` tools are available, look for opportunities to take advantage of them:
* **Vision**: Use `local_vision` to examine screenshots, UI states, diagrams, or any image on disk.
* **Image generation**: Use `local_image` to create images locally with Flux2 or Z-Image.
* **Translation**: Use `local_translate` for translating text or files between languages.
* **Audio**: Use `local_transcribe` to turn audio files into text with Whisper v3.
* **Bulk work**: Use `local_generate` for boilerplate, scaffolding, repetitive transforms, or large amounts of code.
* **Second opinions**: Use `local_review` or `local_candidates` for a different model's perspective.
* **Parallel reasoning**: Use `local_dispatch` to start a local model thinking while you continue working. Call `local_collect` when ready for its answer.
* **Large files**: Use `local_summarize` before reading huge files.
* **Semantic search**: Use `local_similarity_search` to find files related to a concept without reading every file. Use `local_embed` for raw embeddings.
* **Discovery**: Call `local_models_status` if you're unsure what's available.

Don't delegate complex reasoning, architecture decisions, or subtle debugging — do that yourself.
```

## License

GPLv3. See [LICENSE](LICENSE).
