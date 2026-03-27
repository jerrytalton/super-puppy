# Super Puppy

Local AI model infrastructure for Claude Code. Routes tasks to the best available model — local when it's good enough, cloud when it matters.

## What It Does

```
claude-smart ──> Claude Code Router ──┬──> Local Ollama (:11434)
                                      ├──> Local MLX-OpenAI-Server (:8000)
                                      └──> Anthropic API (cloud Claude)
```

Every request is classified by task type and routed automatically:

| Task Type | Default | Why |
|-----------|---------|-----|
| Routine (edits, completions) | Local Qwen 3.5 | Fast, free |
| Complex Reasoning | Cloud Claude Opus | Best quality for hard problems |
| Background (boilerplate) | Local Llama 3B | Speed over quality |
| Long Context | Cloud Claude Opus | Reliable at length |
| Vision | Cloud Claude Opus | Best-in-class |
| Web Search | Local Qwen 3.5 | open-webSearch MCP does the work |

You control the split from the menu bar app. Pull a new model and it's immediately available for routing.

## Quick Start

```bash
git clone https://github.com/jerrytalton/super-puppy.git ~/super-puppy
cd ~/super-puppy
./install.sh

ollama pull qwen3.5
start-local-models
claude-smart --check
claude-smart
```

### Dependencies

```bash
brew install ollama
uv tool install --python 3.12 mlx-openai-server
npm install -g @musistudio/claude-code-router
```

## Three Environments

The system auto-detects where you are:

| Environment | What happens |
|-------------|-------------|
| **Desktop** (512GB) | All models run locally. Serves to LAN. |
| **Laptop at home** | Routes to desktop over LAN (same models, zero cost). |
| **Laptop away** | Local models where capable, cloud where not. Tells you what to install. |

## Menu Bar App

A puppy icon in the menu bar provides:

- **Status** — Ollama/MLX running, loading, or down
- **Routing** — pick which model handles each task type, with provider icons, param counts, context windows
- **Services** — Speech-to-Text (Whisper v3), Web Search (open-webSearch)
- **Model Discovery** — checks HuggingFace hourly for trending models that fit your hardware

## Commands

```bash
claude-smart              # smart-routed Claude (the one you want)
claude-smart --check      # show what's available, what's missing, what to install

start-local-models        # start Ollama + MLX (auto-detects hardware)
start-local-models --status
start-local-models --stop
```

## Structure

```
super-puppy/
├── app/
│   ├── menubar.py          # Menu bar app (PEP 723, runs via uv)
│   ├── icon.png            # Menu bar icon
│   └── icons/              # Provider icons (Ollama, Claude, MLX)
├── bin/
│   ├── claude-smart        # Smart-routed Claude launcher
│   ├── start-local-models  # Service manager
│   ├── pal-mcp-detect      # MCP wrapper with network detection
│   └── local-models-menubar # App launcher
├── config/
│   ├── mlx-server/         # MLX configs (desktop + laptop)
│   ├── claude-code-router/ # CCR config + role filters
│   ├── local-models/       # Network config (desktop hostname)
│   └── launchd/            # LaunchAgent plists
├── install.sh
└── README.md
```

## Configuration

### Routing Defaults

Edit `config/claude-code-router/config.json` to change which model handles each task type.

### Role Filters

Edit `config/claude-code-router/role_filters.json` to control which models appear in each role's menu (min/max params, context requirements, vision requirement).

### MLX Models

Edit `config/mlx-server/config.yaml` (desktop) or `config-laptop.yaml` (laptop) to add MLX models. Set `on_demand: true` for models that should only load when requested.

### LAN Serving

The desktop hostname is in `config/local-models/network.conf`. Ollama listens on `0.0.0.0` via a LaunchAgent (desktop only). If the macOS firewall is enabled, allow Ollama through:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/ollama
```

## Adding a New Model

**Ollama**: Just pull it. It's immediately available in routing menus.
```bash
ollama pull some-new-model
```

**MLX**: Add an entry to `config/mlx-server/config.yaml` and restart:
```bash
# Edit the config, then:
pkill -f mlx-openai-server
start-local-models
```

**The menu bar app** also checks HuggingFace hourly for trending models and offers to install them.
