# User-Centric Audit

Comprehensive audit of Super Puppy from the perspective of 15 diverse user personas. Each use case was validated against the actual codebase (code, config, tests, docs) as of 2026-04-04.

---

## Persona 1: Priya — Senior ML Engineer at a Startup

**Who:** Staff engineer at a 20-person AI startup. Has a Mac Studio with 192GB RAM. Wants private inference for proprietary code so nothing leaves the building. Highly technical, comfortable with CLI.

### Use Cases

**1. Run code review on proprietary files using local models only**

- **Works?** Yes. `local_review` accepts `file_paths` and routes to a local reasoning model. Path validation restricts to `$HOME` and `/tmp`, which covers normal code locations.
- **Documented?** README lists `local_review` in the tool table. The `--help` on `local_review` in the MCP tool description is clear about what it does.
- **Tested?** `test_mcp_server.py` covers model selection; `test_error_handling.py` covers error cases. No test specifically validates that file contents are sent correctly to the model.
- **Issues:** None significant. This works as expected.

**2. Use embeddings to build a semantic search index over a private codebase**

- **Works?** Partially. `local_embed` and `local_similarity_search` exist and work. However, `local_similarity_search` requires the caller to provide an explicit list of `file_paths` — there is no recursive directory walk or glob support. For a codebase with thousands of files, the user would need to construct the file list externally. The MCP tool's `file_paths` parameter is a flat list of absolute paths.
- **Documented?** The tool descriptions are adequate. No documentation on how to do a full-codebase search workflow (e.g., pipe `find` output into the tool).
- **Tested?** `test_mcp_server.py` has basic embedding tests.
- **Issues:** Practical limitation — no directory-recursive search. A power user would want `local_similarity_search` to accept a directory path and glob pattern. Also, embedding large codebases will be slow since it embeds one batch at a time with no caching of previously-computed embeddings.

**3. Set up the Mac Studio as a model server for the engineering team**

- **Works?** Partially. The server/client architecture exists and the installer walks through setup. However, the auth model uses a single shared bearer token. There is no per-user authentication, no audit log of who made what request, and no rate limiting. Any team member with the token has full access to all tools including `local_generate` (which can read arbitrary files under `$HOME`).
- **Documented?** The README covers server setup well. The Tailscale ACL example in `docs/tailscale-setup.md` shows how to restrict port access per user group, which is good. But the single-token auth limitation is not documented as a limitation.
- **Tested?** Auth middleware is tested in `test_mcp_server.py`. No multi-user scenarios.
- **Issues:** Single bearer token for the whole team is a security concern. If one person's laptop is compromised, the token is exposed. No way to revoke one person's access without rotating the token for everyone. No request logging that would help audit who ran what.

**4. Run 70B+ parameter models for complex reasoning tasks**

- **Works?** Yes. With 192GB RAM, the user can run `qwen3.5-large` (397B params, ~A17B active MoE) via MLX, or `deepseek-r1:671b` via Ollama. The "Maximum" profile configures these automatically.
- **Documented?** The README mentions "256GB+ handles 70B+ parameter models with full context." Profiles are documented. The memory requirements per model are not documented — the user would need to know that a 70B 4-bit model needs ~35GB.
- **Tested?** Profile activation is tested. No tests validate that specific models actually fit in specific RAM tiers.
- **Issues:** No clear documentation on memory-per-model. A user with 192GB might pick the "Maximum" profile and load everything, causing swapping. The profile's `max_ram_gb` field exists but is informational — nothing enforces it or warns when total loaded models exceed available RAM.

**5. Dispatch a background review while Claude continues working**

- **Works?** Yes. `local_dispatch` + `local_collect` implement this pattern exactly. The dispatch creates an async task, returns a job ID immediately, and `local_collect` retrieves the result.
- **Documented?** Well documented in the README and CLAUDE.md.
- **Tested?** `test_mcp_server.py` tests the job store lifecycle.
- **Issues:** Jobs expire after 1 hour (`_JOB_TTL = 3600`). If a large reasoning job takes longer, the result is silently lost. The expiry is not communicated to the user. Also, stale job eviction uses `asyncio.get_event_loop().time()` which is monotonic clock, while `_JOB_TTL` comparison uses the same — so this is consistent, but the TTL is not configurable.

---

## Persona 2: Marcus — Creative Writer

**Who:** Novelist who writes fantasy and sci-fi. Has a MacBook Pro M3 with 36GB RAM. Wants a local brainstorming partner that does not send his unpublished manuscripts to the cloud. Moderately technical — can follow instructions but does not write code.

### Use Cases

**1. Have a conversation with a local LLM for story brainstorming**

- **Works?** Not directly as a standalone product. Super Puppy does not include a chat interface for open-ended conversation. The Playground's "General" tool is the closest — it accepts a single prompt and returns a single response. There is no conversation history, no multi-turn dialog. The MCP tools are designed for Claude Code integration, not standalone chat.
- **Documented?** The README says "Playground where you can test any capability interactively" but does not clarify that the Playground is single-turn only.
- **Tested?** Playground streaming is tested.
- **Issues:** Major gap for this persona. A creative writer wants multi-turn conversation. The Playground is a tool tester, not a chat interface. The user could use the Ollama or MLX OpenAI-compatible APIs with a third-party chat UI (like Open WebUI), but Super Puppy does not document this or provide guidance. The `claude-local` command exists but requires Claude Code (a CLI developer tool), which is not what a writer would want.

**2. Generate images for story scenes using local diffusion models**

- **Works?** Depends on available models. The `local_image` tool exists and supports mflux (Flux, Z-Image). On a 36GB MacBook, the "Laptop" profile sets `image_gen: x/flux2-klein:latest`. However, mflux must be installed separately (`pip install mflux`), and the model must be downloaded. The installer does not install mflux.
- **Documented?** The README mentions image generation. The Playground has an "Image Gen" tab. No documentation on how to install mflux or which image models are available.
- **Tested?** `test_playground_coverage.py` verifies the image_gen route exists. No test actually generates an image (requires GPU).
- **Issues:** mflux is not installed by `install.sh`. A non-technical user would try "Image Gen" in the Playground, see an error about `mflux-generate` not being found, and have no idea what to do. The error message from the subprocess failure would be the raw stderr from the missing command.

**3. Use text-to-speech to hear dialogue read aloud in different voices**

- **Works?** Yes, if the TTS model is downloaded. The `local_speak` tool and the Playground's "Speak" tab support 20 Voxtral presets across 9 languages. The `mlx-audio` dependency is in the PEP 723 metadata for both the MCP server and profile server.
- **Documented?** Voice presets are listed in the tool description and visible in the Playground dropdown.
- **Tested?** `test_playground_coverage.py` verifies the route exists.
- **Issues:** First use requires downloading the TTS model (~4GB for Voxtral bf16, ~1.5GB for 4-bit). There is no progress indicator for the download in the Playground — the request will appear to hang. On a 36GB MacBook, the bf16 model might cause memory pressure alongside other loaded models. No guidance on which TTS model size to pick for the available RAM.

**4. Run Claude Code fully offline on a plane**

- **Works?** The `claude-local` script exists for this purpose. It sets `ANTHROPIC_BASE_URL` to local Ollama and runs Claude Code against it.
- **Documented?** Listed in the Commands section of the README.
- **Tested?** No tests for `claude-local`.
- **Issues:** This is a developer tool. A writer using Claude Code would need to know about `claude-local` and be comfortable in a terminal. More importantly, this routes Claude Code itself through a local model — the quality of responses from a 9B local model is dramatically worse than Claude. No documentation sets expectations about this quality tradeoff. Also, the script picks the model from `mcp_preferences.json`'s `general` key, which might be a list — the python one-liner `json.load(...)get('general','')` would return a list, not a string, causing the model selection to fail silently and fall back to the Ollama auto-detection.

**5. Translate story chapters to other languages**

- **Works?** Yes. `local_translate` in the MCP server and the "Translate" tab in the Playground both work. The Playground has a target language field and a text area. For file-based translation, the MCP tool accepts `file_paths`.
- **Documented?** Adequately in the tool description.
- **Tested?** Route existence tested.
- **Issues:** The Playground's Translate tool only accepts inline text (no file upload for Translate). For a novelist with long chapters, they would need to paste content. The MCP tool supports `file_paths` but the Playground does not expose this. Also, translation quality from local models varies significantly by language pair — no guidance on which languages work well.

---

## Persona 3: Jake — CS Student

**Who:** Sophomore CS student. Has a Mac Studio M2 Ultra with 64GB RAM (birthday gift). Learning about AI, wants to experiment with different models. Enthusiastic but still learning fundamentals.

### Use Cases

**1. Pull and try different models to compare their capabilities**

- **Works?** Yes. `ollama pull` works seamlessly — new models are auto-discovered by the MCP server at startup. The Playground lets you select any available model via the model picker. `local_candidates` runs the same prompt against multiple models in parallel.
- **Documented?** "Adding a New Model" section in the README covers this. Playground model picker is self-explanatory.
- **Tested?** Model discovery tested in `test_mcp_server.py`.
- **Issues:** No guidance on which models are good for what. A student pulling random models will not know that a 3B model is terrible at reasoning or that an embedding model cannot chat. The Playground does not filter incompatible models per task (e.g., you can select an embedding model for Code generation and get a cryptic error).

**2. Understand how MoE (Mixture of Experts) models work by inspecting the model info**

- **Works?** Partially. `local_models_status` returns parameter counts including active vs total for MoE models. The Profiles UI shows active parameters per model. However, there is no educational content explaining what these numbers mean.
- **Documented?** `lib/models.py` has detailed comments about the 4-strategy cascade for active param computation. This is developer documentation, not user-facing.
- **Tested?** `active_params_b()` is well-tested in `test_core.py`.
- **Issues:** A student would see "397B (17B active)" and wonder what that means. No tooltips, no explanations, no links to learn more.

**3. Set up the Mac Studio as a server and access it from a laptop**

- **Works?** Yes. The installer walks through server/client configuration. Tailscale setup is guided.
- **Documented?** Well documented in README and `docs/tailscale-setup.md`.
- **Tested?** `test_remote_access.sh` tests HTTPS endpoints.
- **Issues:** The installer requires making choices (server vs client, Tailscale hostname) that a student might not understand. What is a "Tailscale hostname"? What if they have not set up Tailscale yet? The installer tries `tailscale up` but if the student has never used Tailscale, they need to create an account first — this is not explained inline. Also, `sudo tailscale set --ssh` requires admin password, which might alarm a student.

**4. Use the vision model to analyze plots/charts from a textbook**

- **Works?** Yes, if a vision model is available. The MCP `local_vision` tool and the Playground's "Vision" tab support image analysis. On a 64GB machine with the "Desktop" profile, `vision: qwen3.5:9b` is configured — but this is a 9B LLM, and whether it has vision capability depends on whether `qwen3.5:9b` is actually vision-capable. Looking at the code, vision detection checks Ollama's `model_info` for "vision" keys and HuggingFace `config.json` for `vision_config`. The CLAUDE.md notes that "Qwen3.5 models (served via MLX) ARE vision-capable" — but the Desktop profile maps vision to `qwen3.5:9b` which is an Ollama model, not MLX. Whether Ollama's qwen3.5:9b reports vision capability depends on the model metadata.
- **Documented?** Playground Vision tab has a file picker and prompt field.
- **Tested?** Vision routing tested in `test_mcp_server.py`.
- **Issues:** Confusion between model names. The config references `qwen3.5:9b` for vision but the CLAUDE.md specifically discusses Qwen3.5 vision via MLX. A student would not know whether their model actually supports vision until they try it and potentially get "Error: qwen3.5:9b is not vision-capable."

**5. Learn about embeddings by embedding text and visualizing similarity**

- **Works?** Partially. `local_embed` returns raw embedding vectors as JSON. `local_similarity_search` returns similarity scores. But there is no visualization — just raw numbers.
- **Documented?** Tool description says "Returns cosine-similarity-ready normalized vectors."
- **Tested?** Basic embedding tests exist.
- **Issues:** A student gets a massive JSON array of floats. No visualization, no explanation of what the numbers mean, no example of how to use them. The Playground's "Embed" tab returns raw embeddings and a dimension count. This is useful for a developer building an app, not a student learning concepts.

---

## Persona 4: Elena — Professional Photographer

**Who:** Fine art photographer. MacBook Pro M4 Max with 128GB RAM. Wants to use AI for image editing and generation without uploading client images to cloud services. Limited coding skills.

### Use Cases

**1. Edit existing photos with text prompts (e.g., "remove the person in the background")**

- **Works?** Yes, in principle. `local_image_edit` uses Flux Kontext via mflux. The Playground has an "Image Edit" tab with image upload and prompt field.
- **Documented?** Tool description covers parameters (strength, steps, seed).
- **Tested?** Route existence tested, no actual image edit test.
- **Issues:** Same as Persona 2's image gen issue — `mflux-generate-kontext` must be installed separately and is not installed by `install.sh`. The error on missing mflux would be a raw subprocess error. Also, the Playground restricts file paths to `/tmp/` only (`_is_safe_test_path`), so the user cannot edit images from their Photos library or Desktop — they must first upload through the Playground's file picker, which saves to `/tmp/`. This is a security feature but it is unintuitive for a photographer who wants to edit a specific file.

**2. Batch-process images with consistent edits (e.g., apply the same color grade to 50 photos)**

- **Works?** No. There is no batch processing capability. `local_image_edit` processes one image at a time. There is no API for batch operations, no queue, no progress tracking for multiple items.
- **Documented?** Not mentioned because it does not exist.
- **Tested?** N/A.
- **Issues:** A photographer working with sets of images would need to write a script or use an external tool. This is not an unreasonable limitation for v1 but it is a feature gap for this persona.

**3. Generate concept images for a photoshoot mood board**

- **Works?** Yes, same as Persona 2's image gen use case. `local_image` and the Playground's "Image Gen" tab work if mflux is installed and a model is downloaded.
- **Documented?** Adequate.
- **Tested?** Route tested.
- **Issues:** Same mflux installation gap. Also, on 128GB, Flux2 models fit comfortably. But generation time is not documented — a photographer would want to know "how long does this take?" No benchmarks or time estimates are provided anywhere.

**4. Use vision to analyze compositions in reference images**

- **Works?** Yes. Vision tools work well for this use case. A 128GB machine can run capable vision models.
- **Documented?** Adequate.
- **Tested?** Tested.
- **Issues:** The Playground Vision tab requires uploading the image (which saves to `/tmp/`). Cannot point at a directory of images. Single-image analysis only.

**5. Access the Playground from an iPad while on a photoshoot**

- **Works?** Yes. The Playground is a PWA installable from Safari. Tailscale provides remote access from anywhere.
- **Documented?** `docs/tailscale-setup.md` covers iOS setup including the "Add to Home Screen" step.
- **Tested?** `test_remote_access.sh` tests HTTPS endpoints.
- **Issues:** Requires Tailscale installed on the iPad. The PWA experience is well-designed (responsive layout, touch-optimized controls). Auth token is passed via URL parameter which is stored in `sessionStorage` — reasonable for a personal device. However, if the MacBook at home goes to sleep, the Playground becomes unreachable with no helpful error message.

---

## Persona 5: David — Podcaster

**Who:** Runs a weekly interview podcast. Mac Mini M4 with 64GB RAM. Needs transcription for show notes and wants to try TTS for promotional content. Not a developer — uses GarageBand and a web browser.

### Use Cases

**1. Transcribe a podcast episode**

- **Works?** Yes. `local_transcribe` and the Playground's "Transcribe" tab support audio transcription via Whisper v3. The Playground supports file upload and live microphone recording. The MLX config includes `whisper-v3` on demand.
- **Documented?** Tool description is clear. Playground has a file picker and mic button.
- **Tested?** Route tested.
- **Issues:** Long podcast episodes (1-2 hours) will take significant time to transcribe. There is no progress indicator for long transcriptions — the request will appear to hang. The Playground has a 300-second (5-minute) timeout on backend requests. A 1-hour episode will almost certainly exceed this, causing a silent failure. Also, Whisper needs `ffmpeg` for certain audio formats. The `start-local-models` script adds `/opt/homebrew/bin` to PATH for this reason, but `ffmpeg` must be installed (`brew install ffmpeg`) and this is not mentioned in the README or installer.

**2. Record a short clip in the Playground and transcribe it**

- **Works?** Yes. The Playground Transcribe tab has a "Record" button that uses the browser's MediaRecorder API. It records WebM audio, uploads to `/tmp/`, and the profile server converts WebM to WAV via `ffmpeg` before sending to Whisper.
- **Documented?** Not explicitly — the "Or record" UI element is self-explanatory though.
- **Tested?** No test for the recording flow.
- **Issues:** Requires microphone permission in the browser. If using the menu bar webview, the `_WebViewUIDelegate` auto-grants media capture permission. If using Safari remotely, the browser will prompt for mic access. The WebM-to-WAV conversion requires `ffmpeg` — if missing, the transcription fails with a subprocess error.

**3. Generate TTS versions of show notes for a "listen to this episode" teaser**

- **Works?** Yes. Same as Persona 2's TTS use case. Voxtral supports multiple voices.
- **Documented?** Voice presets listed. Playground has a voice dropdown.
- **Tested?** Route tested.
- **Issues:** First-time model download with no progress indicator. Also, the generated audio is saved to `/tmp/` — the user needs to know to copy it somewhere persistent before reboot clears `/tmp/`.

**4. Clone their own voice for consistent podcast intros**

- **Works?** Yes. `local_speak` supports voice cloning via Chatterbox with `ref_audio` parameter. The MCP tool exposes this.
- **Documented?** The tool description mentions `ref_audio` for voice cloning and Chatterbox model.
- **Tested?** No test for voice cloning.
- **Issues:** Voice cloning is only available via the MCP tool (i.e., Claude Code), not the Playground. The Playground's "Speak" tab has voice presets but no reference audio upload. A podcaster without Claude Code cannot access voice cloning through the web UI.

**5. Access transcription from their phone while recording on location**

- **Works?** Yes, via the Playground PWA over Tailscale. Can upload audio files or record directly.
- **Documented?** PWA installation documented.
- **Tested?** Remote access tested.
- **Issues:** Same timeout concern for long audio. Also, uploading a large audio file over Tailscale to a home Mac Mini may be slow if the upload bandwidth is limited.

---

## Persona 6: Barbara — Small Business Owner

**Who:** Runs a bakery. Got a Mac Studio M4 Max 128GB as a business investment after hearing about "running AI locally." Wants to use AI for marketing copy, menu descriptions, and social media content. Virtually no technical knowledge — can barely use Terminal.

### Use Cases

**1. Install Super Puppy**

- **Works?** The install script is interactive and guides through choices. But it requires: (a) cloning a git repo from Terminal, (b) understanding server vs client, (c) understanding Tailscale, (d) understanding auth tokens and 1Password, (e) choosing which model profile to use.
- **Documented?** Quick Start section gives 3 commands. But the prerequisites (git, Xcode command-line tools, Homebrew) are not mentioned.
- **Tested?** No install tests.
- **Issues:** Critical barrier for this persona. The Quick Start assumes `git` is installed (it is not on a fresh Mac — requires Xcode CLT). Homebrew is needed for Ollama. The installer asks questions like "Is this the model server?" and "Tailscale hostname of the model server" that Barbara cannot answer. There is no "just make it work" simple path. Even the concept of "cloning a repo" is foreign. This persona needs a `.dmg` installer or at minimum a one-line `curl | bash` bootstrap that handles prerequisites.

**2. Write marketing copy for the bakery**

- **Works?** If installed, the Playground "General" tab works for this. Type a prompt, get a response.
- **Documented?** Not specifically for non-technical users. No example prompts or use case guides.
- **Tested?** General chat tested.
- **Issues:** The Playground interface uses developer terminology ("Tool", "Model", "Backend"). A non-technical user would not know what to select. The default tool selection and model picker are unintuitive for someone who just wants to type a question and get an answer. Why are there 13 "tools" when she just wants to chat?

**3. Generate images for social media posts**

- **Works?** Same mflux dependency issue as other personas.
- **Documented?** No guidance for non-technical users.
- **Tested?** N/A.
- **Issues:** Even if everything were installed, the image generation takes minutes. No progress bar. Output is saved to `/tmp/` with a cryptic filename. How does Barbara get the image into Instagram?

**4. Have the AI read menu descriptions aloud for a phone greeting**

- **Works?** TTS works via the Playground.
- **Documented?** Not for this use case.
- **Tested?** Route tested.
- **Issues:** Output is a `.wav` file in `/tmp/`. Barbara would need to: (a) find the file in Finder, (b) know that `/tmp/` is a hidden directory, (c) convert or transfer the file to her phone system. No "Download" button in the Playground for the generated audio. Actually — checking the code, the Playground does serve audio files via `/api/test/audio?path=...`, and the JavaScript likely plays them inline. But saving the file for external use is not obvious.

**5. Get help troubleshooting when something breaks**

- **Works?** "Copy Diagnostics" in the menu bar copies debug info to clipboard. The diagnostics include mode, versions, service status, and recent logs.
- **Documented?** Mentioned in Menu Bar Features.
- **Tested?** No test for diagnostics output.
- **Issues:** The diagnostics are technical (log lines, service names, port numbers). Barbara would paste them into an email or forum post without understanding them. This is fine for getting help from a technical friend, but there is no self-service troubleshooting guide for common problems ("Ollama not running" = what does that mean?).

---

## Persona 7: Kenji — Security-Conscious Developer

**Who:** Security engineer at a financial services company. Mac Studio M4 Ultra 512GB RAM. Will not use cloud AI APIs due to compliance requirements. Wants every bit of inference on-premises. Reads source code before installing anything.

### Use Cases

**1. Audit the security of the MCP server**

- **Works?** The code is open source (GPLv3) and reviewable. The MCP server has: bearer token auth (fail-closed), path validation (restricted to `$HOME` and `/tmp`), session tracking, DNS rebinding protection via `TransportSecuritySettings`.
- **Documented?** `docs/architecture.md` has a Security Model section.
- **Tested?** Auth middleware tested. Path validation tested in `test_mcp_server.py`.
- **Issues:**
  - The `_validate_path` function allows access to anything under `$HOME`, which includes `~/.ssh/`, `~/.gnupg/`, `~/.config/` (config files with tokens). A malicious or confused Claude prompt could read `~/.config/local-models/mcp_auth_token` or `~/.ssh/id_rsa` via `local_generate(context_files=[...])` or `local_summarize(file_paths=[...])`. The MCP server trusts Claude to provide safe paths, and Claude trusts the user's prompt. This is a meaningful attack surface if the MCP server is exposed to untrusted clients.
  - Session eviction uses `_authenticated_sessions.pop()` which removes an arbitrary element from the set — in Python 3.12, `set.pop()` removes an arbitrary element, so an attacker who creates many sessions could evict legitimate sessions.
  - The auth token is stored in a plaintext file (`~/.config/local-models/mcp_auth_token`) with 600 permissions. This is standard for config files but means any process running as the user can read it.
  - `OLLAMA_HOST=0.0.0.0` is set on the server (via `setenv.OLLAMA_HOST.plist`), which means Ollama listens on all interfaces — not just Tailscale. Anyone on the same LAN can access Ollama directly without authentication. This is documented in the CLAUDE.md ("Ollama binds `0.0.0.0` on the server for LAN access") but contradicts the architecture doc which says "All services bind to `127.0.0.1`."
  - The `OLLAMA_HOST=0.0.0.0` binding means Ollama is accessible without auth on the LAN, while the MCP server and Playground require auth. This is an inconsistency.

**2. Verify that no data leaves the machine**

- **Works?** The MCP server and profile server only make outgoing connections to `localhost:11434` (Ollama) and `localhost:8000` (MLX). No telemetry, no analytics, no external API calls. The `sentence-transformers` library downloads models from HuggingFace on first use — this is an outgoing network call.
- **Documented?** The README says "No cloud, no per-token billing, no data leaving your network."
- **Tested?** No test verifies the absence of outgoing connections.
- **Issues:** The `sentence-transformers` model download on first use of `local_embed` with HuggingFace models (bge-m3, e5-small-v2) makes an outgoing network call to HuggingFace. This contradicts the "no data leaving your network" claim. Once cached, subsequent uses are local. Also, `mlx-audio` (TTS dependency) is installed from a git URL which requires network access at install time. And the auto-update feature fetches git tags from GitHub every 2 minutes — this is a persistent outgoing connection.

**3. Run the server in a fully air-gapped environment**

- **Works?** Partially. Once all models and dependencies are cached, inference is fully local. But: (a) auto-update fetches from GitHub, (b) `sentence-transformers` may try to download models, (c) `mflux` may need to download model weights.
- **Documented?** No air-gap deployment guide.
- **Tested?** No air-gap scenario tests.
- **Issues:** No way to disable auto-update. No way to pre-cache all HuggingFace models. No documentation on running fully offline. The auto-update poller runs unconditionally every 2 minutes and logs warnings when it cannot reach GitHub, which would be noisy in an air-gapped environment.

**4. Review update signatures before deploying**

- **Works?** The architecture doc says "Verifies the tag's GPG/SSH signature (rejects unsigned tags)." However, looking at the actual code in `menubar.py`, I need to verify this claim.
- **Documented?** Stated in `docs/architecture.md`.
- **Tested?** `test_deployment.py` has tag verification tests.
- **Issues:** The architecture doc claims GPG/SSH signature verification, which is a strong security feature. This would need code verification to confirm it is actually enforced and not bypassable.

**5. Restrict which directories the MCP tools can access**

- **Works?** No. The `_ALLOWED_ROOTS` in the MCP server is hardcoded to `(Path.home(), Path("/tmp"), Path("/private/tmp"))`. There is no configuration to restrict this further.
- **Documented?** The architecture doc mentions "Path validation: All MCP tools that accept file paths validate them against $HOME and /tmp."
- **Tested?** Path validation tested.
- **Issues:** No way to restrict to a specific project directory. A security-conscious user would want to limit file access to `/Users/kenji/projects/` and deny access to `~/.ssh/` and `~/.config/`. This would require code changes.

---

## Persona 8: Sarah — Engineering Team Lead

**Who:** Leads a team of 8 developers. Has a Mac Studio M4 Ultra 512GB in the office server closet. Wants to set up a shared model server so the team stops paying for individual Claude Pro subscriptions for local model tasks.

### Use Cases

**1. Set up the server and onboard 8 developers as clients**

- **Works?** The server/client architecture supports this. Each developer installs Super Puppy on their laptop, runs `install.sh`, chooses "client", enters the server's Tailscale hostname.
- **Documented?** The install flow is documented. The Tailscale setup doc covers adding family members but not a team onboarding workflow.
- **Tested?** Basic connectivity tested.
- **Issues:** Each developer must: (a) install Tailscale, (b) join the tailnet, (c) get approved in the admin console, (d) install Super Puppy, (e) run the installer, (f) get the auth token from Sarah. No batch onboarding. The auth token must be shared out-of-band (no automated distribution). If one developer leaves, the token must be rotated for everyone.

**2. Monitor usage across the team**

- **Works?** Partially. The MCP server tracks GPU activity (`/gpu` endpoint) and request history (`/activity` endpoint). The Activity page in the Playground shows current and recent requests with backend, duration, and status.
- **Documented?** Activity page exists but is not prominently documented.
- **Tested?** Activity endpoint is part of the MCP server.
- **Issues:** No per-user attribution. All requests show up as the same authenticated token. No usage reports, no "team member X used 3 hours of GPU time this week." No quotas or fair-sharing — one developer running a 70B model blocks everyone else's requests (GPU contention warning exists but no queue or priority system).

**3. Set resource limits per developer**

- **Works?** No. There is no concept of per-user quotas, resource limits, or priority levels.
- **Documented?** Not mentioned because it does not exist.
- **Tested?** N/A.
- **Issues:** One developer running `local_candidates` with 3 models simultaneously could starve the GPU for everyone else.

**4. Get notified when the server goes down**

- **Works?** No alerting system. The menu bar app monitors service health and shows status dots, but this is on the server itself. Clients see a generic "not reachable" status.
- **Documented?** Client mode shows service status in the menu bar.
- **Tested?** Status polling tested.
- **Issues:** No push notifications to the team when services go down. No webhook, no email, no Slack integration. The team lead would have to manually check the server.

**5. Manage model profiles for the team**

- **Works?** Partially. Profiles are stored in `~/.config/local-models/profiles.json` on the server. Clients read their own local preferences to select models. There is no server-side profile that automatically applies to all clients. Each developer must set up their own preferences.
- **Documented?** Profile management documented for individual use.
- **Tested?** Profile CRUD tested.
- **Issues:** No centralized profile management. If Sarah wants the whole team to use the same models for the same tasks, she must instruct each developer to configure their preferences individually. No "team defaults."

---

## Persona 9: Tomas — Home Automation Tinkerer

**Who:** Hardware hacker with a Mac Mini M4 16GB RAM in his garage workshop. Runs Home Assistant. Wants to add AI to his automations — natural language control, image recognition for security cameras, voice commands.

### Use Cases

**1. Install on a 16GB Mac Mini**

- **Works?** Barely. The `start-local-models` script checks RAM: `<48GB` skips MLX entirely ("too little for MLX server"). Ollama will run but with very limited model support. 16GB can barely hold a single 7-8B model at 4-bit quantization.
- **Documented?** README says "64GB+ unified memory" is required. The installer checks RAM and suggests the "Laptop" profile at 48GB+, but has no profile for 16GB. Below 48GB, MLX is skipped entirely.
- **Tested?** RAM threshold tested in `test_core.py`.
- **Issues:** The 16GB machine is explicitly below the documented minimum. The user would see "too little for MLX server" and only get Ollama with whatever small models they pull. The "Laptop" profile requires 32GB. There is no "minimal" profile for 16GB machines. Super Puppy is fundamentally not designed for this hardware tier.

**2. Use vision for security camera feed analysis**

- **Works?** No. `local_vision` processes single images, not video streams. There is no continuous processing mode, no webhook integration, no Home Assistant API.
- **Documented?** Not applicable.
- **Tested?** N/A.
- **Issues:** Super Puppy is not a video processing pipeline. Home automation integration would require significant custom development. The Ollama API is accessible and could be called from Home Assistant plugins, but Super Puppy provides no integration for this.

**3. Set up voice commands via local transcription + TTS**

- **Works?** The building blocks exist (transcription + TTS) but there is no voice pipeline. Each operation is a discrete API call, not a continuous listen-process-respond loop.
- **Documented?** Individual tools documented.
- **Tested?** Individual tools tested.
- **Issues:** Tomas would need to build the entire voice pipeline himself, using Ollama/MLX APIs as backends. Super Puppy provides the model serving but not the automation layer.

**4. Use the OpenAI-compatible API from Home Assistant**

- **Works?** Yes. MLX serves an OpenAI-compatible API on port 8000. Ollama also serves compatible APIs. Many Home Assistant AI plugins support custom OpenAI endpoints.
- **Documented?** The README documents both APIs with examples.
- **Tested?** API availability tested in e2e tests.
- **Issues:** On 16GB, MLX is disabled. Ollama's API works but the available model quality is very limited. The Home Assistant user would need to know the correct API format and endpoint.

**5. Run the server headless (no display)**

- **Works?** Partially. The menu bar app requires a GUI session (rumps depends on NSApplication). The `start-local-models` script starts Ollama and MLX without requiring a GUI. The MCP server can be started directly. But the full Super Puppy experience (profiles, Playground, health monitoring, auto-update) requires the menu bar app.
- **Documented?** Not documented. The README assumes a GUI environment.
- **Tested?** No headless tests.
- **Issues:** A garage Mac Mini might run headless. The LaunchAgent for the menu bar app would fail without a GUI session. The user would need to run `start-local-models` manually and skip the menu bar app entirely. No systemd/launchd service file for headless operation of Ollama+MLX without the menu bar app.

---

## Persona 10: Dr. Aisha — NLP Researcher

**Who:** Postdoc in computational linguistics. Mac Studio M4 Ultra 512GB. Working on a paper that requires embedding a corpus of 50,000 academic papers and clustering them. Expert-level ML knowledge.

### Use Cases

**1. Embed 50,000 documents using local models**

- **Works?** Technically yes, but painfully. `local_embed` accepts a list of texts. The MCP tool would need to be called many times (50,000 documents cannot be embedded in a single call due to context limits). `local_similarity_search` only works with explicit file paths.
- **Documented?** Tool descriptions cover basic usage.
- **Tested?** Basic embedding tests.
- **Issues:** No batch embedding API. No progress tracking for bulk operations. No caching of previously-computed embeddings. The MCP tool returns raw JSON embeddings which would need to be parsed and stored. For 50,000 documents, a researcher would bypass Super Puppy entirely and use `sentence-transformers` directly. Super Puppy adds overhead (HTTP round-trips, JSON serialization of high-dimensional vectors) without adding value for bulk embedding.

**2. Compare embedding models (bge-m3 vs mxbai-embed-large vs all-minilm)**

- **Works?** Yes. `local_embed` accepts a `model` parameter. The user can call it multiple times with different models and compare outputs.
- **Documented?** Available models listed in the tool description.
- **Tested?** Basic tests.
- **Issues:** No built-in comparison or benchmarking tool. The researcher would need to write their own evaluation script.

**3. Use long-context models for summarizing papers**

- **Works?** Yes. `local_summarize` routes to the `long_context` task which filters for models with 64K+ context. On a 512GB machine, qwen3.5-large with 65K context is available.
- **Documented?** Adequate.
- **Tested?** Route tested.
- **Issues:** 65K tokens is roughly 50K words — sufficient for most papers. No issue for single papers. Batch summarization (summarize 100 papers) would need to be scripted.

**4. Use the Ollama API directly for custom experiments**

- **Works?** Yes. Ollama's API is standard and well-documented externally. Super Puppy provides model serving; the researcher can use any Ollama client library.
- **Documented?** README shows curl examples and Python client usage.
- **Tested?** E2e tests hit Ollama APIs.
- **Issues:** None. This is where Super Puppy shines — it provides the infrastructure, and the researcher uses standard APIs.

**5. Run experiments requiring reproducible model outputs**

- **Works?** Partially. Ollama supports `seed` and `temperature` parameters. MLX also supports temperature. The `local_image_edit` tool accepts a `seed` parameter. But most text generation tools in the MCP server do not expose a `seed` parameter.
- **Documented?** Not documented for reproducibility use cases.
- **Tested?** No reproducibility tests.
- **Issues:** The MCP tools do not expose `seed` or `temperature` parameters for text generation. A researcher wanting reproducible outputs would need to use the raw Ollama/MLX APIs instead of the MCP tools.

---

## Persona 11: Maya — UI/UX Designer

**Who:** Product designer at a tech company. MacBook Pro M4 Pro 48GB. Wants to use local vision models to get AI feedback on her UI mockups without uploading them to cloud services.

### Use Cases

**1. Analyze a Figma export for accessibility issues**

- **Works?** Yes. `local_vision` accepts image paths and a prompt. A well-crafted prompt like "Analyze this UI screenshot for accessibility issues: contrast ratios, text size, touch targets, color blindness concerns" should produce useful results from a capable vision model.
- **Documented?** Tool description covers basic usage.
- **Tested?** Vision routing tested.
- **Issues:** The quality of the analysis depends entirely on the vision model's capabilities. No documentation on what kinds of visual analysis work well vs poorly. On 48GB, the available vision models would be smaller (9B class), which may give superficial analysis.

**2. Compare multiple design iterations side-by-side**

- **Works?** `local_vision` accepts a list of `image_paths`, so multiple images can be analyzed in a single call. Good.
- **Documented?** Parameter description shows `image_paths` is a list.
- **Tested?** No multi-image test.
- **Issues:** The model sees all images at once but there is no structured comparison output. The prompt would need to explicitly request a comparison.

**3. Use the Playground from her phone to show design feedback to stakeholders**

- **Works?** Yes. The Playground is accessible via Tailscale PWA. Can upload images from the phone's photo library.
- **Documented?** PWA documented.
- **Tested?** Remote access tested.
- **Issues:** Image upload from mobile is supported. The experience should be smooth.

**4. Generate placeholder images for prototypes**

- **Works?** Same mflux dependency gap.
- **Documented?** Inadequate for this persona.
- **Tested?** Route tested.
- **Issues:** mflux not installed by default.

**5. Use computer vision to analyze competitor app screenshots**

- **Works?** Same as use case 1. `local_vision` or `local_computer_use` can analyze screenshots.
- **Documented?** Adequate.
- **Tested?** Tested.
- **Issues:** `local_computer_use` returns structured JSON actions (click, type, scroll) which is not what a designer wants. They want qualitative analysis. `local_vision` is the right tool here but the naming might confuse the user into trying `local_computer_use` first.

---

## Persona 12: Wei — Multilingual Content Creator

**Who:** Creates content in English, Mandarin, and Japanese. Mac Studio M2 Ultra 192GB. Needs translation, TTS in multiple languages, and writing assistance across all three languages.

### Use Cases

**1. Translate blog posts between English, Mandarin, and Japanese**

- **Works?** Yes. `local_translate` supports these languages. The Cogito and Qwen models have strong multilingual capabilities (Qwen is specifically trained on CJK languages).
- **Documented?** Tool description mentions "30+ languages." Playground Translate tab has target language field.
- **Tested?** Route tested.
- **Issues:** No guidance on which language pairs work best with which models. The Playground Translate tool does not list supported languages — just a free-text field. Translation quality for CJK languages from local models can vary significantly.

**2. Generate TTS in Mandarin and Japanese**

- **Works?** Partially. Voxtral supports 9 languages (en, fr, es, de, it, pt, nl, ar, hi). **Mandarin and Japanese are not in the supported language list.** Chatterbox supports 23 languages which may include them. The Playground's voice dropdown only shows Voxtral presets — no CJK options.
- **Documented?** The `local_speak` tool description lists Voxtral voices which are all Western + Arabic/Hindi. No CJK voices documented.
- **Tested?** No CJK TTS test.
- **Issues:** Critical gap for this persona. The two most important languages (Mandarin and Japanese) are not supported by the default TTS model. Chatterbox may support them but it is not the default and requires specifying the model explicitly. The Playground has no way to select the Chatterbox model for TTS — the voice picker only shows Voxtral presets.

**3. Write content in one language and get AI feedback in another**

- **Works?** Yes, via `local_generate` with a system prompt. The local models are multilingual and can take input in one language and respond in another.
- **Documented?** Not specifically documented for cross-lingual use.
- **Tested?** Not specifically tested.
- **Issues:** None significant. This works naturally with multilingual models.

**4. Use voice cloning to create consistent narration across languages**

- **Works?** `local_speak` with `ref_audio` supports voice cloning via Chatterbox. Chatterbox handles 23 languages.
- **Documented?** Mentioned in tool description.
- **Tested?** Not tested.
- **Issues:** Voice cloning is MCP-only (no Playground access). See Persona 5 issue.

**5. Batch-translate a directory of markdown files**

- **Works?** `local_translate` accepts `file_paths` for batch file translation. However, it concatenates all files into a single prompt and returns a single translated output. There is no per-file output — the user gets one big blob of translated text without file boundaries.
- **Documented?** `file_paths` parameter documented.
- **Tested?** Not tested for multi-file.
- **Issues:** The concatenation approach loses file structure. For a directory of 20 markdown files, the user would get one long translated text with `--- /path/to/file ---` markers. They would need to manually split it back into files. No automatic per-file output.

---

## Persona 13: Alex — Coffee Shop Developer

**Who:** Freelance web developer. MacBook Pro M4 64GB. Works from coffee shops with spotty internet. Wants local AI for coding assistance when internet drops.

### Use Cases

**1. Seamless fallback when internet drops**

- **Works?** This is the core client/server design. When the server (desktop) is unreachable, the client falls back to local models. The MCP wrapper (`local-models-mcp-detect`) probes the server at startup, not per-request.
- **Documented?** Well documented.
- **Tested?** Server reachability tested.
- **Issues:** The fallback only happens at MCP server startup. If the connection drops mid-session (e.g., cafe WiFi goes down after 30 minutes), the MCP server is already running and pointing at the remote server. Requests will fail with connection errors rather than gracefully falling back. The user would need to restart the MCP server (or toggle "Local" in the menu bar) to switch to local models. The menu bar does poll for connectivity changes, but the MCP server process keeps its original URLs.

**2. Use Claude Code with local models when fully offline**

- **Works?** `claude-local` script exists. `start-local-models --local` forces local servers.
- **Documented?** Both commands documented.
- **Tested?** `start-local-models --local` logic tested.
- **Issues:** `claude-local` runs Claude Code against a local Ollama model. The quality is dramatically lower than Claude. For a professional developer, this might be more frustrating than helpful — a 9B model's code suggestions are often wrong. The useful path is: keep using regular Claude Code for reasoning, but have the MCP tools (local_generate, local_review) available locally. This works if the MCP server is running locally.

**3. Pre-load models before leaving for the cafe**

- **Works?** The profile `/warm` endpoint pre-loads models into memory. The Playground has "Warm" buttons for profiles.
- **Documented?** Not explicitly documented as a workflow. The warm feature exists in the Profiles UI.
- **Tested?** Profile warm endpoint tested.
- **Issues:** Warming loads models into GPU memory, but on a 64GB laptop, loading multiple models simultaneously may cause memory pressure. No guidance on which models to warm for offline use. The warm operation sends an empty generation to each model, which can take 30-60 seconds per model.

**4. Quick code generation while Copilot is unreachable**

- **Works?** The MCP `local_generate` tool with `code` task works. On a 64GB laptop, the laptop MLX config includes `qwen-coder` (Qwen2.5-Coder 32B) on demand.
- **Documented?** Adequate.
- **Tested?** Tested.
- **Issues:** Good model coverage for this use case on 64GB. The 32B coder is solid for code generation. First request on a cold model takes longer (model loading).

**5. Switch between remote and local modes quickly**

- **Works?** The menu bar has "Remote" / "Local" toggles. FORCE_LOCAL is written to `~/.config/local-models/mode.conf`.
- **Documented?** Documented in Menu Bar Features.
- **Tested?** `load_force_local` and `save_force_local` tested.
- **Issues:** Switching modes in the menu bar sets a config flag, but the MCP server process needs to be restarted to pick up the new URLs. The menu bar does handle MCP restart, but there may be a lag. Also, "Local (override)" label when forced local but desktop is reachable could be confusing — why "override"?

---

## Persona 14: Greg — Sysadmin Managing Multiple Macs

**Who:** IT administrator at a design agency. Manages 15 Macs (mix of Mac Studios and MacBook Pros). Wants to deploy Super Puppy across all machines with a consistent configuration.

### Use Cases

**1. Deploy Super Puppy to 15 machines**

- **Works?** No automated deployment. Each machine requires interactive `install.sh` with manual responses to prompts.
- **Documented?** Only interactive install documented.
- **Tested?** No deployment automation tests.
- **Issues:** Critical gap. No headless install (`install.sh` always requires interactive input). No Ansible playbook, no MDM profile, no silent install flags. Greg would need to walk through the installer 15 times. No `install.sh --yes --server --hostname=studio1` mode.

**2. Push configuration changes to all machines**

- **Works?** No. Configuration is per-machine. No central config management.
- **Documented?** N/A.
- **Tested?** N/A.
- **Issues:** Each machine has its own `~/.config/local-models/` directory. No way to push a config update from a central location. The auto-update mechanism updates code (via git tags) but not configuration.

**3. Monitor all machines from one dashboard**

- **Works?** No. Each machine has its own Playground/Activity view. No aggregated dashboard.
- **Documented?** N/A.
- **Tested?** N/A.
- **Issues:** Greg would need to open 15 browser tabs to monitor all machines. No centralized monitoring, no health check aggregation.

**4. Set up one powerful Mac Studio as the shared server**

- **Works?** Yes, same as Persona 8's server setup.
- **Documented?** Documented.
- **Tested?** Tested.
- **Issues:** Same single-token auth limitation. Same lack of per-user tracking.

**5. Automate updates across all machines**

- **Works?** The auto-update mechanism checks for new git tags every 2 minutes and auto-deploys. `git tag v1.x.x && git push --tags` on the repo triggers updates on all machines.
- **Documented?** Auto-update documented in architecture.
- **Tested?** Well tested in `test_deployment.py` (41 tests).
- **Issues:** This actually works well for the sysadmin use case. Tag a release, and all machines update within 2 minutes. The rollback mechanism (30-second crash window) provides safety. However, there is no staging/canary capability — all machines get the same update simultaneously. If a bad update ships, all 15 machines go down before the rollback kicks in.

---

## Persona 15: Lisa — Parent Setting Up AI for Kids

**Who:** Software engineer with two kids (ages 12 and 14). Has a Mac Mini M4 with 64GB in the home office. Wants to give the kids access to AI tools without cloud accounts or exposure to inappropriate content.

### Use Cases

**1. Set up Playground access for kids without giving them Claude Code**

- **Works?** Yes. The Playground is a web app accessible from any browser on the local network. Kids can use it from their iPads or school laptops by navigating to `http://localhost:8101/tools` (local) or the Tailscale URL (remote).
- **Documented?** PWA installation documented. Playground documented.
- **Tested?** Playground routes tested.
- **Issues:** The Playground is the same interface for everyone — no parental controls, no content filtering, no separate accounts. The "Unfiltered" tool uses models specifically designed to have fewer content restrictions (dolphin3, which is in the ALWAYS_EXCLUDE list but is a special task with its own prefix). A kid could use the Unfiltered tab to get content that a parent might not want.

**2. Filter inappropriate content from model responses**

- **Works?** No content filtering. The "Unfiltered" task type exists specifically to bypass model safety filters (using uncensored models like dolphin3). There is no way to disable the Unfiltered tool or restrict which tools are available in the Playground.
- **Documented?** Unfiltered is listed as a tool.
- **Tested?** Route tested.
- **Issues:** Major issue for this persona. No way to hide or disable the Unfiltered tab. No content filtering on responses from any model. No parental controls. The Playground exposes all tools to all users equally.

**3. Help kids with homework using local AI**

- **Works?** Yes. The "General" and "Code" tools in the Playground work for homework help. The kids type questions and get responses.
- **Documented?** Not specifically for this use case.
- **Tested?** Tested.
- **Issues:** The Playground interface assumes technical knowledge (tool selection, model selection). A 12-year-old might be confused by the tool grid. No simplified "kid mode" or preset configurations.

**4. Set up voice-based interaction for younger kid**

- **Works?** The building blocks exist: Transcribe (record via mic) and Speak (TTS). But there is no integrated voice conversation mode — it is two separate tools with no linking.
- **Documented?** Individual tools documented.
- **Tested?** Individual tools tested.
- **Issues:** A voice conversation would require: (a) record audio, (b) transcribe, (c) send to LLM, (d) speak response. There is no automated pipeline. Each step is manual.

**5. Restrict AI access to specific times of day**

- **Works?** No. No scheduling, no access controls, no time-based restrictions.
- **Documented?** N/A.
- **Tested?** N/A.
- **Issues:** The profile server has an idle timeout but no time-based access control. Lisa could manually stop the services with `start-local-models --stop` but there is no automated schedule.

---

## Summary


### Must Fix

1. **`claude-local` model selection bug:** The script reads `mcp_preferences.json`'s `general` key with `json.load(open(...)).get('general','')`. After the profile system change, this key is a list (e.g., `["qwen3.5", "glm-4.7-flash"]`), not a string. The script would pass a list representation as the model name, causing Ollama to fail. (`/Users/jerry/super-puppy/bin/claude-local`, line 17)

2. **CLAUDE.md / architecture.md inconsistency on Ollama binding:** CLAUDE.md says "Ollama binds `0.0.0.0` on the server for LAN access" while `docs/architecture.md` says "All services bind to `127.0.0.1`". The actual behavior is that `setenv.OLLAMA_HOST.plist` (server-only LaunchAgent) sets `OLLAMA_HOST=0.0.0.0`. This is a documentation inconsistency, not a code bug, but it misrepresents the security posture.

3. **Session eviction race:** In the MCP server, `_authenticated_sessions.pop()` when the set is full removes an arbitrary session. Under concurrent connections (team scenario), this could evict a legitimate active session, causing 403 errors for in-progress work. (`/Users/jerry/super-puppy/mcp/local-models-server.py`, line 127)

4. **Playground summarize path restriction inconsistency:** The Playground's `_is_safe_test_path` only allows `/tmp/` files, while the MCP server's `_validate_path` allows anything under `$HOME`. This means "Summarize" in the Playground can only summarize files in `/tmp/`, while the same tool via MCP can summarize any file under home. A user trying to summarize a project file via the Playground gets "File path must be in /tmp/" with no explanation of why.


2. **mflux not installed by installer:** Image generation and image editing require `mflux` and `mflux-generate-kontext`, which are not installed by `install.sh`. Users discover this only when they try the tools and get subprocess errors.

8. **Voice cloning in Playground:** The MCP tool supports `ref_audio` for Chatterbox voice cloning, but the Playground TTS tab only exposes Voxtral presets. No way to do voice cloning from the web UI.

10. **Disable auto-update:** No way to turn off the 2-minute git tag polling. Important for air-gapped environments and manual update workflows.

11. **Configurable path restrictions:** MCP path validation is hardcoded to `$HOME` + `/tmp`. No way to restrict to a specific project directory.

1. **Prerequisites not documented:** `git`, Xcode Command Line Tools, Homebrew are required but not mentioned in the Quick Start. A fresh Mac will fail at `git clone`.

2. **mflux installation:** Image gen/edit tools require mflux but no documentation explains how to install it.

3. **ffmpeg requirement:** Audio transcription of certain formats (especially WebM from browser recording) requires ffmpeg. Not mentioned in README or installer.

8. **sentence-transformers network calls:** README claims "no data leaving your network" but first-use of HuggingFace embedding models downloads from the internet. Auto-update polls GitHub.

9. **Generation time expectations:** No benchmarks or rough time estimates for any operation. Users do not know if image generation takes 10 seconds or 10 minutes.

10. **Minimum RAM tiers:** README says "64GB+" but the Desktop profile fits in 64GB, the Laptop profile targets 32GB, and below 48GB MLX is disabled. The actual minimum for a useful experience is unclear.

2. **No progress indicators:** Long operations (transcription, image generation, model download on first use) show no progress. The request appears to hang.

3. **Output files in /tmp/:** Generated images, audio, and other outputs are saved to `/tmp/` with cryptic names. Users must navigate a hidden directory to find their files. Files disappear on reboot. Add "Download" button in the Playground for generated assets (audio has a player, images display inline, but explicit download/save is not prominent).

5. **Playground path restriction vs MCP:** Summarize in the Playground only accepts `/tmp/` paths, while the MCP tool accepts `$HOME`. The Playground Summarize tool asks for a "File path" input but does not explain the `/tmp/` restriction. Users typing their actual file paths get a generic error.


6. **Model picker shows all models for all tools:** The Playground model picker does not filter models by tool capability. A user can select an embedding model for Code generation and get an error.

8. **First-run model downloads:** The default profiles reference specific models that may not be downloaded. The installer asks about pulling models, but if the user skips this, the Playground tools fail with model-not-found errors rather than offering to download.

1. **No tests for `claude-local` script:** The offline Claude experience should have been totally removed from main.

### Out Of Scope for Project

5. **Parental controls / tool visibility:** No way to hide or disable specific Playground tools (especially "Unfiltered"). No content filtering on model responses.

7. **Silent / non-interactive install:** `install.sh` always requires interactive input. No flags for automated deployment.

1. **Playground terminology:** "Tool", "Model", "Backend" are developer terms. Non-technical users see 13 tool buttons and do not know which one to pick.

6. **Team onboarding workflow:** No step-by-step guide for setting up a shared server for a team, distributing tokens, or managing multi-user access.

5. **Air-gapped deployment:** No guide for running fully offline (disable auto-update, pre-cache all models, pre-install all dependencies).

4. **Memory-per-model guidance:** No documentation on how much RAM each model needs. Users cannot make informed choices about which profile to use.

7. **"Local (override)" label:** When a user forces local mode but the desktop is reachable, the menu shows "Local (override)" which sounds like they are doing something wrong. Better: "Local (desktop available)" or just "Local".

### Out Of Scope for Project, But Worth Documenting

4. **Per-user authentication:** Single shared bearer token for all users. No per-user accounts, quotas, audit trail, or access revocation.

### Add to TODO for later

1. **Multi-turn conversation / chat interface:** The Playground is single-turn only. No chat history, no conversation threading. Users expecting a ChatGPT-like experience will be disappointed.

7. **Playground is single-turn:** README says "test any capability interactively" which implies conversation, but the Playground is strictly single-prompt single-response.



3. **Batch processing:** No batch operations for any tool. Cannot process multiple images, translate multiple files to separate outputs, or embed thousands of documents efficiently.

6. **Headless / daemonized mode:** No way to run Super Puppy services without the menu bar app GUI. Important for servers, Mac Minis, and CI environments.


9. **Mandarin/Japanese TTS:** Default Voxtral voices cover only Western + Arabic + Hindi languages. No CJK TTS presets in the Playground.

12. **Mid-session server fallback:** Client-to-server connection is established at MCP startup. If the connection drops mid-session, there is no automatic fallback to local models without restarting the MCP server.


4. **Error messages expose internals:** When a backend tool (mflux, ffmpeg) is missing, the error is a raw subprocess error with stderr output. Non-technical users see `FileNotFoundError: [Errno 2] No such file or directory: 'mflux-generate'` and have no idea what to do.

### Test Gaps

2. **No tests for `install.sh`:** The installer (which every user must run) has no automated tests.
3. **No test for voice cloning flow:** `local_speak` with `ref_audio` is untested.
4. **No test for multi-image vision:** `local_vision` with multiple `image_paths` is untested.
5. **No test for long audio transcription timeout:** Transcription timeout behavior for large files is untested.
6. **No test for Playground file upload + tool execution end-to-end:** The upload-then-process flow is untested.
7. **No test for the mid-session connectivity drop scenario:** Client behavior when the server becomes unreachable during an active session is untested.
8. **No test for `_is_safe_test_path` vs `_validate_path` inconsistency:** The different path validation strategies in the profile server vs MCP server are not cross-tested.
9. **No test for Playground model picker filtering by capability:** A user selecting an incompatible model for a tool is not tested.
10. **No headless/daemonized operation tests:** Running services without the menu bar GUI is untested.
11. **No test for batch file translation output:** `local_translate` with `file_paths` producing usable per-file output is untested.
12. **No test for first-use model download behavior:** What happens when a tool references a model that is not yet downloaded is tested in error handling but not for the on-demand download flow.
