# local_video Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `local_video` MCP tool that generates video from text, images, or text+audio using locally-running MLX video models (Wan2.2, LTX-2), with full profile/playground integration.

**Architecture:** Two packages (`mlx-video` for Wan2.2/LTX-2 T2V/I2V, `mlx-video-with-audio` for audio-synced LTX-2) behind a single `local_video` tool. Mode auto-detected from parameters: `image_path` → I2V, `audio_genre` → audio-synced, else T2V. Execution via subprocess (same pattern as mflux image gen). Models discovered from HuggingFace cache via `hf_scanner.py` and visible in profiles.

**Tech Stack:** mlx-video (git), mlx-video-with-audio (PyPI), Python 3.12, pytest

---

### Task 1: Add `video` to SPECIAL_TASKS and ALWAYS_EXCLUDE

**Files:**
- Modify: `lib/models.py:126-167`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Add a new test class at the end of `tests/test_core.py`:

```python
class TestVideoTask:
    def test_video_in_special_tasks(self):
        from lib.models import SPECIAL_TASKS
        assert "video" in SPECIAL_TASKS

    def test_video_has_prefixes(self):
        from lib.models import SPECIAL_TASKS
        task = SPECIAL_TASKS["video"]
        assert "label" in task
        assert "prefixes" in task
        assert len(task["prefixes"]) > 0

    def test_video_prefixes_match_known_models(self):
        from lib.models import SPECIAL_TASKS
        prefixes = SPECIAL_TASKS["video"]["prefixes"]
        test_names = ["wan2.2-i2v", "ltx-video-2b", "Wan2.1-T2V"]
        for name in test_names:
            assert any(name.lower().startswith(p.lower()) for p in prefixes), (
                f"{name} should match a video prefix")

    def test_video_models_excluded_from_general_tasks(self):
        from lib.models import ALWAYS_EXCLUDE
        assert "wan2" in ALWAYS_EXCLUDE
        assert "ltx" in ALWAYS_EXCLUDE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest tests/test_core.py::TestVideoTask -v`
Expected: FAIL — `"video" not in SPECIAL_TASKS`

- [ ] **Step 3: Add video to SPECIAL_TASKS and ALWAYS_EXCLUDE**

In `lib/models.py`, add to `SPECIAL_TASKS` dict (after the `"computer_use"` entry, before the closing `}`):

```python
    "video": {
        "label": "Video",
        "prefixes": ["wan2", "ltx"],
    },
```

In `lib/models.py`, add `"wan2"` and `"ltx"` to `ALWAYS_EXCLUDE`:

```python
ALWAYS_EXCLUDE: list[str] = [
    "vl", "flux", "z-image", "whisper", "ocr", "embed", "minilm",
    "tinyllama", "goonsai", "nsfw", "dolphin",
    "wan2", "ltx",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest tests/test_core.py::TestVideoTask -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add lib/models.py tests/test_core.py
git commit -m "feat: add video task type to SPECIAL_TASKS"
```

---

### Task 2: Add video model detection to hf_scanner

**Files:**
- Modify: `lib/hf_scanner.py:15-31`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Add to the `TestVideoTask` class in `tests/test_core.py`:

```python
    def test_hf_scanner_classifies_wan_video(self):
        from lib.hf_scanner import _classify_model
        # Wan2.2 transformer config has _class_name
        config = {"_class_name": "WanTransformer3DModel"}
        assert _classify_model(config, "Wan2.2-T2V-14B") == "video"

    def test_hf_scanner_classifies_ltx_video(self):
        from lib.hf_scanner import _classify_model
        config = {"_class_name": "LTXVideoTransformer3DModel"}
        assert _classify_model(config, "Lightricks/LTX-Video-2") == "video"

    def test_hf_scanner_classifies_ltx_by_name(self):
        from lib.hf_scanner import _classify_model
        # Some LTX models may not have a diffusers class
        config = {}
        assert _classify_model(config, "ltx-video-2b-v0.9.5") == "video"

    def test_hf_scanner_classifies_wan_by_name(self):
        from lib.hf_scanner import _classify_model
        config = {}
        assert _classify_model(config, "Wan2.1-T2V-1.3B") == "video"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest tests/test_core.py::TestVideoTask::test_hf_scanner_classifies_wan_video -v`
Expected: FAIL — returns `None`

- [ ] **Step 3: Add video detection to hf_scanner.py**

In `lib/hf_scanner.py`, add to `_DIFFUSERS_CLASS_TASKS` (line 22-25):

```python
_DIFFUSERS_CLASS_TASKS = {
    "FluxTransformer2DModel": "image_gen",
    "Flux2Transformer2DModel": "image_gen",
    "WanTransformer3DModel": "video",
    "LTXVideoTransformer3DModel": "video",
}
```

Add to `_NAME_TASK_OVERRIDES` (line 28-31):

```python
_NAME_TASK_OVERRIDES = {
    "Kontext": "image_edit",
    "Fill": "image_edit",
    "ltx-video": "video",
    "ltx2": "video",
    "wan2": "video",
    "Wan2": "video",
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest tests/test_core.py::TestVideoTask -v`
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add lib/hf_scanner.py tests/test_core.py
git commit -m "feat: add video model detection to HuggingFace scanner"
```

---

### Task 3: Add video to MCP server model discovery and pick_model

**Files:**
- Modify: `mcp/local-models-server.py:383-401`
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_mcp_server.py`. Find the test class that tests model discovery or pick_model, and add:

```python
    def test_video_in_task_backends(self):
        """Video task type is registered for HF cache discovery."""
        # Re-import to get the _TASK_BACKENDS dict
        import mcp  # the server module
        # The _TASK_BACKENDS dict is defined inside _discover_models.
        # Instead, verify that a video model discovered from HF cache
        # gets the right backend by checking SPECIAL_TASKS integration.
        from lib.models import SPECIAL_TASKS
        assert "video" in SPECIAL_TASKS
        assert SPECIAL_TASKS["video"]["label"] == "Video"
```

- [ ] **Step 2: Add video to _TASK_BACKENDS**

In `mcp/local-models-server.py`, modify the `_TASK_BACKENDS` dict (around line 383):

```python
        _TASK_BACKENDS = {
            "tts": "mlx-audio",
            "transcription": "mlx",
            "image_edit": "mflux",
            "image_gen": "mflux",
            "video": "mlx-video",
        }
```

- [ ] **Step 3: Run existing tests to verify nothing breaks**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_mcp_server.py -v`
Expected: All existing tests pass

- [ ] **Step 4: Commit**

```bash
git add mcp/local-models-server.py tests/test_mcp_server.py
git commit -m "feat: add video to MCP server model discovery"
```

---

### Task 4: Implement the `local_video` MCP tool

**Files:**
- Modify: `mcp/local-models-server.py` (insert after `local_image_edit`, around line 1036)
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing test**

Add test to `tests/test_mcp_server.py`:

```python
class TestLocalVideoTool:
    """Tests for local_video tool parameter validation and mode detection."""

    def test_local_video_mode_text_to_video(self):
        """No image_path and no audio_genre → text-to-video mode."""
        # Mode detection is embedded in the tool. Test via the tool's
        # command construction by mocking subprocess.
        pass  # Placeholder — real test after implementation

    def test_local_video_mode_image_to_video(self):
        """image_path provided → image-to-video mode."""
        pass

    def test_local_video_mode_audio(self):
        """audio_genre provided → audio-synced mode via mlx-video-with-audio."""
        pass
```

These will be fleshed out after we see the implementation shape. For now, the key test is that the tool function exists:

```python
    def test_local_video_tool_exists(self):
        """local_video is registered as an MCP tool."""
        src = Path("mcp/local-models-server.py").read_text()
        assert "async def local_video(" in src
        assert "@mcp.tool()" in src.split("async def local_video(")[0][-50:]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --with pytest pytest tests/test_mcp_server.py::TestLocalVideoTool::test_local_video_tool_exists -v`
Expected: FAIL — `"async def local_video(" not in src`

- [ ] **Step 3: Implement local_video tool**

Insert after the `local_image_edit` tool (around line 1036) in `mcp/local-models-server.py`:

```python
@mcp.tool()
async def local_video(
    prompt: str,
    image_path: str | None = None,
    output_path: str | None = None,
    model: str | None = None,
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
    audio_genre: str | None = None,
) -> str:
    """Generate video locally using MLX (Wan2.2, LTX-2).

    Auto-detects mode from parameters:
    - Text prompt only → text-to-video
    - image_path provided → image-to-video (animates the image)
    - audio_genre provided → video with synchronized audio

    Output is an MP4 file saved to disk.

    Args:
        prompt: Description of the video to generate.
        image_path: Optional input image for image-to-video mode.
        output_path: Where to save the video. Defaults to /tmp/local_video_<timestamp>.mp4.
        model: Optional model override. Defaults to best available video model.
        width: Output width in pixels (must be divisible by 64). Default: model's native resolution.
        height: Output height in pixels (must be divisible by 64). Default: model's native resolution.
        num_frames: Number of frames to generate. Default: 65 (~2.7s at 24fps).
        audio_genre: Music genre for audio-synced video (e.g. 'electronic', 'jazz', 'ambient').
            When provided, uses mlx-video-with-audio for synchronized audio generation.
    """
    try:
        selected, backend = pick_model("video", model)
    except ValueError:
        return ("Error: no video model available. Install mlx-video or "
                "mlx-video-with-audio and download a model (e.g. Wan2.2-T2V-14B).")

    if image_path:
        err = _validate_path(image_path)
        if err:
            return f"Error: {err}"

    if not output_path:
        import time as _time
        output_path = f"/tmp/local_video_{int(_time.time())}.mp4"
    err = _validate_path(output_path, must_exist=False)
    if err:
        return f"Error: {err}"

    mode = "audio" if audio_genre else ("i2v" if image_path else "t2v")
    logging.info("generate video %s (%s, %s): %s", selected, backend, mode, prompt[:50])

    with _gpu_request("mlx", f"video:{selected}"):
        warning = _gpu_contention_warning("mlx")
        loop = asyncio.get_event_loop()

        if mode == "audio":
            # mlx-video-with-audio CLI
            cmd = [
                sys.executable, "-m", "mlx_video_with_audio",
                "--prompt", prompt,
                "--output", output_path,
            ]
            if width:
                cmd.extend(["--width", str(width)])
            if height:
                cmd.extend(["--height", str(height)])
            if num_frames:
                cmd.extend(["--num-frames", str(num_frames)])
            if audio_genre:
                cmd.extend(["--audio-genre", audio_genre])
            timeout = 1200
        else:
            # mlx-video CLI (Wan2.2 / LTX-2)
            cmd = [
                sys.executable, "-m", "mlx_video",
                "--model", selected,
                "--prompt", prompt,
                "--output", output_path,
            ]
            if image_path:
                cmd.extend(["--image", image_path])
            if width:
                cmd.extend(["--width", str(width)])
            if height:
                cmd.extend(["--height", str(height)])
            if num_frames:
                cmd.extend(["--num-frames", str(num_frames)])
            timeout = 900

        try:
            proc = await loop.run_in_executor(None, lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
            ))
            if proc.returncode != 0:
                return f"Error: video generation failed:\n{proc.stderr[-500:]}"
        except subprocess.TimeoutExpired:
            return f"Error: video generation timed out after {timeout // 60} minutes."

    if not Path(output_path).exists():
        return f"Error: output video was not created at {output_path}"

    size = Path(output_path).stat().st_size
    mb = size / (1024 * 1024)
    return f"{warning}[{selected} via {backend}]\n\nVideo saved to {output_path} ({mb:.1f} MB)"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_mcp_server.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add mcp/local-models-server.py tests/test_mcp_server.py
git commit -m "feat: add local_video MCP tool"
```

---

### Task 5: Add video to profile server model discovery and default profiles

**Files:**
- Modify: `app/profile-server.py:384-390` (_TASK_BACKENDS)
- Modify: `app/profile-server.py:461-540` (DEFAULT_PROFILES)
- Test: `tests/test_profile_server.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_profile_server.py`:

```python
class TestVideoProfiles:
    def test_video_in_task_backends(self, client):
        """Video task type is in the profile server's model discovery."""
        from app import profile_server  # adjust import as needed
        # Check DEFAULT_PROFILES include video
        for name, profile in profile_server.DEFAULT_PROFILES["profiles"].items():
            if name in ("everyday", "maximum"):
                assert "video" in profile["tasks"], (
                    f"Profile '{name}' should have a video task slot")

    def test_video_models_in_api_tasks(self, client):
        """GET /api/tasks includes video as a special task."""
        resp = client.get("/api/tasks")
        data = resp.get_json()
        assert "video" in data.get("special", {}), "video missing from special tasks"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py::TestVideoProfiles -v`
Expected: FAIL

- [ ] **Step 3: Add video to profile server**

In `app/profile-server.py`, add to `_TASK_BACKENDS` (around line 384):

```python
    _TASK_BACKENDS = {
        "tts": "mlx-audio",
        "transcription": "mlx",
        "image_edit": "mflux",
        "image_gen": "mflux",
        "video": "mlx-video",
    }
```

Add `"video"` to DEFAULT_PROFILES. In `everyday` and `maximum` profiles (the 512GB machines), add:

```python
                "video": "Wan2.2-T2V-14B",
```

In `desktop` (64GB), add:

```python
                "video": "Wan2.1-T2V-1.3B",
```

In `laptop` (32GB), omit video (too memory-heavy for 32GB).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add app/profile-server.py tests/test_profile_server.py
git commit -m "feat: add video to profile server and default profiles"
```

---

### Task 6: Add video playground card to tools.html

**Files:**
- Modify: `app/tools.html:245-280` (TOOLS object)
- Modify: `app/tools.html:747` (SLOW_TOOLS)

- [ ] **Step 1: Add video tool entry to TOOLS object**

In `app/tools.html`, add after the `image_gen` entry (around line 247):

```javascript
  video:       { label: 'Video', task: 'video', fields: [
    {id:'prompt', label:'Prompt', type:'textarea', placeholder:'A puppy running through a field of flowers...'},
    {id:'image_path', label:'Source image (optional)', type:'file', accept:'image/*'},
    {id:'audio_genre', label:'Audio genre (optional)', type:'select', options: [
      '', 'ambient', 'classical', 'electronic', 'jazz', 'pop', 'rock',
      'cinematic', 'lo-fi', 'hip-hop', 'folk', 'reggae', 'blues',
    ]},
    {id:'width', label:'Width', type:'text', placeholder:'720'},
    {id:'height', label:'Height', type:'text', placeholder:'480'},
    {id:'num_frames', label:'Frames', type:'text', placeholder:'65'},
  ]},
```

- [ ] **Step 2: Add video to SLOW_TOOLS**

In `app/tools.html`, modify the `SLOW_TOOLS` line (around line 747):

```javascript
      const SLOW_TOOLS = {image_gen: 'Generating image', image_edit: 'Editing image', speak: 'Generating speech', transcribe: 'Transcribing audio', video: 'Generating video'};
```

- [ ] **Step 3: Add video output handler**

In the response handling section of `tools.html`, add video playback support. Find the `else if (data.image_path)` blocks (there are two — one for the stashed-result path and one for the active-result path). After each `image_path` block, add a `video_path` block:

For the stashed-result path (around line 772):
```javascript
        } else if (data.video_path) {
          const vidUrl = `/api/test/video?path=${encodeURIComponent(data.video_path)}`;
          html = data.result +
            `<video controls src="${vidUrl}" style="display:block;margin-top:8px;width:100%;max-height:400px"></video>` +
            `<a href="#" onclick="downloadFile('${vidUrl}','video.mp4');return false" class="download-btn">Download video</a>`;
```

For the active-result path (around line 808):
```javascript
      } else if (data.video_path) {
        output.textContent = data.result;
        const vidUrl = '/api/test/video?path=' + encodeURIComponent(data.video_path);
        const vid = document.createElement('video');
        vid.controls = true; vid.style.cssText = 'display:block;margin-top:8px;width:100%;max-height:400px';
        vid.src = vidUrl;
        output.appendChild(vid);
        const dl = document.createElement('a');
        dl.href = '#'; dl.className = 'download-btn'; dl.textContent = 'Download video';
        dl.onclick = (e) => { e.preventDefault(); downloadFile(vidUrl, 'video.mp4'); };
        output.appendChild(dl);
```

- [ ] **Step 4: Commit**

```bash
git add app/tools.html
git commit -m "feat: add video playground card with preview player"
```

---

### Task 7: Add video API route to profile server

**Files:**
- Modify: `app/profile-server.py` (in `/api/test` handler, after image_gen block)
- Modify: `app/profile-server.py` (add `/api/test/video` serving endpoint)
- Modify: `app/profile-server.py:980-984` (_MISSING_TOOL_HELP)

- [ ] **Step 1: Add video handler to /api/test**

In `app/profile-server.py`, after the `elif tool == "image_gen":` block (around line 1420), add:

```python
        elif tool == "video":
            model, backend = _pick("video")
            image_path = body.get("image_path", "")
            if image_path and not _is_safe_test_path(image_path):
                return jsonify({"error": _PLAYGROUND_PATH_ERROR}), 403
            audio_genre = body.get("audio_genre", "")
            import time as _time
            out = f"/tmp/playground_video_{int(_time.time())}.mp4"

            width_str = body.get("width", "")
            height_str = body.get("height", "")
            frames_str = body.get("num_frames", "")

            mode = "audio" if audio_genre else ("i2v" if image_path else "t2v")

            with _track_playground("video", model, backend):
                try:
                    if mode == "audio":
                        cmd = [
                            sys.executable, "-m", "mlx_video_with_audio",
                            "--prompt", body["prompt"],
                            "--output", out,
                        ]
                        if width_str:
                            cmd.extend(["--width", width_str])
                        if height_str:
                            cmd.extend(["--height", height_str])
                        if frames_str:
                            cmd.extend(["--num-frames", frames_str])
                        if audio_genre:
                            cmd.extend(["--audio-genre", audio_genre])
                    else:
                        cmd = [
                            sys.executable, "-m", "mlx_video",
                            "--model", model,
                            "--prompt", body["prompt"],
                            "--output", out,
                        ]
                        if image_path:
                            cmd.extend(["--image", image_path])
                        if width_str:
                            cmd.extend(["--width", width_str])
                        if height_str:
                            cmd.extend(["--height", height_str])
                        if frames_str:
                            cmd.extend(["--num-frames", frames_str])

                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=1200,
                        env={**os.environ,
                             "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"},
                    )
                except FileNotFoundError:
                    return jsonify({"error": "mlx-video is not installed. Install with: pip install git+https://github.com/Blaizzy/mlx-video.git"})
                if result.returncode != 0:
                    return jsonify({"error": f"video: generation failed:\n{result.stderr[-300:]}"})

            if not Path(out).exists():
                return jsonify({"error": f"video: output was not created at {out}"})
            return jsonify({"result": f"Saved to {out}", "video_path": out, "model": model})
```

- [ ] **Step 2: Add /api/test/video serving endpoint**

After the `/api/test/audio` route (around line 1617), add:

```python
@app.route("/api/test/video")
def api_test_video():
    path = request.args.get("path", "")
    if not path or not _is_safe_test_path(path) or not Path(path).exists():
        return "Not found", 404
    as_download = "download" in request.args
    return send_file(path, mimetype="video/mp4", as_attachment=as_download,
                     download_name=Path(path).name if as_download else None)
```

- [ ] **Step 3: Run tests to verify nothing breaks**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/test_profile_server.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add app/profile-server.py
git commit -m "feat: add video API route and serving endpoint to profile server"
```

---

### Task 8: Update playground coverage tests

**Files:**
- Modify: `tests/test_playground_coverage.py:30-42`

- [ ] **Step 1: Update MCP_TO_PLAYGROUND mapping**

In `tests/test_playground_coverage.py`, add the `local_video` mapping:

```python
MCP_TO_PLAYGROUND = {
    "local_generate": {"code", "general", "unfiltered"},
    "local_review": {"review"},
    "local_vision": {"vision"},
    "local_computer_use": {"computer_use"},
    "local_image": {"image_gen"},
    "local_image_edit": {"image_edit"},
    "local_transcribe": {"transcribe"},
    "local_speak": {"speak"},
    "local_translate": {"translate"},
    "local_summarize": {"summarize"},
    "local_embed": {"embed"},
    "local_video": {"video"},
}
```

- [ ] **Step 2: Run playground coverage tests**

Run: `uv run --with pytest pytest tests/test_playground_coverage.py -v`
Expected: 4 PASSED (all coverage tests pass with the new mapping)

- [ ] **Step 3: Commit**

```bash
git add tests/test_playground_coverage.py
git commit -m "test: add local_video to playground coverage mapping"
```

---

### Task 9: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add video to task types list**

In `CLAUDE.md`, find the "Task types" section under "### Task types". Add `video` to the special tasks list:

```
- **Special tasks** (matched by model capability): `vision`, `computer_use`, `image_gen`, `image_edit`, `video`, `transcription`, `tts`, `embedding`, `unfiltered`
```

- [ ] **Step 2: Add video tool to the global CLAUDE.md**

In `~/.claude/CLAUDE.md`, find the "Local Model Cluster" section. Add after the `local_image` bullet:

```
* **Video generation**: Use `local_video` to generate video from text prompts, animate still images, or create video with synchronized audio. Supports Wan2.2 and LTX-2 via MLX.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add video task type and local_video tool"
```

---

### Task 10: Add package dependencies

**Files:**
- Modify: `mcp/local-models-server.py:1-3` (PEP 723 inline deps)

- [ ] **Step 1: Add mlx-video and mlx-video-with-audio to dependencies**

In `mcp/local-models-server.py`, update the PEP 723 metadata at the top of the file. Add the two video packages to the `dependencies` list:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]==1.26.0", "httpx==0.28.1", "sentence-transformers==5.3.0", "torch==2.11.0", "pyyaml==6.0.3", "mlx-audio[tts] @ git+https://github.com/Blaizzy/mlx-audio.git@e42e1431fcf89af313375296c46d03a0153c4aa7", "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git", "mlx-video-with-audio"]
# ///
```

**Note:** `mlx-video` is git-only (not on PyPI). `mlx-video-with-audio` is on PyPI. Pin to a specific commit hash after verifying the install works — same pattern as `mlx-audio`.

- [ ] **Step 2: Verify the packages can be resolved**

Run: `uv pip compile --python 3.12 - <<< "mlx-video-with-audio"` to verify PyPI resolution.
Run: `pip install --dry-run "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git"` to verify git resolution.

- [ ] **Step 3: Commit**

```bash
git add mcp/local-models-server.py
git commit -m "feat: add mlx-video and mlx-video-with-audio dependencies"
```

---

### Task 11: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `uv run --with pytest --with flask --with pyyaml --with requests pytest tests/ -v`
Expected: All tests pass. No regressions.

- [ ] **Step 2: Verify package installs work (manual)**

On the Mac Studio:
```bash
uv pip install "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git"
uv pip install mlx-video-with-audio
```

- [ ] **Step 3: Test end-to-end with a real model (manual)**

```bash
python -m mlx_video --model Wan2.1-T2V-1.3B --prompt "A puppy playing in snow" --output /tmp/test_video.mp4
```

Verify the output is a valid MP4.
