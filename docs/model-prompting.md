# Model Prompting & Internal Behavior

Prompt-construction details, output formats, and model-specific failure modes for the foundation models behind Super Puppy's MCP wrappers, profile server, and direct-API code paths.

**Audience:** anyone editing wrapper code (`mcp/local-models-server.py`), the agent code (`src/agent/...`), or the model-server config. If you're consuming these models through MCP tools (`local_vision`, `local_computer_use`, etc.) the wrappers handle prompting and parsing — see `~/.claude/model-playbook.md` for the consumer-side cheat sheet instead.

Keep entries terse — no narrative, no motivating sessions (those belong in commits), no changelog (use `git log`).

---

## UI-TARS (`ui-tars-72b`)

GUI grounding / computer use. Screenshot in, next action out. Train res is 1920×1080 — smaller viewports compress buttons below grounding precision.

### Output format — native only

```
Thought: <one sentence>
Action: funcname(arg1='value1')
```

Do **not** ask for JSON or wrap in a custom schema. Coordinates are per-mille (0–1000) inside `<|box_start|>(x,y)<|box_end|>`; rescale `x_px = x/1000 * viewport_width`.

### Native action vocabulary

```
click(start_box='<|box_start|>(x,y)<|box_end|>')
left_double(start_box=...)
right_single(start_box=...)
type(content='text')
hotkey(key='Enter')                # 'ctrl a', 'ArrowDown', 'Tab', etc.
scroll(start_box=..., direction='down')   # up|down|left|right
drag(start_box=..., end_box=...)
wait
finished(content='<reason>')       # terminal
```

Teach custom actions with a one-line inline comment next to the signature. Do not over-explain.

### ⚠️ Compact prompts only

Verbose system prompts measurably degrade grounding accuracy (observed: ~400 px click offset from a 450→1745 char schema bloat). Rules:

- Function signatures with at-most-one-line comments per action.
- No "Rules" sections, no prose about coordinates or output format.
- If you feel like adding clarifying prose, delete it.

### Prompting

- **System**: task context + compact action schema. Tight.
- **User**: image first, then short text footer. History as one-liners (`[1] agent: click(640,145)`). End with "Emit the next Thought and Action."
- **Temperature**: 0 for debugging, 0.2 for diversity.
- **max_tokens**: 256–1024 (actual output ~50–100 tokens).
- Native `finished` takes no args. `finished(content=...)` is a custom extension — teach via inline comment.

### Dense form y-undershoot (~20–40 px)

On dense vertical stacks (inputs + button), y undershoots by ~20–40 px — clicks land above the target, often inside the element one row up. x stays accurate. Sparse pages are fine (~3 px). Post-hoc snap to nearest interactive doesn't fix it: containment tiebreaker snaps to the wrong row.

**Fix that works**: split planning from grounding. Use Qwen3.5:122b as the planner (emits natural-language intents like "click the Next button") and Qwen3.5:9b as the grounder (locates the described element in the screenshot and returns per-mille coordinates). The 9b grounder grounds within ~3 px on all tested layouts including dense forms. See `~/super-puppy/src/agent/model/qwen-fara.ts`.

### Gotchas

- On sparse pages, attention drifts toward image center. Compact prompts mitigate.
- `Thought:` line may come back in Chinese; `Action:` is unaffected.
- Large model, slow under contention. Smoke loops with > 5 min gaps between calls hit cold reloads — keep cadence tight or warm up first.

---

## Qwen3.5 wrapper notes

These apply when wrapping Qwen3.5 (or other thinking VLMs returning structured output) for downstream consumers. General Qwen3.5 prompting (token budgets, suppressing thinking, structured-output penalty params) stays in `~/.claude/model-playbook.md`.

### Grounding output format

When asked to return element coordinates, Qwen3.5 returns 0–1000 per-mille values regardless of whether the prompt says "pixels" or "per-mille." Always assume per-mille and rescale: `x_px = x/1000 * viewport_width`. Detect by checking if both coords are ≤ 1000.

### Default-shaped fallback on unparseable output

"Respond ONLY with valid JSON" is unreliable. The model returns empty strings, prose, or `<think>` blocks. Retries + repair passes help but aren't enough. Wrappers must return a fully-populated default object (every field your downstream expects, sensible empty values) on unparseable input — never `None`, never a partial dict. When logging parse failures, include `raw[:200]`; bare `JSONDecodeError` tells you nothing.

### Coerce field types at the boundary

Schema fields drift across calls: a string field comes back as a list, int, or null; a list field comes back as a bare string. Concatenations and iterations crash the row. Normalize every field at the boundary before touching it — assume nothing about types even when your prompt specified them.

---

## MAI-UI (`mai-ui-8b`)

RL-trained GUI agent fine-tuned from Qwen3-VL. 73.5% on ScreenSpot-Pro vs UI-TARS-72B's ~23%.

### Output format — pixel coordinates

Uses **pixel coords**, not per-mille. Native schema: `click(x=N, y=N)`, `type(content='...')`, `swipe(direction='...')`, `long_press(x, y)`, `system_button(key)`, `wait`, `terminate(status)`.

### Gotchas

**Parser**: model often emits `click(x=358,43)` — mixed named/positional args. Parser must fall back to extracting any two numbers from the raw args text.

**Grounding fails on wide desktop layouts.** Despite ScreenSpot-Pro scores, misses navigation links by ~300 px on 1920×1080 web pages. ScreenSpot-Pro is dominated by mobile screenshots with large touch targets. Desktop UIs with small text links in centered layouts are out-of-distribution.

**Reasoning insufficient for multi-step UX flows.** With DOM action menu (grounding eliminated), correctly navigates to pages and finds buttons, but clicks the same action repeatedly without checking results, navigating to verify, or emitting findings. Chain-of-thought shows it *recognizes* it's stuck ("I've been repeatedly clicking the same spot") but cannot reason about what to do instead. An 8B model lacks the task-planning capacity needed for UX regression testing. **Grounding is not the bottleneck — reasoning is.**
