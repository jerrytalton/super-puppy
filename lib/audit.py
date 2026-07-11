"""Super Puppy configuration audit: verify (and optionally fix) that installed
agent tools are wired to use SP — MCP registration, guidance blocks, session hook.

Design note (spec §S2): guidance blocks are text an AI agent treats as
instructions. Fixes here are NEVER automatic — a caller (sp-doctor / menubar /
install.sh opt-in) applies them explicitly, with a diff. The managed block is
minimal and mechanical; changing GUIDANCE_TEXT is a reviewed code diff.
"""

GUIDANCE_MARKERS = ("<!-- >>> super-puppy >>> -->", "<!-- <<< super-puppy <<< -->")

GUIDANCE_TEXT = """\
## Local Models (super-puppy)

Reach for the `local-models` MCP server whenever a task needs capabilities you \
lack (vision, audio, image/video) or cheap parallel compute (bulk transforms, \
second opinions, delegated reasoning while you keep working). Call \
`local_models_status` for what's live.

**Do things in parallel on the server and with the local cluster whenever it \
makes sense.** While you work, the local GPUs should be busy too — fire off \
`local_dispatch` before you start reasoning, not after you finish.

| Need | Tool |
|------|------|
| Look at an image or screenshot | `local_vision` |
| Plan a GUI interaction | `local_computer_use` |
| Generate / edit an image | `local_image`, `local_image_edit` |
| Generate video | `local_video` |
| Translate / transcribe / speak | `local_translate`, `local_transcribe`, `local_speak` |
| Bulk boilerplate / repetitive transforms | `local_generate` |
| Second opinion on code or designs | `local_review`, `local_candidates` |
| Summarize before reading a huge file | `local_summarize` |
| Delegate reasoning while you keep working | `local_dispatch` then `local_collect` |
| Find files by concept | `local_similarity_search`, `local_embed` |

Model/hardware specifics live in each tool's runtime description and \
`local_models_status` — trust those over anything static.
"""


def render_block() -> str:
    return f"{GUIDANCE_MARKERS[0]}\n{GUIDANCE_TEXT}\n{GUIDANCE_MARKERS[1]}"


def upsert_guidance(text: str) -> str:
    block = render_block()
    start, end = GUIDANCE_MARKERS
    if start in text and end in text:
        pre = text[: text.index(start)]
        post = text[text.index(end) + len(end):]
        return f"{pre}{block}{post}"
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return f"{text}{sep}{block}\n"
