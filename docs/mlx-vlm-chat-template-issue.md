# Bug: Qwen2VL processor missing chat_template causes HTTP 500 on multimodal inference

## Repository

`Blaizzy/mlx-vlm`

## Summary

`Qwen2VLProcessor.from_pretrained()` loads the chat template into the **tokenizer** via `load_chat_template()` but does not pass it through to the `ProcessorMixin.__init__`, leaving `processor.chat_template = None`. Any caller that invokes `processor.apply_chat_template()` (as `mlx-openai-server`'s app layer does for multimodal models) gets:

```
ValueError: Cannot use apply_chat_template because this processor does not have a chat template.
```

This affects **all Qwen2VL-architecture models** served through mlx-openai-server in multi-handler mode (isolated subprocesses), including `mlx-community/UI-TARS-72B-DPO-4bit`.

## Environment

- mlx-openai-server 1.7.0
- mlx-vlm (bundled with mlx-openai-server 1.7.0)
- transformers 5.4.0
- macOS 15.4 / Apple Silicon (M3 Ultra)
- Python 3.12

## Steps to reproduce

1. Configure mlx-openai-server with a Qwen2VL-based multimodal model:
   ```yaml
   - model_path: mlx-community/UI-TARS-72B-DPO-4bit
     model_type: multimodal
     served_model_name: ui-tars-72b
     context_length: 8192
     on_demand: true
   ```

2. Send a multimodal chat completion request:
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "ui-tars-72b",
       "messages": [{
         "role": "user",
         "content": [
           {"type": "image_url", "image_url": {"url": "data:image/png;base64,<valid_png>"}},
           {"type": "text", "text": "What do you see?"}
         ]
       }],
       "max_tokens": 100
     }'
   ```

3. Server returns HTTP 500:
   ```json
   {
     "detail": {
       "error": {
         "message": "Failed to generate multimodal response: Cannot use apply_chat_template because this processor does not have a chat template.",
         "type": "server_error"
       }
     }
   }
   ```

## Root cause

In `mlx_vlm/models/qwen2_vl/processing_qwen2_vl.py`, the `from_pretrained` classmethod:

```python
# Line 137-140: loads chat template into tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
load_chat_template(tokenizer, pretrained_model_name_or_path)

# Line 166-169: constructs processor WITHOUT passing chat_template
return cls(
    image_processor=image_processor,
    tokenizer=tokenizer,
)
```

The `__init__` accepts `chat_template` and forwards it to `ProcessorMixin.__init__` (line 51):
```python
super().__init__(image_processor, tokenizer, chat_template=chat_template)
```

But since `from_pretrained` never passes it, `chat_template` defaults to `None`.

### Why this isn't caught by `AutoProcessor`

When loaded via `transformers.AutoProcessor.from_pretrained()` (transformers 5.4.0), the **transformers-native** `Qwen2VLProcessor` is used, which correctly loads `chat_template.json` via its own `ProcessorMixin` machinery. The bug only manifests when `mlx_vlm.load()` is used, because `mlx_vlm` has its own `Qwen2VLProcessor` that shadows the transformers one and uses a manual `from_pretrained` that misses the template.

### Call chain that triggers the error

```
mlx-openai-server app/handler/mlx_vlm.py:_build_inference_context()
  -> app/models/mlx_vlm.py:create_input_prompt()
    -> processor.apply_chat_template()          # ProcessorMixin from transformers
      -> checks self.chat_template              # None!
        -> raises ValueError
```

Note: the `mlx_vlm.prompt_utils.get_chat_template()` function has fallback logic that checks `processor.tokenizer.chat_template`, but `mlx-openai-server`'s app layer calls `processor.apply_chat_template()` directly (via `ProcessorMixin`), bypassing that fallback.

## Fix

One-line change in `mlx_vlm/models/qwen2_vl/processing_qwen2_vl.py`:

```diff
         return cls(
             image_processor=image_processor,
             tokenizer=tokenizer,
+            chat_template=getattr(tokenizer, "chat_template", None),
         )
```

### Verification

Before fix:
```python
from mlx_vlm import load
model, processor = load('mlx-community/UI-TARS-72B-DPO-4bit', lazy=False)
print(processor.chat_template)  # None
processor.apply_chat_template(...)  # ValueError
```

After fix:
```python
from mlx_vlm import load
model, processor = load('mlx-community/UI-TARS-72B-DPO-4bit', lazy=False)
print(processor.chat_template[:40])  # '{% set image_count = namespace(value=0) ...'
processor.apply_chat_template(...)    # works
```

## Other potentially affected processors

Several other mlx_vlm processor `from_pretrained` methods follow the same pattern of calling `load_chat_template(tokenizer, ...)` without forwarding the result to `cls()`. A grep for `load_chat_template` followed by `return cls(` without `chat_template=` would identify all of them. At minimum, any processor that inherits from `ProcessorMixin` and defines `valid_kwargs = ["chat_template"]` should be checked.
