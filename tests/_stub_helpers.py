"""Scoped sys.modules stubbing for the unit suites.

The MCP and profile servers import heavy optional dependencies (mcp, httpx,
starlette, torch, sentence_transformers) at module level. The unit suites
don't need the real ones, so they install MagicMocks under those names before
executing the server module.

Installing them *permanently* couples every test module to the ones collected
after it, in two directions that both fail silently:

  * A leaked mock satisfies a later module's `import X`, so a guard that means
    "only run if the real X is installed" passes against the mock. That is how
    the smoke suite ran fake TTS inside the release gate.
  * A leaked mock poisons any real library that imports from the stubbed
    package. `huggingface_hub` does `from httpx import HTTPError` and then
    subclasses it, which raises `TypeError: metaclass conflict` when httpx is
    a MagicMock.

So stub for the duration of the import and put sys.modules back afterwards.
Anything a test needs mocked at *call* time should patch locally
(`patch.dict(sys.modules, ...)`), where the scope is visible at the call site.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def stubbed_modules(stubs: dict[str, object]) -> Iterator[None]:
    """Install `stubs` into sys.modules, then remove exactly what we added.

    Names already present are left alone on both entry and exit — a real
    dependency that happens to be installed keeps priority, and we never
    evict a module some other importer is relying on.
    """
    added: list[str] = []
    for name, stub in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = stub
            added.append(name)
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)
