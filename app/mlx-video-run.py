#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "mlx-video @ git+https://github.com/Blaizzy/mlx-video.git@9ab4826d20e39286af13a26615c33b403d48be72",
#   "mlx-video-with-audio==0.1.33",
# ]
# ///
"""Runner shim so the profile server can invoke mlx-video submodules without
having those heavy deps in its own uv environment.  Usage:

    uv run --script app/mlx-video-run.py <module> [args...]

e.g. `uv run --script app/mlx-video-run.py mlx_video.wan_2.generate --prompt ...`
"""
import runpy
import sys

if len(sys.argv) < 2:
    sys.stderr.write("usage: mlx-video-run.py <module> [args...]\n")
    sys.exit(2)

module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
