# /// script
# requires-python = ">=3.12"
# dependencies = ["flask==3.1.3", "laya==0.3.3"]
# ///
"""
Laya decision server for Super Puppy.

A dedicated, persistent typed-decision service — the `laya` backend. Laya
(convaiinnovations) is the open-weight implementation of the "Jev" System-One
model class: given a state (text/JSON) plus typed questions, it returns
calibrated probabilities / typed answers in one non-autoregressive pass. No
text generation, so no parsing errors and no hallucination.

It lives in its own process (not in-process in the MCP/profile servers) so its
heavy torch/transformers dependency stays isolated, and it stays resident so
inference is ~15ms rather than paying the ~24s model load per call. Bound to
localhost only and never added to `tailscale serve` — the trust boundary is the
same as ds4/Ollama/MLX (no bearer auth here; only the served 8100/8101 carry
the token). The MCP boundary validates the question schema before dispatch.

Endpoints:
  POST /decide     {model, state, questions} -> laya predict() result
  GET  /v1/models  ready served models (discovery parity with other backends)
  GET  /health     liveness + load state

Started by bin/start-local-models (resident only in server/offline mode).
"""

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request

from lib.models import (
    LAYA_ENGLISH_REPO,
    LAYA_MULTILINGUAL_REPO,
    LAYA_SERVED_MODELS,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"level":"%(levelname)s","event":"%(message)s","module":"laya-server"}',
)
log = logging.getLogger("laya-server")

app = Flask(__name__)

# Serialize model access: torch inference is not reentrant-safe across
# threads, and predict() is fast (~15ms) so serializing costs little — same
# posture ds4 takes (it serializes requests too).
_lock = threading.Lock()
_agents: dict[str, object] = {}          # served-name -> loaded laya agent
_loading: set[str] = set()               # served-names currently loading
_load_errors: dict[str, str] = {}        # served-name -> last load error

_VALID_TYPES = {"choice", "score", "noul"}


def _load_agent(served_name: str) -> None:
    """Load one laya agent into the resident cache. Fail loud but keep the
    service up so /health can report the error rather than the port dying."""
    import laya  # deferred: keeps torch import cost off module load
    with _lock:
        if served_name in _agents or served_name in _loading:
            return
        _loading.add(served_name)
    try:
        log.info("loading %s", served_name)
        agent = laya.load(served_name)
        with _lock:
            _agents[served_name] = agent
            _load_errors.pop(served_name, None)
        log.info("loaded %s", served_name)
    except Exception as e:  # noqa: BLE001 — record and surface, don't crash the service
        with _lock:
            _load_errors[served_name] = f"{type(e).__name__}: {e}"
        log.error("load failed %s: %s", served_name, e)
    finally:
        with _lock:
            _loading.discard(served_name)


def _validate_questions(questions):
    """Return None if the question schema is valid, else an error string.

    Fails loud (no silent coercion): every question needs a valid type, and
    choice/score need their criteria shape.
    """
    if not isinstance(questions, dict) or not questions:
        return "questions must be a non-empty object"
    for name, q in questions.items():
        if not isinstance(q, dict):
            return f"question {name!r} must be an object"
        qtype = q.get("type")
        if qtype not in _VALID_TYPES:
            return f"question {name!r} has invalid type {qtype!r} (expected one of {sorted(_VALID_TYPES)})"
        if qtype == "choice" and not isinstance(q.get("criteria"), dict):
            return f"choice question {name!r} needs a 'criteria' object (option -> description)"
        if qtype == "score" and not isinstance(q.get("criteria"), list):
            return f"score question {name!r} needs a 'criteria' list (ordinal labels)"
    return None


@app.route("/health")
def health():
    with _lock:
        return jsonify({
            "status": "ok",
            "ready": sorted(_agents),
            "loading": sorted(_loading),
            "errors": dict(_load_errors),
        })


@app.route("/v1/models")
def v1_models():
    """Only models actually loaded and ready — discovery must not advertise a
    model that would 503 on first use during the ~24s cold load."""
    with _lock:
        ready = sorted(_agents)
    return jsonify({"object": "list",
                    "data": [{"id": m, "object": "model", "owned_by": "laya"} for m in ready]})


@app.route("/decide", methods=["POST"])
def decide():
    body = request.get_json(silent=True) or {}
    model = body.get("model") or LAYA_MULTILINGUAL_REPO
    state = body.get("state")
    questions = body.get("questions")

    if model not in LAYA_SERVED_MODELS:
        return jsonify({"error": f"unknown model {model!r}; served: {list(LAYA_SERVED_MODELS)}"}), 400
    if not isinstance(state, dict) or not state:
        return jsonify({"error": "state must be a non-empty object of text fields"}), 400
    schema_err = _validate_questions(questions)
    if schema_err:
        return jsonify({"error": schema_err}), 400

    with _lock:
        agent = _agents.get(model)
        loading = model in _loading
        err = _load_errors.get(model)
    if agent is None:
        if err:
            return jsonify({"error": f"model {model!r} failed to load: {err}"}), 503
        # English is lazy-loaded on first request; kick it off.
        if not loading:
            threading.Thread(target=_load_agent, args=(model,), daemon=True).start()
        return jsonify({"error": f"model {model!r} is still loading; retry shortly"}), 503

    with _lock:  # serialize inference
        try:
            result = agent.predict(state, questions)
        except Exception as e:  # noqa: BLE001
            log.error("predict failed for %s: %s", model, e)
            return jsonify({"error": f"predict failed: {type(e).__name__}: {e}"}), 500
    return jsonify(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--model", default=LAYA_MULTILINGUAL_REPO,
                    help="served model to load resident at startup")
    args = ap.parse_args()

    # Load the default (multilingual) resident in the background so the port
    # binds immediately (/health responds) while the ~24s load proceeds;
    # /decide 503s until it's ready. English (LAYA_ENGLISH_REPO) loads lazily
    # on first request for it.
    threading.Thread(target=_load_agent, args=(args.model,), daemon=True).start()
    log.info("laya-server starting on %s:%s (loading %s)", args.host, args.port, args.model)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
