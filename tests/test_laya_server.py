"""Unit tests for app/laya-server.py — the typed-decision service.

The real laya model is never loaded here (24s + torch); the agent cache is
populated with a fake so the endpoint logic, schema validation, and
load-state handling are tested in isolation.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parent.parent / "app" / "laya-server.py"
spec = importlib.util.spec_from_file_location("laya_server", str(_path))
ls = importlib.util.module_from_spec(spec)
sys.modules["laya_server"] = ls
spec.loader.exec_module(ls)

from lib.models import LAYA_MULTILINGUAL_REPO, LAYA_ENGLISH_REPO


class _FakeAgent:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {name: {"type": q["type"]} for name, q in questions.items()},
                "usage": {"input_tokens": 1, "output_tokens": 0}}


@pytest.fixture(autouse=True)
def _reset():
    ls._agents.clear()
    ls._loading.clear()
    ls._load_errors.clear()
    yield
    ls._agents.clear()
    ls._loading.clear()
    ls._load_errors.clear()


@pytest.fixture()
def client():
    ls.app.config["TESTING"] = True
    with ls.app.test_client() as c:
        yield c


def _good_body(**over):
    body = {
        "model": LAYA_MULTILINGUAL_REPO,
        "state": {"body": "server is on fire"},
        "questions": {"urgency": {"type": "score", "instructions": "?", "criteria": ["low", "high"]}},
    }
    body.update(over)
    return body


class TestDecide:
    def test_dispatches_to_loaded_agent(self, client):
        ls._agents[LAYA_MULTILINGUAL_REPO] = _FakeAgent()
        r = client.post("/decide", json=_good_body())
        assert r.status_code == 200
        assert "urgency" in r.get_json()["answers"]

    def test_unknown_model_400(self, client):
        ls._agents[LAYA_MULTILINGUAL_REPO] = _FakeAgent()
        r = client.post("/decide", json=_good_body(model="some/other-model"))
        assert r.status_code == 400
        assert "unknown model" in r.get_json()["error"]

    def test_empty_state_400(self, client):
        ls._agents[LAYA_MULTILINGUAL_REPO] = _FakeAgent()
        assert client.post("/decide", json=_good_body(state={})).status_code == 400

    @pytest.mark.parametrize("questions", [
        {},                                                        # empty
        {"q": {"type": "bogus"}},                                  # invalid type
        {"q": {"type": "choice"}},                                 # choice missing criteria dict
        {"q": {"type": "choice", "criteria": ["a", "b"]}},         # choice criteria wrong shape
        {"q": {"type": "score"}},                                  # score missing criteria list
        {"q": {"type": "score", "criteria": {"a": "b"}}},          # score criteria wrong shape
    ])
    def test_bad_question_schema_400(self, client, questions):
        ls._agents[LAYA_MULTILINGUAL_REPO] = _FakeAgent()
        r = client.post("/decide", json=_good_body(questions=questions))
        assert r.status_code == 400

    def test_noul_needs_no_criteria(self, client):
        ls._agents[LAYA_MULTILINGUAL_REPO] = _FakeAgent()
        r = client.post("/decide", json=_good_body(
            questions={"cancel": {"type": "noul", "instructions": "?"}}))
        assert r.status_code == 200

    def test_503_while_loading_and_kicks_lazy_load(self, client, monkeypatch):
        started = []
        monkeypatch.setattr(ls.threading, "Thread",
                            lambda target, args, daemon: type("T", (), {"start": lambda s: started.append(args)})())
        r = client.post("/decide", json=_good_body(model=LAYA_ENGLISH_REPO))
        assert r.status_code == 503
        assert "still loading" in r.get_json()["error"]
        assert started and started[0][0] == LAYA_ENGLISH_REPO  # lazy load kicked

    def test_503_reports_load_error(self, client):
        ls._load_errors[LAYA_MULTILINGUAL_REPO] = "RuntimeError: boom"
        r = client.post("/decide", json=_good_body())
        assert r.status_code == 503
        assert "failed to load" in r.get_json()["error"]


class TestDiscovery:
    def test_v1_models_lists_only_ready(self, client):
        ls._agents[LAYA_MULTILINGUAL_REPO] = _FakeAgent()
        ls._loading.add(LAYA_ENGLISH_REPO)
        ids = [m["id"] for m in client.get("/v1/models").get_json()["data"]]
        assert ids == [LAYA_MULTILINGUAL_REPO]  # loading model not advertised

    def test_health_reports_state(self, client):
        ls._agents[LAYA_MULTILINGUAL_REPO] = _FakeAgent()
        ls._loading.add(LAYA_ENGLISH_REPO)
        h = client.get("/health").get_json()
        assert h["status"] == "ok"
        assert h["ready"] == [LAYA_MULTILINGUAL_REPO]
        assert h["loading"] == [LAYA_ENGLISH_REPO]
