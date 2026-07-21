import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import s2t_bench.engines.faster_whisper_local as fw  # noqa: E402
from s2t_bench.server import build_app  # noqa: E402


class StubEngine:
    """Scripted hypotheses; no Whisper, no audio hardware."""

    model_size = "base"

    def __init__(self) -> None:
        self.calls = 0

    def _get_model(self):
        return None

    def transcribe_words(self, audio):  # noqa: ARG002
        seq = [
            [("the", 0.0, 1.0)],
            [("the", 0.0, 1.0), ("olt", 1.0, 2.0)],
            [("the", 0.0, 1.0), ("olt", 1.0, 2.0), ("port", 2.0, 3.0)],
        ]
        out = seq[min(self.calls, len(seq) - 1)]
        self.calls += 1
        return out


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(fw, "FasterWhisperEngine", lambda **kw: StubEngine())
    app = build_app(mount_agent=False, interval_s=0.0)
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["sample_rate"] == 16000


def test_ui_and_worklet_served(client):
    idx = client.get("/ui/")
    assert idx.status_code == 200
    assert "recorder-worklet.js" in idx.text  # page references the worklet
    wk = client.get("/ui/recorder-worklet.js")
    assert wk.status_code == 200
    assert "registerProcessor" in wk.text
    # worklets must be served as JS or the browser refuses to load them
    assert "javascript" in wk.headers.get("content-type", "")


def test_root_redirects_to_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/ui/"


def test_websocket_streams_committed_and_partial(client):
    """Regression: the ws route must bind and stream.

    This previously failed silently because `from __future__ import annotations`
    stringized `ws: WebSocket`, which FastAPI could not resolve against module
    globals (fastapi is imported lazily inside build_app).
    """
    frame = (np.random.RandomState(0).randn(16000) * 0.2 * 32768).astype(np.int16).tobytes()
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(frame)
        first = ws.receive_json()
        assert first["committed"] == []  # nothing stable after one decode
        assert first["partial"] == "the"

        ws.send_bytes(frame)
        second = ws.receive_json()
        assert second["committed"] == ["the"]  # agreed across two decodes
        assert second["partial"] == "olt"


def test_stt_routes_attach_to_any_app():
    """The unification mechanism: STT routes must graft onto a base app that
    already has its own routes (stands in for ADK's get_fast_api_app result),
    so agent + transcription endpoints share one app and one Swagger."""
    from fastapi import FastAPI

    from s2t_bench.server import _add_stt_routes

    base = FastAPI()

    @base.post("/run")
    async def run():
        return {"ok": True}

    _add_stt_routes(base, StubEngine(), glossary=["XGS-PON"], interval_s=0.0)
    spec = TestClient(base).get("/openapi.json").json()["paths"]
    assert "/run" in spec  # pre-existing (agent) route survived
    assert "/transcribe" in spec and "/health" in spec  # STT routes added


def test_ws_emits_sentence_with_raw_and_corrected(monkeypatch):
    """The correction pipeline must surface both S2T (raw) and LLM (corrected)."""
    import s2t_bench.correction as corr
    from s2t_bench.correction import CorrectionResult

    class SentenceStub:
        model_size = "base"
        calls = 0
        def _get_model(self): return None
        def transcribe_words(self, audio):
            SentenceStub.calls += 1
            w = [("the", 0.0, 1.0), ("cin", 1.0, 2.0), ("broken.", 2.0, 3.0)]
            return w  # same both decodes -> commits, ends with '.'

    monkeypatch.setattr(fw, "FasterWhisperEngine", lambda **kw: SentenceStub())
    monkeypatch.setattr(
        corr, "review_and_correct",
        lambda text, glossary=None, **kw: CorrectionResult("corrected", text.replace("cin", "SIM"), True),
    )
    app = build_app(mount_agent=False, interval_s=0.0)
    frame = (np.random.RandomState(1).randn(16000) * 0.2 * 32768).astype(np.int16).tobytes()
    with TestClient(app).websocket_connect("/ws/transcribe?correct=true") as ws:
        sentence = None
        for _ in range(6):
            ws.send_bytes(frame)
            m = ws.receive_json()
            if m.get("type") == "sentence":
                sentence = m
                break
        assert sentence is not None
        assert sentence["raw"] == "the cin broken."          # what Whisper heard
        assert sentence["corrected"] == "the SIM broken."    # what Gemini returned
        assert sentence["changed"] is True


def test_ws_signals_lagging_when_flooded(monkeypatch):
    """Backpressure: if far more audio arrives than can be decoded, the server
    drops stale frames and flags lagging rather than falling behind forever."""
    import time as _t

    class SlowStub:
        model_size = "base"
        def _get_model(self): return None
        def transcribe_words(self, audio):
            _t.sleep(0.05)  # simulate a slow decode
            return [("x", 0.0, 1.0)]

    monkeypatch.setattr(fw, "FasterWhisperEngine", lambda **kw: SlowStub())
    # tiny buffer so max_chunks is small and easily exceeded
    app = build_app(mount_agent=False, interval_s=0.01)
    # shrink the transcriber buffer via monkeypatching default isn't trivial;
    # instead flood many 100ms frames in one go.
    frame = (np.random.RandomState(2).randn(1600) * 0.2 * 32768).astype(np.int16).tobytes()
    with TestClient(app).websocket_connect("/ws/transcribe") as ws:
        for _ in range(400):        # ~40s of audio slammed in at once
            ws.send_bytes(frame)
        saw_lag = False
        for _ in range(10):
            m = ws.receive_json()
            if m.get("type") == "live" and m.get("lagging"):
                saw_lag = True
                break
        assert saw_lag  # server reported it couldn't keep up
