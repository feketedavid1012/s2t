"""FastAPI server exposing on-prem transcription, optionally unified with ADK.

Three surfaces in one process:
- REST + realtime WebSocket for on-prem Whisper (this module)
- the ADK agent endpoints (/run, /run_sse, sessions) via get_fast_api_app
- Swagger UI at /docs (FastAPI auto-generates it)

Why not ADK's built-in streaming for the audio? ADK bidi-streaming requires a
Gemini Live API model and streams audio to Google's cloud, bypassing the local
Whisper — wrong for an on-prem deployment. So the realtime transcription is our
own WebSocket, and ADK contributes the conversational agent alongside it.

NOTE: this module deliberately does NOT use `from __future__ import annotations`.
FastAPI resolves endpoint annotations via get_type_hints() against the *module*
globals; fastapi is imported lazily inside build_app(), so stringized annotations
like "WebSocket" would be unresolvable and routes would silently fail to bind.

Requires:  pip install "s2t-bench[server,local]"   (fastapi, uvicorn)
With ADK:  pip install "s2t-bench[server,local,agent]"
Run:       s2t-bench serve --model small --compute-type int8
           s2t-bench serve --with-agent          # unified with the ADK agent
"""
import asyncio
import time
from pathlib import Path

SAMPLE_RATE = 16000
ADK_APPS_DIR = str(Path(__file__).resolve().parent / "adk_apps")
WEB_DIR = str(Path(__file__).resolve().parent / "web")


def _mount_ui(app):
    """Serve the static front-end (index.html + audio worklet) at /ui."""
    from fastapi.staticfiles import StaticFiles

    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")


def _build_engine(model, device, compute_type, extra_terms):
    from .domain import build_initial_prompt, merge_glossary
    from .engines.faster_whisper_local import FasterWhisperEngine

    glossary = merge_glossary(extra_terms)
    engine = FasterWhisperEngine(
        model=model,
        device=device,
        compute_type=compute_type,
        initial_prompt=build_initial_prompt(glossary),
    )
    return engine, glossary


def _add_stt_routes(app, engine, glossary, interval_s):
    """Attach transcription routes to any FastAPI app (ours or ADK's)."""
    import threading

    import numpy as np
    from fastapi import UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    from .realtime import RealtimeTranscriber

    # One shared model instance -> serialize decodes. CTranslate2 models are not
    # safe for concurrent transcribe() calls from multiple threads.
    decode_lock = threading.Lock()

    @app.on_event("startup")
    async def _warmup():
        # Load weights at boot, not on the first user's request.
        await asyncio.get_running_loop().run_in_executor(None, engine._get_model)

    @app.get("/health", tags=["stt"])
    async def health():
        return {"status": "ok", "model": engine.model_size, "sample_rate": SAMPLE_RATE}

    @app.post("/transcribe", tags=["stt"])
    async def transcribe_file(file: UploadFile, correct: bool = False):
        """Transcribe an uploaded audio file; optionally Gemini-correct it."""
        import tempfile

        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        loop = asyncio.get_running_loop()

        def _locked_transcribe():
            with decode_lock:
                return engine.transcribe(tmp_path)

        result = await loop.run_in_executor(None, _locked_transcribe)
        payload = {
            "text": result.text,
            "audio_seconds": round(result.audio_seconds, 2),
            "rtf": round(result.rtf, 3),
        }
        if correct:
            from .correction import review_and_correct

            review = await loop.run_in_executor(
                None, lambda: review_and_correct(result.text, glossary=glossary)
            )
            payload["corrected"] = review.status == "corrected"
            payload["text"] = review.text if review.status == "corrected" else result.text
        return payload

    @app.websocket("/ws/transcribe")
    async def ws_transcribe(ws: WebSocket):
        """Realtime transcription with optional per-sentence Gemini correction.

        Messages sent to the client:
          {"type":"live","committed":[...],"partial":"..."}   every tick
          {"type":"sentence","raw":"...","corrected":"...","changed":bool}
                                                              at each sentence
        Enable correction with the query param ?correct=true (needs a Gemini key).
        """
        await ws.accept()
        correct = ws.query_params.get("correct", "false").lower() in ("1", "true", "yes")
        transcriber = RealtimeTranscriber(engine)
        loop = asyncio.get_running_loop()
        last = time.monotonic()
        sentence_words: list[str] = []
        last_commit = time.monotonic()
        correct_disabled = False  # set if the key is missing, to avoid spamming

        from .realtime import _apply_backpressure

        chunk_s = 0.1
        max_chunks = int(transcriber.max_buffer_s / chunk_s)

        # Warm the model so its one-time load isn't counted as decode lag.
        await loop.run_in_executor(None, engine._get_model)

        # Reader task: pull frames off the socket as fast as they arrive into a
        # queue, so a slow decode can't cause them to pile up unread. The decode
        # loop drains the queue and drops stale audio when it can't keep up.
        audio_q: "asyncio.Queue[bytes]" = asyncio.Queue()
        disconnected = asyncio.Event()

        async def _reader():
            try:
                while True:
                    audio_q.put_nowait(await ws.receive_bytes())
            except WebSocketDisconnect:
                disconnected.set()
            except Exception:
                disconnected.set()

        reader = asyncio.create_task(_reader())

        def _drain():
            chunks = []
            while True:
                try:
                    chunks.append(audio_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            return chunks

        def _correct(text):
            from .correction import review_and_correct

            review = review_and_correct(text, glossary=glossary)
            if review.status == "corrected" and review.text:
                return review.text, True
            return text, False

        async def _flush_sentence():
            nonlocal sentence_words, correct_disabled
            raw = " ".join(sentence_words).strip()
            sentence_words = []
            if not raw:
                return
            corrected, changed = raw, False
            if correct and not correct_disabled:
                try:
                    corrected, changed = await loop.run_in_executor(None, _correct, raw)
                except Exception as exc:
                    correct_disabled = True
                    await ws.send_json(
                        {"type": "sentence", "raw": raw, "corrected": None,
                         "error": f"{type(exc).__name__}: {exc}"}
                    )
                    return
            await ws.send_json(
                {"type": "sentence", "raw": raw, "corrected": corrected, "changed": changed}
            )

        try:
            while not disconnected.is_set():
                await asyncio.sleep(min(interval_s, 0.1))
                chunks, dropped = _apply_backpressure(_drain(), max_chunks)
                for data in chunks:
                    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    transcriber.insert_audio(pcm)

                now = time.monotonic()
                if sentence_words and now - last_commit > 1.5:
                    await _flush_sentence()

                if now - last < interval_s:
                    continue
                last = now

                def _locked_process():
                    with decode_lock:
                        return transcriber.process()

                t0 = time.monotonic()
                update = await loop.run_in_executor(None, _locked_process)
                lagging = dropped > 0 or (
                    interval_s > 0 and (time.monotonic() - t0) > interval_s * 1.5
                )

                if update.committed:
                    last_commit = now
                    sentence_words.extend(w.text for w in update.committed)
                    joined = " ".join(sentence_words).rstrip()
                    if joined[-1:] in ".?!":
                        await _flush_sentence()
                if update.committed or update.partial or lagging:
                    await ws.send_json(
                        {
                            "type": "live",
                            "committed": [w.text for w in update.committed],
                            "partial": update.partial_text,
                            "lagging": lagging,
                        }
                    )
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await ws.send_json({"error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            reader.cancel()


    return app


def build_app(
    model="base",
    device="auto",
    compute_type="default",
    interval_s=1.0,
    extra_terms="",
    with_adk=False,
    mount_agent=True,
):
    """Construct the FastAPI app.

    with_adk=True  -> ADK's get_fast_api_app is the base; STT routes are added to
                      it, so /docs shows both agent and transcription endpoints and
                      ADK's dev UI is served at /. Requires google-adk.
    with_adk=False -> our own FastAPI app; ADK (if installed) is best-effort
                      mounted as a sub-app at /agent, and the demo UI is at /.
    """
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    engine, glossary = _build_engine(model, device, compute_type, extra_terms)

    if with_adk:
        from google.adk.cli.fast_api import get_fast_api_app

        app = get_fast_api_app(agents_dir=ADK_APPS_DIR, web=True)
        app.title = "s2t-bench + ADK"
        _add_stt_routes(app, engine, glossary, interval_s)
        _mount_ui(app)
        return app

    app = FastAPI(title="s2t-bench", version="0.1.0")
    _add_stt_routes(app, engine, glossary, interval_s)
    _mount_ui(app)

    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse(url="/ui/")

    if mount_agent:
        try:
            from google.adk.cli.fast_api import get_fast_api_app

            app.mount("/agent", get_fast_api_app(agents_dir=ADK_APPS_DIR, web=True))
        except Exception:
            pass  # ADK not installed or API changed; realtime server still runs

    return app


