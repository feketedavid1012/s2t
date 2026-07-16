"""OpenAI Whisper (hosted API) engine.

Requires:  pip install "s2t-bench[whisper]"   (openai)
Auth:      OPENAI_API_KEY
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TranscriptionEngine


class WhisperAPIEngine(TranscriptionEngine):
    name = "whisper_api"

    def __init__(self, model: str = "whisper-1", **kwargs: Any) -> None:
        super().__init__(model=model, **kwargs)
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            self._client = OpenAI()
        return self._client

    def _transcribe(self, audio_path: str) -> tuple[str, dict[str, Any], str | None]:
        client = self._get_client()
        with Path(audio_path).open("rb") as fh:
            resp = client.audio.transcriptions.create(model=self.model, file=fh)
        text = (getattr(resp, "text", "") or "").strip()
        language = getattr(resp, "language", None)
        return text, {"model": self.model}, language
