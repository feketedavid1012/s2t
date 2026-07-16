"""Gemini multimodal transcription engine.

Uses the Google GenAI SDK to transcribe audio with a Gemini model. Useful as a
second "Google-based" data point alongside Cloud STT, and handy when you want
prompt-steerable transcription (formatting, diarization hints, etc.).

Requires:  pip install "s2t-bench[google]"   (google-genai)
Auth:      GOOGLE_API_KEY  (AI Studio)  OR  Vertex AI env vars
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from .base import TranscriptionEngine

_DEFAULT_PROMPT = (
    "Transcribe the speech in this audio verbatim. "
    "Return only the transcript text with no commentary, labels, or timestamps."
)


class GeminiEngine(TranscriptionEngine):
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-flash-latest",
        prompt: str = _DEFAULT_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.model = model
        self.prompt = prompt
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai  # lazy import

            self._client = genai.Client()
        return self._client

    def _transcribe(self, audio_path: str) -> tuple[str, dict[str, Any], str | None]:
        from google.genai import types

        client = self._get_client()
        data = Path(audio_path).read_bytes()
        mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"

        response = client.models.generate_content(
            model=self.model,
            contents=[
                self.prompt,
                types.Part.from_bytes(data=data, mime_type=mime),
            ],
        )
        text = (response.text or "").strip()
        return text, {"model": self.model}, None
