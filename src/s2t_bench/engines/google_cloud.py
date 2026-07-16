"""Google Cloud Speech-to-Text v2 engine (Chirp / Chirp 2 models).

Requires:  pip install "s2t-bench[google]"   (google-cloud-speech)
Auth:      GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import TranscriptionEngine


class GoogleCloudSTTEngine(TranscriptionEngine):
    name = "google_cloud"

    def __init__(
        self,
        model: str = "chirp_2",
        language_codes: list[str] | None = None,
        location: str = "us-central1",
        project: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, location=location, **kwargs)
        self.model = model
        self.location = location
        self.language_codes = language_codes or ["en-US"]
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project:
            raise ValueError(
                "GoogleCloudSTTEngine needs a project id "
                "(pass project= or set GOOGLE_CLOUD_PROJECT)."
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            # Lazy import so the package loads without the SDK installed.
            from google.cloud.speech_v2 import SpeechClient
            from google.api_core.client_options import ClientOptions

            api_endpoint = f"{self.location}-speech.googleapis.com"
            self._client = SpeechClient(
                client_options=ClientOptions(api_endpoint=api_endpoint)
            )
        return self._client

    def _transcribe(self, audio_path: str) -> tuple[str, dict[str, Any], str | None]:
        from google.cloud.speech_v2.types import cloud_speech

        client = self._get_client()
        audio_bytes = Path(audio_path).read_bytes()

        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=self.language_codes,
            model=self.model,
        )
        request = cloud_speech.RecognizeRequest(
            recognizer=(
                f"projects/{self.project}/locations/{self.location}/recognizers/_"
            ),
            config=config,
            content=audio_bytes,
        )
        response = client.recognize(request=request)

        text = " ".join(
            r.alternatives[0].transcript
            for r in response.results
            if r.alternatives
        ).strip()
        language = self.language_codes[0] if self.language_codes else None
        return text, {"results": len(response.results)}, language
