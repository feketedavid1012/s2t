"""On-prem / self-hosted Whisper via faster-whisper (CTranslate2).

Runs fully offline on CPU or GPU. This is the reference "on-prem" engine.

Supports:
- file transcription (the standard TranscriptionEngine interface),
- array transcription (`transcribe_array`) so the streaming layer can feed it
  live mic segments without touching disk,
- `initial_prompt` biasing, which nudges the decoder toward domain vocabulary
  (SKUs, XGS-PON, OLT/ONT names) and noticeably improves rare-term accuracy.

Requires:  pip install "s2t-bench[local]"   (faster-whisper)
"""
from __future__ import annotations

from typing import Any

from .base import TranscriptionEngine

# faster-whisper expects mono audio at this rate when given a raw array.
WHISPER_SAMPLE_RATE = 16000


class FasterWhisperEngine(TranscriptionEngine):
    name = "faster_whisper"

    def __init__(
        self,
        model: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        beam_size: int = 5,
        language: str | None = None,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, device=device, **kwargs)
        self.model_size = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self.initial_prompt = initial_prompt
        self.vad_filter = vad_filter
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def _run(self, audio: Any, vad_filter: bool | None = None) -> tuple[str, Any]:
        model = self._get_model()
        segments, info = model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            initial_prompt=self.initial_prompt,
            vad_filter=self.vad_filter if vad_filter is None else vad_filter,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, info

    def _transcribe(self, audio_path: str) -> tuple[str, dict[str, Any], str | None]:
        text, info = self._run(audio_path)
        raw = {"model": self.model_size, "device": self.device}
        return text, raw, getattr(info, "language", None)

    def transcribe_array(self, audio: "Any") -> tuple[str, str | None]:
        """Transcribe an in-memory float32 mono array at 16 kHz.

        Used by the streaming layer to transcribe live mic utterances. VAD
        filtering is disabled here because the streaming segmenter has already
        trimmed the audio to a single utterance.
        """
        text, info = self._run(audio, vad_filter=False)
        return text, getattr(info, "language", None)
