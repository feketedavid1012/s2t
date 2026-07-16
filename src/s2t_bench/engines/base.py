from __future__ import annotations

import abc
import contextlib
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    """Normalized output returned by every engine."""

    text: str
    engine: str
    audio_seconds: float
    latency_seconds: float
    language: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio duration. Lower is faster."""
        if not self.audio_seconds:
            return float("nan")
        return self.latency_seconds / self.audio_seconds


def audio_duration_seconds(audio_path: str | Path) -> float:
    """Best-effort audio duration.

    Uses the stdlib `wave` module for .wav (no deps). Falls back to `soundfile`
    for other formats if it is installed; otherwise returns 0.0 so RTF is simply
    reported as NaN rather than crashing the run.
    """
    path = Path(audio_path)
    if path.suffix.lower() == ".wav":
        with contextlib.suppress(Exception):
            with contextlib.closing(wave.open(str(path), "rb")) as wf:
                return wf.getnframes() / float(wf.getframerate())
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


class TranscriptionEngine(abc.ABC):
    """Abstract base for all speech-to-text engines."""

    #: Stable identifier used in configs, CLI flags and result tables.
    name: str = "base"

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abc.abstractmethod
    def _transcribe(self, audio_path: str) -> tuple[str, dict[str, Any], str | None]:
        """Do the actual transcription.

        Returns (text, raw_provider_payload, detected_language).
        Timing and duration are handled by `transcribe`.
        """

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """Transcribe one audio file and wrap the output with timing metadata."""
        audio_path = str(audio_path)
        duration = audio_duration_seconds(audio_path)
        start = time.perf_counter()
        text, raw, language = self._transcribe(audio_path)
        latency = time.perf_counter() - start
        return TranscriptionResult(
            text=text,
            engine=self.name,
            audio_seconds=duration,
            latency_seconds=latency,
            language=language,
            raw=raw,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
