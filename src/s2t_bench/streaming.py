"""On-the-fly streaming transcription for the on-prem Whisper engine.

faster-whisper is a batch transcriber, so "live" transcription works by:
  mic -> 16 kHz mono frames -> VAD segmenter -> finalized utterance -> Whisper

The engineer talks continuously; the VAD (voice-activity detector) cuts an
utterance whenever they pause, and that utterance is transcribed immediately.
Optionally each finalized utterance is passed through the Gemini domain
corrector before being emitted.

`VADSegmenter` takes an injectable `is_speech` callable so its boundary logic is
testable without a mic or webrtcvad. The default uses webrtcvad when available
and automatically falls back to a pure-NumPy energy VAD otherwise.

Requires:  pip install "s2t-bench[stream]"   (sounddevice, numpy; webrtcvad optional)
"""
from __future__ import annotations

import queue
import warnings
from dataclasses import dataclass
from typing import Callable, Iterator

SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad accepts 10, 20, or 30 ms frames


@dataclass
class StreamChunk:
    text: str
    corrected: bool = False
    original: str | None = None  # set when correction changed the text


class EnergyVAD:
    """Dependency-light voice-activity detector (RMS energy vs adaptive floor).

    Pure NumPy, works on any Python version. Less robust in noise than
    webrtcvad, but has no legacy `pkg_resources`/setuptools dependency, so it is
    the automatic fallback when webrtcvad can't be imported.
    """

    def __init__(
        self,
        threshold: float | None = None,
        floor_factor: float = 3.0,
        min_threshold: float = 150.0,
    ) -> None:
        self.threshold = threshold  # fixed threshold; if None, adapt to noise
        self.floor_factor = floor_factor
        self.min_threshold = min_threshold
        self._noise: float | None = None

    def __call__(self, frame: bytes) -> bool:
        import numpy as np

        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(samples * samples)))
        if self.threshold is not None:
            return rms > self.threshold
        if self._noise is None:
            self._noise = rms
        thresh = max(self._noise * self.floor_factor, self.min_threshold)
        speaking = rms > thresh
        if not speaking:  # update the noise floor only during silence
            self._noise = 0.95 * self._noise + 0.05 * rms
        return speaking


def default_vad(aggressiveness: int = 2) -> Callable[[bytes], bool]:
    """Return webrtcvad if importable, otherwise fall back to EnergyVAD."""
    try:
        import webrtcvad  # lazy import; fails without pkg_resources/setuptools

        vad = webrtcvad.Vad(aggressiveness)
        return lambda frame: vad.is_speech(frame, SAMPLE_RATE)
    except Exception as exc:  # ImportError incl. missing pkg_resources, or init err
        warnings.warn(
            f"webrtcvad unavailable ({exc}); using energy-based VAD. "
            "For better noise robustness, install setuptools "
            "(provides pkg_resources): pip install setuptools",
            RuntimeWarning,
            stacklevel=2,
        )
        return EnergyVAD()


class VADSegmenter:
    """Frame-driven state machine that yields complete utterances.

    Push fixed-size 16-bit PCM frames via `push`; it returns the utterance's
    PCM bytes once a pause (trailing silence) or the max length is reached,
    otherwise None. Call `flush` at end-of-stream to emit any buffered speech.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = FRAME_MS,
        silence_ms: int = 600,
        min_utterance_ms: int = 200,
        max_utterance_ms: int = 30000,
        is_speech: Callable[[bytes], bool] | None = None,
        vad_aggressiveness: int = 2,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2  # int16
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_frames = max(1, min_utterance_ms // frame_ms)
        self.max_frames = max(1, max_utterance_ms // frame_ms)
        self._is_speech = is_speech
        self._vad_aggressiveness = vad_aggressiveness
        self._buf: list[bytes] = []
        self._trailing_silence = 0
        self._triggered = False

    def _speech(self, frame: bytes) -> bool:
        if self._is_speech is None:
            self._is_speech = default_vad(self._vad_aggressiveness)
        return self._is_speech(frame)

    def push(self, frame: bytes) -> bytes | None:
        speech = self._speech(frame)
        if not self._triggered:
            if speech:
                self._triggered = True
                self._buf = [frame]
                self._trailing_silence = 0
            return None

        self._buf.append(frame)
        self._trailing_silence = 0 if speech else self._trailing_silence + 1

        if self._trailing_silence >= self.silence_frames or len(self._buf) >= self.max_frames:
            return self._finalize()
        return None

    def _finalize(self) -> bytes | None:
        frames = self._buf
        self._buf = []
        self._triggered = False
        self._trailing_silence = 0
        if len(frames) < self.min_frames:
            return None
        return b"".join(frames)

    def flush(self) -> bytes | None:
        if self._triggered and self._buf:
            return self._finalize()
        return None


def _pcm_to_float32(pcm: bytes):
    import numpy as np  # lazy import

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _iter_frames_from_bytes(data: bytes, frame_bytes: int) -> Iterator[bytes]:
    for i in range(0, len(data) - frame_bytes + 1, frame_bytes):
        yield data[i : i + frame_bytes]


def _transcribe_utterance(engine, pcm: bytes, correct, glossary) -> StreamChunk | None:
    audio = _pcm_to_float32(pcm)
    text, _ = engine.transcribe_array(audio)
    text = text.strip()
    if not text:
        return None
    if correct:
        from .correction import review_and_correct

        result = review_and_correct(text, glossary=glossary)
        if result.status == "corrected" and result.text:
            return StreamChunk(text=result.text, corrected=True, original=text)
    return StreamChunk(text=text)


def stream_microphone(
    engine,
    correct: bool = False,
    glossary: list[str] | None = None,
    silence_ms: int = 600,
    device: int | None = None,
) -> Iterator[StreamChunk]:
    """Yield transcribed utterances from the microphone as the user speaks.

    `engine` must be a FasterWhisperEngine (uses `transcribe_array`). Set
    `correct=True` to run each finalized utterance through Gemini domain review.
    """
    import sounddevice as sd  # lazy import

    segmenter = VADSegmenter(silence_ms=silence_ms)
    audio_q: "queue.Queue[bytes]" = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        audio_q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=segmenter.frame_samples,
        dtype="int16",
        channels=1,
        callback=callback,
        device=device,
    ):
        buffer = b""
        while True:
            buffer += audio_q.get()
            while len(buffer) >= segmenter.frame_bytes:
                frame, buffer = buffer[: segmenter.frame_bytes], buffer[segmenter.frame_bytes :]
                utterance = segmenter.push(frame)
                if utterance:
                    chunk = _transcribe_utterance(engine, utterance, correct, glossary)
                    if chunk:
                        yield chunk


def stream_wav_file(
    engine,
    wav_path: str,
    correct: bool = False,
    glossary: list[str] | None = None,
    silence_ms: int = 600,
) -> Iterator[StreamChunk]:
    """Simulate live streaming from a 16 kHz mono WAV file (no mic needed).

    Handy for demos, tests, and CI: it drives the exact same VAD + transcription
    path as the microphone, just fed from a file.
    """
    import wave

    with wave.open(wav_path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1:
            raise ValueError(
                "stream_wav_file expects 16 kHz mono WAV; "
                "convert with: ffmpeg -i in.wav -ar 16000 -ac 1 out.wav"
            )
        pcm = wf.readframes(wf.getnframes())

    segmenter = VADSegmenter(silence_ms=silence_ms)
    for frame in _iter_frames_from_bytes(pcm, segmenter.frame_bytes):
        utterance = segmenter.push(frame)
        if utterance:
            chunk = _transcribe_utterance(engine, utterance, correct, glossary)
            if chunk:
                yield chunk
    tail = segmenter.flush()
    if tail:
        chunk = _transcribe_utterance(engine, tail, correct, glossary)
        if chunk:
            yield chunk
