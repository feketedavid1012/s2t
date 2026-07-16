"""Real-time ("as you speak") transcription via LocalAgreement prefix commitment.

Why this exists
---------------
`streaming.py` waits for a pause before transcribing, so text arrives one whole
utterance at a time. This module instead keeps a rolling audio buffer and
re-transcribes it every ~1s. Whisper revises its guess as more audio arrives, so
a word is only *committed* once two consecutive decodes agree on it — the
LocalAgreement-2 policy from Macháček et al.'s whisper-streaming. Committed text
is stable and never rewritten; the uncommitted tail is shown as a live "partial"
that may still change. Typical latency is ~2-3s.

Tradeoff vs. utterance mode: this decodes the buffer repeatedly, so it costs
several times more compute for the same audio. Use a small model (and int8 on
CPU, or a GPU) or the transcriber will fall behind real time.

The commit policy (`LocalAgreement`) is pure and unit-tested; the audio plumbing
is kept thin around it.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

SAMPLE_RATE = 16000


@dataclass
class TimedWord:
    text: str
    start: float
    end: float


@dataclass
class RealtimeUpdate:
    """One refresh of the live view."""

    committed: list[TimedWord] = field(default_factory=list)  # newly stable words
    partial: list[TimedWord] = field(default_factory=list)  # may still change

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self.committed)

    @property
    def partial_text(self) -> str:
        return " ".join(w.text for w in self.partial)


def _norm(word: str) -> str:
    """Compare words ignoring case/punctuation so trivial diffs don't block commit."""
    return re.sub(r"[^\w]", "", word).lower()


class LocalAgreement:
    """Commit the longest prefix that two consecutive hypotheses agree on.

    Pure state machine: feed it each new hypothesis (a full word list for the
    current buffer) and it returns the words newly promoted to stable, plus the
    still-unstable tail. Committed words are never revoked.
    """

    def __init__(self) -> None:
        self._prev: list[TimedWord] = []
        self._n_committed = 0

    @property
    def n_committed(self) -> int:
        return self._n_committed

    def insert(self, words: list[TimedWord]) -> tuple[list[TimedWord], list[TimedWord]]:
        """Returns (newly_committed, partial)."""
        agreed = 0
        for a, b in zip(self._prev, words):
            if _norm(a.text) == _norm(b.text):
                agreed += 1
            else:
                break

        newly: list[TimedWord] = []
        if agreed > self._n_committed:
            newly = words[self._n_committed : agreed]
            self._n_committed = agreed

        self._prev = words
        partial = words[self._n_committed :]
        return newly, partial

    def committed_end_time(self) -> float | None:
        """End timestamp of the last committed word (buffer-relative)."""
        if self._n_committed == 0 or not self._prev:
            return None
        idx = min(self._n_committed, len(self._prev)) - 1
        return self._prev[idx].end

    def reset(self) -> None:
        """Forget state after the audio buffer is trimmed."""
        self._prev = []
        self._n_committed = 0


class RealtimeTranscriber:
    """Rolling-buffer transcriber that emits stable prefixes as you speak."""

    def __init__(
        self,
        engine,
        max_buffer_s: float = 25.0,
        min_audio_s: float = 1.0,
    ) -> None:
        self.engine = engine
        self.max_buffer_s = max_buffer_s
        self.min_audio_s = min_audio_s
        self.agreement = LocalAgreement()
        self.buffer_offset = 0.0  # absolute time at buffer start
        self._audio = None  # lazy: numpy array

    def insert_audio(self, chunk) -> None:
        import numpy as np

        self._audio = chunk if self._audio is None else np.concatenate([self._audio, chunk])

    @property
    def buffer_seconds(self) -> float:
        return 0.0 if self._audio is None else len(self._audio) / SAMPLE_RATE

    def process(self) -> RealtimeUpdate:
        """Re-decode the buffer and return newly committed + partial words."""
        if self._audio is None or self.buffer_seconds < self.min_audio_s:
            return RealtimeUpdate()

        raw = self.engine.transcribe_words(self._audio)
        words = [TimedWord(t, s, e) for t, s, e in raw]
        committed, partial = self.agreement.insert(words)
        self._maybe_trim()

        off = self.buffer_offset
        return RealtimeUpdate(
            committed=[TimedWord(w.text, w.start + off, w.end + off) for w in committed],
            partial=[TimedWord(w.text, w.start + off, w.end + off) for w in partial],
        )

    def _maybe_trim(self) -> None:
        """Drop already-committed audio so the buffer (and decode cost) stays bounded."""
        if self.buffer_seconds <= self.max_buffer_s:
            return
        cut_t = self.agreement.committed_end_time()
        if not cut_t or cut_t <= 0:
            return
        cut = int(cut_t * SAMPLE_RATE)
        if cut >= len(self._audio):
            return
        self._audio = self._audio[cut:]
        self.buffer_offset += cut_t
        # Committed text is already emitted; start fresh on the trimmed buffer.
        self.agreement.reset()

    def finalize(self) -> RealtimeUpdate:
        """At end of stream, promote whatever is left to committed."""
        if self._audio is None:
            return RealtimeUpdate()
        raw = self.engine.transcribe_words(self._audio)
        words = [TimedWord(t, s, e) for t, s, e in raw]
        tail = words[self.agreement.n_committed :]
        off = self.buffer_offset
        return RealtimeUpdate(
            committed=[TimedWord(w.text, w.start + off, w.end + off) for w in tail]
        )


def stream_microphone_realtime(
    engine,
    interval_s: float = 1.0,
    max_buffer_s: float = 25.0,
    device: int | None = None,
) -> Iterator[RealtimeUpdate]:
    """Yield live updates from the mic, re-decoding every `interval_s` seconds.

    Lower `interval_s` = more responsive but more CPU (each tick re-decodes the
    whole buffer). ~1.0s is a reasonable starting point.
    """
    import queue

    import numpy as np
    import sounddevice as sd

    transcriber = RealtimeTranscriber(engine, max_buffer_s=max_buffer_s)
    audio_q: "queue.Queue" = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        audio_q.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * 0.1),
        dtype="float32",
        channels=1,
        callback=callback,
        device=device,
    ):
        last = time.monotonic()
        while True:
            try:
                chunk = audio_q.get(timeout=0.1)
                transcriber.insert_audio(chunk[:, 0].astype(np.float32))
            except queue.Empty:
                pass

            now = time.monotonic()
            if now - last >= interval_s:
                last = now
                update = transcriber.process()
                if update.committed or update.partial:
                    yield update


def render_live(updates: Iterator[RealtimeUpdate], stream=None) -> str:
    """Print committed text as it stabilizes, with the partial tail greyed inline.

    Committed text is printed once and never rewritten; the partial tail is
    redrawn on the current line each tick.
    """
    out = stream or sys.stdout
    committed_all: list[str] = []
    last_len = 0
    for update in updates:
        if update.committed:
            committed_all.extend(w.text for w in update.committed)
        line = " ".join(committed_all[-12:])
        partial = update.partial_text
        display = f"{line} \033[90m{partial}\033[0m" if partial else line
        out.write("\r" + " " * last_len + "\r")
        out.write(display)
        out.flush()
        last_len = len(display)
    return " ".join(committed_all)
