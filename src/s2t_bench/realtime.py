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
    lagging: bool = False  # set when decode can't keep up and audio was dropped

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

    def trim_committed(self) -> None:
        """Called after the audio for committed words is removed from the buffer.

        Drops the committed words from history (their audio is gone) but KEEPS
        the un-committed tail as prior state, so those words aren't re-emitted and
        can commit on the next agreeing decode. This is the fix for the trim bug
        where a plain reset() caused already-committed text to duplicate (or, with
        different timing, the next utterance to be lost).
        """
        self._prev = self._prev[self._n_committed :]
        self._n_committed = 0

    def reset(self) -> None:
        """Forget all state (e.g. a hard sliding-window trim)."""
        self._prev = []
        self._n_committed = 0


class RealtimeTranscriber:
    """Rolling-buffer transcriber that emits stable prefixes as you speak."""

    #: Whisper's stock hallucinations on silence/noise (trained on YouTube).
    #: Dropped when they are the *entire* decode of a buffer.
    HALLUCINATION_PHRASES = frozenset(
        {
            "thank you for watching",
            "thanks for watching",
            "thank you for watching.",
            "thanks for watching!",
            "please subscribe",
            "like and subscribe",
            "you",
            "thank you",
            "thank you.",
            "bye",
            "the end",
        }
    )

    def __init__(
        self,
        engine,
        max_buffer_s: float = 12.0,
        min_audio_s: float = 1.0,
        silence_rms: float = 0.008,
    ) -> None:
        self.engine = engine
        self.max_buffer_s = max_buffer_s
        self.min_audio_s = min_audio_s
        self.silence_rms = silence_rms  # buffers quieter than this aren't decoded
        self.agreement = LocalAgreement()
        self.buffer_offset = 0.0  # absolute time at buffer start
        self._audio = None  # lazy: numpy array

    def insert_audio(self, chunk) -> None:
        import numpy as np

        self._audio = chunk if self._audio is None else np.concatenate([self._audio, chunk])

    @property
    def buffer_seconds(self) -> float:
        return 0.0 if self._audio is None else len(self._audio) / SAMPLE_RATE

    def _too_quiet(self) -> bool:
        """True if the buffer is near-silent (RMS below threshold).

        Decoding near-silence is what makes Whisper emit 'Thank you for
        watching' — so we skip the decode entirely instead.
        """
        import numpy as np

        if self._audio is None or len(self._audio) == 0:
            return True
        rms = float(np.sqrt(np.mean(self._audio.astype(np.float32) ** 2)))
        return rms < self.silence_rms

    def _is_hallucination(self, words) -> bool:
        """True if the whole decode is just a stock Whisper hallucination."""
        text = " ".join(w.text for w in words).strip().lower()
        return text in self.HALLUCINATION_PHRASES

    def process(self) -> RealtimeUpdate:
        """Re-decode the buffer and return newly committed + partial words."""
        if self._audio is None or self.buffer_seconds < self.min_audio_s:
            return RealtimeUpdate()
        if self._too_quiet():
            return RealtimeUpdate()

        raw = self.engine.transcribe_words(self._audio)
        words = [TimedWord(t, s, e) for t, s, e in raw]
        if self._is_hallucination(words):
            return RealtimeUpdate()
        committed, partial = self.agreement.insert(words)
        self._maybe_trim()

        off = self.buffer_offset
        return RealtimeUpdate(
            committed=[TimedWord(w.text, w.start + off, w.end + off) for w in committed],
            partial=[TimedWord(w.text, w.start + off, w.end + off) for w in partial],
        )

    def _maybe_trim(self) -> None:
        """Keep the buffer (and per-tick decode cost) bounded."""
        if self.buffer_seconds <= self.max_buffer_s:
            return
        cut_t = self.agreement.committed_end_time()
        if cut_t and cut_t > 0:
            cut = int(cut_t * SAMPLE_RATE)
            if cut < len(self._audio):
                self._audio = self._audio[cut:]
                self.buffer_offset += cut_t
                # Committed audio removed; keep the uncommitted tail as history.
                self.agreement.trim_committed()
                return
        # Nothing newly committed but the buffer is over cap: force a hard
        # sliding-window trim so decode latency can't run away. Some context is
        # lost at the cut, which is acceptable as a safety valve.
        if self.buffer_seconds > self.max_buffer_s * 1.5:
            keep = int(self.max_buffer_s * SAMPLE_RATE)
            cut = len(self._audio) - keep
            self._audio = self._audio[cut:]
            self.buffer_offset += cut / SAMPLE_RATE
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


def _drain_queue(audio_q, block_timeout):
    """Collect all currently-queued audio chunks, blocking up to block_timeout
    for the first. Returns a list (possibly empty)."""
    import queue as _q

    chunks = []
    try:
        chunks.append(audio_q.get(timeout=block_timeout))
    except _q.Empty:
        return chunks
    while True:
        try:
            chunks.append(audio_q.get_nowait())
        except _q.Empty:
            break
    return chunks


def _apply_backpressure(chunks, max_chunks):
    """Keep only the most recent `max_chunks` chunks. Returns (kept, dropped).

    When decode can't keep up with real time the queue backs up; instead of
    piling stale audio into the buffer (which causes an ever-growing lag), we
    drop the oldest and stay current. Bounded latency beats completeness here.
    """
    if max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks, 0
    dropped = len(chunks) - max_chunks
    return chunks[-max_chunks:], dropped


def stream_microphone_realtime(
    engine,
    interval_s: float = 1.0,
    max_buffer_s: float = 12.0,
    device: int | None = None,
) -> Iterator[RealtimeUpdate]:
    """Yield live updates from the mic, re-decoding every `interval_s` seconds.

    Includes backpressure: if the machine can't decode as fast as audio arrives,
    stale audio is dropped so latency stays bounded instead of lagging further
    behind every tick. Such updates carry lagging=True as a signal to switch to a
    smaller/faster model.
    """
    import queue

    import numpy as np
    import sounddevice as sd

    chunk_s = 0.1
    transcriber = RealtimeTranscriber(engine, max_buffer_s=max_buffer_s)
    audio_q: "queue.Queue" = queue.Queue()
    max_chunks = int(max_buffer_s / chunk_s)  # never queue more than the buffer

    # Load the model now so its one-time load cost isn't counted as "lag" on the
    # first decode (which would trip the lagging warning spuriously).
    if hasattr(engine, "_get_model"):
        engine._get_model()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        audio_q.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * chunk_s),
        dtype="float32",
        channels=1,
        callback=callback,
        device=device,
    ):
        last = time.monotonic()
        while True:
            timeout = max(0.01, interval_s - (time.monotonic() - last))
            chunks = _drain_queue(audio_q, timeout)
            chunks, dropped = _apply_backpressure(chunks, max_chunks)
            for c in chunks:
                transcriber.insert_audio(c[:, 0].astype(np.float32))

            now = time.monotonic()
            if now - last >= interval_s:
                last = now
                t0 = time.monotonic()
                update = transcriber.process()
                decode_s = time.monotonic() - t0
                update.lagging = dropped > 0 or decode_s > interval_s
                if update.committed or update.partial or update.lagging:
                    yield update


def render_live(updates: Iterator[RealtimeUpdate], stream=None) -> str:
    """Print committed text as it stabilizes, with the partial tail greyed inline.

    Committed text is printed once and never rewritten; the partial tail is
    redrawn on the current line each tick.
    """
    out = stream or sys.stdout
    committed_all: list[str] = []
    last_len = 0
    warned = False
    lag_streak = 0
    for update in updates:
        # A single slow tick is normal (backpressure absorbs it); only warn if
        # lag persists across several ticks, i.e. the hardware genuinely can't
        # sustain this model in real time.
        lag_streak = lag_streak + 1 if update.lagging else 0
        if lag_streak >= 3 and not warned:
            warned = True
            out.write(
                "\n\033[33m[warning] this model is too slow for real time on "
                "this machine; audio is being dropped to keep up. Try a smaller "
                "model or run on a GPU.\033[0m\n"
            )
            out.flush()
            last_len = 0
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
