from s2t_bench.realtime import LocalAgreement, RealtimeTranscriber, TimedWord


def W(*texts: str) -> list[TimedWord]:
    return [TimedWord(t, float(i), float(i) + 1.0) for i, t in enumerate(texts)]


def texts(words: list[TimedWord]) -> list[str]:
    return [w.text for w in words]


# ---- LocalAgreement: the commit policy ----

def test_first_hypothesis_commits_nothing():
    # Needs two agreeing runs before anything is stable.
    la = LocalAgreement()
    committed, partial = la.insert(W("the", "olt"))
    assert texts(committed) == []
    assert texts(partial) == ["the", "olt"]


def test_agreeing_prefix_gets_committed():
    la = LocalAgreement()
    la.insert(W("the", "olt"))
    committed, partial = la.insert(W("the", "olt", "port"))
    assert texts(committed) == ["the", "olt"]
    assert texts(partial) == ["port"]


def test_disagreement_stops_commit_at_divergence():
    la = LocalAgreement()
    la.insert(W("the", "synth", "device"))
    committed, partial = la.insert(W("the", "ont", "device"))
    assert texts(committed) == ["the"]  # only up to the divergence
    assert texts(partial) == ["ont", "device"]


def test_committed_words_are_never_revoked():
    la = LocalAgreement()
    la.insert(W("the", "olt"))
    la.insert(W("the", "olt"))  # commits both
    # Model now changes its mind about an already-committed word.
    committed, _ = la.insert(W("the", "ont", "port"))
    assert texts(committed) == []  # nothing new, and nothing taken back
    assert la.n_committed == 2


def test_commit_is_monotonic_across_growing_buffer():
    la = LocalAgreement()
    seen: list[str] = []
    for hyp in [
        W("we"),
        W("we", "swapped"),
        W("we", "swapped", "the"),
        W("we", "swapped", "the", "olt"),
    ]:
        c, _ = la.insert(hyp)
        seen.extend(texts(c))
    assert seen == ["we", "swapped", "the"]  # last word not yet confirmed twice


def test_punctuation_and_case_do_not_block_agreement():
    la = LocalAgreement()
    la.insert(W("the", "OLT"))
    committed, _ = la.insert(W("The", "olt,", "port"))
    assert len(committed) == 2  # normalized comparison still agrees


def test_reset_clears_state():
    la = LocalAgreement()
    la.insert(W("a", "b"))
    la.insert(W("a", "b"))
    assert la.n_committed == 2
    la.reset()
    assert la.n_committed == 0


# ---- RealtimeTranscriber: buffer handling with a stub engine ----

class StubEngine:
    """Returns a scripted hypothesis per call, ignoring the audio."""

    def __init__(self, scripts: list[list[TimedWord]]) -> None:
        self.scripts = scripts
        self.calls = 0

    def transcribe_words(self, audio):  # noqa: ARG002
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        return [(w.text, w.start, w.end) for w in script]


def test_transcriber_waits_for_min_audio():
    import numpy as np

    engine = StubEngine([W("hello")])
    rt = RealtimeTranscriber(engine, min_audio_s=1.0)
    rt.insert_audio(np.zeros(8000, dtype=np.float32))  # 0.5s < min
    update = rt.process()
    assert update.committed == [] and update.partial == []
    assert engine.calls == 0  # didn't bother decoding


def test_transcriber_emits_committed_prefix():
    import numpy as np

    engine = StubEngine([W("the", "olt"), W("the", "olt", "port")])
    rt = RealtimeTranscriber(engine, min_audio_s=0.5, silence_rms=0.0)
    rt.insert_audio(np.zeros(16000, dtype=np.float32))
    assert texts(rt.process().committed) == []
    assert texts(rt.process().committed) == ["the", "olt"]


def test_buffer_trim_bounds_growth_and_offsets_time():
    import numpy as np

    # Two identical hypotheses -> commits, then the buffer exceeds max and trims.
    engine = StubEngine([W("a", "b"), W("a", "b")])
    rt = RealtimeTranscriber(engine, max_buffer_s=2.0, min_audio_s=0.5, silence_rms=0.0)
    rt.insert_audio(np.zeros(SR := 16000 * 3, dtype=np.float32))  # 3s > max 2s
    rt.process()  # nothing committed yet
    before = rt.buffer_seconds
    rt.process()  # commits "a","b" (ends at t=2.0) -> trims
    assert rt.buffer_seconds < before
    assert rt.buffer_offset > 0  # absolute clock advanced


# ---- hallucination / silence guards ----

def _loud(n=16000):
    import numpy as np
    return (np.random.RandomState(0).randn(n) * 0.2).astype("float32")


def _quiet(n=16000):
    import numpy as np
    return (np.random.RandomState(0).randn(n) * 0.001).astype("float32")


class _Phrase:
    def __init__(self, words):
        self._w = words
    def transcribe_words(self, audio):
        return [(w, float(i), float(i) + 1) for i, w in enumerate(self._w)]


def test_silence_is_skipped_before_decoding():
    rt = RealtimeTranscriber(_Phrase(["the", "olt"]), min_audio_s=0.5)
    rt.insert_audio(_quiet())
    assert rt._too_quiet() is True
    assert rt.process().committed == []


def test_stock_hallucination_is_dropped():
    rt = RealtimeTranscriber(_Phrase(["Thank", "you", "for", "watching"]), min_audio_s=0.5)
    rt.insert_audio(_loud())
    assert rt.process().committed == []  # phrase filtered even on loud audio


def test_real_speech_passes_through():
    rt = RealtimeTranscriber(_Phrase(["the", "olt"]), min_audio_s=0.5)
    rt.insert_audio(_loud())
    rt.process()  # first decode
    assert texts(rt.process().committed) == ["the", "olt"]  # agreed twice


# ---- trim continuity: no duplication, no loss across a buffer trim ----

def test_trim_committed_keeps_uncommitted_tail():
    la = LocalAgreement()
    la.insert(W("a", "b", "c"))
    la.insert(W("a", "b", "c"))  # commit a,b,c
    assert la.n_committed == 3
    # simulate: audio for a,b,c removed; d was still partial
    la._prev = W("a", "b", "c", "d")
    la._n_committed = 3
    la.trim_committed()
    assert la.n_committed == 0
    assert texts(la._prev) == ["d"]  # committed words dropped, tail kept


def test_second_utterance_survives_trim():
    """Regression for the live bug: a second sentence spoken after the buffer
    trims must commit exactly once - not duplicate, not vanish."""
    import numpy as np

    SR = 16000
    truth = [("Can", 0, 1), ("you", 1, 2), ("hear", 2, 3), ("me", 3, 4),
             ("now", 4, 5), ("the", 7, 8), ("sim", 8, 9), ("is", 9, 10),
             ("broken", 10, 11)]

    class Sim:
        def __init__(self, rt):
            self.rt = rt
        def transcribe_words(self, audio):
            start = self.rt.buffer_offset
            end = start + len(audio) / SR
            return [(w, s - start, e - start) for w, s, e in truth
                    if s >= start and e <= end + 0.01]

    rt = RealtimeTranscriber(None, min_audio_s=0.5, silence_rms=0.0, max_buffer_s=12.0)
    rt.engine = Sim(rt)
    got = []
    for tick in range(16):
        rt.insert_audio((np.random.RandomState(tick).randn(SR) * 0.2).astype("float32"))
        got += [w.text for w in rt.process().committed]
    assert got == [w for w, _, _ in truth]  # exactly once, in order


def test_hard_ceiling_bounds_runaway_buffer():
    import numpy as np

    # Engine that never lets anything commit -> would grow forever without the cap.
    class NeverAgrees:
        def __init__(self): self.i = 0
        def transcribe_words(self, audio):
            self.i += 1
            return [(f"w{self.i}", 0.0, 1.0)]  # different word every tick

    rt = RealtimeTranscriber(NeverAgrees(), min_audio_s=0.5, silence_rms=0.0, max_buffer_s=4.0)
    for tick in range(20):
        rt.insert_audio(np.ones(16000, dtype="float32") * 0.2)
        rt.process()
    assert rt.buffer_seconds <= 4.0 * 1.5  # hard sliding-window trim kept it bounded


# ---- mic-loop backpressure (prevents the runaway-lag stall) ----

def test_backpressure_keeps_everything_when_not_behind():
    from s2t_bench.realtime import _apply_backpressure
    kept, dropped = _apply_backpressure(list(range(5)), 120)
    assert dropped == 0 and kept == list(range(5))


def test_backpressure_drops_oldest_when_behind():
    from s2t_bench.realtime import _apply_backpressure
    kept, dropped = _apply_backpressure(list(range(300)), 120)
    assert dropped == 180
    assert kept[0] == 180 and kept[-1] == 299  # newest audio retained


def test_drain_queue_collects_all_available():
    import queue
    from s2t_bench.realtime import _drain_queue
    q = queue.Queue()
    for i in range(7):
        q.put(i)
    got = _drain_queue(q, 0.01)
    assert got == list(range(7))
    assert q.empty()


# ---- lag warning: only on sustained lag, not transient blips ----

def _updates(flags):
    from s2t_bench.realtime import RealtimeUpdate
    for lag in flags:
        yield RealtimeUpdate(committed=W("x"), lagging=lag)


def test_no_warning_on_transient_lag():
    import io
    from s2t_bench.realtime import render_live
    buf = io.StringIO()
    render_live(_updates([True, False, True, False, True]), stream=buf)
    assert "[warning]" not in buf.getvalue()


def test_warning_once_on_sustained_lag():
    import io
    from s2t_bench.realtime import render_live
    buf = io.StringIO()
    render_live(_updates([True] * 6), stream=buf)
    v = buf.getvalue()
    assert v.count("[warning]") == 1
    assert "-m base" not in v and "int8" not in v  # no stale flag advice
