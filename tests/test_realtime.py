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
    rt = RealtimeTranscriber(engine, min_audio_s=0.5)
    rt.insert_audio(np.zeros(16000, dtype=np.float32))
    assert texts(rt.process().committed) == []
    assert texts(rt.process().committed) == ["the", "olt"]


def test_buffer_trim_bounds_growth_and_offsets_time():
    import numpy as np

    # Two identical hypotheses -> commits, then the buffer exceeds max and trims.
    engine = StubEngine([W("a", "b"), W("a", "b")])
    rt = RealtimeTranscriber(engine, max_buffer_s=2.0, min_audio_s=0.5)
    rt.insert_audio(np.zeros(SR := 16000 * 3, dtype=np.float32))  # 3s > max 2s
    rt.process()  # nothing committed yet
    before = rt.buffer_seconds
    rt.process()  # commits "a","b" (ends at t=2.0) -> trims
    assert rt.buffer_seconds < before
    assert rt.buffer_offset > 0  # absolute clock advanced
