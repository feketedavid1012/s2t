from s2t_bench.correction import parse_correction
from s2t_bench.domain import build_initial_prompt, merge_glossary
from s2t_bench.streaming import EnergyVAD, VADSegmenter


# ---- domain ----

def test_merge_glossary_dedupes_and_appends():
    g = merge_glossary("MyBox-100, XGS-PON")  # XGS-PON already in defaults
    assert "MyBox-100" in g
    assert sum(1 for t in g if t.upper() == "XGS-PON") == 1


def test_initial_prompt_mentions_terms():
    prompt = build_initial_prompt(["XGS-PON", "SKU"])
    assert "XGS-PON" in prompt and "SKU" in prompt


# ---- correction parsing (pure, no network) ----

def test_parse_ok():
    r = parse_correction('{"status":"ok"}', "some text")
    assert r.status == "ok" and r.changed is False


def test_parse_corrected():
    r = parse_correction('{"status":"corrected","text":"XGS-PON OLT"}', "excess pon o l t")
    assert r.status == "corrected" and r.text == "XGS-PON OLT" and r.changed


def test_parse_corrected_but_identical_is_ok():
    r = parse_correction('{"status":"corrected","text":"same"}', "same")
    assert r.status == "ok"


def test_parse_strips_code_fences():
    raw = '```json\n{"status":"ok"}\n```'
    assert parse_correction(raw, "x").status == "ok"


def test_parse_plain_text_fallback():
    r = parse_correction("XGS-PON provisioning done", "excess pon provisioning done")
    assert r.status == "corrected" and r.changed


# ---- VAD segmentation (stubbed vad, no mic / webrtcvad) ----

def _make_frames(pattern: str, frame_bytes: int) -> list[bytes]:
    # 'S' = speech frame, '.' = silence frame; content is arbitrary bytes.
    return [bytes([1 if c == "S" else 0]) * frame_bytes for c in pattern]


def _stub_vad(frame: bytes) -> bool:
    return frame[0] == 1  # our speech frames are filled with 0x01


def test_segmenter_emits_utterance_on_trailing_silence():
    seg = VADSegmenter(silence_ms=60, min_utterance_ms=30, is_speech=_stub_vad)
    # 5 speech frames then 3 silence frames (silence_ms=60 -> 2 silence frames)
    emitted = []
    for frame in _make_frames("SSSSS...", seg.frame_bytes):
        out = seg.push(frame)
        if out:
            emitted.append(out)
    assert len(emitted) == 1
    # 5 speech + 2 trailing-silence frames (silence_ms=60 -> 2 frames to finalize)
    assert len(emitted[0]) == 7 * seg.frame_bytes


def test_segmenter_ignores_too_short_blip():
    seg = VADSegmenter(silence_ms=60, min_utterance_ms=300, is_speech=_stub_vad)
    emitted = [seg.push(f) for f in _make_frames("S...", seg.frame_bytes)]
    assert all(e is None for e in emitted)


def test_segmenter_flush_emits_trailing_speech():
    seg = VADSegmenter(silence_ms=600, min_utterance_ms=30, is_speech=_stub_vad)
    for f in _make_frames("SSSS", seg.frame_bytes):
        assert seg.push(f) is None  # no trailing silence yet
    tail = seg.flush()
    assert tail is not None and len(tail) == 4 * seg.frame_bytes


# ---- energy VAD fallback (no webrtcvad / setuptools needed) ----

def test_energy_vad_fixed_threshold():
    import numpy as np

    vad = EnergyVAD(threshold=1000.0)
    loud = (np.ones(480) * 6000).astype(np.int16).tobytes()
    quiet = (np.ones(480) * 50).astype(np.int16).tobytes()
    assert vad(loud) is True
    assert vad(quiet) is False


def test_energy_vad_empty_frame_is_silence():
    assert EnergyVAD(threshold=1.0)(b"") is False
