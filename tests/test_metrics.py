from s2t_bench.benchmark.metrics import (
    character_error_rate,
    normalize,
    word_error_rate,
)
from s2t_bench.engines import available_engines, build_engine


def test_normalize_strips_punct_and_case():
    assert normalize("Hello, World!") == "hello world"


def test_wer_perfect_match_is_zero():
    ref = "the quick brown fox"
    assert word_error_rate(ref, "The quick brown fox.") == 0.0


def test_wer_one_substitution():
    # 1 wrong word out of 4 -> 0.25
    assert word_error_rate("the quick brown fox", "the quick red fox") == 0.25


def test_cer_perfect_match_is_zero():
    assert character_error_rate("abc", "abc") == 0.0


def test_registry_lists_expected_engines():
    engines = available_engines()
    for name in ("google_cloud", "gemini", "whisper_api", "faster_whisper"):
        assert name in engines


def test_build_unknown_engine_raises():
    try:
        build_engine("does_not_exist")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown engine")
