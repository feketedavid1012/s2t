"""Accuracy metrics for transcription: WER and CER, with light normalization.

Uses `jiwer` when available (recommended, handles alignment robustly) and falls
back to a small built-in Levenshtein implementation so the package works even
without the optional dependency.
"""
from __future__ import annotations

import re
import string
import unicodedata

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/accents, collapse whitespace.

    Keep this deliberately simple and explicit — normalization choices strongly
    affect WER, so they should be visible and easy to tune per benchmark.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _levenshtein(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str, normalized: bool = True) -> float:
    ref = normalize(reference) if normalized else reference
    hyp = normalize(hypothesis) if normalized else hypothesis
    try:
        import jiwer  # type: ignore

        return float(jiwer.wer(ref, hyp))
    except Exception:
        ref_w, hyp_w = ref.split(), hyp.split()
        if not ref_w:
            return 0.0 if not hyp_w else 1.0
        return _levenshtein(ref_w, hyp_w) / len(ref_w)


def character_error_rate(reference: str, hypothesis: str, normalized: bool = True) -> float:
    ref = normalize(reference) if normalized else reference
    hyp = normalize(hypothesis) if normalized else hypothesis
    try:
        import jiwer  # type: ignore

        return float(jiwer.cer(ref, hyp))
    except Exception:
        if not ref:
            return 0.0 if not hyp else 1.0
        return _levenshtein(list(ref), list(hyp)) / len(ref)
