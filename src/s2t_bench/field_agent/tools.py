"""Tools for the field transcription agent.

Two tools, called in order by the agent:
1. transcribe_audio      -> on-prem Whisper, biased with telecom vocabulary
2. review_and_correct_tool -> Gemini domain review; OK flag or corrected text

Glossary is the built-in telecom set; pass `extra_terms` (comma-separated) to add
your own catalogue / SKU names for a given job.
"""
from __future__ import annotations

from typing import Any

from ..correction import review_and_correct
from ..domain import build_initial_prompt, merge_glossary
from ..engines.faster_whisper_local import FasterWhisperEngine


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    extra_terms: str = "",
) -> dict[str, Any]:
    """Transcribe a telecom field-audio file with the on-prem Whisper model.

    The decoder is biased toward fibre-access vocabulary (XGS-PON, GPON, OLT,
    ONT, ONU, SKUs, catalogue equipment) to improve rare-term recognition.

    Args:
        audio_path: Path to the audio file (wav/mp3/flac/m4a...).
        model_size: Whisper size: tiny, base, small, medium, or large-v3.
            Larger is more accurate but slower. Defaults to "base".
        extra_terms: Optional comma-separated job-specific terms to bias toward,
            e.g. specific SKU codes or product names.

    Returns:
        {"status": "success", "text": <raw transcript>} or
        {"status": "error", "error": <message>}.
    """
    try:
        glossary = merge_glossary(extra_terms)
        engine = FasterWhisperEngine(
            model=model_size,
            initial_prompt=build_initial_prompt(glossary),
        )
        result = engine.transcribe(audio_path)
        return {
            "status": "success",
            "text": result.text,
            "language": result.language,
            "audio_seconds": round(result.audio_seconds, 2),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def review_and_correct_tool(transcript: str, extra_terms: str = "") -> dict[str, Any]:
    """Review a transcript for telecom-domain correctness using Gemini.

    If the transcript is already correct and free of irrelevant content, this
    flags it OK and changes nothing. If there are errors, it returns the FULL
    corrected transcript (no paraphrasing, no summary, no explanations).

    Args:
        transcript: The raw transcript text to review.
        extra_terms: Optional comma-separated job-specific terms to normalise to.

    Returns:
        {"status": "ok", "changed": false, "text": <original>} if correct, or
        {"status": "corrected", "changed": true, "text": <corrected full text>},
        or {"status": "error", "error": <message>}.
    """
    try:
        glossary = merge_glossary(extra_terms)
        result = review_and_correct(transcript, glossary=glossary)
        if result.status == "ok":
            return {"status": "ok", "changed": False, "text": transcript}
        return {"status": "corrected", "changed": True, "text": result.text}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
