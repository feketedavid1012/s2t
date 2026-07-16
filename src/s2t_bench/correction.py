from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .domain import DEFAULT_TELECOM_GLOSSARY

CORRECTION_SYSTEM_PROMPT = """
You review speech-to-text transcripts for a telecom fibre-access field team.
Engineers speak about XGS-PON / GPON networks, OLT / ONT / ONU equipment,
catalogue part names, SKUs, and installation and provisioning work.

Given a RAW transcript, decide whether it is already correct or needs fixing.

Fix ONLY transcription errors:
- Misheard or split domain terms and acronyms; normalise their casing, e.g.
  "excess pon" / "x g s pon" -> "XGS-PON", "sky you" / "s k u" -> "SKU",
  "oh l t" -> "OLT", "o n t" -> "ONT", "g pon" -> "GPON".
- Obvious homophones and word-boundary errors.
- Equipment names, part numbers and SKUs, using the glossary provided.
- Light punctuation and capitalisation for readability.

Do NOT paraphrase, summarise, translate, reorder, add, or remove meaning.
Preserve the engineer's own wording. Remove a fragment only when it is clearly
speech-to-text noise, never when it could be real content.

Respond with STRICT JSON and nothing else:
- If no changes are needed:      {"status":"ok"}
- If you changed anything:        {"status":"corrected","text":"<full corrected transcript>"}
""".strip()


@dataclass
class CorrectionResult:
    status: str  # "ok" or "corrected"
    text: str | None  # full corrected transcript when status == "corrected"
    changed: bool


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_correction(raw_response: str, original: str) -> CorrectionResult:
    """Parse the model response into a CorrectionResult (pure, testable)."""
    cleaned = _strip_fences(raw_response)
    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        data = None

    if isinstance(data, dict):
        status = str(data.get("status", "")).lower()
        if status == "ok":
            return CorrectionResult("ok", None, False)
        if status == "corrected":
            text = (data.get("text") or "").strip()
            if not text or text == original.strip():
                return CorrectionResult("ok", None, False)
            return CorrectionResult("corrected", text, True)

    # Fallback: model returned plain text instead of JSON.
    if cleaned and cleaned != original.strip():
        return CorrectionResult("corrected", cleaned, True)
    return CorrectionResult("ok", None, False)


def _build_user_prompt(transcript: str, glossary: list[str]) -> str:
    terms = ", ".join(glossary)
    return (
        f"Glossary (normalise to these forms when applicable): {terms}\n\n"
        f"RAW transcript:\n{transcript}"
    )


def review_and_correct(
    transcript: str,
    glossary: list[str] | None = None,
    model: str = "gemini-flash-latest",
    client: object | None = None,
) -> CorrectionResult:
    """Review a transcript with Gemini and return an OK flag or corrected text."""
    transcript = (transcript or "").strip()
    if not transcript:
        return CorrectionResult("ok", None, False)

    glossary = glossary or DEFAULT_TELECOM_GLOSSARY

    from google import genai  # lazy import
    from google.genai import types

    client = client or genai.Client()
    config = types.GenerateContentConfig(
        system_instruction=CORRECTION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.0,
    )
    response = client.models.generate_content(
        model=model,
        contents=_build_user_prompt(transcript, glossary),
        config=config,
    )
    return parse_correction(response.text or "", transcript)
