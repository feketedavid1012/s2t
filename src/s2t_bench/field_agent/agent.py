from __future__ import annotations

import os

from .tools import review_and_correct_tool, transcribe_audio

_MODEL = os.environ.get("S2T_AGENT_MODEL", "gemini-flash-latest")

_INSTRUCTION = """
You process field audio from telecom fibre-access service engineers and return a
clean, domain-correct transcript. The audio covers XGS-PON / GPON networks,
OLT / ONT / ONU equipment, catalogue part names, SKUs, and installation work.

When the user gives you an audio file path, follow these steps exactly:

1. Call transcribe_audio(audio_path) to get the raw transcript. If the user
   mentions specific SKUs or product names, pass them as extra_terms.
2. Pass that transcript to review_and_correct_tool(transcript).
3. Return your final answer based ONLY on the review result:
   - If status is "ok": tell the user the transcript looks correct, then show it.
   - If status is "corrected": return ONLY the corrected full transcript text.
     Do not add a preamble, do not explain what you changed, do not summarise.
   - If status is "error": briefly state the error and the likely fix (missing
     dependency, missing API key/credentials, or bad file path).

Never paraphrase, summarise, or invent content. Only the tools decide the text.
""".strip()

try:
    from google.adk import Agent

    root_agent = Agent(
        name="s2t_field_agent",
        model=_MODEL,
        instruction=_INSTRUCTION,
        description=(
            "Transcribes telecom field audio with on-prem Whisper and returns a "
            "Gemini domain-corrected transcript (XGS-PON, SKUs, equipment names)."
        ),
        tools=[transcribe_audio, review_and_correct_tool],
    )
except ImportError:  # ADK optional; tools remain usable directly.
    root_agent = None
