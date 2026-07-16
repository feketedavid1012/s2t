"""ADK root agent for the speech-to-text benchmark.

Run it with:
    adk run src/s2t_bench/agent
    adk web  src/s2t_bench          # then pick the "agent" package in the UI

The agent can transcribe single files and run head-to-head benchmarks by calling
the tools in tools.py. Model is configurable via S2T_AGENT_MODEL.
"""
from __future__ import annotations

import os

from .tools import list_engines, run_benchmark_tool, transcribe

_MODEL = os.environ.get("S2T_AGENT_MODEL", "gemini-flash-latest")

_INSTRUCTION = """
You are the S2T-Bench assistant. You help users transcribe audio and benchmark
speech-to-text engines (Google Cloud STT, Gemini, OpenAI Whisper, and on-prem
faster-whisper) against each other.

Guidelines:
- To see what is available, call list_engines.
- For a single file, call transcribe with the audio path and chosen engine.
- To compare engines, call run_benchmark_tool with a JSONL manifest path and the
  list of engines. WER and CER are lower-is-better; RTF < 1.0 means faster than
  real time. Always present the returned "table" and then briefly interpret it:
  which engine is most accurate, which is fastest, and any tradeoff.
- If a tool returns status "error", explain the likely cause (missing dependency,
  missing API key/credentials, or bad path) and how to fix it.
- Never invent WER/CER numbers; only report values returned by the tools.
""".strip()

try:
    from google.adk import Agent

    root_agent = Agent(
        name="s2t_bench_agent",
        model=_MODEL,
        instruction=_INSTRUCTION,
        description="Transcribe audio and benchmark STT engines (Google, Whisper, on-prem).",
        tools=[list_engines, transcribe, run_benchmark_tool],
    )
except ImportError:  # ADK optional; the rest of the package still works.
    root_agent = None
