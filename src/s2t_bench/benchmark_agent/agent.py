from __future__ import annotations

import os

from s2t_bench.benchmark_agent.prompt import INSTRUCTION

from .tools import list_engines, run_benchmark_tool, transcribe

_MODEL = os.environ.get("S2T_AGENT_MODEL", "gemini-flash-latest")



try:
    from google.adk import Agent

    root_agent = Agent(
        name="s2t_bench_agent",
        model=_MODEL,
        instruction=INSTRUCTION,
        description="Transcribe audio and benchmark STT engines (Google, Whisper, on-prem).",
        tools=[list_engines, transcribe, run_benchmark_tool],
    )
except ImportError: 
    root_agent = None
