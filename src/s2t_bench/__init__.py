"""s2t-bench: pluggable speech-to-text benchmarking with an ADK agent front-end."""
from .engines import (
    TranscriptionEngine,
    TranscriptionResult,
    available_engines,
    build_engine,
    register_engine,
)
from .benchmark import format_table, run_benchmark, word_error_rate, character_error_rate

__version__ = "0.1.0"

__all__ = [
    "TranscriptionEngine",
    "TranscriptionResult",
    "available_engines",
    "build_engine",
    "register_engine",
    "run_benchmark",
    "format_table",
    "word_error_rate",
    "character_error_rate",
    "__version__",
]
