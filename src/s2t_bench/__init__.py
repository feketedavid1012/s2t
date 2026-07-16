from .benchmark_agent.benchmark_tools import (
    character_error_rate,
    format_table,
    run_benchmark,
    word_error_rate,
)
from .engines import (
    TranscriptionEngine,
    TranscriptionResult,
    available_engines,
    build_engine,
    register_engine,
)

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
