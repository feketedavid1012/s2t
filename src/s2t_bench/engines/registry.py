from __future__ import annotations

from typing import Any

from .base import TranscriptionEngine
from .faster_whisper_local import FasterWhisperEngine
from .gemini import GeminiEngine
from .google_cloud import GoogleCloudSTTEngine
from .whisper_openai import WhisperAPIEngine

# Register new engines here (or via register_engine at runtime).
ENGINE_REGISTRY: dict[str, type[TranscriptionEngine]] = {
    GoogleCloudSTTEngine.name: GoogleCloudSTTEngine,
    GeminiEngine.name: GeminiEngine,
    WhisperAPIEngine.name: WhisperAPIEngine,
    FasterWhisperEngine.name: FasterWhisperEngine,
}


def register_engine(cls: type[TranscriptionEngine]) -> type[TranscriptionEngine]:
    """Decorator/utility to add a custom engine to the registry."""
    ENGINE_REGISTRY[cls.name] = cls
    return cls


def available_engines() -> list[str]:
    return sorted(ENGINE_REGISTRY)


def build_engine(name: str, **kwargs: Any) -> TranscriptionEngine:
    """Instantiate an engine by name with keyword config."""
    if name not in ENGINE_REGISTRY:
        raise KeyError(
            f"Unknown engine {name!r}. Available: {', '.join(available_engines())}"
        )
    return ENGINE_REGISTRY[name](**kwargs)
