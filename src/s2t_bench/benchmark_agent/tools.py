from __future__ import annotations

from typing import Any

from ..engines import available_engines, build_engine
from .benchmark_tools.runner import format_table, run_benchmark


def list_engines() -> dict[str, Any]:
    """List the speech-to-text engines available for transcription and benchmarking.

    Returns:
        A dict with "status" and "engines" (list of engine name strings such as
        "google_cloud", "gemini", "whisper_api", "faster_whisper").
    """
    return {"status": "success", "engines": available_engines()}


def transcribe(audio_path: str, engine: str = "faster_whisper") -> dict[str, Any]:
    """Transcribe a single audio file with one engine.

    Args:
        audio_path: Path to the audio file to transcribe (wav/mp3/flac/...).
        engine: Engine name to use. One of the values from list_engines.
            Defaults to the offline "faster_whisper" engine.

    Returns:
        A dict with "status", "text" (the transcript), "engine",
        "audio_seconds", "latency_seconds", and "rtf" (real-time factor).
    """
    try:
        eng = build_engine(engine)
        result = eng.transcribe(audio_path)
        return {
            "status": "success",
            "engine": result.engine,
            "text": result.text,
            "audio_seconds": round(result.audio_seconds, 3),
            "latency_seconds": round(result.latency_seconds, 3),
            "rtf": round(result.rtf, 3),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def run_benchmark_tool(
    manifest_path: str,
    engines: list[str],
    limit: int = 0,
    output_dir: str = "",
) -> dict[str, Any]:
    """Benchmark multiple engines over a labeled dataset and compare accuracy/speed.

    Args:
        manifest_path: Path to a JSONL manifest. Each line has "audio" and "text"
            (the reference transcript), plus an optional "id".
        engines: List of engine names to benchmark against each other.
        limit: If > 0, only evaluate the first N samples (useful for quick runs).
        output_dir: If non-empty, write summary.json and per_sample.jsonl there.

    Returns:
        A dict with "status", a rendered leaderboard "table" (sorted by WER),
        and "summaries": per-engine avg_wer, avg_cer, avg_rtf, and error counts.
    """
    try:
        summaries = run_benchmark(
            manifest_path=manifest_path,
            engine_specs=list(engines),
            output_dir=output_dir or None,
            limit=limit or None,
        )
        return {
            "status": "success",
            "table": format_table(summaries),
            "summaries": [
                {
                    "engine": s.engine,
                    "avg_wer": round(s.avg_wer, 4),
                    "avg_cer": round(s.avg_cer, 4),
                    "avg_rtf": round(s.avg_rtf, 3),
                    "num_samples": s.num_samples,
                    "num_errors": s.num_errors,
                }
                for s in summaries
            ],
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
