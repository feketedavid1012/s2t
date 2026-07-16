"""Benchmark orchestration: run N engines over a dataset and aggregate metrics."""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..engines import TranscriptionEngine, build_engine
from .dataset import Sample, load_manifest
from .metrics import character_error_rate, word_error_rate


@dataclass
class SampleResult:
    engine: str
    sample_id: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    audio_seconds: float
    latency_seconds: float
    rtf: float
    error: str | None = None


@dataclass
class EngineSummary:
    engine: str
    num_samples: int
    avg_wer: float
    avg_cer: float
    avg_rtf: float
    total_audio_seconds: float
    total_latency_seconds: float
    num_errors: int
    per_sample: list[SampleResult] = field(default_factory=list)


def _score_one(engine: TranscriptionEngine, sample: Sample) -> SampleResult:
    try:
        result = engine.transcribe(sample.audio_path)
        return SampleResult(
            engine=engine.name,
            sample_id=sample.id,
            reference=sample.reference,
            hypothesis=result.text,
            wer=word_error_rate(sample.reference, result.text),
            cer=character_error_rate(sample.reference, result.text),
            audio_seconds=result.audio_seconds,
            latency_seconds=result.latency_seconds,
            rtf=result.rtf,
        )
    except Exception as exc:  # keep the run going; record the failure
        return SampleResult(
            engine=engine.name,
            sample_id=sample.id,
            reference=sample.reference,
            hypothesis="",
            wer=float("nan"),
            cer=float("nan"),
            audio_seconds=0.0,
            latency_seconds=0.0,
            rtf=float("nan"),
            error=f"{type(exc).__name__}: {exc}",
        )


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return statistics.fmean(vals) if vals else float("nan")


def summarize(engine_name: str, rows: list[SampleResult]) -> EngineSummary:
    ok = [r for r in rows if r.error is None]
    return EngineSummary(
        engine=engine_name,
        num_samples=len(rows),
        avg_wer=_mean([r.wer for r in ok]),
        avg_cer=_mean([r.cer for r in ok]),
        avg_rtf=_mean([r.rtf for r in ok]),
        total_audio_seconds=sum(r.audio_seconds for r in ok),
        total_latency_seconds=sum(r.latency_seconds for r in ok),
        num_errors=sum(1 for r in rows if r.error is not None),
        per_sample=rows,
    )


def run_benchmark(
    manifest_path: str | Path,
    engine_specs: dict[str, dict[str, Any]] | list[str],
    output_dir: str | Path | None = None,
    limit: int | None = None,
) -> list[EngineSummary]:
    """Run every engine over the dataset and return per-engine summaries.

    engine_specs:
        - list of engine names (default config), e.g. ["faster_whisper", "gemini"]
        - or dict of {engine_name: {**config}} for per-engine settings.
    """
    samples = load_manifest(manifest_path)
    if limit:
        samples = samples[:limit]

    if isinstance(engine_specs, list):
        engine_specs = {name: {} for name in engine_specs}

    summaries: list[EngineSummary] = []
    for name, cfg in engine_specs.items():
        engine = build_engine(name, **cfg)
        rows = [_score_one(engine, s) for s in samples]
        summaries.append(summarize(name, rows))

    if output_dir:
        _write_reports(summaries, output_dir)
    return summaries


def _write_reports(summaries: list[EngineSummary], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            [
                {k: v for k, v in asdict(s).items() if k != "per_sample"}
                for s in summaries
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out / "per_sample.jsonl").open("w", encoding="utf-8") as fh:
        for s in summaries:
            for row in s.per_sample:
                fh.write(json.dumps(asdict(row)) + "\n")


def format_table(summaries: list[EngineSummary]) -> str:
    """Render a compact leaderboard sorted by WER (best first)."""
    header = f"{'engine':<18}{'WER':>8}{'CER':>8}{'RTF':>8}{'errs':>6}{'n':>5}"
    lines = [header, "-" * len(header)]
    for s in sorted(summaries, key=lambda x: (x.avg_wer != x.avg_wer, x.avg_wer)):
        lines.append(
            f"{s.engine:<18}"
            f"{s.avg_wer:>8.3f}{s.avg_cer:>8.3f}{s.avg_rtf:>8.2f}"
            f"{s.num_errors:>6}{s.num_samples:>5}"
        )
    return "\n".join(lines)
