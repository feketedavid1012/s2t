"""Evaluate Whisper model sizes on accuracy + speed + hardware under concurrency.

For each model size: transcribe a labeled manifest under N concurrent workers,
measuring WER/CER (accuracy), latency/throughput (speed), and CPU/RAM (footprint,
including the model's RSS once loaded). CPU-only friendly.
"""
from __future__ import annotations

from .concurrency import run_concurrent
from .hardware import HardwareMonitor


def evaluate_model(
    model_size: str,
    samples,
    concurrency: int = 3,
    compute_type: str = "int8",
    engine_factory=None,
) -> dict:
    """Evaluate one Whisper size. `engine_factory(model, compute_type)` is
    injectable for testing; defaults to a real FasterWhisperEngine."""
    from ..benchmark.metrics import character_error_rate, word_error_rate

    if engine_factory is None:
        from ..engines.faster_whisper_local import FasterWhisperEngine

        def engine_factory(model, compute_type):  # noqa: E731
            return FasterWhisperEngine(model=model, compute_type=compute_type)

    engine = engine_factory(model_size, compute_type)

    def _task(sample):
        result = engine.transcribe(sample.audio_path)
        return {
            "wer": word_error_rate(sample.reference, result.text),
            "cer": character_error_rate(sample.reference, result.text),
            "rtf": result.rtf,
            "audio_s": result.audio_seconds,
            "reference": sample.reference,
            "hypothesis": result.text,         
        }

    with HardwareMonitor() as hw:
        # Warm up / load weights before the measured run.
        if hasattr(engine, "_get_model"):
            try:
                engine._get_model()
            except Exception:
                pass
        conc = run_concurrent(_task, samples, concurrency)

    ok = [o.result for o in conc.outcomes if o.ok]
    errors = [o.error for o in conc.outcomes if not o.ok]
    per_sample = [
        {
            "id": samples[o.index].id,
            "wer": round(o.result["wer"], 3),
            "reference": o.result["reference"],
            "hypothesis": o.result["hypothesis"],
        }
        for o in conc.outcomes if o.ok
    ]

    def _mean(key):
        vals = [r[key] for r in ok if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    return {
        "model": model_size,
        "compute_type": compute_type,
        "concurrency": concurrency,
        "num_samples": len(samples),
        "accuracy": {"avg_wer": round(_mean("wer"), 4), "avg_cer": round(_mean("cer"), 4)},
        "speed": {"avg_rtf": round(_mean("rtf"), 3), **conc.as_dict()},
        "hardware": hw.summary().as_dict(),
        "num_errors": len(errors),
        "first_error": errors[0] if errors else None,
        "per_sample": per_sample,
    }


def evaluate_models(
    model_sizes,
    manifest_path,
    concurrency=3,
    compute_type="int8",
    limit=None,
    output_dir=None,
):
    from ..benchmark.dataset import load_manifest

    samples = load_manifest(manifest_path)
    if limit:
        samples = samples[:limit]
    results = [evaluate_model(m, samples, concurrency, compute_type) for m in model_sizes]
    if output_dir:
        from .report import write_results

        write_results(results, output_dir, name="whisper_eval")
    return results
