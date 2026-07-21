"""Evaluate local Gemma models (via Ollama) on two tasks under concurrency:

1. correction  - clean up a noisy telecom transcript
2. json        - convert the report context into the fault-report schema

Measures task quality (WER for correction; parse/schema-valid/field accuracy for
JSON), latency/throughput under N concurrent requests, and CPU/RAM (tracking the
ollama process for the model's memory footprint).
"""
from __future__ import annotations

import json

from ..concurrency import run_concurrent
from ..hardware import HardwareMonitor
from . import schema, scorer
from .ollama_client import generate
from .samples import load_samples

CORRECTION_SYSTEM = (
    "You clean up noisy speech-to-text transcripts from telecom fibre-access "
    "engineers (XGS-PON, GPON, OLT, ONT, SKUs). Fix mis-heard terms, acronym "
    "casing, and spoken numbers. Do not add or remove meaning. Reply with only "
    "the corrected transcript."
)

JSON_SYSTEM = (
    "You convert a telecom fault report into a strict JSON object. Use only the "
    "information given. Output JSON only, matching the required fields exactly."
)


def _json_prompt(context: str) -> str:
    fields = ", ".join(schema.ALL_FIELDS)
    return (
        f"Fault report context:\n{context}\n\n"
        f"Produce a JSON object with exactly these fields: {fields}. "
        "The *_flag fields are booleans. faulty_components and used_components are "
        'arrays of {"item": string, "sku": string}. Categories should be concise.'
    )


def _run_correction(model, host, sample):
    out = generate(model, sample.raw_text, system=CORRECTION_SYSTEM, host=host)
    return scorer.score_correction(sample.corrected_text, out)


def _run_json(model, host, sample):
    out = generate(
        model,
        _json_prompt(sample.corrected_text),
        system=JSON_SYSTEM,
        fmt=schema.json_schema(),  # constrain to the schema
        host=host,
    )
    return scorer.score_json(sample.expected, scorer.extract_json(out))


def evaluate_model(
    model: str,
    tasks: list[str],
    concurrency: int = 3,
    host: str = "http://localhost:11434",
    samples=None,
    generate_fn=None,
) -> dict:
    """Evaluate one model. `generate_fn` is injectable for testing."""
    global generate
    if generate_fn is not None:
        generate = generate_fn  # type: ignore

    samples = samples or load_samples()
    result: dict = {"model": model, "concurrency": concurrency, "num_samples": len(samples)}

    with HardwareMonitor(track_process_names=["ollama"]) as hw:
        # Warm up (Ollama loads the model into RAM on first call).
        try:
            generate(model, "ping", host=host)
        except Exception:
            pass

        if "correction" in tasks:
            conc = run_concurrent(lambda s: _run_correction(model, host, s), samples, concurrency)
            agg = scorer.aggregate_correction([o.result for o in conc.outcomes if o.ok])
            result["correction"] = {"quality": {"wer": round(agg.wer, 4)}, "perf": conc.as_dict()}

        if "json" in tasks:
            conc = run_concurrent(lambda s: _run_json(model, host, s), samples, concurrency)
            agg = scorer.aggregate_json([o.result for o in conc.outcomes if o.ok])
            result["json"] = {
                "quality": {
                    "parse_rate": round(agg.parse_rate, 3),
                    "schema_valid_rate": round(agg.schema_valid_rate, 3),
                    "field_accuracy": round(agg.field_accuracy, 3),
                    "per_field": agg.per_field,
                },
                "perf": conc.as_dict(),
            }

    result["hardware"] = hw.summary().as_dict()
    return result


def evaluate_models(models, tasks, concurrency=3, host="http://localhost:11434", output_dir=None):
    results = [evaluate_model(m, tasks, concurrency, host) for m in models]
    if output_dir:
        from ..report import write_results

        write_results(results, output_dir, name="gemma_eval")
    return results
