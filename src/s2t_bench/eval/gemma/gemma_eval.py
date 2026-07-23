"""Evaluate local Gemma models (via Ollama) on two tasks under concurrency:

1. correction  - clean up a noisy telecom transcript
2. json        - convert the report context into the fault-report schema

Measures task quality (WER for correction; parse/schema-valid/field accuracy for
JSON), latency/throughput under N concurrent requests, and CPU/RAM (tracking the
ollama process for the model's memory footprint).

Per-sample outputs are retained so you can inspect expected vs actual and tell
whether a low score is a real model failure or a flaw in the ground truth.
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
    "casing, and spoken numbers. Do not add or remove meaning. Preserve the "
    "original wording exactly except where it is wrong; do not rephrase, "
    "summarise, or add commentary. Reply with only the corrected transcript."
)

JSON_SYSTEM = (
    "You convert a telecom fault report into a strict JSON object. Use only the "
    "information given. Do not invent part numbers or SKUs that do not appear in "
    "the text; use an empty string for sku if none is stated. "
    "Output JSON only, matching the required fields exactly."
)


def _json_prompt(context: str) -> str:
    fields = ", ".join(schema.ALL_FIELDS)
    return (
        f"Fault report context:\n{context}\n\n"
        f"Produce a JSON object with exactly these fields: {fields}. "
        f"rc_hl_category must be exactly one of: {', '.join(schema.RC_HL_CATEGORIES)}. "
        f"rc_ll_category must be exactly one of: {', '.join(schema.RC_LL_CATEGORIES)}. "
        "fault_reference is the bare number only, with no prefix. "
        "The *_flag fields are booleans. faulty_components and used_components are "
        'arrays of {"item": string, "sku": string}.'
    )


def _run_correction(model, host, sample):
    out = generate(model, sample.raw_text, system=CORRECTION_SYSTEM, host=host)
    score = scorer.score_correction(sample.corrected_text, out)
    score["id"] = sample.id
    score["expected"] = sample.corrected_text
    score["got"] = out
    return score


def _run_json(model, host, sample):
    out = generate(model, _json_prompt(sample.corrected_text),
                   system=JSON_SYSTEM, fmt=schema.json_schema(), host=host)
    parsed = scorer.extract_json(out)
    score = scorer.score_json(sample.expected, parsed)
    score["id"] = sample.id
    score["expected"] = sample.expected
    score["got"] = parsed
    return score


def _correction_per_sample(conc):
    return [
        {
            "id": o.result["id"],
            "wer": round(o.result["wer"], 3),
            "expected": o.result["expected"],
            "got": o.result["got"],
        }
        for o in conc.outcomes if o.ok
    ]


def _json_per_sample(conc):
    rows = []
    for o in conc.outcomes:
        if not o.ok:
            continue
        exp = o.result["expected"]
        got = o.result["got"] or {}
        rows.append({
            "id": o.result["id"],
            "field_accuracy": round(o.result["field_accuracy"], 3),
            "fields": o.result.get("fields", {}),
            "expected_ref": exp.get("fault_reference"),
            "got_ref": got.get("fault_reference"),
            "expected_faulty": exp.get("faulty_components"),
            "got_faulty": got.get("faulty_components"),
            "expected_used": exp.get("used_components"),
            "got_used": got.get("used_components"),
            "expected_hl": exp.get("rc_hl_category"),
            "got_hl": got.get("rc_hl_category"),
            "expected_ll": exp.get("rc_ll_category"),
            "got_ll": got.get("rc_ll_category"),
            "expected_correct_flag": exp.get("reported_issue_correct_flag"),
            "got_correct_flag": got.get("reported_issue_correct_flag"),
        })
    return rows


def evaluate_model(
    model: str,
    tasks: list[str],
    concurrency: int = 3,
    host: str = "http://localhost:11434",
    samples=None,
    generate_fn=None,
    limit: int | None = None,
) -> dict:
    """Evaluate one model. `generate_fn` is injectable for testing."""
    global generate
    if generate_fn is not None:
        generate = generate_fn  # type: ignore

    samples = samples or load_samples()
    if limit:
        samples = samples[:limit]

    result: dict = {"model": model, "concurrency": concurrency, "num_samples": len(samples)}

    with HardwareMonitor(track_process_names=["ollama"]) as hw:
        # Warm up (Ollama loads the model into RAM on first call).
        try:
            generate(model, "ping", host=host)
        except Exception:
            pass

        if "correction" in tasks:
            conc = run_concurrent(lambda s: _run_correction(model, host, s), samples, concurrency)
            ok = [o.result for o in conc.outcomes if o.ok]
            errs = [o.error for o in conc.outcomes if not o.ok]
            agg = scorer.aggregate_correction(ok)
            result["correction"] = {
                "quality": {"wer": round(agg.wer, 4)},
                "perf": conc.as_dict(),
                "num_errors": len(errs),
                "first_error": errs[0] if errs else None,
                "per_sample": _correction_per_sample(conc),
            }

        if "json" in tasks:
            conc = run_concurrent(lambda s: _run_json(model, host, s), samples, concurrency)
            ok = [o.result for o in conc.outcomes if o.ok]
            errs = [o.error for o in conc.outcomes if not o.ok]
            agg = scorer.aggregate_json(ok)
            result["json"] = {
                "quality": {
                    "parse_rate": round(agg.parse_rate, 3),
                    "schema_valid_rate": round(agg.schema_valid_rate, 3),
                    "field_accuracy": round(agg.field_accuracy, 3),
                    "per_field": agg.per_field,
                },
                "perf": conc.as_dict(),
                "num_errors": len(errs),
                "first_error": errs[0] if errs else None,
                "per_sample": _json_per_sample(conc),
            }

    result["hardware"] = hw.summary().as_dict()
    return result


def evaluate_models(models, tasks, concurrency=3, host="http://localhost:11434",
                    output_dir=None, limit=None):
    results = [evaluate_model(m, tasks, concurrency, host, limit=limit) for m in models]
    if output_dir:
        from ..report import write_results

        write_results(results, output_dir, name="gemma_eval")
    return results