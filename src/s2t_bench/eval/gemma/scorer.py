"""Scoring for the two Gemma tasks.

Correction task:  compare produced corrected text to the expected corrected text
                  (word error rate, reusing the benchmark metrics).
JSON task:        parse rate, schema-valid rate, and field-level accuracy against
                  the expected structured report.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import schema


def extract_json(text: str):
    """Best-effort parse of a JSON object from model output.

    Handles code fences and leading/trailing prose by grabbing the outermost
    {...} span. Returns the parsed object or None.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            return None
    return None


def _norm(s) -> str:
    return re.sub(r"[^\w]", "", str(s)).lower()


def _components_match(expected: list, got: list) -> float:
    """Set-overlap (F1-ish) on normalized SKUs of a component list."""
    exp = {_norm(c.get("sku", "")) for c in expected if isinstance(c, dict)}
    gotset = {_norm(c.get("sku", "")) for c in got if isinstance(c, dict)}
    exp.discard("")
    gotset.discard("")
    if not exp and not gotset:
        return 1.0
    if not exp or not gotset:
        return 0.0
    tp = len(exp & gotset)
    prec = tp / len(gotset)
    rec = tp / len(exp)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def score_json(expected: dict, got) -> dict:
    """Field-level scoring of one produced report against the expected one."""
    parsed_ok = got is not None
    valid = schema.is_valid(got) if parsed_ok else False
    fields: dict[str, float] = {}
    if parsed_ok and isinstance(got, dict):
        for f in schema.STRING_FIELDS:
            # substring/exact on normalized text is too strict for verbose fields;
            # score exact for short categorical fields, presence for verbose ones.
            if f in ("rc_hl_category", "rc_ll_category", "fault_reference"):
                fields[f] = 1.0 if _norm(got.get(f)) == _norm(expected.get(f)) else 0.0
            else:
                fields[f] = 1.0 if str(got.get(f, "")).strip() else 0.0
        for f in schema.BOOL_FIELDS:
            fields[f] = 1.0 if got.get(f) == expected.get(f) else 0.0
        for f in schema.COMPONENT_LIST_FIELDS:
            fields[f] = _components_match(expected.get(f, []), got.get(f, []))
    field_mean = sum(fields.values()) / len(fields) if fields else 0.0
    return {
        "parsed": parsed_ok,
        "schema_valid": valid,
        "field_accuracy": field_mean,
        "fields": fields,
    }


def score_correction(expected_text: str, got_text: str) -> dict:
    from ...benchmark.metrics import word_error_rate

    return {"wer": word_error_rate(expected_text, got_text or "")}


@dataclass
class TaskAggregate:
    n: int = 0
    parse_rate: float = 0.0
    schema_valid_rate: float = 0.0
    field_accuracy: float = 0.0
    wer: float = float("nan")
    per_field: dict = field(default_factory=dict)


def aggregate_json(scores: list[dict]) -> TaskAggregate:
    n = len(scores)
    if not n:
        return TaskAggregate()
    per_field: dict[str, list] = {}
    for s in scores:
        for f, v in s.get("fields", {}).items():
            per_field.setdefault(f, []).append(v)
    return TaskAggregate(
        n=n,
        parse_rate=sum(s["parsed"] for s in scores) / n,
        schema_valid_rate=sum(s["schema_valid"] for s in scores) / n,
        field_accuracy=sum(s["field_accuracy"] for s in scores) / n,
        per_field={f: round(sum(v) / len(v), 3) for f, v in per_field.items()},
    )


def aggregate_correction(scores: list[dict]) -> TaskAggregate:
    n = len(scores)
    if not n:
        return TaskAggregate()
    return TaskAggregate(n=n, wer=sum(s["wer"] for s in scores) / n)
