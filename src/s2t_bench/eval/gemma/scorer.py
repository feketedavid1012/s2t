"""Scoring for the two Gemma tasks.

Correction task:  compare produced corrected text to the expected corrected text
                  (word error rate, reusing the benchmark metrics).
JSON task:        parse rate, schema-valid rate, and field-level accuracy against
                  the expected structured report.

Scoring notes:
- Components are matched on the *item name* first, with the SKU as partial credit.
  Matching on SKU alone punished models that correctly declined to invent a part
  number that never appeared in the source text.
- fault_reference is compared on digits only, so "1001", "Fault 1001" and
  "FLT-1001" all count as correct.
- Verbose free-text fields are scored by content-word recall against the expected
  text, not merely by being non-empty.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import schema

# Fields where we expect prose, scored by content overlap rather than exact match.
VERBOSE_FIELDS = ("reported_issue_summary", "rc_fault_story_verbose")
# Fields compared exactly (after normalization).
EXACT_FIELDS = ("rc_hl_category", "rc_ll_category")

_STOPWORDS = {
    "a", "an", "the", "was", "were", "is", "are", "be", "been", "being", "and",
    "or", "but", "of", "to", "in", "on", "at", "for", "with", "by", "from",
    "as", "that", "this", "it", "its", "had", "has", "have", "no", "not",
    "after", "before", "which", "into", "out", "up", "down", "then", "so",
}

# Weighting for component scoring: item name carries most of the credit,
# the SKU adds the rest. Tune here if your priorities differ.
ITEM_WEIGHT = 0.7
SKU_WEIGHT = 0.3


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


def _tokens(s) -> set[str]:
    return {t for t in re.split(r"[^\w]+", str(s).lower()) if t}


def _content_tokens(s) -> set[str]:
    return {t for t in _tokens(s) if t not in _STOPWORDS and len(t) > 1}


def _digits(s) -> str:
    """Digits only, so 'FLT-1001' / 'Fault 1001' / '1001' all compare equal."""
    return re.sub(r"\D", "", str(s))


def _items_equivalent(a, b) -> bool:
    """True if two component item names refer to the same thing.

    Uses token subset / overlap so 'ONT' matches 'ONT device' and 'splitter'
    matches '1:32 splitter'.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= 0.5


def _pair_score(exp_c: dict, got_c: dict) -> float:
    """Credit for one matched component: item name plus optional SKU bonus."""
    score = ITEM_WEIGHT
    exp_sku, got_sku = _norm(exp_c.get("sku", "")), _norm(got_c.get("sku", ""))
    if exp_sku and exp_sku == got_sku:
        score += SKU_WEIGHT
    elif not exp_sku and not got_sku:
        # Neither side claims a SKU - full credit, no fabrication expected.
        score += SKU_WEIGHT
    return score


def _components_match(expected: list, got: list) -> float:
    """F1 over component lists, matched by item name with SKU as partial credit."""
    exp = [c for c in (expected or []) if isinstance(c, dict)]
    got_l = [c for c in (got or []) if isinstance(c, dict)]
    if not exp and not got_l:
        return 1.0  # correctly reported no components
    if not exp or not got_l:
        return 0.0

    matched: set[int] = set()
    tp = 0.0
    for e in exp:
        for i, g in enumerate(got_l):
            if i in matched:
                continue
            if _items_equivalent(e.get("item"), g.get("item")):
                tp += _pair_score(e, g)
                matched.add(i)
                break

    prec = tp / len(got_l)
    rec = tp / len(exp)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _verbose_score(expected: str, got: str) -> float:
    """Content-word recall of the expected text within the produced text."""
    exp_t = _content_tokens(expected)
    got_t = _content_tokens(got)
    if not exp_t:
        return 1.0 if not got_t else 0.0
    if not got_t:
        return 0.0
    return len(exp_t & got_t) / len(exp_t)


def score_json(expected: dict, got) -> dict:
    """Field-level scoring of one produced report against the expected one."""
    parsed_ok = got is not None
    valid = schema.is_valid(got) if parsed_ok else False
    fields: dict[str, float] = {}

    if parsed_ok and isinstance(got, dict):
        for f in schema.STRING_FIELDS:
            if f == "fault_reference":
                e, g = _digits(expected.get(f)), _digits(got.get(f))
                fields[f] = 1.0 if e and e == g else 0.0
            elif f in EXACT_FIELDS:
                fields[f] = 1.0 if _norm(got.get(f)) == _norm(expected.get(f)) else 0.0
            else:  # verbose prose
                fields[f] = _verbose_score(expected.get(f, ""), got.get(f, ""))

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