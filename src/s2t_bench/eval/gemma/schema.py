"""The fault-report output schema (from the target spec) and a dependency-light
validator + scorer helpers.

Schema fields:
  fault_reference            str   - ticket / fault id
  reported_issue_summary     str   - short summary of the reported issue
  reported_issue_correct_flag bool - was the originally reported issue accurate
  valid_issue_flag           bool  - is this a valid/actionable issue
  rc_hl_category             str   - root-cause high-level category
  rc_ll_category             str   - root-cause low-level category
  rc_fault_story_verbose     str   - verbose root-cause narrative
  faulty_components          list of {item, sku}
  used_components            list of {item, sku}
"""
from __future__ import annotations

STRING_FIELDS = [
    "fault_reference",
    "reported_issue_summary",
    "rc_hl_category",
    "rc_ll_category",
    "rc_fault_story_verbose",
]
BOOL_FIELDS = ["reported_issue_correct_flag", "valid_issue_flag"]
COMPONENT_LIST_FIELDS = ["faulty_components", "used_components"]
ALL_FIELDS = STRING_FIELDS + BOOL_FIELDS + COMPONENT_LIST_FIELDS

# High-level root-cause categories used in scoring / prompting.
RC_HL_CATEGORIES = ["Hardware", "Software", "Network", "Physical Plant", "Configuration", "Power"]


def json_schema() -> dict:
    """JSON-schema dict to hand to the model (Ollama `format`)."""
    component = {
        "type": "object",
        "properties": {"item": {"type": "string"}, "sku": {"type": "string"}},
        "required": ["item", "sku"],
    }
    props = {f: {"type": "string"} for f in STRING_FIELDS}
    props.update({f: {"type": "boolean"} for f in BOOL_FIELDS})
    props.update({f: {"type": "array", "items": component} for f in COMPONENT_LIST_FIELDS})
    return {"type": "object", "properties": props, "required": ALL_FIELDS}


def validate(obj) -> list[str]:
    """Return a list of schema violations; empty list means valid."""
    errors: list[str] = []
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    for f in STRING_FIELDS:
        if f not in obj:
            errors.append(f"missing {f}")
        elif not isinstance(obj[f], str):
            errors.append(f"{f} not a string")
    for f in BOOL_FIELDS:
        if f not in obj:
            errors.append(f"missing {f}")
        elif not isinstance(obj[f], bool):
            errors.append(f"{f} not a boolean")
    for f in COMPONENT_LIST_FIELDS:
        if f not in obj:
            errors.append(f"missing {f}")
        elif not isinstance(obj[f], list):
            errors.append(f"{f} not a list")
        else:
            for i, c in enumerate(obj[f]):
                if not isinstance(c, dict) or "item" not in c or "sku" not in c:
                    errors.append(f"{f}[{i}] not a valid component (needs item, sku)")
    return errors


def is_valid(obj) -> bool:
    return not validate(obj)
