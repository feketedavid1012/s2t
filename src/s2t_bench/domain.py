from __future__ import annotations

import json
from pathlib import Path

# Seed vocabulary for XGS-PON / GPON fibre-access service engineering.
# Replace / extend with your actual catalogue equipment names and SKUs.
DEFAULT_TELECOM_GLOSSARY: list[str] = [
    "XGS-PON", "XG-PON", "GPON", "PON", "NG-PON2",
    "OLT", "ONT", "ONU", "ODN", "ONT ID",
    "SKU", "MPN", "part number", "serial number", "S/N",
    "SFP", "SFP+", "XFP", "line card", "uplink card", "chassis", "shelf",
    "splitter", "1:32 splitter", "1:64 splitter", "patch panel", "ODF",
    "fibre", "drop cable", "feeder cable", "distribution cable", "pigtail",
    "connector", "SC/APC", "LC/APC", "attenuation", "dBm", "optical budget",
    "wavelength", "downstream", "upstream", "uplink", "downlink",
    "VLAN", "OMCI", "MAC address", "DHCP", "provisioning", "commissioning",
    "cabinet", "street cabinet", "DSLAM", "MDU", "MDF", "riser",
]


def merge_glossary(extra_terms: str | list[str] | None = None) -> list[str]:
    """Combine the default glossary with ad-hoc extra terms.

    `extra_terms` may be a comma-separated string or a list. Duplicates are
    removed while preserving order.
    """
    terms: list[str] = list(DEFAULT_TELECOM_GLOSSARY)
    if extra_terms:
        if isinstance(extra_terms, str):
            extra = [t.strip() for t in extra_terms.split(",") if t.strip()]
        else:
            extra = [str(t).strip() for t in extra_terms if str(t).strip()]
        terms.extend(extra)
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def load_glossary(path: str | Path) -> list[str]:
    """Load a JSON list of terms and merge with the defaults."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Glossary file must be a JSON list of strings")
    return merge_glossary([str(t) for t in data])


def build_initial_prompt(glossary: list[str] | None = None) -> str:
    """Build a faster-whisper `initial_prompt` that biases toward domain terms."""
    glossary = glossary or DEFAULT_TELECOM_GLOSSARY
    terms = ", ".join(glossary)
    return (
        "Telecom fibre-access service engineering notes covering XGS-PON and "
        f"GPON networks. Expect terms such as: {terms}."
    )
