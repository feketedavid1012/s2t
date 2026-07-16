from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Sample:
    id: str
    audio_path: str
    reference: str


def load_manifest(manifest_path: str | Path) -> list[Sample]:
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    samples: list[Sample] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            audio = row.get("audio") or row.get("audio_path")
            if not audio:
                raise ValueError(f"{manifest_path}:{lineno} missing 'audio' field")
            audio_path = Path(audio)
            if not audio_path.is_absolute():
                audio_path = base / audio_path
            samples.append(
                Sample(
                    id=str(row.get("id") or audio_path.name),
                    audio_path=str(audio_path),
                    reference=row.get("text") or row.get("reference") or "",
                )
            )
    return samples


def iter_manifest(manifest_path: str | Path) -> Iterator[Sample]:
    yield from load_manifest(manifest_path)
