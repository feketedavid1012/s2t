"""Persist evaluation results as JSON plus a readable Markdown summary."""
from __future__ import annotations

import json
from pathlib import Path


def write_results(results: list[dict], output_dir: str | Path, name: str = "eval") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / f"{name}.md").write_text(_markdown(results, name), encoding="utf-8")


def _markdown(results: list[dict], name: str) -> str:
    lines = [f"# {name}", ""]
    for r in results:
        lines.append(f"## {r.get('model', '?')}")
        hw = r.get("hardware", {})
        lines.append(
            f"- hardware: peak RAM {hw.get('system_ram_used_peak_mb', '?')} MB, "
            f"proc RSS {hw.get('process_rss_peak_mb', '?')} MB, "
            f"CPU mean {hw.get('cpu_percent_mean', '?')}% / peak {hw.get('cpu_percent_peak', '?')}% "
            f"({hw.get('num_cpus', '?')} cores)"
        )
        if "accuracy" in r:  # whisper
            a, s = r["accuracy"], r["speed"]
            lines.append(f"- accuracy: WER {a['avg_wer']}, CER {a['avg_cer']}")
            lines.append(
                f"- speed: RTF {s['avg_rtf']}, p95 latency {s['p95_latency_s']}s, "
                f"throughput {s['throughput_per_s']}/s, errors {s['num_errors']}"
            )
        if "correction" in r:
            c = r["correction"]
            lines.append(
                f"- correction: WER {c['quality']['wer']}, "
                f"p95 latency {c['perf']['p95_latency_s']}s, "
                f"throughput {c['perf']['throughput_per_s']}/s"
            )
        if "json" in r:
            j = r["json"]
            q = j["quality"]
            lines.append(
                f"- json: parse {q['parse_rate']}, schema-valid {q['schema_valid_rate']}, "
                f"field-acc {q['field_accuracy']}, "
                f"p95 latency {j['perf']['p95_latency_s']}s"
            )
            if q.get("per_field"):
                lines.append(f"  - per-field: {q['per_field']}")
        lines.append("")
    return "\n".join(lines)
