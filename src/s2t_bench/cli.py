"""Command-line interface: `s2t-bench transcribe|benchmark|engines`."""
from __future__ import annotations

import argparse
import sys

from .benchmark_agent.benchmark_tools.runner import format_table, run_benchmark
from .engines import available_engines, build_engine


def _cmd_engines(_: argparse.Namespace) -> int:
    print("\n".join(available_engines()))
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    engine = build_engine(args.engine)
    result = engine.transcribe(args.audio)
    print(f"[{result.engine}] rtf={result.rtf:.2f} "
          f"({result.latency_seconds:.2f}s / {result.audio_seconds:.2f}s)")
    print(result.text)
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    summaries = run_benchmark(
        manifest_path=args.manifest,
        engine_specs=args.engines,
        output_dir=args.output,
        limit=args.limit,
    )
    print(format_table(summaries))
    if args.output:
        print(f"\nReports written to {args.output}/")
    return 0


def _cmd_correct(args: argparse.Namespace) -> int:
    from .correction import review_and_correct
    from .domain import merge_glossary

    result = review_and_correct(args.text, glossary=merge_glossary(args.extra_terms))
    if result.status == "ok":
        print("[ok] transcript looks correct")
    else:
        print(result.text)
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    from .domain import build_initial_prompt, merge_glossary
    from .engines.faster_whisper_local import FasterWhisperEngine
    from .streaming import stream_microphone, stream_wav_file

    glossary = merge_glossary(args.extra_terms)
    engine = FasterWhisperEngine(
        model=args.model, initial_prompt=build_initial_prompt(glossary)
    )
    if args.source == "mic":
        source = stream_microphone(engine, correct=args.correct, glossary=glossary)
        print("Listening... (Ctrl+C to stop)")
    else:
        source = stream_wav_file(
            engine, args.source, correct=args.correct, glossary=glossary
        )
    try:
        for chunk in source:
            tag = "corrected" if chunk.corrected else "final"
            print(f"[{tag}] {chunk.text}")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="s2t-bench", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("engines", help="List available engines").set_defaults(
        func=_cmd_engines
    )

    t = sub.add_parser("transcribe", help="Transcribe one audio file")
    t.add_argument("audio")
    t.add_argument("-e", "--engine", default="faster_whisper")
    t.set_defaults(func=_cmd_transcribe)

    b = sub.add_parser("benchmark", help="Benchmark engines over a manifest")
    b.add_argument("manifest")
    b.add_argument("-e", "--engines", nargs="+", required=True)
    b.add_argument("-o", "--output", default=None, help="Directory for reports")
    b.add_argument("-n", "--limit", type=int, default=None)
    b.set_defaults(func=_cmd_benchmark)

    c = sub.add_parser("correct", help="Domain-review/correct a transcript with Gemini")
    c.add_argument("text")
    c.add_argument("--extra-terms", default="", help="Comma-separated extra vocab")
    c.set_defaults(func=_cmd_correct)

    s = sub.add_parser("stream", help="Live on-the-fly transcription (mic or wav)")
    s.add_argument(
        "--source",
        default="mic",
        help="'mic' for microphone, or a path to a 16 kHz mono WAV to simulate",
    )
    s.add_argument("-m", "--model", default="base", help="Whisper model size")
    s.add_argument(
        "--correct",
        action="store_true",
        help="Run each finalized utterance through Gemini domain correction",
    )
    s.add_argument("--extra-terms", default="", help="Comma-separated extra vocab")
    s.set_defaults(func=_cmd_stream)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
