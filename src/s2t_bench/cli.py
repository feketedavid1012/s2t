"""Command-line interface: `s2t-bench transcribe|benchmark|engines`."""
from __future__ import annotations

import argparse
import sys

from .benchmark.runner import format_table, run_benchmark
from .engines import available_engines, build_engine


def _cmd_engines(_: argparse.Namespace) -> int:
    print("\n".join(available_engines()))
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    kwargs: dict = {}
    if args.engine == "faster_whisper":
        kwargs = {
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
        }
        if args.domain:
            from .domain import build_initial_prompt, merge_glossary

            kwargs["initial_prompt"] = build_initial_prompt(
                merge_glossary(args.extra_terms)
            )
    engine = build_engine(args.engine, **kwargs)
    result = engine.transcribe(args.audio)
    print(f"[{result.engine}] rtf={result.rtf:.2f} "
          f"({result.latency_seconds:.2f}s / {result.audio_seconds:.2f}s)")
    if not args.correct:
        print(result.text)
        return 0

    from .correction import review_and_correct
    from .domain import merge_glossary

    print(f"[raw] {result.text}")
    review = review_and_correct(
        result.text, glossary=merge_glossary(args.extra_terms)
    )
    if review.status == "ok":
        print("[ok] no correction needed")
    else:
        print(f"[corrected] {review.text}")
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
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        initial_prompt=build_initial_prompt(glossary),
    )
    if args.realtime:
        if args.source != "mic":
            print("--realtime currently supports --source mic only")
            return 2
        from .realtime import render_live, stream_microphone_realtime

        print("Listening (real-time)... (Ctrl+C to stop)\n")
        try:
            render_live(stream_microphone_realtime(engine, interval_s=args.interval))
        except KeyboardInterrupt:
            print("\nStopped.")
        return 0

    if args.source == "mic":
        source = stream_microphone(
            engine,
            correct=args.correct,
            glossary=glossary,
            silence_ms=args.silence_ms,
        )
        print("Listening... (Ctrl+C to stop)")
    else:
        source = stream_wav_file(
            engine,
            args.source,
            correct=args.correct,
            glossary=glossary,
            silence_ms=args.silence_ms,
        )
    try:
        for chunk in source:
            tag = "corrected" if chunk.corrected else "final"
            print(f"[{tag}] {chunk.text}")
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import build_app

    app = build_app(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        interval_s=args.interval,
        extra_terms=args.extra_terms,
        with_adk=args.with_agent,
        silence_rms=args.silence_rms,
        vad=not args.no_vad,
    )
    where = "unified with ADK agent" if args.with_agent else "standalone"
    print(f"Serving ({where}) on http://{args.host}:{args.port}")
    print(f"  Web UI:      http://{args.host}:{args.port}/ui/")
    print(f"  Swagger UI:  http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_eval_whisper(args: argparse.Namespace) -> int:
    from .eval.whisper_eval import evaluate_models

    results = evaluate_models(
        model_sizes=args.models,
        manifest_path=args.manifest,
        concurrency=args.concurrency,
        compute_type=args.compute_type,
        limit=args.limit,
        output_dir=args.output,
    )
    for r in results:
        print(f"{r['model']:>10}  WER {r['accuracy']['avg_wer']:.3f}  "
              f"RTF {r['speed']['avg_rtf']:.2f}  "
              f"p95 {r['speed']['p95_latency_s']:.2f}s  "
              f"RAM {r['hardware']['process_rss_peak_mb']:.0f}MB  "
              f"CPU {r['hardware']['cpu_percent_peak']:.0f}%"
              f"{r['num_errors']}/{r['num_samples']} failed — e.g. {r['first_error']}")
    if args.output:
        print(f"\nSaved to {args.output}/")
    return 0


def _cmd_eval_gemma(args: argparse.Namespace) -> int:
    from .eval.gemma.gemma_eval import evaluate_models

    results = evaluate_models(
        models=args.models,
        tasks=args.tasks,
        concurrency=args.concurrency,
        host=args.host,
        output_dir=args.output,
    )
    for r in results:
        line = f"{r['model']:>14}"
        if "correction" in r:
            line += f"  corr-WER {r['correction']['quality']['wer']:.3f}"
        if "json" in r:
            q = r["json"]["quality"]
            line += f"  json-valid {q['schema_valid_rate']:.2f}  field-acc {q['field_accuracy']:.2f}"
        line += f"  RAM {r['hardware']['tracked_rss_peak_mb'].get('ollama', 0):.0f}MB"
        print(line)
    if args.output:
        print(f"\nSaved to {args.output}/")
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
    t.add_argument(
        "-m", "--model", default="base",
        help="Whisper size: tiny|base|small|medium|large-v3 (bigger = more accurate)",
    )
    t.add_argument("--device", default="auto", help="auto|cpu|cuda")
    t.add_argument(
        "--compute-type", default="default",
        help="default|int8|int8_float16|float16|float32 ('int8' is much faster on CPU)",
    )
    t.add_argument(
        "--domain", action="store_true",
        help="Bias the decoder toward the telecom glossary (XGS-PON, OLT, SKU...)",
    )
    t.add_argument(
        "--correct", action="store_true",
        help="Run the transcript through Gemini domain review/correction",
    )
    t.add_argument("--extra-terms", default="", help="Comma-separated extra vocab")
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
    s.add_argument("--device", default="auto", help="auto|cpu|cuda")
    s.add_argument(
        "--compute-type", default="default",
        help="default|int8|int8_float16|float16|float32 ('int8' is much faster on CPU)",
    )
    s.add_argument(
        "--silence-ms", type=int, default=600,
        help="Pause length that ends an utterance (lower = snappier, choppier)",
    )
    s.add_argument(
        "--correct",
        action="store_true",
        help="Run each finalized utterance through Gemini domain correction",
    )
    s.add_argument(
        "--realtime", action="store_true",
        help="Live word-by-word view (~2-3s latency) instead of per-utterance",
    )
    s.add_argument(
        "--interval", type=float, default=1.0,
        help="Realtime re-decode interval in seconds (lower = snappier, more CPU)",
    )
    s.add_argument("--extra-terms", default="", help="Comma-separated extra vocab")
    s.set_defaults(func=_cmd_stream)

    sv = sub.add_parser("serve", help="FastAPI server: REST + realtime WebSocket")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("-m", "--model", default="base", help="Whisper model size")
    sv.add_argument("--device", default="auto", help="auto|cpu|cuda")
    sv.add_argument("--compute-type", default="default")
    sv.add_argument("--interval", type=float, default=1.0)
    sv.add_argument("--extra-terms", default="")
    sv.add_argument(
        "--with-agent", action="store_true",
        help="Unify with the ADK agent (one process, one Swagger); needs [agent]",
    )
    sv.add_argument(
        "--silence-rms", type=float, default=0.005,
        help="Skip decoding buffers quieter than this RMS. Lower = catches quiet "
             "speech but risks silence hallucinations. 0 disables the gate.",
    )
    sv.add_argument(
        "--no-vad", action="store_true",
        help="Disable faster-whisper's VAD filter (use if quiet/short speech is "
             "being dropped as non-speech).",
    )
    sv.set_defaults(func=_cmd_serve)

    ew = sub.add_parser("eval-whisper", help="Evaluate Whisper sizes: accuracy + speed + hardware")
    ew.add_argument("manifest", help="JSONL manifest with audio + reference text")
    ew.add_argument("-m", "--models", nargs="+", default=["tiny", "base", "small"],
                    help="Model sizes to compare")
    ew.add_argument("-c", "--concurrency", type=int, default=3)
    ew.add_argument("--compute-type", default="int8")
    ew.add_argument("-n", "--limit", type=int, default=None)
    ew.add_argument("-o", "--output", default=None, help="Directory for results")
    ew.set_defaults(func=_cmd_eval_whisper)

    eg = sub.add_parser("eval-gemma", help="Evaluate local Gemma (Ollama): correction + JSON schema")
    eg.add_argument("-m", "--models", nargs="+", default=["gemma3:1b", "gemma3:4b"],
                    help="Ollama model tags (pull them first)")
    eg.add_argument("-t", "--tasks", nargs="+", default=["correction", "json"],
                    choices=["correction", "json"])
    eg.add_argument("-c", "--concurrency", type=int, default=3)
    eg.add_argument("--host", default="http://localhost:11434")
    eg.add_argument("-o", "--output", default=None, help="Directory for results")
    eg.set_defaults(func=_cmd_eval_gemma)
    return p


def _load_dotenv() -> None:
    """Load .env from the current dir or repo root, if python-dotenv is present.

    ADK loads .env automatically for `adk run`/`adk web`; this gives the plain
    CLI the same behaviour so keys live in one place.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
