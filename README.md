# s2t-bench

Pluggable **speech-to-text benchmarking** across Google, Whisper, and on-prem
models, with a **Google ADK** agent front-end for driving everything
conversationally.

Every backend implements one small interface (`TranscriptionEngine`), so adding
a model or comparing them head-to-head is uniform. The benchmark harness reports
**WER**, **CER**, and **RTF** (real-time factor) plus latency.

## Engines included

| Name             | Type            | Backend                                   | Extra     |
| ---------------- | --------------- | ----------------------------------------- | --------- |
| `google_cloud`   | Google (cloud)  | Cloud Speech-to-Text v2 (Chirp / Chirp 2) | `google`  |
| `gemini`         | Google (cloud)  | Gemini multimodal transcription           | `google`  |
| `whisper_api`    | Whisper (cloud) | OpenAI Whisper API                        | `whisper` |
| `faster_whisper` | **On-prem**     | faster-whisper (CTranslate2), fully local | `local`   |

Add your own (NeMo, wav2vec2, whisper.cpp, Vosk, ...) by subclassing
`TranscriptionEngine` and registering it — see _Extending_ below.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or pick extras: .[local], .[google], .[whisper], .[agent]
cp .env.example .env             # fill in the keys for the engines you'll use
```

Extras let you install only what you need — e.g. `pip install -e ".[local]"` for
an offline, on-prem-only setup with no cloud SDKs.

## CLI usage

```bash
# What's available?
s2t-bench engines

# Transcribe one file with the on-prem engine
poetry run python -m s2t_bench.cli transcribe sample-speech-1m.mp3 --engine faster_whisper
poetry run python -m s2t_bench.cli stream
s2t-bench transcribe sample-speech-1m.mp3 --engine faster_whisper

# Benchmark several engines over a labeled dataset
s2t-bench benchmark data/manifest.example.jsonl \
    --engines faster_whisper gemini whisper_api \
    --output results/
```

Example leaderboard (sorted by WER, lower is better; RTF < 1.0 = faster than real time):

```
engine                 WER     CER     RTF  errs    n
------------------------------------------------------
faster_whisper       0.041   0.018    0.32     0   50
whisper_api          0.048   0.021    0.55     0   50
gemini               0.062   0.030    0.88     0   50
```

## Dataset format

A JSONL manifest, one sample per line. `audio` is absolute or relative to the
manifest's directory; `text` is the ground-truth reference.

```json
{
  "id": "utt-001",
  "audio": "audio/utt-001.wav",
  "text": "the reference transcript"
}
```

## The ADK agent

The agent wraps the same tools so you can transcribe and benchmark in natural
language ("benchmark faster_whisper against gemini on data/manifest.example.jsonl").

```bash
pip install -e ".[agent]"
adk run src/s2t_bench/agent        # interactive CLI
adk web src/s2t_bench              # browser UI, then pick the "agent" package
```

Tools exposed to the agent (`src/s2t_bench/agent/tools.py`):
`list_engines`, `transcribe`, `run_benchmark_tool`. The agent model defaults to
`gemini-flash-latest` (override with `S2T_AGENT_MODEL`).

## Extending

```python
from s2t_bench.engines import TranscriptionEngine, register_engine

@register_engine
class MyEngine(TranscriptionEngine):
    name = "my_engine"

    def _transcribe(self, audio_path):
        text = ...              # call your model
        return text, {"raw": "..."}, "en"   # (text, raw_payload, language)
```

It's now usable everywhere: `build_engine("my_engine")`, the CLI `--engines`
flag, and the agent's `list_engines` / `run_benchmark_tool`.

## Layout

```
src/s2t_bench/
├── engines/        # TranscriptionEngine interface + one file per backend
├── benchmark/      # metrics (WER/CER), dataset loader, benchmark runner
├── agent/          # ADK root_agent + function tools
└── cli.py          # s2t-bench transcribe|benchmark|engines
```

## Notes

- Metrics use `jiwer` when installed, with a pure-Python fallback so the core
  package works with no heavy deps.
- Engine SDK imports are lazy: the package imports fine even if you only
  installed some extras. You only pay for the engines you actually run.
- Text normalization for WER/CER lives in `benchmark/metrics.py:normalize` and is
  intentionally simple and explicit so you can tune it per benchmark.
