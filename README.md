# s2t-bench

Pluggable **speech-to-text benchmarking** across Google, Whisper, and on-prem
models, with a **Google ADK** agent front-end for driving everything
conversationally.

Every backend implements one small interface (`TranscriptionEngine`), so adding
a model or comparing them head-to-head is uniform. The benchmark harness reports
**WER**, **CER**, and **RTF** (real-time factor) plus latency.

## Engines included

| Name             | Type            | Backend                                  | Extra      |
| ---------------- | --------------- | ---------------------------------------- | ---------- |
| `google_cloud`   | Google (cloud)  | Cloud Speech-to-Text v2 (Chirp / Chirp 2)| `google`   |
| `gemini`         | Google (cloud)  | Gemini multimodal transcription          | `google`   |
| `whisper_api`    | Whisper (cloud) | OpenAI Whisper API                       | `whisper`  |
| `faster_whisper` | **On-prem**     | faster-whisper (CTranslate2), fully local| `local`    |

Add your own (NeMo, wav2vec2, whisper.cpp, Vosk, ...) by subclassing
`TranscriptionEngine` and registering it — see *Extending* below.

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
s2t-bench transcribe sample.wav --engine faster_whisper

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
{"id": "utt-001", "audio": "audio/utt-001.wav", "text": "the reference transcript"}
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

## Field agent: transcribe → domain-correct

A second ADK agent (`src/s2t_bench/field_agent/`) runs the telecom field
pipeline: it transcribes audio with on-prem Whisper (decoder biased toward
XGS-PON / GPON / OLT / ONT / SKU vocabulary), then sends the transcript to Gemini
for a domain review. Gemini either flags it OK and changes nothing, or returns
the **full corrected transcript only** — no paraphrasing, no summaries.

```bash
pip install -e ".[local,google,agent]"
adk run src/s2t_bench/field_agent
# then: "transcribe and correct /path/to/site-notes.wav"
```

The correction step is also a standalone tool/command:

```bash
s2t-bench correct "we swapped the excess pon o l t and updated the sky you"
# -> XGS-PON / OLT / SKU normalised, full corrected line returned
```

Domain vocabulary lives in `src/s2t_bench/domain.py`. Drop your real catalogue /
SKU names into `DEFAULT_TELECOM_GLOSSARY` (or load a JSON list with
`load_glossary`) — those terms bias both Whisper decoding and the Gemini review.
Per-job terms can be passed ad hoc with `--extra-terms "SKU-123, MegaSplit-64"`.

## Live streaming (talk continuously to the mic)

faster-whisper is a batch transcriber, so "live" transcription works by
capturing the mic, using voice-activity detection to cut an utterance at each
pause, and transcribing that utterance immediately. Engineers can talk
continuously and see text appear per utterance.

```bash
pip install -e ".[local,stream]"
s2t-bench stream                       # live mic; prints each utterance
s2t-bench stream --correct             # also Gemini-correct each utterance (needs [google])
s2t-bench stream --source clip.wav     # simulate streaming from a 16 kHz mono WAV
```

The mic path and the file path drive the exact same VAD + transcription code
(`src/s2t_bench/streaming.py`), so you can test the pipeline in CI without audio
hardware. Tune responsiveness with the segmenter's `silence_ms` (how long a pause
ends an utterance) and `max_utterance_ms` (hard cap for very long speech).

> The `stream` extra installs `sounddevice` (needs the system PortAudio library —
> `apt install libportaudio2` on Debian/Ubuntu) plus `numpy`. `webrtcvad` is
> optional: if it can't load (e.g. Python 3.12+ venvs lack `pkg_resources` unless
> `setuptools` is installed), streaming automatically falls back to a pure-NumPy
> energy VAD. Install `setuptools` to enable the more noise-robust webrtcvad.

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
