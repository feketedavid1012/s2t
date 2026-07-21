#!/usr/bin/env python3
"""Augment eval audio with numpy + soundfile only (no librosa/numba/llvmlite).

Encodes the augmentation name in each output filename and in the manifest's
`variant` field, so the eval can be broken down per augmentation.

    poetry run python augment_dataset.py \
        --manifest data/manifest.jsonl \
        --out-audio data/audio_aug \
        --out-manifest data/manifest_aug.jsonl

Deps: numpy, soundfile  (both already installed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
rng = np.random.default_rng(0)


# --- primitive transforms (pure numpy) -------------------------------------

def resample_linear(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample by `factor` via linear interpolation (no scipy)."""
    n_out = int(round(len(x) / factor))
    if n_out < 1:
        return x
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

def time_stretch(x, rate):
    """Change speed AND keep pitch-ish by resample then play at original rate.
    Simpler approach: change duration only (affects pitch slightly). For a
    speed change that Whisper sees as faster/slower speech, resample is fine."""
    return resample_linear(x, rate)

def pitch_shift(x, semitones):
    """Shift pitch by resampling then restoring length (naive but dependency-free).
    Resample by 2^(st/12), then stretch back to original length."""
    factor = 2 ** (semitones / 12.0)
    up = resample_linear(x, 1.0 / factor)       # change pitch+length
    return resample_linear(up, len(up) / len(x))  # restore original length

def add_noise_snr(x, snr_db):
    """Add white noise at a target signal-to-noise ratio."""
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=len(x)).astype(np.float32)
    return x + noise

def gain_db(x, db):
    return (x * (10 ** (db / 20))).astype(np.float32)

def clip_distort(x, drive=3.0):
    """Mild soft-clipping to mimic handset/mic overdrive."""
    return np.tanh(x * drive).astype(np.float32) / np.tanh(drive)


# --- named variants (single or combined) -----------------------------------
# Each is a list of (name, fn) steps; the joined names go into the filename.

VARIANTS = {
    "fast":       [("fast", lambda x: time_stretch(x, 1.25))],
    "slow":       [("slow", lambda x: time_stretch(x, 0.80))],
    "pitch_up":   [("pitchup", lambda x: pitch_shift(x, 3))],
    "pitch_down": [("pitchdn", lambda x: pitch_shift(x, -3))],
    "snr20":      [("snr20", lambda x: add_noise_snr(x, 20))],
    "snr10":      [("snr10", lambda x: add_noise_snr(x, 10))],
    "snr5":       [("snr5",  lambda x: add_noise_snr(x, 5))],
    "quiet":      [("quiet", lambda x: gain_db(x, -12))],
    "distort":    [("distort", lambda x: clip_distort(x, 3.0))],
    # combinations -> names joined with '+'
    "field_sim":  [("fast", lambda x: time_stretch(x, 1.1)),
                   ("pitchdn", lambda x: pitch_shift(x, -2)),
                   ("snr12", lambda x: add_noise_snr(x, 12))],
    "noisy_fast": [("fast", lambda x: time_stretch(x, 1.2)),
                   ("snr10", lambda x: add_noise_snr(x, 10))],
}


def load_mono_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:                       # simple linear resample to 16k
        audio = resample_linear(audio, sr / SR)
    return audio


def apply_variant(x, steps):
    tag_parts = []
    for tag, fn in steps:
        x = fn(x)
        tag_parts.append(tag)
    # normalize to avoid clipping after transforms
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 1.0:
        x = x / peak
    return x.astype(np.float32), "+".join(tag_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out-audio", default="data/audio_aug")
    ap.add_argument("--out-manifest", default="data/manifest_aug.jsonl")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="subset of variant keys; default = all")
    ap.add_argument("--include-clean", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    base_dir = manifest_path.parent
    out_audio = Path(args.out_audio); out_audio.mkdir(parents=True, exist_ok=True)

    variants = VARIANTS if not args.variants else {k: VARIANTS[k] for k in args.variants}

    rows_in = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    out_rows = []

    for row in rows_in:
        audio_rel = row.get("audio") or row.get("audio_path")
        src = Path(audio_rel)
        if not src.is_absolute():
            src = base_dir / src
        if not src.exists():
            print(f"skip missing {src}", file=sys.stderr); continue

        audio = load_mono_16k(src)
        stem = Path(audio_rel).stem
        text = row.get("text") or row.get("reference", "")

        if args.include_clean:
            out_name = f"{stem}__clean.wav"
            sf.write(str(out_audio / out_name), audio, SR)
            out_rows.append({"id": f"{row.get('id', stem)}__clean",
                             "audio": str((out_audio / out_name).relative_to(base_dir)),
                             "text": text, "variant": "clean"})

        for _, steps in variants.items():
            aug, tag = apply_variant(audio, steps)
            out_name = f"{stem}__{tag}.wav"          # <-- augmentation in filename
            sf.write(str(out_audio / out_name), aug, SR)
            out_rows.append({
                "id": f"{row.get('id', stem)}__{tag}",
                "audio": str((out_audio / out_name).relative_to(base_dir)),
                "text": text,
                "variant": tag,                       # <-- augmentation in manifest
            })

    Path(args.out_manifest).write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n", encoding="utf-8")
    print(f"wrote {len(out_rows)} variants from {len(rows_in)} clips -> {args.out_manifest}")


if __name__ == "__main__":
    main()