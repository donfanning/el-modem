#!/usr/bin/env python3
"""Per-second envelope + dominant-tone report for a mono 8 kHz int16-LE raw file.

Used to sanity-check captured/third-party audio before handing it to a modem replay: confirms the
signal is present, at a sensible level, and sitting on the expected carrier band. Compare against
an already-working capture to catch format/split mistakes early.

    python3 analyze_envelope.py <raw> [--top] [--from 0] [--dur 0]
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np

FS = 8000


def dominant_freq(x: np.ndarray, fmin: float = 400.0, fmax: float = 3400.0) -> float:
    """Strongest tone in [fmin, fmax], via a windowed FFT (QAM bursts are broadband, so this
    reports the carrier band rather than a sharp tone)."""
    n = len(x)
    if n < 64:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any():
        return 0.0
    return float(freqs[band][int(np.argmax(spec[band]))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", action="store_true", help="also print dominant tone per second")
    ap.add_argument("--from", dest="start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=0.0)
    args = ap.parse_args()

    raw = open(args.path, "rb").read()
    x = np.array(struct.unpack(f"<{len(raw) // 2}h", raw), dtype=float)
    a = int(args.start * FS)
    b = len(x) if args.dur <= 0 else min(len(x), int((args.start + args.dur) * FS))
    x = x[a:b]
    print(f"{args.path}: {len(x) / FS:.2f}s  peak={np.max(np.abs(x)):.0f}")
    for t in range(int(len(x) // FS)):
        seg = x[t * FS:(t + 1) * FS]
        rms = float(np.sqrt(np.mean(seg * seg)))
        crest = float(np.max(np.abs(seg)) / (rms + 1e-9)) if rms > 1 else 0.0
        line = f"  {t + args.start:5.1f}s rms={rms:7.1f} crest={crest:4.2f}"
        if args.top:
            line += f" tone={dominant_freq(seg):6.0f}Hz"
        print(line)


if __name__ == "__main__":
    sys.exit(main())