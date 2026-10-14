#!/usr/bin/env python3
"""Track a single tone's amplitude and phase over time (complex demodulation), so the V.34 phase-2
A/B reversal structure of a recording can be checked without running the full modem.

    python3 tone_track.py <raw> <freq> [seconds]

Prints one line per ~20 ms with the tone amplitude and the wrapped phase, plus a summary of the
phase-reversal times (where phase jumps by roughly +/-180 degrees while amplitude stays high).
"""
from __future__ import annotations

import struct
import sys

import numpy as np

FS = 8000


def main() -> None:
    path = sys.argv[1]
    f0 = float(sys.argv[2])
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    raw = open(path, "rb").read()
    x = np.array(struct.unpack(f"<{len(raw) // 2}h", raw), dtype=float)
    if secs > 0:
        x = x[:int(secs * FS)]

    win = int(0.020 * FS)          # 20 ms analysis window
    hop = win
    n = np.arange(win)
    ref = np.exp(-2j * np.pi * f0 * n / FS)
    amp, ph, t_axis = [], [], []
    for start in range(0, len(x) - win, hop):
        seg = x[start:start + win]
        c = np.sum(seg * ref) / win
        amp.append(abs(c))
        ph.append(np.angle(c))
        t_axis.append((start + win / 2) / FS)
    amp = np.array(amp)
    ph = np.unwrap(np.array(ph))
    t_axis = np.array(t_axis)

    hi = float(np.percentile(amp, 90))
    thr = max(hi * 0.35, 200.0)
    print(f"{path}  tone={f0:.0f}Hz  amp p90={hi:.0f} threshold={thr:.0f}")
    print(f"  windows={len(amp)}  active={int((amp > thr).sum())}")

    # phase reversals: active, and |dphase| near pi
    d = np.diff(ph)
    rev = []
    for i in range(1, len(d)):
        if amp[i] > thr and amp[i - 1] > thr and abs(d[i]) > 1.6:
            rev.append((t_axis[i], np.degrees(d[i])))
    print(f"  phase reversals: {len(rev)}")
    for t, dg in rev:
        print(f"    {t:6.3f}s  dphase={dg:+7.1f}deg")

    # amplitude on/off edges (for finding B segments and gaps)
    on = amp > thr
    edges = []
    prev = on[0]
    for i, v in enumerate(on):
        if v != prev:
            edges.append((t_axis[i], "ON" if v else "OFF"))
            prev = v
    print(f"  amplitude edges: {len(edges)}")
    for t, s in edges[:40]:
        print(f"    {t:6.3f}s {s}")


if __name__ == "__main__":
    main()