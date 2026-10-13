#!/usr/bin/env python3
"""Convert a stereo third-party V.34 recording into per-direction mono int16-LE raw fixtures.

Left channel is the originating (caller) modem and right channel is the answering modem, so the
answerer channel is what our *caller* receives and the caller channel is what our *answerer* receives.

    python3 make_stereo_fixtures.py <stereo.wav> <out-dir> <name>
"""
from __future__ import annotations

import os
import struct
import sys
import wave


def main() -> None:
    src, outdir, name = sys.argv[1], sys.argv[2], sys.argv[3]
    w = wave.open(src)
    if w.getnchannels() != 2:
        raise SystemExit(f"expected stereo, got {w.getnchannels()} channel(s)")
    if w.getsampwidth() != 2 or w.getframerate() != 8000:
        raise SystemExit(f"expected 8 kHz 16-bit, got {w.getframerate()} Hz "
                         f"{w.getsampwidth() * 8}-bit")
    n = w.getnframes()
    s = struct.unpack(f"<{n * 2}h", w.readframes(n))
    w.close()
    os.makedirs(outdir, exist_ok=True)
    for side, chan in (("caller", 0), ("answerer", 1)):
        path = os.path.join(outdir, f"{name}.{side}.raw")
        with open(path, "wb") as fh:
            fh.write(struct.pack(f"<{n}h", *s[chan::2]))
        print(f"{path}: {n / 8000:.2f}s")


if __name__ == "__main__":
    main()