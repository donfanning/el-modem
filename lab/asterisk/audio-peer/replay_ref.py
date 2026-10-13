#!/usr/bin/env python3
"""Replay a known-good third-party V.34 recording (or a captured PSTN call) through our caller and
stream the handshake events as they happen, so a real commercial answerer's phase-3/4 (MP/E/B1) can
be compared against what we do and against what Astound does.

Events are printed incrementally (the modem appends to ``caller.events`` as it runs) so progress is
visible immediately instead of only at the end.

    python3 -u replay_ref.py <raw> [seconds] [--quiet]
"""
from __future__ import annotations

import struct
import sys
import time

from v34_modem import V34Modem

FS = 8000


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    path = args[0]
    secs = float(args[1]) if len(args) > 1 else 0.0

    raw = open(path, "rb").read()
    rx = struct.unpack(f"<{len(raw) // 2}h", raw)
    limit = len(rx) if secs <= 0 else min(len(rx), int(secs * FS))
    m = V34Modem(debug=True, with_v8=False)
    c = m.caller
    blk = 160
    seen = 0
    t0 = time.time()
    print(f"replay {path}: {limit / FS:.2f}s of audio", flush=True)

    for i in range(0, limit - blk, blk):
        m.tx_samples(blk)
        m.feed_pcm(struct.pack(f"<{blk}h", *rx[i:i + blk]))
        ev = c.events
        while seen < len(ev):
            t, s = ev[seen]
            seen += 1
            if "data rx" in s and quiet:
                continue
            print(f"  {t / FS:6.2f}s {s}", flush=True)
        if i % (blk * 50) == 0:                       # progress every second of audio
            pos = i / FS
            el = time.time() - t0
            print(f"  .. {pos:6.2f}s audio  ({el:5.1f}s wall, {el / max(pos, 1e-9):4.1f}x rt)",
                  flush=True)

    print(f"connected={m.connected} bit_rate={m.bit_rate}", flush=True)
    print(f"done in {time.time() - t0:.1f}s wall for {limit / FS:.2f}s audio", flush=True)


if __name__ == "__main__":
    main()