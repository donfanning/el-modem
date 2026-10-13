#!/usr/bin/env python3
"""Profile only a window of the V.34 replay, so cProfile overhead stays bounded.

    python3 prof_window.py <raw> <warmup_s> <window_s>
"""
import cProfile
import pstats
import struct
import sys

from v34_modem import V34Modem

FS = 8000
path = sys.argv[1]
warm = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
win = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

raw = open(path, "rb").read()
rx = struct.unpack(f"<{len(raw) // 2}h", raw)
blk = 160


def feed_range(m, lo, hi):
    for i in range(lo, hi - blk, blk):
        m.tx_samples(blk)
        m.feed_pcm(struct.pack(f"<{blk}h", *rx[i:i + blk]))


m = V34Modem(debug=True, with_v8=False)
feed_range(m, 0, int(warm * FS))
print(f"warmed to {warm}s: events={len(m.caller.events)}", flush=True)

lo, hi = int(warm * FS), int((warm + win) * FS)
pr = cProfile.Profile()
pr.enable()
feed_range(m, lo, hi)
pr.disable()
print(f"window {warm}..{warm + win}s done: events={len(m.caller.events)}", flush=True)
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(18)