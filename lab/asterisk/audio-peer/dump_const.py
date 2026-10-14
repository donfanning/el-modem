#!/usr/bin/env python3
"""Dump the equalized phase-3 symbols (complex) so the constellation can be examined offline.

After the answerer's TRN the receiver sees *something*; whether that is a 4-point signal (J) or a
larger QAM constellation (phase 4) is the question that decides where the handshake actually is.
Slicing with ``decide4`` hides this, so the raw equalizer output is saved for later analysis.

    python3 -u dump_const.py <raw> <seconds> <out.npy>
"""
from __future__ import annotations

import struct
import sys

import numpy as np

import v34_rx
from v34_modem import V34Modem

FS = 8000


def main() -> None:
    raw_path, secs, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    raw = open(raw_path, "rb").read()
    rx = struct.unpack(f"<{len(raw) // 2}h", raw)
    limit = min(len(rx), int(secs * FS))
    m = V34Modem(debug=True, with_v8=False)
    blk = 160
    ys: list[complex] = []
    saved = {"done": False}

    orig_step = v34_rx.SymbolEq.step

    def step(self, desired=None, decide=None):
        y = orig_step(self, desired=desired, decide=decide)
        ys.append(y)
        return y

    v34_rx.SymbolEq.step = step

    for i in range(0, limit - blk, blk):
        m.tx_samples(blk)
        m.feed_pcm(struct.pack(f"<{blk}h", *rx[i:i + blk]))
        c = m.caller
        if not saved["done"] and getattr(c, "_p3_rcv", None) is not None:
            print(f"  Phase3Receiver active at {i / FS:.2f}s; {len(ys)} symbols so far", flush=True)

    arr = np.array(ys, dtype=np.complex128)
    np.save(out, arr)
    rcv = getattr(m.caller, "_p3_rcv", None)
    print(f"saved {arr.size} equalized symbols to {out}", flush=True)
    if rcv is not None:
        print(f"  _end_est={rcv._end_est}  idx={len(rcv.idx)}  k_trn={rcv.k_trn}  slips={rcv.eq.slips}",
              flush=True)

    # Constellation size over time: a 4-point signal gives 4 tight magnitude clusters, QAM gives a
    # continuous spread. This is what distinguishes "J (4-point)" from "already in phase 4".
    if arr.size:
        print("\n-- magnitude spread per 512 symbols (4-point => 1 tight cluster) --", flush=True)
        for k in range(0, arr.size - 512, 512):
            w = arr[k:k + 512]
            mag = np.abs(w)
            print(f"  sym {k:6d}  |y| p10={np.percentile(mag, 10):6.1f} p50={np.percentile(mag, 50):6.1f} "
                  f"p90={np.percentile(mag, 90):6.1f}  ratio p90/p10={np.percentile(mag, 90) / max(np.percentile(mag, 10), 1e-9):5.2f}",
                  flush=True)


if __name__ == "__main__":
    main()