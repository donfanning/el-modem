#!/usr/bin/env python3
"""Probe phase 3 on a recording: where does the answerer's TRN really end, and is J anywhere?

Phase3Receiver only looks for J within +-12 symbols of ``_end_est``, a first guess latched from the
first burst of TRN mismatches. If that guess is off, J is never found and the receiver just stalls.
This dumps the symbol stream and searches the whole thing, so a missed J is distinguishable from an
absent one.

    python3 -u probe_p3.py <raw> [seconds] [--save <idx.npy>]

With ``--save`` the recovered symbol stream is written out, and passing a saved ``.npy`` as ``<raw>``
re-runs only the analysis on it (fast iteration on the bit stream without replaying the audio again).
"""
from __future__ import annotations

import struct
import sys
import time

import numpy as np

import v34_rx
import v34_tx as tx
from v34_modem import V34Modem

FS = 8000
IDX_CACHE = "/tmp/p3_idx.npy"


def mismatch_profile(idx: list[int], role: str) -> list[tuple[int, float]]:
    """Sliding 32-symbol mismatch fraction against the expected TRN sequence (fraction, not count)."""
    sc = tx.Scrambler(role)
    exp = []
    for _ in idx:
        i1, i2 = sc.step(1), sc.step(1)
        exp.append(2 * i2 + i1)
    exp = np.array(exp, dtype=np.int64)
    got = np.array(idx, dtype=np.int64)
    bad = (got != exp).astype(float)
    c = np.concatenate([[0.0], np.cumsum(bad)])
    out = []
    for i in range(0, max(1, len(idx) - 32), 32):
        out.append((i, (c[i + 32] - c[i]) / 32.0))
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0]
    secs = float(args[1]) if len(args) > 1 and not args[1].endswith(".npy") else 0.0
    save = None
    if "--save" in sys.argv:
        save = sys.argv[sys.argv.index("--save") + 1]

    t0 = time.time()
    if path.endswith(".npy"):
        # cached symbol stream: skip the replay entirely and re-analyse
        idx = np.load(path).tolist()
        role = "answer"
        _end_est = None
        print(f"loaded cached symbol stream {path}: {len(idx)} symbols", flush=True)
    else:
        raw = open(path, "rb").read()
        rx = struct.unpack(f"<{len(raw) // 2}h", raw)
        limit = len(rx) if secs <= 0 else min(len(rx), int(secs * FS))
        m = V34Modem(debug=True, with_v8=False)
        blk = 160
        print(f"probe {path}: {limit / FS:.2f}s of audio", flush=True)

        for i in range(0, limit - blk, blk):
            m.tx_samples(blk)
            m.feed_pcm(struct.pack(f"<{blk}h", *rx[i:i + blk]))

        c = m.caller
        print(f"\n-- events ({len(c.events)}) --", flush=True)
        for t, s in c.events:
            print(f"  {t / FS:6.2f}s {s}", flush=True)

        rcv = getattr(c, "_p3_rcv", None)
        if rcv is None:
            print("\nno Phase3Receiver was created (never reached equalizer training)", flush=True)
            return
        idx = list(rcv.idx)
        role = rcv.role
        _end_est = rcv._end_est
        print(f"\n-- phase 3 state --\n  symbols {len(idx)}  eq.k {rcv.eq.k}  k_trn {rcv.k_trn}  "
              f"slips {rcv.eq.slips}\n  _end_est {_end_est}  j {rcv.j}", flush=True)
        if save:
            np.save(save, np.array(idx, dtype=np.int64))
            print(f"  saved symbol stream to {save}", flush=True)
    if len(idx) < 64:
        print("  too few symbols to analyse", flush=True)
        return

    rcv = type("R", (), {"role": role, "_end_est": _end_est})()
    print("\n-- TRN mismatch fraction per 32 symbols --", flush=True)
    prof = mismatch_profile(idx, role)
    # print only transitions plus a coarse periodic sample so the output stays readable
    prev = None
    for i, f in prof:
        bucket = "TRN" if f < 0.25 else ("mixed" if f < 0.75 else "data")
        if bucket != prev or i % 256 == 0:
            print(f"  {i:6d} {f:4.2f} {bucket}", flush=True)
            prev = bucket

    print("\n-- search for J across the whole symbol stream --", flush=True)
    arr = np.array(idx, dtype=np.int64)
    pat = np.array(tx.J_4PT * 3, dtype=np.int64)
    taps = sorted({v34_rx.descramble_tap(role), v34_rx.descramble_tap("call"), 0, 5, 18})
    # Search every plausible interpretation at once: a J that is absent, present-but-not-descrambled,
    # or present under the other role's polynomial all look like "no J" to the production detector.
    variants = {
        "descrambled": lambda n: v34_rx._descramble_at(arr, role, n, pat.size),
        "raw-differential": lambda n: v34_rx._interleave(
            np.concatenate(([arr[0]], (arr[n - 12:n + 24] - arr[n - 13:n + 23]) % 4))),
        "raw-absolute": lambda n: v34_rx._interleave(arr[max(n - 12, 0):n + 24]),
    }
    for vname, vfn in variants.items():
        hits = []
        for n in range(24, len(arr) - 24):
            seg = vfn(n)
            seg = seg[-pat.size:] if seg.size >= pat.size else seg
            if seg.size == pat.size and np.array_equal(seg, pat):
                hits.append(n)
        print(f"  {vname:16s}: {len(hits)} hit(s){(' at ' + str(hits[:20])) if hits else ''}", flush=True)
        # Try descrambling the raw variants under every tap: catches "J present, wrong polynomial".
        if vname != "descrambled":
            for tap in taps:
                seg = vfn(40).astype(np.int64)
                d = v34_rx._descramble_np(seg, tap)[-pat.size:]
                if d.size == pat.size and np.array_equal(d, pat):
                    print(f"  HIT: J in {vname} after descramble(tap={tap})", flush=True)

    # How close does the *descrambled* stream ever come to J? 24/48 would already be remarkable for random data.
    best = (0, None)
    for n in range(24, min(len(arr), 12000) - 24):
        seg = v34_rx._descramble_at(arr, role, n, pat.size)
        if seg.size < pat.size:
            continue
        agree = int((seg[-pat.size:] == pat).sum())
        if agree > best[0]:
            best = (agree, n)
    print(f"  best descrambled J agreement: {best[0]}/{pat.size} bits at symbol {best[1]}"
          f"  (random data averages ~{pat.size // 2})", flush=True)

    # What do the descrambled bits look like around the TRN end? A real J run shows as a repeated
    # 16-bit pattern; random data does not. Show the longest 16-bit run that repeats.
    print("\n-- descrambled bits after each candidate end (looking for repeating structure) --", flush=True)
    for n in ([rcv._end_est] if rcv._end_est is not None else []) + list(range(300, min(len(arr) - 20, 2400), 100)):
        seg = v34_rx._descramble_at(arr, role, n, 96)
        if seg.size == 0:
            continue
        s = "".join(map(str, seg.tolist()))
        best_run = 0
        run = 1
        for k in range(16, len(s), 16):
            if s[k:k + 16] == s[k - 16:k]:
                run += 1
            else:
                run = 1
            best_run = max(best_run, run)
        tag = " <-- _end_est" if n == rcv._end_est else ""
        print(f"  n={n:5d} first16={s[:16]} best16run={best_run} ones={s.count('1'):3d}/96{tag}", flush=True)

    print(f"\ndone in {time.time() - t0:.1f}s wall", flush=True)


if __name__ == "__main__":
    main()