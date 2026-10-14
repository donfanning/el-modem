#!/usr/bin/env python3
"""Fit the J pattern to a recovered phase-3 symbol stream.

``decode_j`` requires a *perfect* 48-bit match of J_4PT repeated. On a real modem's J the recovered
bits carry a few errors (the equalizer is still settling at the TRN->J boundary), so an exact test
rejects a J that is plainly there. This scores every split point instead, so a near-miss is
distinguishable from an absent J, and reports where the match starts and how long it runs.

    python3 -u jfit.py <idx.npy> [role]
"""
from __future__ import annotations

import sys

import numpy as np

import v34_rx
import v34_tx as tx

REPS = 3


def agreement(idx: np.ndarray, role: str, pat: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Bits of ``pat`` matched at each candidate TRN-end split point in [lo, hi)."""
    out = np.zeros(max(0, hi - lo), dtype=np.int64)
    for k, n in enumerate(range(lo, hi)):
        seg = v34_rx._descramble_at(idx, role, n, pat.size)
        if seg.size == pat.size:
            out[k] = int((seg == pat).sum())
    return out


def main() -> None:
    path = sys.argv[1]
    role = sys.argv[2] if len(sys.argv) > 2 else "answer"
    idx = np.load(path).astype(np.int64)
    pat = np.array(tx.J_4PT * REPS, dtype=np.int64)
    nb = pat.size
    print(f"{path}: {idx.size} symbols, role={role}, J_4PT*3 = {nb} bits", flush=True)
    print(f"  J_4PT  = {''.join(map(str, tx.J_4PT))}", flush=True)
    print(f"  J_PRIME= {''.join(map(str, tx.J_PRIME))}", flush=True)

    hi = min(idx.size - 8, 20000)
    a = agreement(idx, role, pat, 24, hi)
    order = np.argsort(-a)
    print(f"\n-- best split points (random data averages {nb // 2}, ~5.5 sigma at 43) --", flush=True)
    seen: list[int] = []
    for k in order[:400]:
        n = 24 + int(k)
        if any(abs(n - s) < 8 for s in seen):
            continue
        seen.append(n)
        if len(seen) >= 12:
            break
        print(f"  n={n:6d}  {a[k]:2d}/{nb} bits", flush=True)

    # Characterise the winning region: agreement as a function of split point around the best hit.
    best = int(order[0])
    n0 = 24 + best
    print(f"\n-- agreement vs split point around the best hit (n={n0}) --", flush=True)
    lo = max(24, n0 - 200)
    w = agreement(idx, role, pat, lo, min(hi, n0 + 900))
    for k in range(0, w.size, 20):
        n = lo + k
        bar = "#" * int(w[k])
        print(f"  n={n:6d} {w[k]:2d}/{nb} {bar}", flush=True)

    # How long does the high-agreement region last? That is the J run length.
    good = np.flatnonzero(w >= nb - 8)                  # within 8 bits of perfect
    if good.size:
        print(f"\n  split points within 8 bits of perfect: {lo + good[0]}..{lo + good[-1]} "
              f"({good.size} of {w.size})", flush=True)
    else:
        thr = int(np.percentile(w, 99.9))
        print(f"\n  no split point within 8 bits of perfect; 99.9th percentile = {thr}/{nb}", flush=True)

    # Sustained check: for the winning split, how many consecutive symbols keep agreeing?
    print(f"\n-- sustained J at n={n0} (per-symbol agreement over the following 400 symbols) --",
          flush=True)
    run = 0
    best_run = 0
    for s in range(0, 400):
        n = n0 + s
        seg = v34_rx._descramble_at(idx, role, n, 16)
        at = (2 * np.arange(8)) % 16
        want = np.array([tx.J_4PT[(2 * s + i) % 16] for i in range(16)], dtype=np.int64)
        if seg.size == 16:
            agree = int((seg == want).sum())
            run = run + 1 if agree >= 12 else 0
            best_run = max(best_run, run)
            if s < 24 or s % 50 == 0:
                print(f"    +{s:4d} symbols: {agree:2d}/16 (run {run})", flush=True)
        else:
            break
    print(f"  longest run of >=12/16 agreement: {best_run} symbols", flush=True)

    print(f"\n-- same search against J_PRIME, for comparison --", flush=True)
    patp = np.array(tx.J_PRIME * REPS, dtype=np.int64)
    ap = agreement(idx, role, patp, 24, hi)
    print(f"  best J_PRIME agreement {ap.max()}/{patp.size} (J_4PT best was {a.max()}/{nb})", flush=True)


if __name__ == "__main__":
    main()