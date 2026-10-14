#!/usr/bin/env python3
"""Exhaustively search a recovered phase-3 symbol stream for J, under every interpretation.

J_4PT is a 16-bit pattern sent repeatedly, so wherever J really is, the *descrambled* bit stream has
a strong periodicity at lag 16. That periodicity is a parameter-free detector: it does not need to
know where TRN ends or which scrambler polynomial applies, so it can distinguish "J is absent" from
"J is present but our split point or tap is wrong".

Swept dimensions:
  * split point n (where absolute TRN coding gives way to differential coding)
  * descrambler tap (0 = none, 5 = GPa answerer, 18 = GPC caller)
  * coding (absolute vs differential)
  * pattern (J_4PT vs J_PRIME)

    python3 -u jsearch.py <idx.npy> [lo] [hi]
"""
from __future__ import annotations

import sys

import numpy as np

import v34_rx
import v34_tx as tx

TAPS = (0, 5, 18)
PATTERNS = {"J_4PT": tx.J_4PT, "J_PRIME": tx.J_PRIME}


def bits_from(idx: np.ndarray, n: int, span_sym: int, tap: int, differential: bool) -> np.ndarray:
    """``span_sym`` interleaved bit pairs from symbol ``n`` under the given interpretation."""
    s0 = max(n - 12, 0)
    s1 = min(n + span_sym + 1, idx.size)
    if s1 - s0 < 2:
        return np.empty(0, dtype=np.int64)
    k = np.arange(s0, s1)
    if differential:
        v = idx[k].copy()
        if v.size > 1:
            v[1:] = (idx[k[1:]] - idx[k[:-1]]) % 4
    else:
        v = idx[k]
    cut = 2 * (n - s0)
    buf = np.concatenate((np.zeros(cut, dtype=np.int64), v34_rx._interleave(v)))
    if tap == 0:
        return buf
    return v34_rx._descramble_np(buf, tap)


def lag16(b: np.ndarray, nb: int = 256) -> float:
    """Normalized autocorrelation at lag 16 over the first ``nb`` bits: J is strongly periodic."""
    if b.size < nb + 16:
        nb = b.size - 16
    if nb < 32:
        return 0.0
    w = b[:nb].astype(np.float64) * 2.0 - 1.0
    a, c = w[:-16], w[16:]
    den = np.sqrt((a * a).sum() * (c * c).sum())
    return float((a * c).sum() / den) if den > 0 else 0.0


def agreement(b: np.ndarray, pat: np.ndarray) -> int:
    if b.size < pat.size:
        return 0
    return int((b[:pat.size] == pat).sum())


def main() -> None:
    path = sys.argv[1]
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else min(12000, 0)
    idx = np.load(path).astype(np.int64)
    if hi <= lo:
        hi = min(idx.size - 260, 20000)
    print(f"{path}: {idx.size} symbols; sweeping n={lo}..{hi} x tap{TAPS} x 2 codings", flush=True)
    for name, p in PATTERNS.items():
        print(f"  {name} = {''.join(map(str, p))}", flush=True)

    # ---- primary: lag-16 periodicity, which needs no knowledge of n or tap -------------------
    print("\n-- lag-16 periodicity of the descrambled stream (random data ~0, J is strongly >0) --",
          flush=True)
    best = []
    for tap in TAPS:
        for diff in (True, False):
            for n in range(lo, hi, 4):
                b = bits_from(idx, n, 140, tap, diff)
                if b.size >= 140:
                    best.append((lag16(b), n, tap, diff))
    best.sort(reverse=True)
    print(f"  (scanned {len(best)} hypotheses)", flush=True)
    for v, n, tap, diff in best[:15]:
        print(f"  ac16={v:+.3f}  n={n:6d} tap={tap:2d} differential={diff}", flush=True)

    # ---- secondary: exact/partial agreement with each J pattern --------------------------------
    print("\n-- best pattern agreement per (tap, coding) --", flush=True)
    pats = {k: np.array(v * 3, dtype=np.int64) for k, v in PATTERNS.items()}
    for tap in TAPS:
        for diff in (True, False):
            line = []
            for name, pat in pats.items():
                bv = (0, None)
                for n in range(lo, hi, 4):
                    b = bits_from(idx, n, 24, tap, diff)
                    a = agreement(b, pat)
                    if a > bv[0]:
                        bv = (a, n)
                line.append(f"{name} {bv[0]:2d}/{pat.size} @n={bv[1]}")
            print(f"  tap={tap:2d} differential={str(diff):5s}  " + "   ".join(line), flush=True)

    # ---- raw dump around the transition, so the bits can be eyeballed ------------------------
    ce = 6392
    print(f"\n-- raw descrambled bits around symbol {ce} (differential, tap=5) --", flush=True)
    b = bits_from(idx, ce - 40, 600, 5, True)
    for k in range(0, min(b.size, 600), 16):
        chunk = "".join(map(str, b[k:k + 16].tolist()))
        j4 = "".join(map(str, tx.J_4PT))
        tag = " <== J_4PT" if chunk == j4 else ""
        print(f"  sym {ce - 40 + k // 2:6d}: {chunk}{tag}", flush=True)

    print("\n-- rotation indices in the same region (4 distinct values expected) --", flush=True)
    w = idx[ce - 40:ce + 260]
    print("  " + " ".join(map(str, w[:120].tolist())), flush=True)
    print(f"  distinct values in that window: {sorted(set(w.tolist()))}", flush=True)


if __name__ == "__main__":
    main()