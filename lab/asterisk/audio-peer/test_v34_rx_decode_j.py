#!/usr/bin/env python3
"""Differential test for the vectorized decode_j against the original pure-Python version."""
from __future__ import annotations

import time
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_rx
    import v34_tx
    from v34_rx import decode_j, descramble

    from test_v34_rx_decode import build_with_match


def decode_j_ref(idx, trn_end_estimate, role="answer", reps=3, lead=128, patterns=("J_4PT", "J_16PT")):
    """Original pure-Python implementation, kept as the reference oracle."""
    pats = [(name, getattr(v34_tx, name) * reps) for name in patterns]
    lo = max(trn_end_estimate - 12, 24)
    hi = min(trn_end_estimate + lead, len(idx) - 1)

    def descr(split):
        """Descrambled bits of a stream whose symbols before ``split`` are absolute, the rest differential."""
        bits = []
        for k in range(len(idx)):
            i_k = int(idx[k]) if k < split else (int(idx[k]) - int(idx[k - 1])) % 4
            bits += [i_k & 1, i_k >> 1]
        return descramble(bits, role)

    for off in (0, -8, 8, -16, 16):
        split = trn_end_estimate + off
        if split < 24 or split >= len(idx):
            continue
        d = descr(split)
        run = d[2 * split:]                                   # what one _descrambled_run() would return
        best = None
        for name, pat in pats:
            n_hi = min(hi, split + (len(run) - len(pat)) // 2)
            for n in range(max(lo, split), n_hi + 1):
                if run[2 * (n - split):2 * (n - split) + len(pat)] == pat:
                    if best is None or n < best[0]:
                        best = (n, name, split)
        if best is not None:
            n, name, split = best
            tail = getattr(v34_tx, name) * 200
            seen = 0
            while 2 * n + (seen + 1) * 2 <= len(d) and \
                    d[2 * n + seen * 2:2 * n + seen * 2 + 2] == tail[seen * 2:seen * 2 + 2]:
                seen += 1
            return n, seen, name, split
    return None


def build_with_j(role: str, trn_len: int, j_len: int = 30, n_extra: int = 15, seed: int = 0,
                 pattern: str = "J_4PT", j_gap: int = 0):
    """Symbol history whose differential coding starts at ``trn_len`` and whose descrambled bits at
    ``trn_len + j_gap`` start with ``pattern`` repeated.

    ``j_gap`` is the point of the real-world bug: J does not begin at the TRN end, so the descrambler
    split and the J start are two different symbol numbers, and the 23-bit descrambler lookback of the
    first J bits reaches back across that gap into symbols that are themselves differential.

    Built by inverting the descrambler, so the match is guaranteed rather than left to chance.
    """
    rng = np.random.default_rng(seed)
    tap = v34_rx.descramble_tap(role)
    one = getattr(v34_tx, pattern)
    pat = one * 3
    j_start = trn_len + j_gap
    n_sym = j_start + j_len + n_extra
    d = rng.integers(0, 2, size=2 * n_sym)
    d[2 * j_start:2 * j_start + len(pat)] = pat             # J pattern at the J start
    tail = np.array(one * 200, dtype=np.int64)
    for s in range(j_len):                                  # and keep matching it after that
        a = (2 * s) % tail.size
        d[2 * (j_start + s):2 * (j_start + s) + 2] = tail[a:a + 2]
    b = np.zeros(2 * n_sym, dtype=np.int64)
    for i in range(2 * n_sym):
        b[i] = d[i] ^ (b[i - tap] if i >= tap else 0) ^ (b[i - 23] if i >= 23 else 0)
    sym = (b[0::2] | (b[1::2] << 1)).astype(np.int64)
    idx = np.empty(n_sym, dtype=np.int64)
    idx[:trn_len] = sym[:trn_len]
    for k in range(trn_len, n_sym):
        idx[k] = (idx[k - 1] + sym[k]) % 4
    return idx, j_start


@unittest.skipIf(np is None, "numpy not installed")
class DecodeJEquivTests(unittest.TestCase):
    def test_matches_reference_on_random_histories(self):
        rng = np.random.default_rng(4242)
        for lead in (24, 64, 128):
            for trial in range(300):
                role = "answer" if trial % 2 == 0 else "call"
                n_sym = int(rng.integers(30, 250))
                idx = rng.integers(0, 4, size=n_sym).astype(np.int64)
                end = int(rng.integers(24, max(25, n_sym)))
                a = decode_j_ref(idx, end, role, lead=lead)
                b = decode_j(idx, end, role, lead=lead)
                self.assertEqual(a, b, f"lead={lead} trial {trial}: mismatch (ref={a}, new={b})")

    def test_matches_reference_when_j_is_present(self):
        hits = 0
        for pattern in ("J_4PT", "J_16PT"):
            for trial in range(100):
                role = "answer" if trial % 2 == 0 else "call"
                trn_len = 30 + trial % 25
                # J starts well after the differential transition, and the transition itself is only
                # known to within a few symbols: the real-world case from the reference capture.
                gap = 14 + trial % 60
                idx, truth = build_with_j(role, trn_len, j_len=10 + trial % 40, seed=trial,
                                          pattern=pattern, j_gap=gap)
                end = trn_len + (trial % 11) - 5
                a = decode_j_ref(idx, end, role)
                b = decode_j(idx, end, role)
                self.assertIsNotNone(a, f"{pattern} trial {trial}: oracle missed the constructed J")
                self.assertEqual(a, b, f"{pattern} trial {trial}: mismatch (ref={a}, new={b})")
                self.assertEqual(a[0], truth, f"{pattern} trial {trial}: wrong J start")
                self.assertEqual(a[2], pattern, f"{pattern} trial {trial}: wrong pattern reported")
                self.assertLessEqual(abs(a[3] - trn_len), 8, f"{pattern} trial {trial}: split off the TRN end")
                hits += 1
        self.assertEqual(hits, 200)

    def test_j_start_exactly_at_the_transition(self):
        """The self-loop case: J begins at the differential transition, so a split a few symbols late
        (which poisons everything after it) must not be mistaken for a match."""
        for pattern in ("J_4PT", "J_16PT"):
            idx, truth = build_with_j("answer", 400, j_len=30, seed=5, pattern=pattern, j_gap=0)
            got = decode_j(idx, 400, "answer")
            self.assertIsNotNone(got, f"{pattern}: J at the transition went undetected")
            self.assertEqual(got[0], truth, f"{pattern}: wrong J start")
            self.assertEqual(got[2], pattern)

    def test_j_16pt_is_found_past_the_trn_end_and_j_4pt_is_not(self):
        """The reference-capture bug: a 33600 modem signals the 16-point phase-4 constellation with
        J_16PT, and J starts tens of symbols after the TRN end estimate. Both the wide window and the
        second pattern are needed; a J_4PT-only receiver finds nothing at all."""
        idx, j_start = build_with_j("answer", 200, j_len=40, seed=7, pattern="J_16PT", j_gap=54)
        got = decode_j(idx, 192, "answer")
        self.assertIsNotNone(got, "J_16PT 54 symbols past the TRN end estimate went undetected")
        self.assertEqual(got[0], j_start)
        self.assertEqual(got[2], "J_16PT")
        self.assertGreaterEqual(got[1], 16)
        self.assertIsNone(decode_j(idx, 192, "answer", patterns=("J_4PT",)),
                          "a J_4PT-only receiver must not claim this history")
        # and the same history with the window narrowed the way it used to be is still missed
        self.assertIsNone(decode_j(idx, 192, "answer", lead=24, patterns=("J_16PT",)),
                          "the +-12 symbol window should still be too narrow for this")

    def test_reported_start_is_the_j_start_not_the_trn_end_estimate(self):
        """The phase-4 receive chain needs the split point (TRN end), not the J start; make sure the two
        cannot be silently conflated."""
        idx, j_start = build_with_j("answer", 300, j_len=30, seed=3, pattern="J_16PT", j_gap=46)
        n_j, _, _, split = decode_j(idx, 300, "answer")
        self.assertEqual(n_j, j_start)
        self.assertNotEqual(n_j, 300)
        self.assertEqual(split, 300, "split must be the abs/differential transition, not the J start")

    def test_cost_is_independent_of_history_length(self):
        rng = np.random.default_rng(11)
        idx = rng.integers(0, 4, size=40000).astype(np.int64)
        times = {}
        for n in (2000, 4000, 40000):
            sub = idx[:n]
            t0 = time.perf_counter()
            for _ in range(5):
                decode_j(sub, 100, "answer")
            times[n] = (time.perf_counter() - t0) / 5
        self.assertLess(times[40000], times[2000] * 6,
                        f"per-call cost still grows with history: {times}")


@unittest.skipIf(np is None, "numpy not installed")
class DescrambleSplitTests(unittest.TestCase):
    """``_descramble_at`` with an explicit split must agree with ``_full_descramble`` at that split."""

    def test_abs_end_matches_full_descramble(self):
        rng = np.random.default_rng(99)
        for role in ("answer", "call"):
            idx = rng.integers(0, 4, size=900).astype(np.int64)
            full = v34_rx._full_descramble(idx, role, 500)
            for n in range(510, 880, 7):
                got = v34_rx._descramble_at(idx, role, n, 64, abs_end=500)
                self.assertEqual(got.tolist(), full[2 * n:2 * n + 64].tolist(),
                                 f"role={role} n={n}: abs_end window disagrees with _full_descramble")

    def test_default_abs_end_is_backwards_compatible(self):
        rng = np.random.default_rng(1234)
        idx = rng.integers(0, 4, size=500).astype(np.int64)
        for n in (100, 200, 333):
            self.assertEqual(
                v34_rx._descramble_at(idx, "answer", n, 32).tolist(),
                v34_rx._full_descramble(idx, "answer", n)[2 * n:2 * n + 32].tolist(),
                f"n={n}: default abs_end must behave like a split at n")


if __name__ == "__main__":
    unittest.main()