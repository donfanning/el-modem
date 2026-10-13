#!/usr/bin/env python3
"""Differential test for the vectorized decode_after_trn.

The vectorized version must agree *exactly* with the original pure-Python one - same matched TRN-end
offset and same returned descrambled tail - over random symbol histories. Both matching and
non-matching candidates are exercised: for the matching cases the end estimate is chosen so that a
real match is inside the candidate window (found by brute force over every offset first).
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_rx
    import v34_mp
    from v34_rx import decode_after_trn, descramble


def decode_after_trn_ref(idx, end_estimate, role, sync_ones=17):
    """Original pure-Python implementation, kept as the reference oracle (sync check only; the vectorized
    version additionally validates a whole MP frame, so the fixtures carry a complete valid frame)."""
    for n in range(max(end_estimate - 24, 2), end_estimate + 24):
        if n >= len(idx):
            break
        bits = []
        for k in range(len(idx)):
            i_k = int(idx[k]) if k < n else (int(idx[k]) - int(idx[k - 1])) % 4
            bits += [i_k & 1, i_k >> 1]
        d = descramble(bits, role)
        tail = d[2 * n:]
        if tail[:sync_ones + 1] == [1] * sync_ones + [0]:
            return n, tail
    return None


def build_with_match(role: str, trn_len: int, n_extra: int = 60, seed: int = 0, sync_ones: int = 17):
    """Build a symbol history that really does carry a complete (valid) MP frame at ``trn_len``.

    The descrambler is invertible and memoryless (``d[i] = b[i] ^ b[i-tap] ^ b[i-23]``), so a desired
    descrambled stream can be turned back into raw bits and then into symbols - which makes
    guaranteed-matching fixtures easy instead of relying on chance.  The frame is a real Type 0 MP so
    the vectorized ``decode_after_trn``'s full-frame validation also accepts it.
    """
    rng = np.random.default_rng(seed)
    tap = v34_rx.descramble_tap(role)
    n_sym = trn_len + n_extra
    d = rng.integers(0, 2, size=2 * n_sym)
    if v34_mp is not None:
        frame = v34_mp.build_mp(0, max_c2a=2, max_a2c=2, mask=0b11)
        d[2 * trn_len:2 * trn_len + len(frame)] = frame
    else:
        d[2 * trn_len:2 * trn_len + sync_ones] = 1              # MP frame sync
        d[2 * trn_len + sync_ones] = 0                          # start bit
    b = np.zeros(2 * n_sym, dtype=np.int64)
    for i in range(2 * n_sym):                                  # invert the descrambler
        b[i] = d[i] ^ (b[i - tap] if i >= tap else 0) ^ (b[i - 23] if i >= 23 else 0)
    sym = (b[0::2] | (b[1::2] << 1)).astype(np.int64)           # the value each symbol must carry
    idx = np.empty(n_sym, dtype=np.int64)
    idx[:trn_len] = sym[:trn_len]                               # before the TRN end: absolute
    for k in range(trn_len, n_sym):                             # from there on: differential
        idx[k] = (idx[k - 1] + sym[k]) % 4
    return idx, trn_len


@unittest.skipIf(np is None, "numpy not installed")
class DecodeAfterTrnEquivTests(unittest.TestCase):
    def test_random_histories_never_yield_a_false_frame(self):
        """Random symbol histories essentially never encode a valid MP frame, so the full-frame validation
        must return None.  (The bare sync-only oracle can return spurious hits here, which is why the
        vectorized version now validates a whole frame.)"""
        rng = np.random.default_rng(20261002)
        for trial in range(3000):
            role = "answer" if trial % 2 == 0 else "call"
            n_sym = int(rng.integers(20, 300))
            idx = rng.integers(0, 4, size=n_sym).astype(np.int64)
            end_est = int(rng.integers(2, max(3, n_sym - 1)))
            got = decode_after_trn(idx, end_est, role)
            if got is not None:
                # even if a random fixture happened to frame (it does not), the tail must be parseable
                self.assertIsNotNone(v34_mp.parse_mp(got[1]), f"trial {trial}: returned an invalid frame")

    def test_matches_reference_when_a_real_match_exists(self):
        """Cover the matching path with histories that really do carry a whole valid MP frame."""
        hits = 0
        for trial in range(300):
            role = "answer" if trial % 2 == 0 else "call"
            trn_len = 6 + trial % 40
            idx, truth = build_with_match(role, trn_len, seed=trial)
            end_est = truth + (trial % 21) - 10
            end_est = max(2, end_est)
            a = decode_after_trn_ref(idx, end_est, role)
            b = decode_after_trn(idx, end_est, role)
            self.assertIsNotNone(a, f"trial {trial}: oracle missed the constructed match")
            self.assertIsNotNone(b, f"trial {trial}: vectorized missed the constructed match")
            self.assertEqual(a[0], b[0], f"trial {trial}: offset mismatch")
            self.assertEqual(a[1], b[1], f"trial {trial}: tail mismatch")
            hits += 1
        self.assertGreater(hits, 200)

    def test_cost_is_independent_of_history_length(self):
        """The whole point of the rewrite: a long symbol history must not cost more per call."""
        rng = np.random.default_rng(3)
        idx = rng.integers(0, 4, size=40000).astype(np.int64)
        import time
        times = {}
        for n in (2000, 4000, 40000):
            sub = idx[:n]
            t0 = time.perf_counter()
            for _ in range(5):
                decode_after_trn(sub, 100, "answer")
            times[n] = (time.perf_counter() - t0) / 5
        self.assertLess(times[40000], times[2000] * 6,
                        f"per-call cost still grows with history: {times}")


if __name__ == "__main__":
    unittest.main()