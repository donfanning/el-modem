#!/usr/bin/env python3
"""Differential test for the vectorized find_jprime / jprime_search.

``find_jprime`` originally built the whole interleaved bit list in a Python loop and descrambled it
with a second Python loop, over the entire stream every call. The vectorized version must agree
*exactly* with that oracle - same symbol index or same None - including on streams that really do
contain a J-aligned J', which are constructed here by inverting the descrambler rather than waiting
for an 18-bit pattern to appear by chance in random data.
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_rx
    import v34_tx as tx
    from v34_rx import descramble, find_jprime, jprime_search


def find_jprime_ref(idx, n_trn, role):
    """Original pure-Python implementation, kept verbatim as the reference oracle."""
    idx = list(idx)
    bits = []
    for k in range(len(idx)):
        i_k = idx[k] if k < n_trn else (idx[k] - idx[k - 1]) % 4
        bits += [i_k & 1, i_k >> 1]
    d = descramble(bits, role)[2 * n_trn:]
    s = "".join(map(str, d))
    jp = "".join(map(str, tx.J_PRIME))
    pat = "".join(map(str, tx.J_4PT))
    pos = s.find(jp)
    while pos >= 0:
        if pos % 16 == 0 and pos >= 16 and s[pos - 16:pos] == pat and len(s) >= pos + 16:
            return n_trn + pos // 2
        pos = s.find(jp, pos + 1)
    return None


def build_with_jprime(role: str, n_trn: int, j_units: int, seed: int = 0):
    """Build a stream whose descrambled bits are J_4PT repeated, then J_PRIME at a 16-bit boundary.

    The descrambler is invertible and memoryless, so the desired descrambled stream is inverted back
    into raw bits and then into symbols: a guaranteed match instead of an astronomically unlikely one.
    """
    rng = np.random.default_rng(seed)
    tap = v34_rx.descramble_tap(role)
    n_sym = n_trn + 8 * (j_units + 2)
    d = rng.integers(0, 2, size=2 * n_sym)
    d[2 * n_trn:] = 0
    j_start = 2 * n_trn
    for u in range(j_units):                                    # J run, 16 bits per 8 symbols
        d[j_start + 16 * u:j_start + 16 * u + 16] = tx.J_4PT
    jp_at = j_start + 16 * j_units                             # J' on the next 16-bit boundary
    d[jp_at:jp_at + 16] = tx.J_PRIME
    b = np.zeros(2 * n_sym, dtype=np.int64)
    for i in range(2 * n_sym):                                  # invert the descrambler
        b[i] = d[i] ^ (b[i - tap] if i >= tap else 0) ^ (b[i - 23] if i >= 23 else 0)
    sym = (b[0::2] | (b[1::2] << 1)).astype(np.int64)
    idx = np.empty(n_sym, dtype=np.int64)
    idx[:n_trn] = sym[:n_trn]                                   # before the TRN end: absolute
    for k in range(n_trn, n_sym):                               # from there on: differential
        idx[k] = (idx[k - 1] + sym[k]) % 4
    return idx, n_trn + 8 * j_units


@unittest.skipIf(np is None, "numpy not installed")
class FindJPrimeEquivTests(unittest.TestCase):
    def test_matches_reference_on_random_histories(self):
        rng = np.random.default_rng(20261003)
        for trial in range(2000):
            role = "answer" if trial % 2 == 0 else "call"
            n_sym = int(rng.integers(20, 300))
            idx = rng.integers(0, 4, size=n_sym).astype(np.int64)
            n_trn = int(rng.integers(2, max(3, n_sym - 12)))
            a = find_jprime_ref(idx, n_trn, role)
            b = find_jprime(idx, n_trn, role)
            self.assertEqual(a, b, f"trial {trial}: random history mismatch (n_trn={n_trn})")

    def test_matches_reference_when_a_real_jprime_exists(self):
        """Cover the matching path with streams that really do carry a J-aligned J'."""
        hits = 0
        for trial in range(300):
            role = "answer" if trial % 2 == 0 else "call"
            n_trn = 4 + trial % 40
            idx, truth = build_with_jprime(role, n_trn, j_units=2 + trial % 9, seed=trial)
            a = find_jprime_ref(idx, n_trn, role)
            b = find_jprime(idx, n_trn, role)
            self.assertIsNotNone(a, f"trial {trial}: oracle missed the constructed J'")
            self.assertEqual(a, b, f"trial {trial}: mismatch on a constructed match")
            self.assertEqual(a, truth, f"trial {trial}: wrong J' offset")
            hits += 1
        self.assertGreater(hits, 250)

    def test_jprime_search_agrees_with_find_jprime(self):
        """jprime_search is the cheap form for rescanning; it must return the same answer.

        ``bits`` here is the full ``_full_descramble`` output (not sliced): jprime_search works in
        absolute bit coordinates and adds ``n_trn`` itself, so a one-shot search from 0 must agree
        exactly with find_jprime.
        """
        rng = np.random.default_rng(7)
        checked = 0
        for trial in range(200):
            role = "answer" if trial % 2 == 0 else "call"
            if trial % 2:
                n_trn = 4 + trial % 30
                idx, _ = build_with_jprime(role, n_trn, j_units=2 + trial % 7, seed=trial)
            else:
                n_sym = int(rng.integers(40, 200))
                idx = rng.integers(0, 4, size=n_sym).astype(np.int64)
                n_trn = int(rng.integers(2, n_sym - 24))
            bits = v34_rx._full_descramble(np.asarray(idx), role, n_trn)
            got, _ = jprime_search(bits, n_trn)
            self.assertEqual(got, find_jprime(idx, n_trn, role),
                             f"trial {trial}: jprime_search disagrees with find_jprime")
            checked += 1
        self.assertEqual(checked, 200)

    def test_jprime_search_cursor_rescans_only_new_bits(self):
        """Rescanning a growing stream via the cursor must find the same J' as a full scan.

        This is the whole point of the cursor: a caller that polls once per feed block advances
        ``start_bit``, so each poll only inspects the bits that arrived since the last one.
        """
        role = "answer"
        n_trn, j_units = 6, 4
        idx, truth = build_with_jprime(role, n_trn, j_units=j_units, seed=99)
        bits = v34_rx._full_descramble(np.asarray(idx), role, n_trn)
        # one-shot full scan
        one_shot, _ = jprime_search(bits, n_trn)
        self.assertEqual(one_shot, truth)
        # incremental scan: the stream grows, as it would live; each poll passes only what has arrived
        cursor, found, prev, polls = 16, None, 16, 0
        for have in range(16, bits.size + 1, 16):
            found, cursor = jprime_search(bits[:have], n_trn, cursor)
            if found is not None:
                break
            self.assertGreaterEqual(cursor, prev, "cursor must never go backwards")
            prev = cursor
            polls += 1
        self.assertEqual(found, truth, "incremental cursor scan missed the constructed J'")
        self.assertGreater(polls, 1, "expected several polls before reaching J'")

    def test_cursor_scan_cost_is_bounded_per_chunk(self):
        """Each incremental rescan must cost only the size of the new chunk, not the whole prefix."""
        import time
        rng = np.random.default_rng(11)
        idx = rng.integers(0, 4, size=40000).astype(np.int64)
        n_trn = 100
        bits = v34_rx._full_descramble(idx, "answer", n_trn)
        # rescan in fixed 16-bit steps up to each horizon; total work should grow linearly, not
        # quadratically, in the horizon if the cursor is doing its job.
        times = {}
        for horizon in (2000, 8000, 32000):
            t0 = time.perf_counter()
            cursor = 16
            for have in range(16, horizon + 1, 16):        # stream grows 16 bits per poll
                _, cursor = jprime_search(bits[:have], n_trn, cursor)
            times[horizon] = time.perf_counter() - t0
        # linear would be 1x / 4x / 16x; quadratic would be 1x / 16x / 256x
        self.assertLess(times[32000], times[8000] * 8,
                        f"incremental rescan still looks super-linear: {times}")


@unittest.skipIf(np is None, "numpy not installed")
class JPrimeScannerTests(unittest.TestCase):
    """JPrimeScanner is what the answerer polls while it waits for the caller's J'; it must find
    exactly what find_jprime would, on a stream that grows the way a live one does."""

    def test_finds_jprime_that_arrives_after_the_seed(self):
        """Seed while the caller is still repeating J, then feed symbols until J' shows up."""
        checked = 0
        for trial in range(60):
            role = "answer" if trial % 2 == 0 else "call"
            n_trn = 4 + trial % 25
            j_units = 2 + trial % 6
            idx, truth = build_with_jprime(role, n_trn, j_units=j_units, seed=trial)
            # 16 descrambled bits per 8 symbols: the J run starts at symbol n_trn and J' at
            # n_trn + 8 * j_units, so seed partway into the run, strictly before J'.
            seed_at = n_trn + 8 * (j_units // 2)
            scan = v34_rx.JPrimeScanner(idx[:seed_at], role, n_trn)
            self.assertIsNone(scan.found, f"trial {trial}: found J' before it was transmitted")
            got = None
            for have in range(seed_at, idx.size, 8):                # 8 symbols per poll, as live
                got = scan.update(idx[:have])
                if got is not None:
                    break
            self.assertEqual(got, truth, f"trial {trial}: wrong or missed J'")
            # A found J' is sticky, so a later poll need not remember it.
            self.assertEqual(scan.update(idx), truth, f"trial {trial}: lost the J' index")
            checked += 1
        self.assertEqual(checked, 60)

    def test_finds_jprime_already_present_at_the_seed(self):
        """The caller may already have sent J' by the time the answerer gets to phase 4."""
        checked = 0
        for trial in range(60):
            role = "answer" if trial % 2 == 0 else "call"
            n_trn = 5 + trial % 20
            j_units = 2 + trial % 5
            idx, truth = build_with_jprime(role, n_trn, j_units=j_units, seed=1000 + trial)
            seed_at = n_trn + 8 * (j_units + 1)                     # one unit past J'
            scan = v34_rx.JPrimeScanner(idx[:seed_at], role, n_trn)
            self.assertEqual(scan.found, truth, f"trial {trial}: missed a J' already on the wire")
            checked += 1
        self.assertEqual(checked, 60)

    def test_incremental_descrambled_bits_match_a_full_pass(self):
        """The sharpest check: after every poll the scanner's descrambled bits must equal what a single
        ``_full_descramble`` over the whole stream produces.

        The obvious ways to get this wrong all still produce plausible-looking 0/1 streams, and a
        search for J' in one simply returns None - which is indistinguishable from "J' has not arrived
        yet" unless the bits are compared directly.  Comparing answers alone is what let a bug through
        once: differentially coded symbols were all differenced against the *first* predecessor of the
        chunk instead of their own, so every symbol after the first of each chunk decoded to noise, and
        the random-history test below still passed because both sides returned None.
        """
        for trial in range(40):
            role = "answer" if trial % 2 == 0 else "call"
            n_trn = 3 + trial % 30
            if trial % 2:
                n_sym = n_trn + 8 * (2 + trial % 7) + 8
                idx, _ = build_with_jprime(role, n_trn, j_units=2 + trial % 7, seed=trial)
            else:
                rng = np.random.default_rng(trial)
                n_sym = int(rng.integers(n_trn + 8, n_trn + 600))
                idx = rng.integers(0, 4, size=n_sym).astype(np.int64)
            ref = v34_rx._full_descramble(idx, role, n_trn)
            for step in (8, 1, 3, 17):
                with self.subTest(trial=trial, seed=n_trn, step=step):
                    seed_at = n_trn + 1 + (trial % 9)
                    scan = v34_rx.JPrimeScanner(idx[:seed_at], role, n_trn)
                    for have in range(seed_at, n_sym, step):
                        scan.update(idx[:have])
                        got = scan._dec[:scan._ndec]
                        self.assertEqual(got.size, min(2 * have, ref.size))
                        self.assertTrue(np.array_equal(got, ref[:got.size]),
                                        f"trial {trial} step {step}: descrambled bits diverge at "
                                        f"_nsym={scan._nsym}")

    def test_agrees_with_find_jprime_on_random_histories(self):
        """No J' in sight must read as None, exactly as find_jprime does - including when the stream
        ends mid-pattern, which is the state the answerer sits in while it waits.

        Half the trials carry a constructed J' so the two are compared on streams where an answer
        exists as well as on streams where both must read None.
        """
        rng = np.random.default_rng(31337)
        for trial in range(300):
            role = "answer" if trial % 2 == 0 else "call"
            n_trn = 3 + trial % 17
            if trial % 3 == 0:
                idx, _ = build_with_jprime(role, n_trn, j_units=2 + trial % 5, seed=5000 + trial)
                self.assertIsNotNone(find_jprime(idx, n_trn, role))
            else:
                n_sym = int(rng.integers(n_trn + 2, n_trn + 400))
                idx = rng.integers(0, 4, size=n_sym).astype(np.int64)
            seed_at = n_trn + int(rng.integers(1, max(2, len(idx) - n_trn)))
            scan = v34_rx.JPrimeScanner(idx[:seed_at], role, n_trn)
            for have in range(seed_at, len(idx), 8):
                scan.update(idx[:have])
            self.assertEqual(scan.found, find_jprime(idx, n_trn, role),
                             f"trial {trial}: scanner disagrees with find_jprime")

    def test_rejects_a_seed_before_the_trn_end(self):
        """The scanner extends only differentially coded symbols, so it must refuse to start early
        rather than quietly descrambling the wrong thing."""
        idx = np.zeros(10, dtype=np.int64)
        with self.assertRaises(ValueError):
            v34_rx.JPrimeScanner(idx, "answer", n_trn=11)

    def test_poll_cost_is_bounded_as_the_stream_grows(self):
        """The reason this class exists: waiting for J' must not get quadratically more expensive.

        find_jprime rebuilds and rescans the whole history on every call, so a live answerer polling it
        once per 8 symbols would pay O(n^2) over the wait. Here the same number of polls is timed at two
        horizons; the scanner's total should stay roughly proportional to the number of symbols.
        """
        import time
        rng = np.random.default_rng(5)
        n_trn = 64
        # No J' anywhere, so the scanner keeps working for the whole run - the "waiting" case.
        ratios = []
        for n_sym in (8000, 32000):
            idx = rng.integers(0, 4, size=n_sym).astype(np.int64)
            scan = v34_rx.JPrimeScanner(idx[:n_trn + 64], "answer", n_trn)
            t0 = time.perf_counter()
            for have in range(n_trn + 64, n_sym, 8):
                scan.update(idx[:have])
            ratios.append((n_sym, time.perf_counter() - t0))
        (n_a, t_a), (n_b, t_b) = ratios
        self.assertLess(t_b, t_a * 12,
                        f"scanner cost grew {t_b / max(t_a, 1e-9):.1f}x for a {n_b / n_a:.0f}x longer "
                        f"stream ({ratios}); expected roughly linear")


if __name__ == "__main__":
    unittest.main()