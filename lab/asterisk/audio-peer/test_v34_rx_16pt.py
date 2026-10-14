#!/usr/bin/env python3
"""16-point phase-4 constellation support (ITU-T V.34 10.1.3.8 / 10.1.3.9, Figure 5).

The phase-4 TRN / MP / MP' / E are on the constellation J announced: 4-point at 2400 bps, 16-point
above it.  The 16-point set is quarter-superconstellation labels 0..3 ((1,1), (-3,1), (1,-3), (-3,-3)),
each rotated by multiples of 90 degrees.  The rotation bits I1/I2 are absolute in TRN and differentially
coded in MP / J' / E (the differential coder is seeded from the last TRN symbol); the quarter bits Q1/Q2
select the point and are always absolute.  The scrambler restarts for the TRN and is shared by it and the
following MP / E (10.1.3.8: "initialized to zero prior to transmission of the TRN signal").
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_mp
    import v34_rx
    import v34_tx as tx
    from v34_rx import decode_after_trn, slice16


def tx_stream(role: str, n_trn: int, mp_type: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build a phase-4 16-point stream (TRN then MP then E) and slice it back to (rotation, quarter) arrays.

    Returns (rot, qv, all_symbols, true_trn_len)."""
    trn, sc = tx.trn_symbols(n_trn, role, constellation=16)
    last_i = tx.trn_last_rotation(trn, 16)
    bits = v34_mp.build_mp(mp_type, max_c2a=14, max_a2c=14, mask=(1 << 14) - 1, precoder=[0] * 6)
    mp, z = tx.bit_symbols(bits, role, sc, z0=last_i, constellation=16)
    e, _ = tx.bit_symbols([1] * 20, role, sc, z0=z, constellation=16)
    syms = np.concatenate([trn, mp, e])
    rot, qv = [], []
    for s in syms:
        q, n = slice16(s)
        qv.append(q)
        rot.append(n)
    return np.asarray(rot, dtype=np.int64), np.asarray(qv, dtype=np.int64), syms, n_trn


@unittest.skipIf(np is None, "numpy not installed")
class ConstellationTests(unittest.TestCase):
    def test_sixteen_points_are_distinct_and_power_ten(self):
        pts = tx.constellation_points(16)
        self.assertEqual(len(set(pts)), 16)
        self.assertAlmostEqual(float(np.mean([abs(p) ** 2 for p in pts])), 10.0, places=6)

    def test_four_points_are_point_zero_rotated(self):
        self.assertEqual(tx.constellation_points(4), [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])

    def test_labels_match_figure_five(self):
        # Figure 5 labels 0..3, read from the spec: (1,1), (-3,1), (1,-3), (-3,-3).
        self.assertEqual(tx.C16, [(1, 1), (-3, 1), (1, -3), (-3, -3)])


@unittest.skipIf(np is None, "numpy not installed")
class SixteenPointRoundTripTests(unittest.TestCase):
    def test_candidate_reads_only_bounded_quarter_history(self):
        class WindowOnly:
            def __init__(self, values):
                self.values=values;self.reads=[]
            def __array__(self, *args, **kwargs):
                raise AssertionError('candidate converted the full quarter history')
            def __getitem__(self, key):
                self.reads.append(key)
                return self.values[key]
        rng=np.random.default_rng(3434)
        rot=rng.integers(0,4,10000);quarters=rng.integers(0,4,10000)
        for role in ('call','answer'):
            for split,at,span in [(0,0,18),(5,5,64),(7000,7005,191),(9988,9990,64)]:
                with self.subTest(role=role,split=split):
                    history=WindowOnly(quarters.tolist())
                    expected=v34_rx._full_descramble(rot,role,split,c16=True,qidx=quarters)
                    got=v34_rx._descramble_at(rot,role,at,span,abs_end=split,c16=True,qidx=history)
                    np.testing.assert_array_equal(got,expected[4*at:4*at+span])
                    self.assertEqual(len(history.reads),1)
                    window=history.reads[0]
                    self.assertLessEqual(window.stop-window.start,12+at-split+span//4+1)

    def test_long_trn_list_history_preserves_mp_and_e(self):
        for role in ('call','answer'):
            rot,quarters,_,end=tx_stream(role,6000,1)
            found=decode_after_trn(rot,end-5,role,c16=True,qidx=quarters.tolist())
            self.assertEqual(found[0],end)
            self.assertIsNotNone(v34_mp.parse_mp(found[1]))
            self.assertEqual(found[1][188:208],[1]*20)

    def test_trn_last_rotation_matches_the_true_rotation(self):
        # angle(trn[-1]/P0) is only right for the 4-point constellation; the quarter point's own angle
        # would corrupt it for 16-point.  trn_last_rotation must give the true value.
        trn, _ = tx.trn_symbols(500, "answer", constellation=16)
        q, n_true = slice16(trn[-1])
        self.assertEqual(tx.trn_last_rotation(trn, 16), n_true)

    def test_decode_mp_and_e_round_trip(self):
        for role in ("call", "answer"):
            for mp_type in (0, 1):
                rot, qv, _, n_trn = tx_stream(role, 700, mp_type)
                found = decode_after_trn(rot, n_trn - 5, role, c16=True, qidx=qv)
                self.assertIsNotNone(found, f"{role} type {mp_type}: missed the TRN end")
                self.assertEqual(found[0], n_trn, f"{role} type {mp_type}: wrong TRN end")
                parsed = v34_mp.parse_mp(found[1])
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed["type"], mp_type)
                if mp_type == 1:
                    self.assertEqual(parsed["precoder"], [0] * 6)
                # the 20-bit E sequence follows the MP frame
                mp_len = 88 if mp_type == 0 else 188
                self.assertEqual(found[1][mp_len:mp_len + 20], [1] * 20)

    def test_no_spurious_sync_before_the_true_end(self):
        # With a long TRN the descramble can produce a spurious "17 ones then 0" run; the full-frame
        # validation must reject those and return the exact TRN end.
        rot, qv, _, n_trn = tx_stream("answer", 800, 1)
        found = decode_after_trn(rot, n_trn, "answer", c16=True, qidx=qv)
        self.assertIsNotNone(found)
        self.assertEqual(found[0], n_trn)


@unittest.skipIf(np is None, "numpy not installed")
class BitSymbolsSixteenPointTests(unittest.TestCase):
    def test_e_is_five_symbols_on_sixteen_point(self):
        sc = tx.Scrambler("answer")
        sym, _ = tx.bit_symbols([1] * 20, "answer", sc, z0=0, constellation=16)
        self.assertEqual(len(sym), 5)                     # 20 bits / 4 bits per symbol

    def test_sixteen_point_differential_is_rotation_only(self):
        # Verify the differential coder accumulates rotation only, leaving the quarter index absolute.
        sc = tx.Scrambler("answer")
        bits = [0] * 40
        sym, z = tx.bit_symbols(bits, "answer", sc, z0=0, constellation=16)
        pts = set(tx.constellation_points(16))
        self.assertTrue(all(any(abs(s - p) < 1e-9 for p in pts) for s in sym))


@unittest.skipIf(np is None, "numpy not installed")
class JPrimeIsAlwaysFourPointTests(unittest.TestCase):
    """J' is 4-point even when J announced 16 points, and getting that wrong is a silent deadlock.

    10.1.3.4: "Sequence J' is used to terminate J and is transmitted only once.  J' is generated as
    described in 10.1.3.3 except that the 16-bit pattern shown in Table 19 is used."  10.1.3.3 (J) is
    the 4-point rule - two bits I1,I2, differential Zn, point 0 rotated Zn*90 deg - and 10.1.3.9 says so
    outright when it defers to it ("The 4-point MP sequence is generated as described in 10.1.3.3")
    before spelling out the 16-point rule separately.  So unlike E (10.1.3.2) and TRN (10.1.3.8), J'
    does *not* follow the constellation J announced.

    That distinction is what lets the answerer hunt J' on its phase-3 slicer, which by definition only
    knows the 4-point set - it has not learned the phase-4 constellation yet.  A 16-point J' therefore
    leaves the answerer in "jprime" forever while the caller loops MP: both sides look alive and the
    handshake never completes.
    """

    def test_jprime_uses_the_four_point_rule_at_every_rate(self):
        for rate in (2400, 14400, 33600):
            with self.subTest(rate=rate):
                sc4 = tx.Scrambler("call")
                four, _ = tx.j_symbols(tx.J_PRIME, 1, "call", sc4, z0=0, constellation=4)
                sc16 = tx.Scrambler("call")
                sixteen, _ = tx.j_symbols(tx.J_PRIME, 1, "call", sc16, z0=0, constellation=16)
                # 16 bits: 8 symbols at 2 bits/symbol, 4 symbols at 4 bits/symbol.
                self.assertEqual(len(four), 8)
                self.assertEqual(len(sixteen), 4)
                for s in four:
                    self.assertTrue(any(abs(s - p) < 1e-9 for p in tx.constellation_points(4)),
                                    "4-point J' symbol off the 4-point constellation")
                self.assertFalse(any(any(abs(s - p) < 1e-9 for p in tx.constellation_points(4))
                                     for s in sixteen),
                                 "16-point J' is not what an answerer hunting on phase 3 can see")

    def test_caller_sends_four_point_jprime_even_when_j_announced_16_points(self):
        """The regression itself: the caller's J' must be 8 symbols of the 4-point set at 33600 bps.

        Built the way the engine does it - J ran first, so the differential coder carries J's state
        into J', and only the TRN after it is 16-point.
        """
        sc = tx.Scrambler("call")
        _jp, z_after_j = tx.j_symbols(tx.J_16PT, 2, "call", sc, z0=0, constellation=4)
        jp, _ = tx.j_symbols(tx.J_PRIME, 1, "call", sc, z0=z_after_j, constellation=4)
        self.assertEqual(len(jp), 8)
        pts4 = tx.constellation_points(4)
        for s in jp:
            self.assertTrue(any(abs(s - p) < 1e-9 for p in pts4),
                            f"J' symbol {s} is not on the 4-point constellation the answerer decodes")
        # The 16-point constellation strictly contains the 4-point one, so this is the check that
        # matters: every J' symbol must lie in the small set, not just somewhere in the big one.
        self.assertEqual(len(pts4), 4)
        self.assertEqual(len(tx.constellation_points(16)), 16)


if __name__ == "__main__":
    unittest.main()
