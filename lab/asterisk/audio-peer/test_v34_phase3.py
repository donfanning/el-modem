#!/usr/bin/env python3
"""V.34 phase 3/4 pieces: QAM transmitter (block and stream), S/PP/TRN/J receiver, MP frames - all offline."""
import math
import unittest
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_mp as mp
    import v34_rx as rx
    import v34_tx as tx


@unittest.skipIf(np is None, "numpy not installed")
class MpTests(unittest.TestCase):
    def test_type0_and_type1_round_trip(self):
        b = mp.build_mp(0, max_c2a=2, max_a2c=3, mask=0b111, ack=1)
        self.assertEqual(len(b), 88)
        p = mp.parse_mp(b)
        self.assertEqual((p["max_c2a"], p["max_a2c"], p["mask"], p["ack"]), (2, 3, 7, 1))
        b = mp.build_mp(1, max_c2a=9, max_a2c=9, shaping=1, precoder=[1, -2, 3, -4, 5, -6])
        self.assertEqual(len(b), 188)
        p = mp.parse_mp(b)
        self.assertEqual((p["shaping"], p["precoder"]), (1, [1, -2, 3, -4, 5, -6]))

    def test_corrupted_mp_is_rejected(self):
        b = mp.build_mp(0)
        b[40] ^= 1
        self.assertIsNone(mp.parse_mp(b))


@unittest.skipIf(np is None, "numpy not installed")
class SignalTests(unittest.TestCase):
    def test_pp_is_periodic_and_unit_amplitude(self):
        pp = tx.pp_signal()
        self.assertEqual(len(pp), 288)
        self.assertTrue(np.allclose(np.abs(pp), 1))
        self.assertTrue(np.allclose(pp[:48], pp[48:96]))

    def test_trn_and_j_use_the_scrambler(self):
        trn, sc = tx.trn_symbols(100, "answer")
        self.assertEqual(len(trn), 100)
        self.assertTrue(all(abs(abs(v) - 2 ** 0.5) < 1e-9 for v in trn))
        j, z = tx.j_symbols(tx.J_4PT, 2, "answer", sc, 0)
        self.assertEqual(len(j), 16)

    def test_stream_matches_block_modulation(self):
        sy = np.concatenate([tx.s_signal(128), tx.s_signal(16, True), tx.pp_signal(), np.zeros(100)])
        st = tx.StreamQam(2400, 1600, 1000)
        st.push(sy)
        a = st.read(1500)
        b = tx.qam_modulate(sy, 2400, 1600, 1000, span=9, n0=-30, power=2.0)
        self.assertLess(np.max(np.abs(a[100:1400] - b[130:1430])), 40)


@unittest.skipIf(np is None, "numpy not installed")
class ReceiverTests(unittest.TestCase):
    def test_incremental_baseband_matches_full_conversion_at_chunk_boundaries(self):
        rng = np.random.default_rng(8)
        signal = rng.normal(size=5000)
        receiver = rx.Phase3Rx(3428.5714285714284, 1959.1836734693877)
        for n in (1000, 1001, 1160, 1800, 5000, 5000, 900, 1400):
            with self.subTest(samples=n):
                np.testing.assert_allclose(receiver.update_baseband(signal[:n]),
                                           rx.to_baseband(signal[:n], receiver.fc, receiver.bw),
                                           atol=1e-12)

    def test_equalizer_rebases_grid_when_timing_slips_exhaust_either_margin(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(size=2000) + 1j * rng.normal(size=2000)
        for k, shift in ((0, -7), (127, 20)):
            with self.subTest(k=k, shift=shift):
                R = SimpleNamespace(_span=12, taps=np.ones(24, dtype=complex),
                                    spl=8000 / 3428.5714285714284, lag=100,
                                    LEAD=8, delay=0.25, _z=signal)
                eq = rx.SymbolEq(R, 0)
                eq.window(0)
                eq.k = k
                eq.dt = shift * R.spl / 2
                eq._g_shift = shift
                window = eq.window(k)
                t = R.lag + (R.LEAD + k + R.delay) * R.spl + eq.dt
                expected = rx.interp(signal, t + eq._offs)
                self.assertEqual(len(window), 24)
                np.testing.assert_allclose(window, expected, atol=1e-10)

    def test_phase3_receiver_finds_trn_end_and_j_through_noise_and_delay(self):
        rng = np.random.default_rng(5)
        S, Sb, PP = tx.s_signal(128), tx.s_signal(16, True), tx.pp_signal() * 2 ** 0.5
        trn, sc = tx.trn_symbols(900, "answer")
        last = int(round(-np.angle(trn[-1] / tx.P0) / (np.pi / 2))) % 4
        j, _ = tx.j_symbols(tx.J_4PT, 8, "answer", sc, last)
        st = tx.StreamQam(2400, 1600, 1500)
        st.push(np.concatenate([S, Sb, PP, trn, j, np.zeros(200)]))
        sig = st.read(int(0.9 * 8000 + 3 * 8000))
        sig = np.concatenate([np.zeros(1234), sig]) + rng.normal(0, 30, len(sig) + 1234)
        # a mild channel: one echo
        sig = sig + 0.15 * np.concatenate([np.zeros(11), sig[:-11]])
        R = rx.Phase3Rx(2400, 1600)
        got = R.acquire(sig)
        self.assertIsNotNone(got)
        self.assertLess(R.train(n_trn=500), 0.1)
        n = 144 + 288 + 900 + 64 - 60
        idx = rx.slice4(R.equalize(432, 900 + 40))
        exp = np.array([int(round(-np.angle(s / tx.P0) / (np.pi / 2))) % 4 for s in tx.trn_symbols(940, "answer")[0]])
        end = rx.find_trn_end(idx, exp)
        self.assertIsNotNone(end)
        self.assertEqual(end, 900, "the TRN is a known sequence, so it should be pinned to the exact symbol")
        mm = (idx != exp[:len(idx)]).astype(np.int64)
        n_trn, seen, pattern, split = rx.decode_j(idx, end, "answer", mismatch=mm)
        self.assertEqual(n_trn, 900)
        self.assertEqual(pattern, "J_4PT")          # this history was built with J_4PT at 2400 bps
        self.assertLessEqual(abs(split - 900), 8, "reported split should be the TRN end, within the latch window")
        self.assertGreaterEqual(seen, 16)

    def test_decode_j_recovers_the_split_from_the_mismatch_alone(self):
        """A coarse TRN-end estimate is not enough on its own: a couple of symbols out is enough to poison
        J's first dozen descrambled bits. Given the mismatch against the known TRN the split is recovered
        and J_16PT - the pattern a modem above 2400 bps actually sends - is found at its true start."""
        rng = np.random.default_rng(5)
        S, Sb, PP = tx.s_signal(128), tx.s_signal(16, True), tx.pp_signal() * 2 ** 0.5
        trn, sc = tx.trn_symbols(900, "answer")
        last = int(round(-np.angle(trn[-1] / tx.P0) / (math.pi / 2))) % 4
        j, _ = tx.j_symbols(tx.J_16PT, 8, "answer", sc, last)
        st = tx.StreamQam(2400, 1600, 1500)
        st.push(np.concatenate([S, Sb, PP, trn, j, np.zeros(200)]))
        sig = st.read(int(0.9 * 8000 + 3 * 8000))
        sig = np.concatenate([np.zeros(1234), sig]) + rng.normal(0, 30, len(sig) + 1234)
        sig = sig + 0.15 * np.concatenate([np.zeros(11), sig[:-11]])
        R = rx.Phase3Rx(2400, 1600)
        self.assertIsNotNone(R.acquire(sig))
        self.assertLess(R.train(n_trn=500), 0.1)
        idx = rx.slice4(R.equalize(432, 900 + 40))
        exp = np.array([int(round(-np.angle(s / tx.P0) / (math.pi / 2))) % 4
                        for s in tx.trn_symbols(940, "answer")[0]])
        mm = (idx != exp[:len(idx)]).astype(np.int64)
        for bad in (898, 902, 906):
            got = rx.decode_j(idx, bad, "answer", mismatch=mm)
            self.assertIsNotNone(got, f"estimate {bad}: J_16PT not recovered from the mismatch")
            self.assertEqual(got[0], 900, f"estimate {bad}: wrong J start")
            self.assertEqual(got[2], "J_16PT", f"estimate {bad}: wrong pattern")


if __name__ == "__main__":
    unittest.main()
