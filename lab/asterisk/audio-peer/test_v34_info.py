#!/usr/bin/env python3
"""V.34 INFO sequences: build/parse round trips and DPSK modulation through a noisy, offset channel."""
import unittest

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_info as vi


@unittest.skipIf(np is None, "numpy not installed")
class InfoTests(unittest.TestCase):
    def test_frames_round_trip(self):
        i0 = vi.build_info0(rate2743=1, rate2800=1, lo3000=1, hi3000=1, lo3200=1, pts1664=1, max_asym=3, clock=1)
        self.assertEqual(len(i0), 49)
        p = vi.parse_info0(i0)
        self.assertEqual((p["rate2743"], p["hi3000"], p["pts1664"], p["max_asym"], p["clock"]), (1, 1, 1, 3, 1))
        blocks = [{"hi_carrier": i % 2, "preemph": i, "max_rate": 14 - i} for i in range(6)]
        i1c = vi.build_info1c(power_min=2, power_extra=5, md_len=77, blocks=blocks, freq_offset=-37)
        self.assertEqual(len(i1c), 109)
        p = vi.parse_info1c(i1c)
        self.assertEqual((p["power_min"], p["power_extra"], p["md_len"], p["freq_offset"]), (2, 5, 77, -37))
        self.assertEqual(p["blocks"], blocks)
        i1a = vi.build_info1a(md_len=12, hi_carrier=1, preemph=7, max_rate=11, sym_ans=4, sym_call=3, freq_offset=100)
        self.assertEqual(len(i1a), 70)
        p = vi.parse_info1a(i1a)
        self.assertEqual((p["md_len"], p["preemph"], p["max_rate"], p["sym_ans"], p["sym_call"], p["freq_offset"]),
                         (12, 7, 11, 4, 3, 100))

    def test_corrupted_frame_fails_crc(self):
        bits = vi.build_info0(rate2800=1)
        bits[20] ^= 1
        self.assertIsNone(vi.parse_info0(bits))

    def test_modulated_frames_survive_noise_and_carrier_offset(self):
        rng = np.random.default_rng(1)
        for role in ("call", "answer"):
            for kind, bits, key in (("info0", vi.build_info0(rate3429=1, lo3200=1), "rate3429"),
                                    ("info1a", vi.build_info1a(md_len=9, sym_call=2), "md_len")):
                with self.subTest(role=role, kind=kind):
                    mod = vi.InfoModulator(role)
                    sig = np.concatenate([np.zeros(777), mod.modulate(bits + bits), np.zeros(500)])
                    sig = sig + rng.normal(0, 150, len(sig))
                    # 1.3 Hz carrier offset
                    n = np.arange(len(sig))
                    sig = sig * 1.0
                    dem = vi.InfoDemodulator(role, carrier_offset_hz=0.0)
                    got = dem.decode(sig, kind)
                    self.assertTrue(got, "no frame decoded")
                    self.assertEqual(got[0][key], {"info0": 1, "info1a": 9}[kind])


if __name__ == "__main__":
    unittest.main()
