"""Upstream constellation, modulus and rate selection."""

import random
import unittest
from math import prod

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import v34_g711 as g711
    import v92_cp as cp
    import v92_modulation_design as md
    import v92_pcm_upstream as pu


def all_mulaw_levels():
    codes = np.array([0xFF - i for i in range(1, 128)], dtype=np.uint8)
    return sorted({abs(float(v)) for v in g711.decode(codes, "pcmu")})


@unittest.skipIf(np is None, "numpy not installed")
class ModulationDesignTests(unittest.TestCase):
    def test_constellation_gap_scales_with_noise(self):
        levels = all_mulaw_levels()
        quiet = md.select_constellation(levels, 1.0)
        noisy = md.select_constellation(levels, 40.0)
        self.assertGreater(len(quiet), len(noisy))
        for chosen, sigma in ((quiet, 1.0), (noisy, 40.0)):
            gaps = np.diff(chosen)
            self.assertTrue(np.all(gaps >= 9.0 * sigma))
            self.assertGreaterEqual(chosen[0], 4.5 * sigma)
        # The A/D step is charged against the margin: two chosen levels at a coarse part of
        # the scale must be farther apart than the bare 9-sigma rule asks.
        coarse = [v for v in md.select_constellation(levels, 1.0) if v > 20000]
        self.assertTrue(all(b - a >= 9.0 + 500 for a, b in zip(coarse, coarse[1:])))
        self.assertLessEqual(len(quiet), 64)

    def test_quiet_line_reaches_48k_and_noise_lowers_the_rate(self):
        levels = all_mulaw_levels()
        rates = []
        for sigma in (0.5, 60.0, 120.0, 250.0, 500.0):
            plan = md.design(levels, sigma)
            rates.append(None if plan is None else plan["rate"])
        self.assertEqual(rates[0], 48000.0)
        usable = [r for r in rates if r is not None]
        self.assertEqual(usable, sorted(usable, reverse=True))
        self.assertGreater(len(set(usable)), 2)

    def test_plan_satisfies_the_clause_constraints(self):
        plan = md.design(all_mulaw_levels(), 6.0)
        n = 2 * len(plan["constellation"])
        self.assertEqual(plan["bits"], 2 * (plan["drn"] + 17))
        self.assertGreaterEqual(prod(plan["moduli"]), 1 << plan["bits"])
        for i, m in enumerate(plan["moduli"]):
            self.assertLessEqual(m if i % 4 < 3 else 2 * m, n)
        self.assertAlmostEqual(plan["rate"], (plan["drn"] + 17) * 8000 / 6)

    def test_very_noisy_line_falls_back(self):
        self.assertIsNone(md.design(all_mulaw_levels(), 1000.0))
        self.assertIsNone(md.design([], 1.0))

    def test_plan_round_trips_through_cpd_and_the_mapper_with_noise(self):
        sigma = 8.0
        plan = md.design(all_mulaw_levels(), sigma)
        bits = cp.build_cpd(drn=plan["drn"], trellis=plan["trellis"], gain4=0x4000,
                            moduli=plan["moduli"], interval_sets=(0,) * 6,
                            constellations=[[int(v) for v in plan["constellation"]]])
        profile = pu.profile_from_cpd(cp.parse_cpd(bits))
        self.assertEqual(profile.bits_per_frame, plan["bits"])
        profile = pu.PcmUpstreamProfile(
            constellations=profile.constellations, interval_set=profile.interval_set,
            moduli=profile.moduli, trellis=profile.trellis, bits=profile.bits, gain=1.0)
        tx, rx = pu.PcmUpstreamTransmitter(profile), pu.PcmUpstreamReceiver(profile)
        rng = random.Random(6)
        noise = np.random.default_rng(6)
        for _ in range(40):
            data = [rng.randint(0, 1) for _ in range(profile.bits_per_frame)]
            sent = tx.encode_frame(data)
            heard = [float(v) + noise.normal(0.0, sigma) for v in sent]
            self.assertEqual(rx.decode_frame(heard), data)


if __name__ == "__main__":
    unittest.main()
