"""Standards-level checks for the V.90/V.34 Phase-2 probe transmitter."""

import math
import unittest

import numpy as np

from v90_digital_phase2_control import Phase2Action
from v90_digital_phase2_tx import V90DigitalPhase2Transmitter


class V90Phase2ProbeLevelTests(unittest.TestCase):
    def test_l1_is_six_db_above_nominal_l2(self):
        """V.90 §8.2.4 references the V.34 §10.1.2.4 probe-level rule."""
        requested_levels = []

        def probe_builder(duration, rms):
            requested_levels.append((duration, rms))
            count = round(duration * 8000)
            # A constant signal makes this test specifically about requested
            # level scaling, independent of the probe's frequency components.
            return np.full(count, rms, dtype=float)

        nominal_rms = 450.0
        tx = V90DigitalPhase2Transmitter(
            signals=None,
            probe_builder=probe_builder,
            tone_b_peak=900.0,
            probe_rms=nominal_rms,
        )
        l1_samples = int(0.160 * 8000)
        l2_samples = int(0.500 * 8000)
        tx.submit([
            Phase2Action(0, "l1"),
            Phase2Action(l1_samples, "l2", value=l2_samples),
        ])
        rendered = tx.render(l1_samples + l2_samples)

        l1_rms = float(np.sqrt(np.mean(rendered[:l1_samples] ** 2)))
        l2_rms = float(np.sqrt(np.mean(rendered[l1_samples:] ** 2)))
        self.assertAlmostEqual(l1_rms / l2_rms, 2.0, places=12)
        self.assertAlmostEqual(20.0 * math.log10(l1_rms / l2_rms),
                               6.020599913, places=8)
        self.assertEqual(requested_levels, [
            (0.160, nominal_rms * 2.0),
            (l2_samples / 8000, nominal_rms),
        ])


if __name__ == "__main__":
    unittest.main()
