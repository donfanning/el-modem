"""Coordinate and fit-isolation gates for the research frontend comparison."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from frontend_architecture_probe import phase_splitting_design, score


class FrontendArchitectureTests(unittest.TestCase):
    def test_real_iq_filters_demodulate_with_correct_sign(self):
        # At Fs/4, adjacent integer samples are exact quadrature. Independent
        # real I/Q filters must recover +1 after output carrier demodulation.
        n = np.arange(3000)
        pcm = np.cos(2 * np.pi * 2000 * n / 8000)
        centers = np.arange(100, 2000, 3)
        design = phase_splitting_design(pcm, centers, 3, 2000)
        taps = np.zeros(72, complex)
        taps[36], taps[37] = 1, -1j
        np.testing.assert_allclose(design @ taps, 1, atol=2e-12)

    def test_validation_targets_cannot_change_training_fit(self):
        rng = np.random.default_rng(991)
        design = rng.normal(size=(716, 12)) + 1j * rng.normal(size=(716, 12))
        known = design @ (rng.normal(size=12) + 1j * rng.normal(size=12))
        baseline = score(design, known)
        perturbed = known.copy()
        perturbed[480:680] += 1000
        changed = score(design, perturbed)
        self.assertEqual(baseline['train_nmse'], changed['train_nmse'])
        self.assertGreater(changed['validation_nmse'], baseline['validation_nmse'])


if __name__ == '__main__':
    unittest.main()
