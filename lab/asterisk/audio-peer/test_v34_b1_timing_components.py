"""Verify real timing/gain fitting and rejection of incompatible capture domains."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from b1_timing_components import joint_gain_timing, analyse


class B1TimingComponentsTests(unittest.TestCase):
    def test_recovers_real_displacement_with_complex_waveform_derivative(self):
        rng = np.random.default_rng(51)
        reference = rng.normal(size=60) + 1j * rng.normal(size=60)
        derivative = 1j * rng.normal(size=60)  # Imaginary information must survive.
        for beta in (-.04, .025):
            gain = .17 - .23j
            observed = gain * reference + beta * derivative
            fitted_gain, fitted_beta, _ = joint_gain_timing(observed, reference, derivative)
            self.assertAlmostEqual(abs(fitted_gain - gain), 0., places=12)
            self.assertAlmostEqual(fitted_beta, beta, places=12)

    def test_gain_degenerate_derivative_has_no_timing_estimate(self):
        reference = np.array([1+2j, -3+1j, 2-1j])
        with self.assertRaisesRegex(ValueError, 'independent timing'):
            joint_gain_timing(reference, reference, (2+3j)*reference)

    def test_mismatched_live_zero_window_is_rejected_before_fitting(self):
        payload = dict(linear=np.zeros(4), chosen_shift=0, windows=np.ones((4,2)),
                       timing_window_zero=np.ones((4,2)) + .001,
                       timing_window_derivative=np.zeros((4,2)), timing_fd_step=.05)
        with self.assertRaisesRegex(ValueError, 'zero-offset windows disagree'):
            analyse(payload, {}, 'answer', (0,0,0,0))


if __name__ == '__main__':
    unittest.main()
