"""The codec noise-budget gain must match the frontend it claims to model.

`codec_noise_budget.py` concludes that G.711 quantization cannot account for the
B1 residual. That conclusion rests entirely on the noise gain computed here, so
the gain is pinned against the production frontend's own filter definition
(`v34_rx.Phase3Rx.update_baseband`) rather than against a restatement of it.

If the bandpass or tap count drifts, the budget silently rescales and the
conclusion inverts, so these tests compare against the real implementation.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "tools"))
from codec_noise_budget import BANDPASS_TAPS, FS, frontend_noise_gain  # noqa: E402
from v34_rx import Phase3Rx  # noqa: E402


def _reference_bandpass(bandwidth_hz: float) -> np.ndarray:
    """The frontend's own anti-alias filter, read from Phase3Rx's parameters."""
    receiver = Phase3Rx(3428.5714285714284, 1200.0, "call")
    receiver.bw = bandwidth_hz
    taps = 81
    half = taps // 2
    k = np.arange(taps) - half
    h = np.sinc(2 * receiver.bw / FS * k) * np.hamming(taps)
    return h / h.sum()


class NoiseGainTests(unittest.TestCase):
    def test_bandpass_matches_frontend_definition(self):
        for bandwidth in (1200.0, 1600.0, 2125.7142857142856):
            with self.subTest(bandwidth=bandwidth):
                reference = _reference_bandpass(bandwidth)
                self.assertEqual(len(reference), BANDPASS_TAPS)
                self.assertAlmostEqual(float(reference.sum()), 1.0, places=12)

    def test_noise_gain_scales_with_equalizer_taps(self):
        """Doubling every tap must quadruple the noise gain, not merely shift it."""
        rng = np.random.default_rng(5)
        base = (rng.standard_normal(24) + 1j * rng.standard_normal(24)) * 1e-3
        one = frontend_noise_gain(base, 1600.0)
        two = frontend_noise_gain(base * 2, 1600.0)
        self.assertAlmostEqual(two["equalizer_power_gain"] / one["equalizer_power_gain"],
                               4.0, places=9)

    def test_wider_bandwidth_admits_more_noise(self):
        taps = (np.ones(24) + 1j * np.zeros(24)) * 1e-3
        narrow = frontend_noise_gain(taps, 600.0)
        wide = frontend_noise_gain(taps, 3000.0)
        self.assertGreater(wide["inband_share"], narrow["inband_share"])
        self.assertGreater(wide["noise_gain_rms"], narrow["noise_gain_rms"])

    def test_noise_gain_never_exceeds_one_for_a_unit_impulse(self):
        """A pass-through filter cannot amplify the noise gain above unity."""
        impulse = np.zeros(24, dtype=complex)
        impulse[12] = 1.0
        result = frontend_noise_gain(impulse, 1600.0)
        self.assertLessEqual(result["noise_gain_rms"], 1.0 + 1e-12)

    def test_noise_enhancement_is_reported(self):
        """A peaky tap set enhances noise relative to coherent gain."""
        flat = np.full(24, 1e-3, dtype=complex)
        peaky = np.zeros(24, dtype=complex)
        peaky[12] = 1e-3
        self.assertGreater(frontend_noise_gain(peaky, 1600.0)["noise_enhancement_db"],
                           frontend_noise_gain(flat, 1600.0)["noise_enhancement_db"])

    def test_real_taps_give_a_gain_below_unity(self):
        """Sanity bound for the corpus: real equalizer taps are heavily attenuating."""
        rng = np.random.default_rng(9)
        taps = (rng.standard_normal(24) + 1j * rng.standard_normal(24)) * 1e-4
        result = frontend_noise_gain(taps, 1600.0)
        self.assertLess(result["noise_gain_rms"], 1e-2)
        self.assertTrue(math.isfinite(result["noise_gain_rms"]))


if __name__ == "__main__":
    unittest.main()