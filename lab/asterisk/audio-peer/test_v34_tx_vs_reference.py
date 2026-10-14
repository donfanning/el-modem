import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
import tx_vs_reference as T


class DataModeTransmitAgainstReferenceTests(unittest.TestCase):
    def test_zero_precoder_without_and_with_shaping_match_exactly(self):
        for shaping in (False, True):
            r = T.compare_frames(33600, 3429, shaping, [0] * 6, 20)
            self.assertEqual(r['mismatches'], 0, (shaping, r))

    def test_the_queens_precoder_with_shaping_matches_exactly_past_b1(self):
        r = T.compare_frames(33600, 3429, True, T.QUEENS_PRECODER, 40, seed=7)
        self.assertEqual((r['mismatches'], r['max_diff']), (0, 0.0), r)
        self.assertGreater(r['symbols'], 40 * r['symbols_per_frame'])

    def test_a_much_larger_precoder_also_matches(self):
        r = T.compare_frames(33600, 3429, True, [c * 40 for c in T.QUEENS_PRECODER], 20, seed=3)
        self.assertEqual(r['mismatches'], 0, r)

    def test_other_symbol_rates(self):
        for baud, rate in ((2400, 14400), (3200, 28800)):
            r = T.compare_frames(rate, baud, True, T.QUEENS_PRECODER, 20, seed=5)
            self.assertEqual(r['mismatches'], 0, (baud, rate, r))

    def test_negative_control_a_different_precoder_in_the_transmitter_is_caught(self):
        r = T.compare_frames(33600, 3429, True, T.QUEENS_PRECODER, 40, seed=7, tx_precoder=[323, 0, -162, 16, -8, 37])
        self.assertGreater(r['mismatches'], 0, r)

    def test_nonlinear_encoder_follows_the_spec_formula(self):
        r = T.nonlinear_report(33600, 3429, True, T.QUEENS_PRECODER, nframes=100)
        self.assertLess(abs(r['energy_ratio'] - 1), 0.01, r)
        self.assertLess(r['max_amplitude_error'], 0.005, r)


if __name__ == '__main__':
    unittest.main()
