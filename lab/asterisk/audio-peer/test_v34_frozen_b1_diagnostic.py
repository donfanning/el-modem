"""Ensure B1 diagnostic traces describe one qualification, at the chosen boundary."""
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from frozen_b1_path_costs import path_costs, stage_observed
from b1_refinement_attribution import decode_with, fit_capacity
from b1_tap_correction import correction_spectrum
from frozen_trn_drift_probe import timing_projection
from v34_negotiation import DirectionParameters
from v34_receive import Receiver


class FrozenB1DiagnosticTests(unittest.TestCase):
    def test_trace_is_one_burst_and_matches_production_qualification(self):
        profile = DirectionParameters(24000, 24000 / 7, 1959.1836734693877)
        receiver = Receiver(profile, 'answer')
        observed = (.7 + .2j) * receiver.linear
        receiver.qualify(observed, trace=True)
        rows, detector = path_costs(observed, asdict(profile), 'answer', (0, 0, 0, 0))
        n = len(observed) // 2
        self.assertEqual(len(rows), n)
        self.assertEqual(detector.m4, n)
        self.assertEqual(len(detector.decided), n)
        self.assertEqual(detector.decided, receiver.b1_detector.decided)
        self.assertEqual(receiver.diagnostics['bit_errors'], 0)
        for got, expected in zip(detector.trace, receiver.b1_detector.trace):
            np.testing.assert_array_equal(got['metric_after'], expected['metric_after'])

    def test_boundary_offset_is_not_applied_twice_to_refined_capture(self):
        payload = dict(linear=np.zeros(4), chosen_shift=2,
                       observed_before=np.arange(8), observed_after=np.arange(4) + 10)
        np.testing.assert_array_equal(stage_observed(payload, 'before'), [2, 3, 4, 5])
        np.testing.assert_array_equal(stage_observed(payload, 'after'), [10, 11, 12, 13])

    def test_explicit_calibration_matches_production_bits_and_mapper_errors(self):
        profile = DirectionParameters(24000, 24000 / 7, 1959.1836734693877)
        receiver = Receiver(profile, 'answer')
        observed = (.7 + .2j) * receiver.linear
        observed[-12:] += 20  # Exercise error reporting, not just a perfect decode.
        gain, _ = receiver.qualify(observed)
        result = decode_with(observed, gain, receiver.energy, asdict(profile), 'answer', (0, 0, 0, 0))
        self.assertEqual(result['bit_errors'], receiver.diagnostics['bit_errors'])
        self.assertEqual(result['error_bit_indices'], receiver.diagnostics['error_bit_indices'])
        self.assertEqual(result['mapper_errors'], receiver.diagnostics['mapper_symbol_errors'])

    def test_capacity_counts_complex_observations_and_actual_ridge_shrinkage(self):
        result = fit_capacity(np.eye(4, 3, dtype=complex))
        self.assertEqual(result['real_observations'], 8)
        self.assertEqual(result['nominal_real_taps'], 6)
        self.assertAlmostEqual(result['conditional_complex_effective_df'], 3 / (1 + 1e-4))

    def test_complex_singular_coordinates_reconstruct_the_correction(self):
        rng = np.random.default_rng(44)
        matrix = rng.normal(size=(16, 8)) + 1j * rng.normal(size=(16, 8))
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        delta = vh.conj().T[:, -1]
        result = correction_spectrum(matrix, delta)
        self.assertEqual(result['dominant_correction_modes'][0], 7)
        self.assertAlmostEqual(result['correction_energy_by_mode'][-1], 1.)
        self.assertLess(result['reconstruction_relative_error'], 1e-12)

    def test_zero_correction_has_no_energy_fraction(self):
        result = correction_spectrum(np.eye(4), np.zeros(4))
        self.assertTrue(result['zero_correction'])
        self.assertIsNone(result['correction_share_top_matrix_energy_modes'])

    def test_real_timing_score_removes_complex_gain_nuisance(self):
        rng = np.random.default_rng(41)
        known = rng.normal(size=64) + 1j * rng.normal(size=64)
        derivative = rng.normal(size=64) + 1j * rng.normal(size=64)
        gain = .3 + .7j
        for offset in (-.1, .08):
            observed = gain * known - offset * derivative + (.05 - .02j) * known
            result = timing_projection(observed, derivative, known, gain)
            self.assertAlmostEqual(result['correction_pcm_samples'], offset, places=12)

    def test_timing_score_reports_no_information_when_derivative_is_gain(self):
        known = np.array([1 + 1j, 2 - 1j, -1 + 3j])
        result = timing_projection(known, 2j * known, known, 1.)
        self.assertIsNone(result['correction_pcm_samples'])


if __name__ == '__main__':
    unittest.main()
