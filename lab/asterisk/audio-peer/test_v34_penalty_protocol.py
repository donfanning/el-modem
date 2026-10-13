"""Penalty protocol checks against production fitting and PCM causality."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'tools'))
from equalizer_penalty_compare import fit, guard_split, check_disjoint, pcm_support, cascade_noise_matrix
from v34_rx import Phase3Rx


class PenaltyProtocolTests(unittest.TestCase):
    def receiver(self, pcm):
        receiver = Phase3Rx(3428.5714285714284, 1959.1836734693877, 'answer')
        receiver.lag, receiver.delay = 512, .25
        receiver._span, receiver._L = 24, 48
        receiver._z = receiver.update_baseband(pcm)
        return receiver

    def test_alpha_matches_actual_production_ridge(self):
        rng = np.random.default_rng(2)
        matrix = rng.standard_normal((120, 48)) + 1j * rng.standard_normal((120, 48))
        known = rng.standard_normal(120) + 1j * rng.standard_normal(120)
        receiver = Phase3Rx(3428.5714285714284, 1959.1836734693877, 'answer')
        receiver._design = lambda start, count: matrix
        receiver.train_known(known, 0, span=24, reg=1e-3)
        np.testing.assert_allclose(fit(matrix, known, 'ridge', 1e-3), receiver.taps, rtol=1e-12, atol=1e-12)

    def test_guards_isolate_validation_from_fit_pcm_perturbation(self):
        pcm = np.random.default_rng(5).standard_normal(4000)
        receiver = self.receiver(pcm)
        blocks = guard_split(716, 24, receiver)
        self.assertFalse(check_disjoint(blocks, 24, receiver))
        touched = pcm_support(blocks['fit'], receiver, 24)
        changed = pcm.copy()
        changed[touched[0]:touched[1]] += 1
        perturbed = self.receiver(changed)
        for name in ('selection', 'validation'):
            lo, hi = blocks[name]
            np.testing.assert_array_equal(receiver._design(lo, hi-lo), perturbed._design(lo, hi-lo))
        old_blocks = guard_split(716, 24)
        self.assertTrue(check_disjoint(old_blocks, 24, receiver))

    def test_noise_retains_realization_axis(self):
        receiver = self.receiver(np.zeros(4000))
        capture = dict(k_start=0, y=np.zeros(12))
        matrix = cascade_noise_matrix(receiver, capture, 48)
        self.assertEqual(matrix.shape, (12, 12, 48))
        self.assertFalse(np.array_equal(matrix[0], matrix[1]))

    def test_penalties_agree_without_regularization(self):
        rng = np.random.default_rng(7)
        matrix = rng.standard_normal((80, 48)) + 1j * rng.standard_normal((80, 48))
        known = rng.standard_normal(80)
        np.testing.assert_allclose(fit(matrix, known, 'ridge', 0), fit(matrix, known, 'difference', 0))


if __name__ == '__main__':
    unittest.main()
