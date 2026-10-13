"""Phase units, forward prediction and averaging-aware injection gates."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from trn_jitter_probe import basis,fit_phase,phase_blocks,analyse


class TRNJitterTests(unittest.TestCase):
    def test_recovers_signed_sine_cosine_and_linear_nuisance(self):
        t=np.linspace(-.2,.2,180)
        coef=np.array([.3,.07,.01,-.02,.04,.03,-.015,.01,.003,-.005])
        np.testing.assert_allclose(fit_phase(t,basis(t)@coef),coef,atol=1e-12)

    def test_phase_blocks_use_complex_gain_not_mean_noisy_angles(self):
        target=np.array([1,2,1,2,1,2,1,2],complex)
        t=np.arange(8)/3200
        times,phase=phase_blocks(target*(.7*np.exp(.4j)),target,t,0,8)
        np.testing.assert_allclose(phase,[.4],atol=1e-12)
        self.assertEqual(times[0],t.mean())

    def test_coherent_jitter_predicts_later_block_and_injection_is_filtered(self):
        t=np.arange(5500)/3200;known=np.ones(len(t),complex)
        y=known*np.exp(1j*(.2+.1*t+.05*np.sin(2*np.pi*60*t)))
        result=analyse(y,known,t)
        self.assertGreater(result['prediction_improvement_fraction'],.95)
        self.assertTrue(result['predictive_lead'])
        injection=result['injection']
        self.assertLess(injection['expected_filtered_sin_deg'],3.)
        self.assertAlmostEqual(injection['recovered_sin_deg'],injection['expected_filtered_sin_deg'],places=3)

    def test_rank_deficient_fit_and_zero_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'rank-deficient'):fit_phase(np.zeros(20),np.zeros(20))
        with self.assertRaisesRegex(ValueError,'zero reference'):
            phase_blocks(np.ones(8),np.zeros(8),np.arange(8),0,8)


if __name__=='__main__':unittest.main()
