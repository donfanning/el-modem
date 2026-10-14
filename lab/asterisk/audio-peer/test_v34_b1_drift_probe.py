"""Physical slope fitting and exact live-context interpolation gates."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from b1_drift_probe import fit, exact_observation, block_noise
from v34_rx import interp


class B1DriftProbeTests(unittest.TestCase):
    def test_real_slopes_are_separated_from_complex_gain_and_static_timing(self):
        rng=np.random.default_rng(49); n=64
        ref=rng.normal(size=n)+1j*rng.normal(size=n)
        derivative=rng.normal(size=n)+1j*rng.normal(size=n)
        t=np.linspace(-.01,.01,n); phase_ref=(.2+.3j)*ref
        truth=np.array([.2,.3,-.02,.8,-4.])
        y=(truth[0]+1j*truth[1])*ref+(truth[2]+truth[3]*t)*derivative+1j*truth[4]*t*phase_ref
        coef,pred,_=fit(y,ref,derivative,t,phase_ref)
        np.testing.assert_allclose(coef,truth,atol=1e-12)
        np.testing.assert_allclose(pred,y,atol=1e-12)

    def test_exact_interpolator_zero_parity_and_support_rejection(self):
        z=np.exp(.17j*np.arange(50)); pos=np.array([15.1,16.2,17.3]); taps=np.array([.2,.7,.1j])
        p=dict(timing_sources=np.array([z,z]),timing_positions=np.array([pos,pos+.2]),
               timing_rotations=np.array([1.,1j]),timing_exact_radius=2.)
        actual=exact_observation(p,taps,np.zeros(2),np.zeros(2))
        expected=np.array([interp(z,pos)@taps,1j*(interp(z,pos+.2)@taps)])
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'support'):
            exact_observation(p,taps,np.array([0.,2.1]),np.zeros(2))

    def test_block_noise_preserves_complex_pair_relationship(self):
        r=np.arange(60)*(1+2j)
        out=block_noise(r,8,np.random.default_rng(1))
        self.assertEqual(len(out),60)
        np.testing.assert_allclose(out.imag,2*out.real)

    def test_rank_deficiency_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            fit(np.ones(60),np.ones(60),np.ones(60),np.arange(60),np.ones(60))
