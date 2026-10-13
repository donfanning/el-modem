"""Independent channel interventions for calibration and state-specific refits."""
import unittest
import numpy as np
import v34_receive as receive
from v34_negotiation import DirectionParameters


class StagingTests(unittest.TestCase):
    def test_block_fit_does_not_see_other_half(self):
        x=np.tile(np.array([1+1j,3-1j,-1-3j,-3+3j]),20)
        y=x.copy();y[len(x)//2:]*=2*np.exp(.3j)
        first=receive.calibrate(y,x,False)[0]
        second=receive.calibrate(y,x,False,fit_indices=np.arange(len(x)//2,len(x)))[0]
        self.assertAlmostEqual(abs(first-1),0)
        self.assertAlmostEqual(abs(second-2*np.exp(.3j)),0)
        for ids in ([],[0,0],[-1],[len(x)],[.5]):
            with self.assertRaises(ValueError):receive.calibrate(y,x,False,fit_indices=ids)

    def test_refinement_target_depends_on_state_and_not_late_input(self):
        profile=DirectionParameters(33600,3429,1959)
        r=receive.Receiver(profile,'call')
        alternate=receive.Receiver(profile,'call',(0,1,0,0))
        rng=np.random.default_rng(712)
        W=rng.normal(size=(len(r.linear),8))+1j*rng.normal(size=(len(r.linear),8))
        prior=np.zeros(8,complex);prior[4]=1
        observed=W@prior
        _,h=receive.refine_taps(r,W,observed,prior)
        corrupted=observed.copy();corrupted[len(corrupted)//2:]*=17j
        _,unchanged=receive.refine_taps(r,W,corrupted,prior)
        np.testing.assert_array_equal(h,unchanged)
        _,other=receive.refine_taps(alternate,W,observed,prior)
        self.assertGreater(np.linalg.norm(h-other),1e-4)


if __name__=='__main__':unittest.main()
