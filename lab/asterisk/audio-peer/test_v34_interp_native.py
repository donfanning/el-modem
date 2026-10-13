"""Independent NumPy parity at fractional coordinates and clipped edges."""
import unittest
import os
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from v34_interp_native import interpolate, warmup
from v34_rx import interp


class NativeInterpTests(unittest.TestCase):
    def test_centered_phase_and_direct_support_limits_at_audio_scale(self):
        rng=np.random.default_rng(3429)
        z=10000*(rng.normal(size=512)+1j*rng.normal(size=512))
        fractions=np.array([0.,1e-12,1e-8,1e-7,2e-7,.0003,.01,
                            .5-1e-12,.5,.5+1e-12,.99,.9997,
                            1-2e-7,1-1e-7,1-1e-8,1-1e-12])
        positions=(np.arange(-10,522)[:,None]+fractions).ravel()
        # Support4 takes the original formula, supports2/6/8 are factored.
        for half in (2,4,6,8):
            np.testing.assert_allclose(interpolate(z,positions,half),interp(z,positions,half),
                                       atol=3e-11,rtol=3e-14)

    def test_symbol_window_opt_in_and_numpy_fallback(self):
        from v34_rx import SymbolEq
        import v34_interp_native as native
        rng=np.random.default_rng(26400)
        r=SimpleNamespace(_span=24,spl=7/3,lag=50,LEAD=8,delay=.5,
                          taps=np.ones(48,complex),
                          _z=rng.normal(size=4096)+1j*rng.normal(size=4096))
        with patch.dict(os.environ,{'V34_SYMBOL_INTERP_NATIVE':'0'}):
            reference=SymbolEq(r,40)
        with patch.dict(os.environ,{'V34_SYMBOL_INTERP_NATIVE':'1'}):
            candidate=SymbolEq(r,40)
            self.assertEqual(candidate._native_interp,native.njit is not None)
            with patch.object(native,'njit',None):
                fallback=SymbolEq(r,40)
                self.assertFalse(fallback._native_interp)
        for dt in (.27,-.12,.0003):
            reference.dt=candidate.dt=fallback.dt=dt
            expected=reference.window(40)
            np.testing.assert_allclose(candidate.window(40),expected,atol=3e-15,rtol=3e-14)
            np.testing.assert_array_equal(fallback.window(40),expected)
        # A fitted symbol clock changes the spacing after construction.
        r.spl=2.5
        center=r.lag+(r.LEAD+40+r.delay)*r.spl+candidate.dt
        expected=interp(r._z,center+(np.arange(candidate.L)-candidate.span)*r.spl/2)
        np.testing.assert_allclose(candidate.window(40),expected,atol=3e-15,rtol=3e-14)

    def test_retained_row_windows_and_fit_match_numpy(self):
        from v34_trn_clock import windows, fit_clock
        rng=np.random.default_rng(340302)
        source=10000*(rng.normal(size=(96,80))+1j*rng.normal(size=(96,80)))
        positions=np.tile(12.37+np.arange(24)*7/6,(96,1))
        positions[0,0]=5.;positions[1,-1]=73.
        rotation=np.exp(1j*rng.uniform(-3,3,96))
        shift=rng.uniform(-.1,.1,96);shift[:2]=0
        with patch.dict(os.environ,{'V34_TRN_CLOCK_NATIVE':'0'}):
            expected=windows(source,positions,rotation,shift)
        with patch.dict(os.environ,{'V34_TRN_CLOCK_NATIVE':'1'}):
            actual=windows(source,positions,rotation,shift)
            with self.assertRaisesRegex(ValueError,'support'):
                windows(source,positions,rotation,np.full(96,100.))
        np.testing.assert_allclose(actual,expected,atol=3e-11,rtol=3e-14)
        positions=np.tile(12.37+np.arange(24)*7/6,(96,1))
        prior=rng.normal(size=24)+1j*rng.normal(size=24)
        with patch.dict(os.environ,{'V34_TRN_CLOCK_NATIVE':'0'}):
            wanted=windows(source,positions,rotation,shift)@prior
        fit=np.arange(96)<48
        models=[]
        for flag in ('0','1'):
            with patch.dict(os.environ,{'V34_TRN_CLOCK_NATIVE':flag}):
                models.append(fit_clock(source,positions,rotation,np.zeros(96),
                                        np.arange(96),wanted,prior,fit))
        for key in ('offset_samples','samples_per_symbol_correction'):
            self.assertAlmostEqual(models[0][key],models[1][key],places=12)

    def test_random_fractional_integer_and_clipped_positions(self):
        rng=np.random.default_rng(340301)
        z=rng.normal(size=1024)+1j*rng.normal(size=1024)
        positions=np.r_[rng.uniform(-12,1036,2048),np.arange(-10,1034),
                        np.arange(20,40)+1e-12,np.arange(20,40)-1e-12]
        for half in (2,6,8):
            np.testing.assert_allclose(interpolate(z,positions,half),interp(z,positions,half),
                                       atol=3e-15,rtol=3e-14)

    def test_production_finite_difference_at_audio_scale(self):
        rng=np.random.default_rng(33600)
        z=10000*(rng.normal(size=1000)+1j*rng.normal(size=1000))
        positions=np.arange(24)*7/6+400.37
        actual=(interpolate(z,positions+.05)-interpolate(z,positions-.05))/.1
        expected=(interp(z,positions+.05)-interp(z,positions-.05))/.1
        np.testing.assert_allclose(actual,expected,atol=2e-10,rtol=2e-13)
        self.assertEqual(len(interpolate(z,np.empty(0))),0)


if __name__=='__main__':unittest.main()
