"""Parity checks for the opt-in fused data-mode decision/feedback kernel."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import v34_feedback_native as feedback
import v34_receive as receive
from v34_negotiation import DirectionParameters
from v34_rx import Phase3Rx,SymbolEq


class FusedDecisionFeedbackTests(unittest.TestCase):
    def test_decision_and_equalizer_state_match(self):
        rng=np.random.default_rng(730241)
        for nonlinear,coeff in ((False,(0,0,0,0,0,0)),
                                (True,(129,188,-297,-160,222,-69))):
            profile=DirectionParameters(33600,24000/7,1959.1836734693877,
                shaping=True,nonlinear=nonlinear,precoder=coeff)
            baseline=receive.Receiver(profile,'answer');candidate=receive.Receiver(profile,'answer')
            baseline.energy=candidate.energy=675.
            baseline.slicer_hist=candidate.slicer_hist=[.31-.22j,-.17+.09j,.04+.12j]
            h0=(rng.normal(size=48)+1j*rng.normal(size=48)).astype(np.complex128)*.01
            received=np.concatenate(((rng.normal(size=96)+1j*rng.normal(size=96))*2,
                                     np.array([0j,2+0j,2+2j,100+0j,-100+0j])))
            windows=(rng.normal(size=(len(received),48))+1j*rng.normal(size=(len(received),48))).astype(np.complex128)
            for i,(yr,w) in enumerate(zip(received,windows)):
                h_ref=h0.copy();h_fused=h0.copy();phase0=.17+i*1e-5;freq0=-.00013
                d=baseline.decide(complex(yr))
                phase_ref,freq_ref=feedback.feedback(h_ref,w,complex(yr),d,.2,.01,5e-6,phase0,freq0)
                eq=SimpleNamespace(h=h_fused,mu=.2,kp=.01,ki=5e-6)
                d2,phase_fused,freq_fused=candidate.decide_and_feedback(eq,w,complex(yr),phase0,freq0)
                # The fused and split implementations evaluate equivalent
                # complex expressions in a different order; allow roundoff
                # at the last few binary64 bits, while keeping a tight bound.
                np.testing.assert_allclose(d2,d,rtol=2e-14,atol=2e-14)
                np.testing.assert_allclose(h_fused,h_ref,rtol=2e-14,atol=2e-14)
                np.testing.assert_allclose((phase_fused,freq_fused),(phase_ref,freq_ref),
                                           rtol=2e-14,atol=2e-14)
                np.testing.assert_allclose(candidate.slicer_hist,baseline.slicer_hist,
                                           rtol=2e-14,atol=2e-14)

    def test_symbol_eq_trajectory_matches_split_path(self):
        rng=np.random.default_rng(730242)
        profile=DirectionParameters(33600,24000/7,1959.1836734693877,
            shaping=True,nonlinear=True,precoder=(129,188,-297,-160,222,-69))
        baseline=receive.Receiver(profile,'answer');fused_rx=receive.Receiver(profile,'answer')
        baseline.energy=fused_rx.energy=675.
        initial=[.31-.22j,-.17+.09j,.04+.12j]
        baseline.slicer_hist=initial.copy();fused_rx.slicer_hist=initial.copy()
        source=(rng.normal(size=6000)+1j*rng.normal(size=6000)).astype(np.complex128)
        def make_eq():
            R=Phase3Rx(24000/7,1959.1836734693877,'answer');R._z=source.copy()
            R._span=24;R.taps=np.zeros(48,complex);R.taps[24]=1;R.lag=80;R.delay=0
            return SymbolEq(R,24)
        with patch.dict('os.environ',{'V34_NATIVE_FEEDBACK':'1','V34_NATIVE_DECISION_FEEDBACK':'1'}):
            split=make_eq();fused=make_eq()
        split._native_feedback=fused._native_feedback=True
        split._native_decision_feedback=False;fused._native_decision_feedback=True
        split.adapt=fused.adapt=True
        left=[];right=[]
        for _ in range(240):
            left.append(split.step(decide=baseline.decide))
            right.append(fused.step(decide=fused_rx.decide))
        np.testing.assert_array_equal(right,left);np.testing.assert_array_equal(fused.h,split.h)
        np.testing.assert_array_equal(fused_rx.slicer_hist,baseline.slicer_hist)
        for key in ('phase','freq','dt','vel'):self.assertEqual(getattr(fused,key),getattr(split,key))


if __name__=='__main__':unittest.main()
