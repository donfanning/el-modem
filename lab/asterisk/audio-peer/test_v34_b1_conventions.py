"""B1 qualification under both differential-encoder conventions (10.1.3.1 reset, or continued from E).

Synthetic B1 streams come from the independent reference with a chosen initial differential
rotation Z0. An odd Z0 must qualify under the dual-convention policy as the 'continued'
convention (initial state [0,1,0,0] with a free complex gain) and be rejected by the standard
policy; an even Z0 stays the standard zero-state match. The data detector must start from the
matched convention's end-of-B1 state. See lab/V34_B1_INITIAL_ROTATION.md.
"""
import os
import unittest
from types import SimpleNamespace
import numpy as np
import v34_b1_reference as ref
import v34_negotiation as vn
import v34_receive as rx

COEFF=(835,259,-842,-106,775,-157)
GAIN=.11*np.exp(1j*.6)


class StubEq:
    """Minimal equalizer: one tap, window k is the k-th synthetic B1 observation."""
    class R:
        lag=0;LEAD=0;delay=0;spl=2.;fc=1959.
    def __init__(self,observed):
        self.W=np.asarray(observed,complex)[:,None];self.h=np.array([1+0j])
        self.k=0;self.phase=0.;self.freq=0.;self.dt=0.;self.adapt=True;self.began=None
    def available(self):return 10**6
    def window(self,k):return self.W[k%len(self.W)]
    def step(self,decide=None):
        y=complex(self.window(self.k)@self.h);self.k+=1;return y
    def timing_snapshot(self):
        return dict(sample_position=0.,samples_per_symbol=2.,timing_velocity=0.,carrier_phase=self.phase)
    def begin_data_tracking(self,precoded):self.began=precoded


def run(z0,policy):
    profile=vn.DirectionParameters(33600,3429,1959,shaping=True,nonlinear=False,precoder=COEFF)
    built=ref.build(33600,3429,'call',True,COEFF,initial_rotation=z0,trace=True)
    observed=GAIN*np.asarray(built['x'])
    engine=SimpleNamespace(negotiated=SimpleNamespace(rx=profile),_rx_b1=0,_p4_x0=0,_receive_trace=False,
        rx_qualification={},failed_reason=None,_ev=lambda m:None,_local_mp={'mask':0})
    eq=StubEq(observed)
    ready=rx.initialize(engine,eq,'call',3429,policy=policy)
    return ready,engine,eq,built


class ConventionTests(unittest.TestCase):
    def test_dual_policy_definition_and_default_policy_unchanged(self):
        d=rx.SearchPolicy.dual_convention()
        self.assertEqual(d.initial_states,((0,0,0,0),(0,1,0,0)))
        self.assertFalse(d.rate_search);self.assertEqual(d.boundary_offsets,(0,))
        self.assertEqual(rx.SearchPolicy(),rx.SearchPolicy('standard',False,((0,0,0,0),),(0,)))

    def test_live_default_is_dual_and_can_be_reverted(self):
        old=os.environ.pop('V34_B1_CONVENTIONS',None)
        try:
            self.assertEqual(rx.live_policy().label,rx.DUAL_CONVENTION)
            os.environ['V34_B1_CONVENTIONS']='standard'
            self.assertEqual(rx.live_policy(),rx.SearchPolicy())
        finally:
            os.environ.pop('V34_B1_CONVENTIONS',None)
            if old is not None:os.environ['V34_B1_CONVENTIONS']=old

    def test_even_rotation_picks_zero_state_and_is_standard(self):
        for z0 in (0,2):
            ready,engine,_,_=run(z0,rx.SearchPolicy.dual_convention())
            q=engine.rx_qualification
            self.assertTrue(ready)
            self.assertEqual(q['bit_errors'],0)
            self.assertEqual(q['initial_trellis_state'],[0,0,0,0])
            self.assertEqual(q['differential_convention'],'reset')
            self.assertTrue(q['standard_qualification'])

    def test_odd_rotation_qualifies_as_continued_and_is_not_standard(self):
        for z0 in (1,3):
            ready,engine,_,_=run(z0,rx.SearchPolicy.dual_convention())
            q=engine.rx_qualification
            self.assertTrue(ready)
            self.assertEqual(q['bit_errors'],0)
            self.assertEqual(q['initial_trellis_state'],[0,1,0,0])
            self.assertEqual(q['differential_convention'],'continued')
            self.assertFalse(q['standard_qualification'])
            self.assertEqual(q['convention_policy'],rx.DUAL_CONVENTION)

    def test_standard_policy_rejects_odd_rotation(self):
        for z0 in (1,3):
            ready,engine,_,_=run(z0,rx.SearchPolicy())
            self.assertFalse(ready)
            self.assertGreater(engine.rx_qualification.get('bit_errors',999),100)
        self.assertTrue(run(0,rx.SearchPolicy())[0])

    def test_selection_uses_calibration_only(self):
        # The scores that rank the two states are first-half fits; corrupt the second half
        # so B1 errors are large and the choice must still follow the first half.
        built=ref.build(33600,3429,'call',True,COEFF,initial_rotation=1)
        wave=GAIN*np.asarray(built['x']);wave[len(wave)//2:]*=-1
        linear=np.asarray([ref.build(33600,3429,'call',True,COEFF,s)['x'] for s in ((0,0,0,0),(0,1,0,0))])
        scores=rx.calibration_scores(np.asarray([wave]),linear,False)
        self.assertEqual(int(np.argmin(scores[:,0])),1)

    def test_data_detector_starts_from_the_matched_conventions_end_state(self):
        for z0,state in ((0,(0,0,0,0)),(1,(0,1,0,0)),(3,(0,1,0,0))):
            ready,engine,_,built=run(z0,rx.SearchPolicy.dual_convention())
            self.assertTrue(ready)
            truth=ref.build(33600,3429,'call',True,COEFF,state,trace=True)['alignment'][-1]['state_after']
            drx=engine._rx_state['drx']
            self.assertEqual(list(drx.conv.d),truth)
            start=engine._rx_state['receiver'].detector(start=True)
            self.assertEqual(int(np.argmin(start.metric)),sum(v<<(3-i) for i,v in enumerate(truth)))


if __name__=='__main__':unittest.main()
