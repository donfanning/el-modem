"""Clause-level checks with scripted peers, not our other modem as oracle."""
import itertools
import unittest
from unittest.mock import patch
import numpy as np
import v34_ans
import v34_call
import v34_data as vd
import v34_negotiation as vn
from v34_preemphasis import PreEmphasis
from v34_preemphasis_coefficients import COEFFICIENTS
from v34_tx import SYMBOL_RATES, CARRIERS

class DuplexTests(unittest.TestCase):
    def test_answerer_initiates_then_answers_b_after_40ms(self):
        a=v34_ans.V34Answerer(); a.info0c={}; a.stage='toneA'
        a._src=v34_ans.ToneASrc(a.amp); a._t_a=500
        a.t,a.rx_t=900,1900
        with patch.object(a.tone_b,'steady_for',return_value=True): a._step()
        self.assertEqual(a.stage,'wait_first_b')
        a.t,a.rx_t=1300,2300; a.tone_b.reversals=[2280]; a._step()
        self.assertEqual(a.rtd_samples,60)
        self.assertEqual(a._actions[0][0],1600)
        a.tx(301); self.assertEqual(a._actions[0][0],1680)

    def test_cessation_is_not_a_phase_reversal(self):
        tr=v34_ans.ToneTracker(1200); t=np.arange(800)/8000
        tr.feed(np.concatenate((1000*np.cos(2*np.pi*1200*t),np.zeros(160))))
        self.assertEqual(tr.reversals,[])

    def test_caller_requires_second_a_for_ranging(self):
        c=v34_call.V34Caller(); c.stage='b_done'; c.t=c.rx_t=1000
        c._b_reversal_rx=900; c.a_reversals=[500]; c._step()
        self.assertEqual(c.stage,'b_done')
        c.a_reversals.append(1300)
        with patch.object(c,'_probing'): c._step()
        self.assertEqual(c.rtd_samples,80)

class NegotiationTests(unittest.TestCase):
    def test_asymmetric_waveforms_decode_the_remote_rate(self):
        caller = v34_call.V34Caller(data_rate=33600)
        answerer = v34_ans.V34Answerer(data_rate=33600)
        answerer._local_mp['max_c2a'] = 3
        sent = False
        outgoing, incoming = bytes(range(128)), bytes(reversed(range(128)))
        for _ in range(1000):
            c, a = caller.tx(160), answerer.tx(160)
            caller.rx(a); answerer.rx(c)
            if caller.stage == answerer.stage == 'data' and not sent:
                caller.feed_bytes(outgoing); answerer.feed_bytes(incoming); sent = True
            if sent and bytes(caller.rx_bytes) == incoming and bytes(answerer.rx_bytes) == outgoing:
                break
        self.assertEqual((caller.data_rate, caller.rx_rate), (7200, 33600))
        self.assertEqual((answerer.data_rate, answerer.rx_rate), (33600, 7200))
        self.assertEqual(bytes(caller.rx_bytes), incoming)
        self.assertEqual(bytes(answerer.rx_bytes), outgoing)

    def resolve(self,role='call',**updates):
        return vn.resolve(vn.offer(33600),{**vn.offer(33600),**updates},role,
                          SYMBOL_RATES[5],SYMBOL_RATES[5])
    def test_asymmetric_both_roles(self):
        for role,rates in [('call',(7200,33600)),('answer',(33600,7200))]:
            n=self.resolve(role,max_c2a=3)
            self.assertEqual((n.tx.rate,n.rx.rate),rates)
    def test_symmetric_if_either_declines(self):
        n=self.resolve(asym=0,max_c2a=3)
        self.assertEqual((n.tx.rate,n.rx.rate),(7200,7200))
        local=vn.offer(33600); local['asym']=0
        self.assertFalse(vn.resolve(local,vn.offer(9600),'call',2400,2400).asymmetric)
    def test_mask_holes_and_invalid_selections(self):
        self.assertEqual(self.resolve(mask=(1<<11)|(1<<1)).tx.rate,28800)
        for fields in [dict(mask=0),dict(max_c2a=0),dict(max_a2c=15),dict(trellis=3)]:
            with self.subTest(fields=fields),self.assertRaises(ValueError): self.resolve(**fields)
    def test_all_symbol_rates_have_representable_rates(self):
        for baud in SYMBOL_RATES.values():
            self.assertTrue(any(vn.supported(n*2400,baud) for n in range(1,15)))
        self.assertFalse(vn.supported(33600,2400))
    def test_peer_fields_command_our_transmitter(self):
        n=self.resolve(trellis=2,shaping=1,nonlinear=1,precoder=[1,2,3,4,5,6])
        self.assertEqual(n.tx.trellis,2); self.assertEqual(n.rx.trellis,0)
        self.assertTrue(n.tx.shaping); self.assertFalse(n.rx.shaping)
    def test_capabilities_stay_stable(self):
        for cls,attr,rxrate in [(v34_call.V34Caller,'mp_ans',33600),(v34_ans.V34Answerer,'mp_call',7200)]:
            m=cls(data_rate=33600); m.call_baud=m.ans_baud=SYMBOL_RATES[5]
            setattr(m,attr,{**vn.offer(33600),'max_c2a':3}); m._prepare_data_mode()
            self.assertEqual(m.rx_rate,rxrate); self.assertEqual(m._local_mp['max_c2a'],14)

class EncoderTests(unittest.TestCase):
    def test_all_delay_states_against_diagram_nodes(self):
        # Independent node-by-node Figure 11/12 transcription. Enumerate both AND gates.
        for selection,size in [(1,5),(2,6)]:
            for state in itertools.product((0,1),repeat=size):
                for value in range(16):
                    y1,y2,y3,y4=[(value>>k)&1 for k in range(4)]
                    if size==5:
                        expect=[state[4],state[0]^y2,state[1]^y1,state[2]^y4,state[3]^y2]
                    else:
                        left=state[0]^state[1]; middle=state[1]^y1
                        expect=[left^(middle&state[2])^y4,left^y3^(y2&state[2])^state[3],
                                middle^state[2],state[2],state[5],state[4]^y2^state[2]]
                    e=vd.convolutional_encoder(selection); e.d=list(state)
                    self.assertEqual(e.y0,state[-1]); e.step(value); self.assertEqual(e.d,expect)
        with self.assertRaises(ValueError): vd.convolutional_encoder(-1)

class PreEmphasisTests(unittest.TestCase):
    def test_all_responses_match_itu_relative_template(self):
        f=np.linspace(0,4000,2049)
        for nominal,high,index in COEFFICIENTS:
            baud=next(v for v in SYMBOL_RATES.values() if round(v)==nominal)
            fc=CARRIERS[nominal][high]; taps=np.array(COEFFICIENTS[(nominal,high,index)])
            band=(f>=fc-.45*baud)&(f<=fc+.45*baud); x=f[band]/baud
            h=np.exp(-2j*np.pi*f[band,None]/8000*np.arange(len(taps)))@taps
            target=2*index*x if index<=5 else np.interp(x,[0,.4,.8,1.2],[0,0,(index-5)*.5,(index-5)*1.5])
            err=20*np.log10(np.abs(h))-target; err-=np.mean(err)
            self.assertLess(np.max(np.abs(err)),1.,(nominal,high,index))
    def test_block_continuity_and_flat_bypass(self):
        x=np.random.default_rng(5).normal(size=1000)
        a=PreEmphasis(3200,CARRIERS[3200][0],5); b=PreEmphasis(3200,CARRIERS[3200][0],5)
        np.testing.assert_allclose(a.process(x),np.concatenate([b.process(x[:23]),b.process(x[23:])]))
        np.testing.assert_array_equal(PreEmphasis(2400,1600,0).process(x),x)
        self.assertEqual(a.delay_samples,64)

if __name__=='__main__': unittest.main()
