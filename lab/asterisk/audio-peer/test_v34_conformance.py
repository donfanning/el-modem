"""Clause-level checks with scripted peers, not our other modem as oracle."""
import itertools
import unittest
from unittest.mock import patch
import numpy as np
import v34_ans
import v34_call
import v34_data as vd
import v34_mp
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
    def test_configured_minimum_excludes_lower_mp_rates(self):
        local = vn.offer(33600, minimum=9600)
        negotiated = vn.resolve(local, vn.offer(33600), 'call', SYMBOL_RATES[5], SYMBOL_RATES[5])
        self.assertGreaterEqual(negotiated.tx.rate, 9600)
        self.assertGreaterEqual(negotiated.rx.rate, 9600)
        with self.assertRaisesRegex(ValueError, 'no common V.34 rate'):
            vn.resolve(local, vn.offer(7200), 'call', SYMBOL_RATES[5], SYMBOL_RATES[5])

    def test_every_min_max_pair_reaches_both_production_mp_offers(self):
        for minimum_index in range(1, 15):
            for maximum_index in range(minimum_index, 15):
                expected_mask = ((1 << maximum_index) - 1) ^ ((1 << (minimum_index - 1)) - 1)
                for cls in (v34_call.V34Caller, v34_ans.V34Answerer):
                    modem = cls(data_rate=maximum_index * 2400,
                                min_data_rate=minimum_index * 2400)
                    mp = v34_mp.parse_mp(v34_mp.build_mp(1, **modem._local_mp))
                    self.assertEqual(mp['max_c2a'], maximum_index)
                    self.assertEqual(mp['max_a2c'], maximum_index)
                    self.assertEqual(mp['mask'], expected_mask)

    def test_asymmetric_waveforms_decode_the_remote_rate(self):
        outgoing = bytes((i * 73 + 19) & 0xff for i in range(4096))
        incoming = bytes((i * 151 + 7) & 0xff for i in range(4096))
        cases = (
            ('caller-to-answerer limited', 'answer', 'max_c2a', 3, (7200, 33600), (33600, 7200)),
            ('answerer-to-caller limited', 'call', 'max_a2c', 5, (33600, 12000), (12000, 33600)),
        )
        for name, limiter, field, cap, caller_rates, answerer_rates in cases:
            with self.subTest(name=name):
                caller = v34_call.V34Caller(data_rate=33600)
                answerer = v34_ans.V34Answerer(data_rate=33600)
                owner = answerer if limiter == 'answer' else caller
                owner._local_mp[field] = cap
                sent = False
                for _ in range(1000):
                    c, a = caller.tx(160), answerer.tx(160)
                    caller.rx(a); answerer.rx(c)
                    if caller.stage == answerer.stage == 'data' and not sent:
                        caller.feed_bytes(outgoing); answerer.feed_bytes(incoming); sent = True
                    if sent and bytes(caller.rx_bytes) == incoming and bytes(answerer.rx_bytes) == outgoing:
                        break
                self.assertEqual((caller.data_rate, caller.rx_rate), caller_rates)
                self.assertEqual((answerer.data_rate, answerer.rx_rate), answerer_rates)
                self.assertTrue(sent)
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

    def test_each_mp_rate_bit_resolves_at_each_symbol_rate_and_role(self):
        # The current mapper supports a contiguous rate range at each baud.
        # These explicit ceilings make mask-to-rate selection observable rather
        # than deriving the expected answer through vn.supported().
        ceilings = (10, 11, 11, 12, 13, 14)
        minimums = (1, 2, 2, 2, 2, 2)
        for symbol_index, (baud, ceiling, minimum) in enumerate(
                zip(SYMBOL_RATES.values(), ceilings, minimums)):
            for rate_index in range(1, 15):
                for role in ('call', 'answer'):
                    local = vn.offer(ceiling * 2400, minimum=minimum * 2400)
                    peer = vn.offer(ceiling * 2400, minimum=minimum * 2400)
                    local['mask'] = peer['mask'] = 1 << (rate_index - 1)
                    if minimum <= rate_index <= ceiling:
                        result = vn.resolve(local, peer, role, baud, baud)
                        self.assertEqual((result.tx.rate, result.rx.rate),
                                         (rate_index * 2400, rate_index * 2400),
                                         (symbol_index, rate_index, role))
                    else:
                        with self.subTest(symbol_index=symbol_index,
                                          rate_index=rate_index, role=role), \
                                self.assertRaises(ValueError):
                            vn.resolve(local, peer, role, baud, baud)

    def test_directional_maxima_map_to_tx_rx_for_both_roles(self):
        local, peer = vn.offer(33600), vn.offer(33600)
        local.update(asym=1, max_c2a=3, max_a2c=5)
        peer.update(asym=1, max_c2a=4, max_a2c=6)
        for role, expected in (('call', (7200, 12000)),
                               ('answer', (12000, 7200))):
            result = vn.resolve(local, peer, role, SYMBOL_RATES[5], SYMBOL_RATES[5])
            self.assertEqual((result.tx.rate, result.rx.rate), expected)
    def test_peer_fields_command_our_transmitter(self):
        n=self.resolve(trellis=2,shaping=1,nonlinear=1,precoder=[1,2,3,4,5,6])
        self.assertEqual(n.tx.trellis,2); self.assertEqual(n.rx.trellis,0)
        self.assertTrue(n.tx.shaping); self.assertFalse(n.rx.shaping)

    def test_profile_fields_are_directional_for_every_supported_selection(self):
        own_precoder = [123, -45, 67, -89, 101, -112]
        peer_precoder = [-321, 54, -76, 98, -109, 210]
        for role in ('call', 'answer'):
            for own_shaping, own_nonlinear, peer_trellis, peer_shaping, peer_nonlinear in itertools.product(
                    (0, 1), (0, 1), (0, 1, 2), (0, 1), (0, 1)):
                local = vn.offer(33600, receive={
                    'trellis': 0, 'shaping': own_shaping, 'nonlinear': own_nonlinear,
                    'precoder': own_precoder})
                peer = vn.offer(33600)
                peer.update(trellis=peer_trellis, shaping=peer_shaping,
                            nonlinear=peer_nonlinear, precoder=peer_precoder)
                n = vn.resolve(local, peer, role, SYMBOL_RATES[5], SYMBOL_RATES[5])
                self.assertEqual((n.tx.trellis, n.tx.shaping, n.tx.nonlinear,
                                  n.tx.precoder),
                                 (peer_trellis, bool(peer_shaping), bool(peer_nonlinear),
                                  tuple(peer_precoder)), role)
                self.assertEqual((n.rx.trellis, n.rx.shaping, n.rx.nonlinear,
                                  n.rx.precoder),
                                 (0, bool(own_shaping), bool(own_nonlinear),
                                  tuple(own_precoder)), role)

    def test_mp_encoder_leaves_reserved_rate_mask_bit_zero(self):
        for mp_type in (0, 1):
            bits = v34_mp.build_mp(mp_type, max_c2a=14, max_a2c=14, mask=0x3fff)
            self.assertEqual(len(bits), 88 if mp_type == 0 else 188)
            self.assertEqual(bits[49], 0)  # Table 20/21: reserved bit 49.
            decoded = v34_mp.parse_mp(bits)
            self.assertEqual(decoded['mask'], 0x3fff)
        for rate_index in range(14):
            bits = v34_mp.build_mp(mask=1 << rate_index)
            self.assertEqual(bits[35 + rate_index], 1)
            self.assertEqual(sum(bits[35:49]), 1)

    def test_mp_encoder_rejects_invalid_or_reserved_selections(self):
        cases = (
            {'mp_type': 2}, {'max_c2a': 0}, {'max_a2c': 15},
            {'trellis': 3}, {'aux': 2}, {'nonlinear': 2},
            {'shaping': 2}, {'ack': 2}, {'asym': 2},
            {'mask': 1 << 14}, {'mask': -1},
            {'mp_type': 1, 'precoder': [0] * 5},
            {'mp_type': 1, 'precoder': [0] * 5 + [32768]},
        )
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                v34_mp.build_mp(**fields)
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
