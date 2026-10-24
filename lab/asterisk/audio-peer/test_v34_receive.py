"""Receive-domain and known-bit tests; expected samples include independent equations."""
import random
import unittest
from itertools import product
import numpy as np
import v34_data as vd
import v34_datarx as dr
import v34_negotiation as vn
import v34_receive as rx

COEFF=(835,259,-842,-106,775,-157)

class ReceiveTests(unittest.TestCase):
    def test_clean_nonlinear_precoded_diagnostics_compare_mapper_domain(self):
        p=vn.DirectionParameters(33600,3429,1959,shaping=True,nonlinear=True,precoder=COEFF)
        r=rx.Receiver(p,'call')
        r.qualify(.7*np.exp(.2j)*rx.projection(r.linear,500),trace=True)
        self.assertTrue(r.diagnostics['qualified'])
        self.assertEqual(r.diagnostics['mapper_symbol_errors'],0)
        self.assertEqual(r.diagnostics['first_decoded_point_deltas'],[])
        self.assertIsNone(r.diagnostics['first_divergence'])
        replay=r.detector(trace=True)
        for a,b in (r.b1_gain*rx.projection(r.linear,500)/r.b1_gain).reshape(-1,2):
            replay.feed(a,b)
        replay.flush()
        self.assertEqual(replay.decided,r.b1_detector.decided)

    def test_error_indices_include_late_bits_beyond_symbol_count(self):
        p=vn.DirectionParameters(33600,3429,1959)
        r=rx.Receiver(p,'answer');wave=r.linear.copy();wave[len(wave)//2:]*=-1
        r.qualify(wave)
        expected=[i for i,(a,b) in enumerate(zip(r.bits,r.expected)) if a!=b]
        self.assertEqual(r.diagnostics['error_bit_indices'],expected)
        self.assertTrue(any(i>=len(wave) for i in expected))

    def test_search_axes_are_independent(self):
        self.assertEqual(rx.SearchPolicy(),rx.SearchPolicy('standard',False,((0,0,0,0),),(0,)))
        rate=rx.SearchPolicy.diagnostic('rate')
        self.assertTrue(rate.rate_search)
        self.assertEqual(rate.initial_states,((0,0,0,0),))
        self.assertEqual(rate.boundary_offsets,(0,))
        reset=rx.SearchPolicy.diagnostic('reset')
        self.assertFalse(reset.rate_search)
        self.assertEqual(len(reset.initial_states),16)
        self.assertEqual(reset.boundary_offsets,(0,))
        boundary=rx.SearchPolicy.diagnostic('boundary')
        self.assertFalse(boundary.rate_search)
        self.assertEqual(boundary.initial_states,((0,0,0,0),))
        self.assertEqual(boundary.boundary_offsets,tuple(range(-8,9)))
        wide=rx.SearchPolicy.diagnostic('combined',boundary_radius=128)
        self.assertEqual(wide.boundary_offsets,tuple(range(-128,129)))
        with self.assertRaises(ValueError):rx.SearchPolicy(rate_search=True)
        with self.assertRaises(ValueError):rx.SearchPolicy.diagnostic('unknown')
        with self.assertRaises(ValueError):rx.SearchPolicy.diagnostic(boundary_radius=129)
        with self.assertRaises(ValueError):rx.SearchPolicy('standard',False,((0,0,0,0),),(1,))

    def test_projection_inverse(self):
        # Independent scalar evaluation of 9-33..35 (not production forward code).
        for energy in (2.,100.,1000.):
            for z in (0j,1+3j,-31+7j,90-60j):
                t=.3125*(z.real*z.real+z.imag*z.imag)/energy
                wave=z*(1+t/6+t*t/120)
                self.assertAlmostEqual(abs(rx.inverse_projection(wave,energy)-z),0,places=9)
        for e in (0,-1,float('nan')):
            with self.assertRaises(ValueError):rx.inverse_projection(1j,e)

    def test_quantization_ties_and_modulo(self):
        for a,b in ((.5,0),(1.5,1),(-.5,0),(-1.5,-1)):
            self.assertEqual(vd._round_half_to_smaller(a,1),b)
        self.assertEqual(vd.modulo_bit(2+0j,0j),1)
        self.assertEqual(vd.modulo_bit(2+2j,0j),0)

    def test_independent_precoder_pair(self):
        # Hand-selected points and history: p0=.75+.5j,c0=0;
        # x0=.25+.5j. Then p1=.125+.25j,c1=0,x1=-1.125+.75j.
        det=__import__('v34_viterbi').TrellisDetector(16,8,12,precoder=[.5+0j,0j,0j],x_hist=[1.5+1j,0j,0j])
        u,c,_=det._unprecode(.25+.5j)
        self.assertEqual((u,c),(1+1j,0j))
        u,c,_=det._unprecode(-1.125+.75j)
        self.assertEqual((u,c),(-1+1j,0j))

    def test_profiles_b1_and_following_frames(self):
        for baud in (2400,2743,2800,3000,3200,3429):
            rate=9600 if baud!=3429 else 33600
            for shaping,nonlinear in product((False,True),repeat=2):
                for role in ('call','answer'):
                    with self.subTest(baud=baud,shaping=shaping,nonlinear=nonlinear,role=role):
                        p=vn.DirectionParameters(rate,baud,1800,shaping=shaping,nonlinear=nonlinear,precoder=COEFF)
                        r=rx.Receiver(p,role)
                        # Own TX is used for continuity; independent modem gates qualify interoperability.
                        tx=vd.DataTx(rate,baud,shaping,precoder=vd.precoder_from_mp(COEFF),nonlinear=nonlinear)
                        b1=np.array(tx.data_frame(r.expected))
                        gain=.65*np.exp(.24j)
                        got,_=r.qualify(gain*b1)
                        self.assertTrue(r.diagnostics['qualified'],r.diagnostics)
                        self.assertLess(r.diagnostics['held_out_nmse'],1e-12)
                        rng=random.Random(25)
                        bits=[];sy=[]
                        for _ in range(3):
                            b=[rng.randrange(2) for _ in range(tx.bits_per_data_frame)]
                            bits.extend(b);sy.extend(tx.data_frame(b))
                        for a,b in np.array(sy).reshape(-1,2):r.vit.feed(a,b)
                        r.vit.flush()
                        points=[x for pair in r.vit.decided for x in pair]
                        out=[]
                        for i in range(0,len(points),8):out.extend(r.drx.decode_frame(points[i:i+8])[0])
                        self.assertEqual(out,bits)

    def test_foreign_normalization_and_noisy_b1(self):
        for seed in range(3):
            for role in ('call','answer'):
                profile=vn.DirectionParameters(33600,3429,1959,shaping=True,nonlinear=True,precoder=COEFF)
                receiver=rx.Receiver(profile,role)
                # Choose a normalization distinct from our TX's estimated power.
                energy=receiver.tx._linear_power*1.8
                gain=.71*np.exp(-.32j)
                rng=np.random.default_rng(seed)
                y=gain*(rx.projection(receiver.linear,energy)+rng.normal(0,.06,len(receiver.linear))+1j*rng.normal(0,.06,len(receiver.linear)))
                receiver.qualify(y)
                self.assertTrue(receiver.diagnostics['qualified'],receiver.diagnostics)
                self.assertLess(abs(receiver.energy/energy-1),.08)

    def test_recorded_initial_state_is_explicit(self):
        profile=vn.DirectionParameters(24000,3429,1959,shaping=True,nonlinear=True,precoder=COEFF)
        receiver=rx.Receiver(profile,'call',(0,1,0,0))
        receiver.qualify(rx.projection(receiver.linear,receiver.tx._linear_power))
        self.assertTrue(receiver.diagnostics['qualified'])
        self.assertFalse(receiver.diagnostics['standard_initialization'])

    def test_recorded_equalized_b1_fixtures(self):
        # These are recorded samples after the frozen equalizer, not generated TX.
        # Raw-audio startup/control regressions remain in test_v34_reference.
        import json
        from pathlib import Path
        cases=json.loads((Path(__file__).parent/'fixtures/v34-reference/b1_receive.json').read_text())
        for case in cases:
            with self.subTest(recording=case['recording'],role=case['role']):
                profile=vn.DirectionParameters(**case['profile'])
                receiver=rx.Receiver(profile,'answer' if case['role']=='call' else 'call',case['initial_trellis_state'])
                observed=np.array([complex(a,b) for a,b in case['observed']])
                receiver.qualify(observed)
                self.assertEqual(receiver.diagnostics['qualified'],case['expected_qualified'])
                self.assertTrue(np.isfinite(receiver.diagnostics['held_out_nmse']))

    def test_held_out_errors_are_not_hidden_by_expected_b1(self):
        p=vn.DirectionParameters(9600,3200,1800)
        r=rx.Receiver(p,'answer')
        wave=r.linear.copy();wave[len(wave)//2:]*=-1
        r.qualify(wave)
        self.assertFalse(r.diagnostics['qualified'])
        self.assertGreater(r.diagnostics['bit_errors'],0)

    def test_unsupported_profile(self):
        with self.assertRaisesRegex(ValueError,'16-state'):
            rx.Receiver(vn.DirectionParameters(9600,3200,1800,trellis=2),'answer')

if __name__=='__main__':unittest.main()
