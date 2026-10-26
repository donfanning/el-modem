"""Known RTP gaps must not train feedback; sample-clock and decoder time continue."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from v34_rx import Phase3Rx, SymbolEq
from v34_transport import (FeedbackValidity, feedback_support, gap_recovery_enabled,
                           note_concealed, trellis_erasure_enabled, trellis_hold_enabled)
from v34_quality import DataQuality, observe
from v34_streaming import update_data_baseband


class TransportFeedbackTests(unittest.TestCase):
    def test_combined_recovery_selects_trellis_only_for_zero_precoder(self):
        with patch.dict('os.environ',{'V34_GAP_RECOVERY':'1'}):
            self.assertTrue(gap_recovery_enabled())
            self.assertTrue(trellis_erasure_enabled(()))
            self.assertTrue(trellis_erasure_enabled((0j,0j,0j)))
            self.assertFalse(trellis_erasure_enabled((.25+0j,0j,0j)))
            self.assertFalse(trellis_hold_enabled((.25+0j,0j,0j)))
        with patch.dict('os.environ',{'V34_GAP_RECOVERY':'0'}):
            self.assertFalse(gap_recovery_enabled())
        with patch.dict('os.environ',{'V34_GAP_RECOVERY':'1','V34_GAP_TRELLIS_ERASURE':'1'}):
            self.assertFalse(trellis_erasure_enabled((.25+0j,0j,0j)))
        with patch.dict('os.environ',{'V34_GAP_RECOVERY':'yes'}):
            with self.assertRaisesRegex(ValueError,'must be 0 or 1'):
                gap_recovery_enabled()

    def test_combined_recovery_overrides_split_switches_as_one_policy(self):
        R=Phase3Rx(3200,1800);R.lag=0.;R.delay=0.;R._span=12
        R.taps=np.zeros(24,complex);R.taps[12]=1.;R._z=np.ones(2000,complex)
        R._pcm_source_offset=10000
        env={'V34_GAP_RECOVERY':'1','V34_GAP_FEEDBACK':'0',
             'V34_GAP_TRELLIS_ERASURE':'0','V34_GAP_TRELLIS_HOLD':'0',
             'V34_INPUT_POWER_GUARD':'0'}
        with patch.dict('os.environ',env):eq=SymbolEq(R,150)
        self.assertEqual(eq._transport_guard.mode,'hold')
        self.assertTrue(eq._transport_guard.hold_dsp)

    def test_trellis_erasure_opt_in_enables_observation_and_marks_tainted_windows(self):
        R=Phase3Rx(3200,1800);R.lag=0.;R.delay=0.;R._span=12
        R.taps=np.zeros(24,complex);R.taps[12]=1.;R._z=np.ones(2000,complex)
        R._pcm_source_offset=10000
        with patch.dict('os.environ',{'V34_GAP_FEEDBACK':'0','V34_GAP_TRELLIS_ERASURE':'1',
                                      'V34_INPUT_POWER_GUARD':'0'}):
            eq=SymbolEq(R,150)
        self.assertEqual(eq._transport_guard.mode,'observe')
        center=eq.R._pcm_source_offset+eq.R.lag+(eq.R.LEAD+eq.k+eq.R.delay)*eq.R.spl+eq.dt
        eq._transport_guard.add(int(center),160)
        self.assertTrue(eq.last_transport_tainted is False)
        eq.step(decide=lambda y:1+0j)
        self.assertTrue(eq.last_transport_tainted)
        eq._transport_guard.record_trellis_erasure(0)
        report=eq._transport_guard.report()
        self.assertEqual(report['trellis_erasure_intervals'],1)
        self.assertEqual(report['first_trellis_erasure'],0)

    def eq(self, mode='hold', timing='centroid'):
        R=Phase3Rx(3200,1800);R.lag=0.;R.delay=0.;R._span=12
        R.taps=np.zeros(24,complex);R.taps[12]=1.;R._z=np.ones(2000,complex)
        R._pcm_source_offset=10000
        with patch.dict('os.environ',{'V34_GAP_FEEDBACK':mode,'V34_INPUT_POWER_GUARD':'0'}):
            eq=SymbolEq(R,150,timing_mode=timing)
        eq.phase=.3;eq.freq=.001;eq.vel=.003;eq.dt=.037
        return eq

    def test_hold_preserves_learned_feedback_and_advances_ncos_without_skipping_symbols(self):
        for native in (False,True):
            for timing in ('centroid','gradient'):
                with self.subTest(native=native,timing=timing):
                    eq=self.eq(timing=timing);eq._native_feedback=native
                    center=eq.R._pcm_source_offset+eq.R.lag+(eq.R.LEAD+eq.k+eq.R.delay)*eq.R.spl+eq.dt
                    eq._transport_guard.add(int(center),160)
                    eq._gradient_rows=[('pre-gap',)];h=eq.h.copy();k=eq.k;n=eq._n;dt=eq.dt;phase=eq.phase
                    for _ in range(40):eq.step(decide=lambda y:1+.25j)
                    np.testing.assert_array_equal(eq.h,h)
                    self.assertEqual(eq.k,k+40);self.assertEqual(eq._n,n)
                    self.assertAlmostEqual(eq.phase,phase+40*.001)
                    self.assertAlmostEqual(eq.dt,dt+40*.003)
                    self.assertEqual(eq.freq,.001);self.assertEqual(eq.vel,.003)
                    self.assertEqual(eq._gradient_rows,[])
                    report=eq._transport_guard.report();self.assertEqual(report['held_symbols'],40)
                    self.assertEqual(report['tap_updates_on_taint'],0)
                    self.assertEqual(report['carrier_updates_on_taint'],0)
                    self.assertEqual(report['timing_velocity_updates_on_taint'],0)
                    # Resume only after the entire filter/equalizer support clears.
                    while eq._transport_guard.tainted(eq):eq.step(decide=lambda y:1+.25j)
                    before=eq.h.copy();eq.step(decide=lambda y:1+.25j)
                    self.assertGreater(float(np.linalg.norm(eq.h-before)),0.)

    def test_observation_is_exactly_passive_and_detects_updates_on_replacement_audio(self):
        eq=self.eq(mode='observe');plain=copy.deepcopy(eq);plain._transport_guard=None
        center=10000+(eq.R.LEAD+eq.k)*eq.R.spl
        eq._transport_guard.add(int(center),160)
        for _ in range(40):
            self.assertEqual(eq.step(decide=lambda y:1+.25j),plain.step(decide=lambda y:1+.25j))
            np.testing.assert_array_equal(eq.h,plain.h)
            for key in ('k','dt','vel','phase','freq','slips','_n'):self.assertEqual(getattr(eq,key),getattr(plain,key))
        r=eq._transport_guard.report();self.assertEqual(r['held_symbols'],0)
        self.assertGreater(r['tap_updates_on_taint'],0);self.assertGreater(r['carrier_updates_on_taint'],0)
        self.assertGreater(r['tap_delta_squared'],0.)

    def test_clean_and_default_paths_keep_bit_exact_feedback(self):
        for mode in ('hold','observe','0'):
            eq=self.eq(mode=mode);plain=copy.deepcopy(eq);plain._transport_guard=None
            for _ in range(40):
                self.assertEqual(eq.step(decide=lambda y:1+.25j),plain.step(decide=lambda y:1+.25j))
            np.testing.assert_array_equal(eq.h,plain.h)
            for key in ('dt','vel','phase','freq','slips'):self.assertEqual(getattr(eq,key),getattr(plain,key))
        self.assertIsNone(eq._transport_guard)

    def test_bounds_cover_actual_raw_pcm_influence_for_direct_cached_trajectory_and_trimmed_windows(self):
        raw=np.random.default_rng(4).normal(size=2000)
        for mode in ('direct','cache','trajectory','trim'):
            def build(samples):
                R=Phase3Rx(3428.5714285714284,1959.1836734693877);R.lag=2.7;R.delay=.3;R._span=12
                R.taps=np.zeros(24,complex);R.taps[12]=1;R._z=R.update_baseband(samples)
                if mode=='trim':R._z=R.update_baseband(samples[200:],origin=200)
                R._pcm_source_offset=10000
                with patch.dict('os.environ',{'V34_GAP_FEEDBACK':'hold','V34_INPUT_POWER_GUARD':'0'}):eq=SymbolEq(R,160)
                if mode=='direct':eq.dt=.037
                if mode=='trajectory':eq._traj=True;eq.vel=.004
                return eq,eq.window(eq.k).copy()
            eq,window=build(raw);left,right=feedback_support(eq)
            if mode=='trajectory':self.assertEqual(eq._tj_k,eq.k)
            # Raw impulses are independent of the support formula: reconstruct
            # the actual FIR and interpolation window and check its influence.
            indices=[left-10000-2,left-10000-1,left-10000,left-10000+1,
                     (left+right)//2-10000,right-10000-1,right-10000,right-10000+1,right-10000+2]
            affected=0
            for index in indices:
                changed=raw.copy();changed[index]+=1e6
                other,actual=build(changed)
                difference=float(np.max(abs(actual-window)))
                guard=FeedbackValidity('hold');guard.add(10000+index,1)
                if difference>1e-8:
                    affected+=1;self.assertTrue(guard.tainted(eq),(mode,index,left,right,difference))
                if not left<=10000+index<=right:self.assertLess(difference,1e-8,(mode,index,difference))
            self.assertGreater(affected,0,mode)

    def test_bounded_storage_coalesces_instead_of_losing_live_gaps(self):
        guard=FeedbackValidity('hold')
        for n in range(1000):guard.add(n*20,2)
        self.assertLessEqual(len(guard.ranges),guard.MAX_RANGES)
        for n in range(1000):self.assertTrue(any(a<=n*20<b for a,b in guard.ranges))
        guard.prune(21000);self.assertEqual(len(guard.ranges),0)
        for args in ((-1,1),(0,0),(0,-1),(1.5,2)):
            with self.assertRaises(ValueError):guard.add(*args)

    def test_core_registration_is_data_only_and_streaming_origin_is_absolute(self):
        eq=self.eq();core=SimpleNamespace(_rx_state=None,_p4rcv=SimpleNamespace(eq=eq))
        note_concealed(core,10300,160);self.assertEqual(eq._transport_guard.registered_gaps,0)
        core._rx_state={};note_concealed(core,10300,160)
        self.assertEqual(eq._transport_guard.registered_gaps,1)
        core._rx=np.ones(1000);core.rx_t=10500;core._rx_origin=9500
        R=Phase3Rx(3200,1800);R.update_baseband(core._rx,origin=0)   # production: the receiver already consumed PCM before the buffer was trimmed
        update_data_baseband(core,R,9000)
        self.assertEqual(R._pcm_source_offset,9000);self.assertEqual(R._bb_origin,500)
        self.assertEqual(len(R._z),1000)

    def test_quality_discards_the_whole_tainted_batch_only_in_hold_mode(self):
        for mode in ('observe','hold'):
            eq=self.eq(mode);detector=SimpleNamespace(m4=0,metric_offset=0.,h=[0]*3,pts=[1,2])
            profile=SimpleNamespace(rate=33600,nonlinear=False)
            quality=DataQuality(detector,profile,3200,min_symbols=4)
            eq._transport_guard.discard_marks=1 if mode=='hold' else 0
            detector.m4=2;detector.metric_offset=4.
            observe(None,{'quality':quality},eq)
            self.assertEqual(quality.discarded_symbols,4 if mode=='hold' else 0)
            self.assertEqual(quality.window_count,0 if mode=='hold' else 1)

    def test_split_modes_separate_dsp_hold_from_quality_discard(self):
        flags={m:(FeedbackValidity(m).hold_dsp,FeedbackValidity(m).discard_quality)
               for m in ('observe','hold','hold-dsp','hold-quality')}
        self.assertEqual(flags,{'observe':(False,False),'hold':(True,True),'hold-dsp':(True,False),'hold-quality':(False,True)})
        for mode in ('hold-dsp','hold-quality'):
            eq=self.eq(mode);before=(eq.h.copy(),eq.phase,eq.freq,eq.vel)
            g=eq._transport_guard;g.record(before,eq,g.hold_dsp)
            self.assertEqual((g.held_symbols,g.discard_marks),(1,0) if mode=='hold-dsp' else (0,1))

    def test_carrier_and_timing_holds_are_independent(self):
        flags={m:(FeedbackValidity(m).hold_carrier,FeedbackValidity(m).hold_timing)
               for m in ('observe','hold','hold-dsp','hold-quality','hold-carrier','hold-timing')}
        self.assertEqual(flags,{'observe':(False,False),'hold':(True,True),'hold-dsp':(True,True),
            'hold-quality':(False,False),'hold-carrier':(True,False),'hold-timing':(False,True)})
        # on the same tainted window the partial holds freeze only their own loop
        def run(mode):
            eq=self.eq(mode);center=10000+(eq.R.LEAD+eq.k)*eq.R.spl
            eq._transport_guard.add(int(center),160)
            h=eq.h.copy();n=eq._n
            for _ in range(40):eq.step(decide=lambda y:1+.25j)
            return eq,h,n
        eq,h,n=run('hold-carrier')
        np.testing.assert_array_equal(eq.h,h);self.assertEqual(eq.freq,.001)       # taps + carrier loop frozen
        self.assertGreater(eq._n,n)                                              # timing recovery still ran
        eq,h,n=run('hold-timing')
        self.assertEqual(eq._n,n)                                                # timing recovery frozen
        self.assertGreater(float(np.linalg.norm(eq.h-h)),0.)                     # taps still adapt


if __name__=='__main__':unittest.main()
