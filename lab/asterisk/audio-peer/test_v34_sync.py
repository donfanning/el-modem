"""Synchronization checks against analytical signals and injected clock truth."""
import unittest
from types import SimpleNamespace
import numpy as np
from v34_rx import SymbolEq,interp
from v34_stream_impair import StreamImpairment


class SynchronizationTests(unittest.TestCase):
    def eq(self):
        n=np.arange(4096);z=np.exp(1j*(.12*n+.00004*n*n))
        r=SimpleNamespace(_span=12,spl=2.5,lag=50,LEAD=8,delay=0,taps=np.ones(24,complex),_z=z,baud=3200,fc=1900)
        return SymbolEq(r,40)

    def test_fractional_nco_and_slips_match_direct_sampling(self):
        for dt in (.01,.25,-.33,1.25,-1.25):
            eq=self.eq();eq.window(eq.k);eq.dt=dt
            expected=interp(eq.R._z,eq.R.lag+(eq.R.LEAD+eq.k)*eq.R.spl+dt+eq._offs)
            np.testing.assert_allclose(eq.window(eq.k),expected,atol=1e-12)

    def test_frozen_feedback_preserves_learned_clock_velocity(self):
        eq=self.eq();eq.adapt=False;eq.vel=.002;eq.freq=.003
        for _ in range(100):eq.step(decide=lambda y:y)
        self.assertAlmostEqual(eq.dt,.2);self.assertAlmostEqual(eq.phase,.3)
        self.assertAlmostEqual(eq.timing_snapshot()['samples_per_symbol'],2.502)

    def test_data_handoff_preserves_position_and_independent_oscillators(self):
        for precoded in (False,True):
            eq=SymbolEq(self.eq().R,40,timing_mode='data-gradient')
            eq.dt=.25;eq.vel=.0002;eq.phase=.1;eq.freq=.003
            before=eq.timing_snapshot();window=eq.window(eq.k).copy()
            eq._gradient_rows=[(1j,1.,.1)]
            old_gains=(eq.kp,eq.ki);eq.begin_data_tracking(precoded)
            after=eq.timing_snapshot()
            for key in ('sample_position','samples_per_symbol','carrier_phase','carrier_radians_per_symbol'):
                self.assertEqual(before[key],after[key])
            np.testing.assert_array_equal(eq.window(eq.k),window)
            self.assertEqual(eq._gradient_rows,[])
            if precoded:self.assertEqual((eq.kp,eq.ki),old_gains)

    def test_refined_tap_centroid_is_not_mistaken_for_clock_error_at_data_handoff(self):
        eq=self.eq();eq.h=np.zeros(eq.L,complex);eq.h[eq.span]=1;eq.h[eq.span+1]=.5j
        eq.c0=-.4;eq.cf=.2;eq.ktp=.003;eq.kti=5e-8
        eq.dt=.25;eq.vel=.0002;eq.phase=.1;eq.freq=.003
        before=(eq.dt,eq.vel,eq.phase,eq.freq);taps=eq.h.copy()
        eq.begin_data_tracking(precoded=True)
        self.assertAlmostEqual(eq.c0,.2);self.assertEqual(eq.cf,0.)
        eq._timing()
        self.assertEqual((eq.dt,eq.vel,eq.phase,eq.freq),before)
        np.testing.assert_array_equal(eq.h,taps)
        # A subsequent genuine centroid movement must still drive recovery.
        eq.h=np.zeros(eq.L,complex);eq.h[eq.span+1]=1;eq.cf=0;eq.calpha=1
        eq._timing();self.assertLess(eq.dt,before[0]);self.assertLess(eq.vel,before[1])

    def test_clock_warp_is_chunk_invariant_and_matches_analytic_carrier(self):
        n=np.arange(12000);x=9000*np.cos(2*np.pi*1100*n/8000)
        for ppm in (-500,-100,100,500):
            outputs=[]
            for chunk in (80,160,320):
                stream=StreamImpairment(clock_ppm=ppm)
                outputs.append(np.concatenate([stream.process(x[k:k+chunk]) for k in range(0,len(x),chunk)]))
            for output in outputs[1:]:np.testing.assert_array_equal(output,outputs[0])
            pos=(n-1024)/(1+ppm*1e-6)
            expected=9000*np.cos(2*np.pi*1100*pos/8000)
            self.assertLess(np.sqrt(np.mean((outputs[0][2000:]-expected[2000:])**2)),1.)

    def test_carrier_shift_is_independent_of_clock_and_causal(self):
        n=np.arange(12000);x=9000*np.cos(2*np.pi*1500*n/8000)
        stream=StreamImpairment(clock_ppm=100,carrier_hz=1.2)
        out=np.concatenate([stream.process(x[k:k+160]) for k in range(0,len(x),160)])
        pos=(n-1024)/(1+100e-6)-63
        expected=9000*np.cos(2*np.pi*1501.2*pos/8000)
        # Finite Hilbert FIR has small amplitude ripple; phase/frequency remain testable.
        self.assertLess(np.sqrt(np.mean((out[3000:]-expected[3000:])**2)),15.)

    def test_delayed_clock_activation_has_no_position_jump(self):
        for ppm in (-500,500):
            stream=StreamImpairment(clock_ppm=ppm,wander_samples=.25,active=False)
            stream.process(np.zeros(4000))
            before=float(stream.positions([stream.index])[0]);stream.activate()
            self.assertEqual(float(stream.positions([stream.index])[0]),before)
            n=stream.index+500
            expected=n-1024+500*(1/(1+ppm*1e-6)-1)-.25*np.sin(2*np.pi*500/16000)
            self.assertAlmostEqual(float(stream.positions([n])[0]),expected)
            stream.activate();self.assertEqual(stream.activation_sample,4000)

    def test_delayed_impairment_matches_analytic_clock_and_chunking(self):
        n=np.arange(12000);x=9000*np.cos(2*np.pi*1100*n/8000)
        outputs=[]
        for chunk in (80,160,320):
            stream=StreamImpairment(clock_ppm=500,active=False);parts=[]
            for k in range(0,len(x),chunk):
                if k==3200:stream.activate()
                parts.append(stream.process(x[k:k+chunk]))
            outputs.append(np.concatenate(parts))
        for y in outputs[1:]:np.testing.assert_array_equal(y,outputs[0])
        pos=n-1024+np.maximum(0,n-3200)*(1/(1+500e-6)-1)
        expected=9000*np.cos(2*np.pi*1100*pos/8000)
        self.assertLess(np.sqrt(np.mean((outputs[0][2000:]-expected[2000:])**2)),1.)

    def test_transport_lookahead_exhaustion_is_explicit(self):
        from v34_stream_impair import ImpairmentBufferError
        stream=StreamImpairment(clock_ppm=-500);stream.latency=64
        with self.assertRaises(ImpairmentBufferError):
            for _ in range(500):stream.process(np.zeros(160))

    def test_gradient_recovers_independent_clock_and_carrier_truth(self):
        from v34_rx import Phase3Rx
        from v34_tx import qam_modulate
        rng=np.random.default_rng(841)
        symbols=rng.choice([1+1j,1-1j,-1+1j,-1-1j],6200)
        for ppm,hz in ((0,0),(100,0),(-100,0),(500,1),(-500,-1),(0,1)):
            with self.subTest(clock_ppm=ppm,carrier_hz=hz):
                spl=2.5;true=spl*(1+ppm*1e-6)
                r=Phase3Rx(3200,1900);r.lag=8*(true-spl);r.delay=0
                r._z=qam_modulate(symbols,8000/true,0,1,baseband=True)
                r._z*=np.exp(2j*np.pi*hz*np.arange(len(r._z))/8000)
                r.train_known(symbols[:300],0)
                eq=SymbolEq(r,320,timing_mode='gradient');errors=[]
                for k in range(320,6000):
                    errors.append(abs(eq.step(desired=symbols[k])-symbols[k])**2)
                self.assertLess(abs(eq.vel/spl*1e6-ppm),10)
                self.assertLess(abs(eq.freq*3200/(2*np.pi)-hz),.01)
                self.assertLess(float(np.mean(errors[-500:])/2),.001)

    def test_extended_model_fits_clock_without_final_holdout(self):
        from test_v34_forward_audit import transmit
        from v34_tx import trn_symbols
        from v34_forward_audit import fit_pulse
        trn,_=trn_symbols(1000,'call',16);spl=8000/(24000/7);carrier=1959.183673469
        pcm=transmit(trn,40,spl*(1+100e-6),carrier+.3,2700,True)
        model,trace=fit_pulse(pcm,trn,40,spl,carrier,extended=True)
        self.assertLess(abs(model.clock_ppm-100),30)
        self.assertLess(abs(model.carrier_offset_hz-.3),.1)
        changed=pcm.copy();changed[trace['validation_samples'][-50:]]+=100
        other,_=fit_pulse(changed,trn,40,spl,carrier,extended=True)
        self.assertEqual(model.selection,other.selection)
        np.testing.assert_array_equal(model.pulse,other.pulse)


if __name__=='__main__':unittest.main()
