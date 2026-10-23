"""Ordinary detector native/reference path and metric parity, including drain."""
import unittest
import numpy as np
import v34_data as data
from v34_viterbi import TrellisDetector
from v34_trellis_native import njit


@unittest.skipIf(njit is None,'optional Numba backend unavailable')
class NativeTrellisTests(unittest.TestCase):
    def test_dynamic_erasure_hold_traceback_matches_python(self):
        rng=np.random.default_rng(221023)
        ref=TrellisDetector(128,8,15,depth=3,backend='python',trace=True,erasure_hold=True)
        native=TrellisDetector(128,8,15,depth=3,backend='numba',trace=True,erasure_hold=True)
        samples=rng.normal(0,8,300)+1j*rng.normal(0,8,300)
        erased=set(range(20,80))
        for i in range(0,len(samples),2):
            flag=i//2 in erased
            ref.feed(*samples[i:i+2],erasure=flag)
            native.feed(*samples[i:i+2],erasure=flag)
            self.assertEqual(ref.depth,native.depth)
            self.assertEqual(ref.decided,native.decided)
            np.testing.assert_allclose(native.metric,ref.metric,rtol=1e-11,atol=1e-10)
            np.testing.assert_array_equal(native.trace[-1]['prev'],ref.trace[-1]['prev'])
        ref.flush();native.flush()
        self.assertEqual(ref.decided,native.decided)
        self.assertEqual(2*len(native.decided),len(samples))

    def test_native_and_python_gap_erasure_metrics_match(self):
        rng=np.random.default_rng(221022)
        ref=TrellisDetector(128,8,15,depth=5,backend='python',trace=True)
        native=TrellisDetector(128,8,15,depth=5,backend='numba',trace=True)
        samples=rng.normal(0,8,80)+1j*rng.normal(0,8,80)
        erased={3,4,5,19}
        for i in range(0,len(samples),2):
            flag=i//2 in erased
            ref.feed(*samples[i:i+2],erasure=flag)
            native.feed(*samples[i:i+2],erasure=flag)
            self.assertEqual(ref.decided,native.decided)
            self.assertEqual(ref.x_hist,native.x_hist)
            np.testing.assert_allclose(native.metric,ref.metric,rtol=1e-11,atol=1e-10)
            np.testing.assert_array_equal(native.trace[-1]['prev'],ref.trace[-1]['prev'])
            np.testing.assert_allclose(native.trace[-1]['branch_cost'],ref.trace[-1]['branch_cost'],
                rtol=1e-13,atol=1e-12)
            self.assertEqual(native.trace[-1]['erasure'],flag)
        ref.flush();native.flush()
        self.assertEqual(ref.decided,native.decided)
        self.assertEqual(native.erased_intervals,len(erased))

    def test_linear_zero_displacement_matches_reference_at_ties(self):
        rng=np.random.default_rng(264336)
        for length in (16,128,1664):
            ref=TrellisDetector(length,8,15,depth=5,backend='python',trace=True)
            native=TrellisDetector(length,8,15,depth=5,backend='numba',trace=True)
            samples=np.r_[np.zeros(8,complex),
                          [2+2j,np.nextafter(2.,3.)+2j,2+np.nextafter(2.,1.)*1j,0j,
                           90+90j,-90+90j],
                          rng.normal(0,9,64)+1j*rng.normal(0,9,64)]
            for i in range(0,len(samples),2):
                for detector in (ref,native):detector.feed(*samples[i:i+2])
                self.assertEqual(native.decided,ref.decided)
                for key in ('prev','s0','s1','p0','U'):
                    np.testing.assert_array_equal(native.trace[-1][key],ref.trace[-1][key])
                np.testing.assert_allclose(native.metric,ref.metric,rtol=1e-11,atol=1e-10)
            ref.flush();native.flush()
            self.assertEqual(native.decided,ref.decided)

    def test_precoded_nonlinear_and_tied_paths_match_every_interval(self):
        rng=np.random.default_rng(340617)
        for nonlinear in (False,True):
            for length in (16,128,1664):
                options=dict(depth=5,precoder=[.25+.125j,-.1j,.0625],w=2,
                    nonlinear=nonlinear,energy=675.,x_hist=[1.5+1j,-1+1j,0j],
                    start_state=(1,0,1,0),m4_start=29,trace=True)
                ref=TrellisDetector(length,8,15,backend='python',**options)
                native=TrellisDetector(length,8,15,backend='numba',**options)
                samples=rng.normal(0,9,100)+1j*rng.normal(0,9,100)
                samples[:8]=0j
                for i in range(0,len(samples),2):
                    for d in (ref,native):d.feed(*samples[i:i+2])
                    self.assertEqual(native.decided,ref.decided)
                    self.assertEqual(native.x_hist,ref.x_hist)
                    np.testing.assert_allclose(native.metric,ref.metric,rtol=1e-11,atol=1e-10)
                    for key in ('prev','s0','s1','p0','U'):
                        np.testing.assert_array_equal(native.trace[-1][key],ref.trace[-1][key])
                    self.assertAlmostEqual(native.metric_offset,ref.metric_offset,places=8)
                ref.flush();native.flush()
                self.assertEqual(native.decided,ref.decided)
                self.assertEqual(native.hist,[])
                for a,b in zip(native.snapshots,ref.snapshots):
                    for key in ('interval','state_before','state_after','s0','s1','p0','U0'):
                        self.assertEqual(a[key],b[key])

    def test_independent_arbitrary_payload_matches_mapper(self):
        rng=np.random.default_rng(340618)
        for rate,baud in ((9600,3200),(24000,3429),(33600,3429)):
            tx=data.DataTx(rate,baud,True)
            samples=[]
            for _ in range(4):samples+=tx.data_frame(rng.integers(0,2,tx.bits_per_data_frame).tolist())
            d=TrellisDetector(4*tx.M*2**tx.q,tx.J,tx.P,backend='numba')
            for i in range(0,len(samples),2):d.feed(*samples[i:i+2])
            d.flush()
            self.assertEqual([v for pair in d.decided for v in pair],samples)

    def test_native_refuses_a_subclass_that_overrides_unprecode(self):
        # The njit kernel hardcodes the production unprecode and its x_hist
        # update; it cannot honour an override. Refusing beats silently decoding
        # a different detector than the caller asked for.
        class NoFeedback(TrellisDetector):
            def _unprecode(self,y):
                return y,0j,0j
        with self.assertRaises(ValueError):
            NoFeedback(16,8,15,backend='numba')
        # The production class is still accepted, and the Python path still runs
        # the override.
        TrellisDetector(16,8,15,backend='numba')
        d=NoFeedback(16,8,15,backend='python')
        d.feed(3.1+1.2j,5.3-3.1j)
        self.assertEqual(d.x_hist,[0j]*3)

    def test_native_refuses_any_overridden_detector_method(self):
        # The kernel bypasses every one of these, not only _unprecode.
        for name in ('_argbest','_distances','_layout','_best'):
            sub=type('Override',(TrellisDetector,),{name:lambda self,*a,**k:None})
            with self.assertRaises(ValueError) as ctx:
                sub(16,8,15,backend='numba')
            self.assertIn(name,str(ctx.exception))
            sub(16,8,15,backend='python')

    def test_suppressed_stream_snapshots_preserve_bounded_traceback(self):
        a=TrellisDetector(16,8,15,depth=3,backend='numba')
        b=TrellisDetector(16,8,15,depth=3,backend='numba');b.record_snapshots=False
        for i in range(100):
            for d in (a,b):d.feed(complex(i%7,i%3),complex(i%5,i%11))
            self.assertEqual(a.decided,b.decided)
            self.assertLessEqual(len(b.hist),3)
            self.assertEqual(b.snapshots,[])
        a.flush();b.flush();self.assertEqual(a.decided,b.decided)


if __name__=='__main__':unittest.main()
