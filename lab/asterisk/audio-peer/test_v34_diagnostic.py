import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from v34_diagnostic import DspCapture
from v34_negotiation import DirectionParameters
from v34_receive import Receiver
from v34_rx import Phase3Rx, SymbolEq
from verify_v34_dsp_capture import verify


class DiagnosticTests(unittest.TestCase):
    def test_owned_source_support_matches_clipped_operator_and_survives_revision(self):
        from v34_diagnostic import owned_support
        for length in (1,4,100):
            for stride in (1,2):
                for start in (-200,-1,0,1,length-1,length,length+2):
                    for width in (0,1,24,120):
                        with self.subTest(length=length,stride=stride,start=start,width=width):
                            source=np.arange(length*stride,dtype=float)[::stride].astype(complex)
                            if stride==2:source=np.repeat(source,2)[::2]
                            expected=source[np.clip(np.arange(start,start+width),0,length-1)].copy()
                            got=owned_support(source,start,width)
                            np.testing.assert_array_equal(got,expected)
                            self.assertFalse(np.shares_memory(got,source))
                            source[:]=99+7j
                            np.testing.assert_array_equal(got,expected)

    def test_single_window_observation_preserves_fractional_feedback_state(self):
        from unittest.mock import patch
        for adapting in (False,True):
            with self.subTest(adapting=adapting), tempfile.TemporaryDirectory() as directory:
                rng=np.random.default_rng(412)
                R=Phase3Rx(3200,1800,'answer')
                R._z=rng.normal(size=2400)+1j*rng.normal(size=2400)
                R._span=12;R.taps=np.zeros(24,complex);R.taps[12]=1
                R.lag=50;R.delay=0
                eq=SymbolEq(R,20);eq.phase=.12;eq.freq=.0003
                eq.dt=.175;eq.vel=.0001;eq.adapt=adapting;eq._traj=False
                plain=copy.deepcopy(eq)
                cap=DspCapture(Path(directory)/'capture',SimpleNamespace(caller=
                    SimpleNamespace(_p3=R,info1c_sent_t=100,rx_t=1000,events=[]),events=[]))
                decisions=[]
                def decide(y):
                    decisions.append(y)
                    return 1+1j
                class Input:
                    def feed_pcm(self,pcm):
                        with cap.phase('phase3'):
                            return eq.step(decide=decide)
                with patch.object(eq,'window',wraps=eq.window) as window:
                    for _ in range(80):
                        expected=plain.step(decide=lambda y:1+1j)
                        self.assertEqual(cap.feed(b'',Input()),expected)
                        np.testing.assert_array_equal(eq.h,plain.h)
                        self.assertEqual(eq.timing_snapshot(),plain.timing_snapshot())
                    self.assertEqual(window.call_count,80)
                self.assertEqual(len(decisions),80)
                R._z+=33+7j
                cap.close()
                self.assertTrue(verify(cap.prefix)['valid'])

    def test_sampled_training_rows_preserve_live_trajectory_and_exact_evidence(self):
        from unittest.mock import patch
        R=Phase3Rx(3200,1800)
        R._z=np.random.default_rng(20261009).normal(size=4000).astype(complex)
        R._span=12;R.taps=np.zeros(24,complex);R.taps[12]=1
        R.lag=50;R.delay=0
        eq=SymbolEq(R,200);eq.dt=.175;eq.phase=.12;eq.freq=.0003
        eq._traj=False
        plain=copy.deepcopy(eq)
        with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'EL_MODEM_DSP_TRAINING_STRIDE':'4'}):
            cap=DspCapture(Path(directory)/'capture',SimpleNamespace(caller=SimpleNamespace(rx_t=0,events=[]),events=[]))
            class Input:
                def feed_pcm(self,pcm):
                    with cap.phase('phase3'):
                        return eq.step(decide=lambda y:1+1j)
            for _ in range(80):
                expected=plain.step(decide=lambda y:1+1j)
                self.assertEqual(cap.feed(b'',Input()),expected)
                np.testing.assert_array_equal(eq.h,plain.h)
                self.assertEqual(eq.timing_snapshot(),plain.timing_snapshot())
            self.assertEqual(cap.row_count,20)
            self.assertEqual(cap.training_rows_skipped,60)
            cap.close()
            self.assertEqual(json.loads(Path(str(cap.prefix)+'.json').read_text())['training_stride'],4)
            self.assertTrue(verify(cap.prefix)['valid'])

    def test_epoch_profiles_own_negotiation_and_mp_before_core_is_replaced(self):
        old=SimpleNamespace(rx_t=320,events=[],stage='data',
            negotiated={'tx':{'rate':33600}},_local_mp={'mask':16383},mp_ans={'max_a2c':14})
        new=SimpleNamespace(rx_t=0,events=[],stage='phase3')
        engine=SimpleNamespace(caller=old,events=[])
        with tempfile.TemporaryDirectory() as directory:
            cap=DspCapture(Path(directory)/'capture',engine)
            class Input:
                def feed_pcm(self,pcm):engine.caller=new
            cap.feed(b'',Input())
            old.negotiated['tx']['rate']=24000
            old._local_mp['mask']=255;old.mp_ans['max_a2c']=10
            self.assertEqual(cap.epochs[0]['negotiated']['tx']['rate'],33600)
            self.assertEqual(cap.epochs[0]['protocol']['_local_mp']['mask'],16383)
            self.assertEqual(cap.epochs[0]['protocol']['mp_ans']['max_a2c'],14)
            cap.close()

    def test_progress_uses_absolute_production_center_and_reports_buffered_b1(self):
        R=SimpleNamespace(lag=10.,LEAD=4,delay=2,spl=2.,role='answer')
        eq=SimpleNamespace(R=R,k=20,dt=.25,available=lambda:60)
        core=SimpleNamespace(rx_t=200,events=[],_p4R=R,_p4_x0=30,
            _p4rcv=SimpleNamespace(eq=eq,trn_end=40,b1_symbol=55))
        with tempfile.TemporaryDirectory() as directory:
            cap=DspCapture(Path(directory)/'capture',SimpleNamespace(caller=core,events=[]))
            got=cap.progress()
            self.assertEqual(got['next_symbol_pcm_center'],92.25)
            self.assertEqual(got['received_minus_next_symbol_center_samples'],107.75)
            self.assertEqual(got['available_minus_b1_symbols'],5)
            cap.close()

    def test_retrain_core_replacement_clears_b1_and_rearms_bounded_data_capture(self):
        old=SimpleNamespace(rx_t=320,events=[],stage='data',rx_qualification={'bit_errors':0})
        new=SimpleNamespace(rx_t=0,events=[],stage='phase3',_p3={'x0':123})
        engine=SimpleNamespace(caller=old,events=[])
        with tempfile.TemporaryDirectory() as directory:
            cap=DspCapture(Path(directory)/'capture',engine)
            cap.data_rows=1024
            cap.reference={'old':True};cap.b1_inputs={'old':True};cap.detector_trace={'old':True}
            class Input:
                def feed_pcm(self,pcm):
                    engine.caller=new
                    return 'retraining'
            self.assertEqual(cap.feed(bytes(320),Input()),'retraining')
            self.assertIs(cap.core,new)
            self.assertTrue(new._receive_trace)
            self.assertEqual(cap.epoch,1)
            self.assertEqual(cap.data_rows,0)
            self.assertTrue(cap.requires_exact_windows())
            self.assertEqual(cap.source_offset(object()),123)
            self.assertIsNone(cap.reference)
            self.assertIsNone(cap.b1_inputs)
            self.assertIsNone(cap.detector_trace)
            self.assertEqual(cap.epochs[0]['qualification'],{'bit_errors':0})
            self.assertEqual(cap.epochs[0]['initial_data_symbols'],1024)
            cap.close()
            meta=json.loads(Path(str(cap.prefix)+'.json').read_text())
            self.assertIsNone(meta['qualification'])
            self.assertEqual(meta['current_epoch'],1)
            self.assertEqual(len(meta['epochs']),2)

    def test_stage_timing_keeps_epoch_cursor_and_exception_without_replacing_result(self):
        from unittest.mock import patch
        core=SimpleNamespace(rx_t=100,events=[],stage='phase4')
        with tempfile.TemporaryDirectory() as directory:
            cap=DspCapture(Path(directory)/'capture',SimpleNamespace(caller=core,events=[]))
            with patch('v34_diagnostic.time.monotonic',side_effect=[10.,10.02]):
                with cap.measure('phase4_feed'):
                    core.rx_t=260
            with self.assertRaisesRegex(ValueError,'sentinel'):
                with patch('v34_diagnostic.time.monotonic',side_effect=[11.,11.01]):
                    with cap.measure('phase4_feed'):
                        raise ValueError('sentinel')
            record=cap.epochs[0]['timings']['phase4_feed']
            self.assertEqual(record['count'],2)
            self.assertAlmostEqual(record['total_seconds'],.03)
            self.assertEqual(record['worst']['core_rx_before'],100)
            self.assertEqual(record['worst']['core_rx_after'],260)
            self.assertFalse(record['worst']['failed'])
            cap.close()

    def test_trajectory_resumes_after_exact_data_capture_limit(self):
        from unittest.mock import patch
        rng = np.random.default_rng(93)
        R = Phase3Rx(3200, 1800, 'answer')
        R._z = rng.normal(size=2000)+1j*rng.normal(size=2000)
        R._span=12; R.taps=np.zeros(24, complex); R.taps[12]=1
        R.lag=50; R.delay=0
        eq=SymbolEq(R, 20); eq.vel=.001; eq.adapt=False; eq._traj=True
        core=SimpleNamespace(_rx_state=object(), rx_t=0, events=[])
        with tempfile.TemporaryDirectory() as directory, patch('v34_diagnostic.INITIAL_DATA_SYMBOLS', 2):
            cap=DspCapture(Path(directory)/'capture', SimpleNamespace(caller=core, events=[]))
            class Input:
                def feed_pcm(self, pcm):
                    return eq.step(decide=lambda y:y)
            cap.feed(b'', Input())
            self.assertFalse(eq._g_traj)
            self.assertTrue(eq._traj)
            cap.feed(b'', Input())
            self.assertFalse(eq._g_traj)
            cap.feed(b'', Input())
            self.assertTrue(eq._g_traj)
            self.assertEqual(cap.data_rows, 2)
            cap.close()
            self.assertTrue(verify(cap.prefix)['valid'])

    def test_cached_source_survives_revision_without_changing_dsp_or_callbacks(self):
        rng = np.random.default_rng(911)
        R = Phase3Rx(3200,1800,'answer')
        R._z = rng.normal(size=2000)+1j*rng.normal(size=2000)
        R._span=12; R.taps=np.zeros(24,complex); R.taps[12]=1
        R.lag=50; R.delay=0
        eq=SymbolEq(R,20); eq.phase=.12
        plain=copy.deepcopy(eq)
        control=[plain.step(decide=lambda y:1+1j) for _ in range(3)]
        callbacks=[]
        def decision(y):
            callbacks.append(y)
            return 1+1j
        # Caller phase-three state is a receiver object, unlike the answerer's dict.
        core=SimpleNamespace(_p3=R,info1c_sent_t=100,rx_t=1000,events=[])
        engine=SimpleNamespace(caller=core,events=[])
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'capture'
            cap=DspCapture(prefix,engine)
            self.assertEqual(cap.source_offset(R),100)
            class Input:
                def feed_pcm(self, pcm):
                    with cap.phase('phase4'):
                        return [eq.step(decide=decision) for _ in range(3)]
            measured=cap.feed(bytes(320),Input())
            np.testing.assert_array_equal(measured,control)
            np.testing.assert_array_equal(eq.h,plain.h)
            self.assertEqual(eq.timing_snapshot(),plain.timing_snapshot())
            self.assertEqual(len(callbacks),3)
            # Change all live baseband values without invalidating its length.
            # Production still uses the batch built before this change.
            R._z+=17+3j
            cap.feed(bytes(320),Input())
            cap.close()
            report=verify(prefix)
            self.assertTrue(report['valid'],report)
            self.assertLess(report['groups'][0]['window_max_error'],1e-12)
            data=np.load(str(prefix)+'.npz')
            self.assertEqual(len(data['eq0_observed']),6)
            self.assertTrue(np.all(data['eq0_adaptation'][:,2]==False))

    def test_b1_is_qualified_once_and_replays_identically(self):
        profile=DirectionParameters(24000,24000/7,1959.183673469)
        control=Receiver(profile,'answer')
        observed=control.linear*(1.03+.04j)
        control.qualify(observed)
        core=SimpleNamespace(rx_t=0,events=[])
        engine=SimpleNamespace(caller=core,events=[])
        with tempfile.TemporaryDirectory() as directory:
            cap=DspCapture(Path(directory)/'capture',engine)
            receiver=Receiver(profile,'answer')
            class Input:
                def feed_pcm(self, pcm):
                    return receiver.qualify(observed)
            cap.feed(b'',Input())
            self.assertEqual(receiver.bits,control.bits)
            self.assertEqual(len(cap.detector_trace['trace'])*2,len(observed))
            self.assertEqual(len(cap.detector_trace['snapshots'])*2,len(observed))
            core.rx_qualification = receiver.diagnostics
            cap.close()
            meta=json.loads(Path(str(cap.prefix)+'.json').read_text())
            self.assertTrue(meta['closed'])
            self.assertEqual(meta['qualification_remote_role'],'answer')
            # A qualifier can run without a tracking group; replay must use its
            # actual remote identity rather than infer it from a training loop.
            self.assertTrue(verify(cap.prefix)['valid'])

    def test_b1_snapshot_survives_detector_reuse_and_defers_json_conversion(self):
        from unittest.mock import patch
        from v34_diagnostic import json_value
        profile=DirectionParameters(24000,24000/7,1959.183673469)
        receiver=Receiver(profile,'answer')
        observed=receiver.linear*(1.03+.04j)
        core=SimpleNamespace(rx_t=0,events=[])
        with tempfile.TemporaryDirectory() as directory:
            cap=DspCapture(Path(directory)/'capture',SimpleNamespace(caller=core,events=[]))
            class Input:
                def feed_pcm(self, pcm):
                    return receiver.qualify(observed)
            with patch('v34_diagnostic.json_value',side_effect=AssertionError('conversion in receive')):
                cap.feed(b'',Input())
            detector=receiver.b1_detector
            expected=json_value(dict(trace=detector.trace,snapshots=detector.snapshots,
                                     decided=detector.decided))
            # Simulate buffer reuse plus consumption/replacement of containers.
            def mutate(value):
                if isinstance(value,np.ndarray):value.fill(123)
                elif isinstance(value,dict):
                    for child in value.values():mutate(child)
                    value.clear()
                elif isinstance(value,(list,tuple)):
                    for child in value:mutate(child)
                    if isinstance(value,list):value.clear()
            mutate(detector.trace);mutate(detector.snapshots);mutate(detector.decided)
            self.assertEqual(json_value(cap.detector_trace),expected)
            core.rx_qualification=receiver.diagnostics
            cap.close()
            saved=json.loads(Path(str(cap.prefix)+'.detector.json').read_text())
            self.assertEqual(saved,expected)
            self.assertTrue(verify(cap.prefix)['valid'])

    def test_direct_fractional_windows_include_baseband_origin(self):
        rng=np.random.default_rng(42)
        R=Phase3Rx(3200,1800,'answer')
        R._z=rng.normal(size=1000)+1j*rng.normal(size=1000)
        R._bb_origin=100; R._span=12; R.taps=np.ones(24)/24
        R.lag=100; R.delay=0
        eq=SymbolEq(R,30); eq.vel=.003; eq.adapt=False
        core=SimpleNamespace(_p4R=R,_p4_x0=20,rx_t=1000,events=[])
        with tempfile.TemporaryDirectory() as directory:
            cap=DspCapture(Path(directory)/'capture',SimpleNamespace(caller=core,events=[]))
            class Input:
                def feed_pcm(self, pcm):
                    with cap.phase('b1'):
                        return [eq.step(decide=lambda y:y) for _ in range(4)]
            cap.feed(bytes(320),Input());cap.close()
            self.assertTrue(verify(cap.prefix)['valid'])


if __name__=='__main__':
    unittest.main()
